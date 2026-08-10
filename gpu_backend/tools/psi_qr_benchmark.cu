#include "psi_qr_snapshot.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char* where) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(where) + ": cuBLAS status " + std::to_string(status));
    }
}

void check_solver(cusolverStatus_t status, const char* where) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(where) + ": cuSOLVER status " + std::to_string(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            cudaFree(data_);
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    void allocate(std::size_t count) {
        cudaFree(data_);
        data_ = nullptr;
        count_ = count;
        if (count > 0) check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
    }
    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return count_; }
private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

struct Snapshot {
    sgpu::PsiQrSnapshotHeader header{};
    std::vector<float> matrix;
    std::vector<float> rhs;
    std::vector<double> scale;
};

Snapshot read_snapshot(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open snapshot: " + path);
    Snapshot snapshot;
    stream.read(reinterpret_cast<char*>(&snapshot.header), sizeof(snapshot.header));
    if (!stream || std::memcmp(snapshot.header.magic, sgpu::kPsiQrSnapshotMagic, 8) != 0 ||
        snapshot.header.version != sgpu::kPsiQrSnapshotVersion ||
        snapshot.header.scalar_type != sgpu::kPsiQrScalarFloat32 ||
        snapshot.header.layout != sgpu::kPsiQrLayoutColumnMajor) {
        throw std::runtime_error("invalid psi QR snapshot header");
    }
    const std::uint64_t expected_matrix = snapshot.header.rows * snapshot.header.cols * sizeof(float);
    const std::uint64_t expected_rhs = snapshot.header.rows * sizeof(float);
    const std::uint64_t expected_scale = snapshot.header.cols * sizeof(double);
    if (snapshot.header.matrix_bytes != expected_matrix || snapshot.header.rhs_bytes != expected_rhs ||
        snapshot.header.scale_bytes != expected_scale || snapshot.header.data_rows > snapshot.header.rows) {
        throw std::runtime_error("inconsistent psi QR snapshot dimensions");
    }
    snapshot.matrix.resize(static_cast<std::size_t>(snapshot.header.rows * snapshot.header.cols));
    snapshot.rhs.resize(static_cast<std::size_t>(snapshot.header.rows));
    snapshot.scale.resize(static_cast<std::size_t>(snapshot.header.cols));
    stream.read(reinterpret_cast<char*>(snapshot.matrix.data()), static_cast<std::streamsize>(snapshot.header.matrix_bytes));
    stream.read(reinterpret_cast<char*>(snapshot.rhs.data()), static_cast<std::streamsize>(snapshot.header.rhs_bytes));
    stream.read(reinterpret_cast<char*>(snapshot.scale.data()), static_cast<std::streamsize>(snapshot.header.scale_bytes));
    if (!stream) throw std::runtime_error("truncated psi QR snapshot");
    char trailing = 0;
    if (stream.read(&trailing, 1)) throw std::runtime_error("psi QR snapshot contains trailing bytes");
    return snapshot;
}

double wall_ms(const std::function<void()>& operation) {
    check_cuda(cudaDeviceSynchronize(), "pre-timing synchronize");
    const auto started = Clock::now();
    operation();
    check_cuda(cudaDeviceSynchronize(), "post-timing synchronize");
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double position = q * static_cast<double>(values.size() - 1);
    const std::size_t low = static_cast<std::size_t>(std::floor(position));
    const std::size_t high = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(low);
    return values[low] * (1.0 - fraction) + values[high] * fraction;
}

struct Timings {
    std::vector<double> restore_ms;
    std::vector<double> factor_ms;
    std::vector<double> apply_ms;
    std::vector<double> triangular_ms;
    std::vector<double> solve_ms;
};

struct ErrorMetrics {
    double coefficient_relative = 0.0;
    double augmented_residual_relative = 0.0;
    double physical_residual_relative = 0.0;
    double backward_error = 0.0;
    double normal_residual_relative = 0.0;
};

struct Context {
    int m = 0;
    int n = 0;
    int data_rows = 0;
    DeviceBuffer<float> matrix;
    DeviceBuffer<float> rhs;
    DeviceBuffer<float> reference_solution;
    cublasHandle_t blas = nullptr;
    float matrix_norm = 0.0f;
    float rhs_norm = 0.0f;

    Context(const Snapshot& snapshot)
        : m(static_cast<int>(snapshot.header.rows)),
          n(static_cast<int>(snapshot.header.cols)),
          data_rows(static_cast<int>(snapshot.header.data_rows)),
          matrix(static_cast<std::size_t>(m) * n), rhs(m), reference_solution(n) {
        check_cublas(cublasCreate(&blas), "cublasCreate");
        check_cuda(cudaMemcpy(matrix.data(), snapshot.matrix.data(), matrix.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "copy snapshot matrix");
        check_cuda(cudaMemcpy(rhs.data(), snapshot.rhs.data(), rhs.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "copy snapshot rhs");
        check_cublas(cublasSnrm2(blas, static_cast<int>(matrix.size()), matrix.data(), 1, &matrix_norm), "matrix norm");
        check_cublas(cublasSnrm2(blas, m, rhs.data(), 1, &rhs_norm), "rhs norm");
        check_cuda(cudaDeviceSynchronize(), "snapshot upload synchronize");
    }
    ~Context() { cublasDestroy(blas); }
};

void verify_info(const DeviceBuffer<int>& info, const char* where) {
    int host_info = 0;
    check_cuda(cudaMemcpy(&host_info, info.data(), sizeof(int), cudaMemcpyDeviceToHost), where);
    if (host_info != 0) throw std::runtime_error(std::string(where) + ": info=" + std::to_string(host_info));
}

class HouseholderSolver {
public:
    HouseholderSolver(Context& context, bool generic)
        : context_(context), generic_(generic), matrix_(context.matrix.size()), rhs_(context.m),
          tau_(context.n), info_(1) {
        check_solver(cusolverDnCreate(&solver_), "cusolverDnCreate");
        int legacy_geqrf = 0;
        int legacy_ormqr = 0;
        check_solver(cusolverDnSgeqrf_bufferSize(solver_, context_.m, context_.n, matrix_.data(), context_.m,
                                                 &legacy_geqrf), "Sgeqrf_bufferSize");
        check_solver(cusolverDnSormqr_bufferSize(
            solver_, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, context_.m, 1, context_.n,
            matrix_.data(), context_.m, tau_.data(), rhs_.data(), context_.m, &legacy_ormqr),
            "Sormqr_bufferSize");
        legacy_work_.allocate(static_cast<std::size_t>(std::max(legacy_geqrf, legacy_ormqr)));
        if (generic_) {
            check_solver(cusolverDnCreateParams(&params_), "cusolverDnCreateParams");
            check_solver(cusolverDnXgeqrf_bufferSize(
                solver_, params_, context_.m, context_.n, CUDA_R_32F, matrix_.data(), context_.m,
                CUDA_R_32F, tau_.data(), CUDA_R_32F, &generic_device_bytes_, &generic_host_bytes_),
                "Xgeqrf_bufferSize");
            generic_device_.allocate(generic_device_bytes_);
            generic_host_.resize(generic_host_bytes_);
        }
    }

    ~HouseholderSolver() {
        if (params_) cusolverDnDestroyParams(params_);
        cusolverDnDestroy(solver_);
    }

    void run_once(Timings* timings) {
        const double restore = wall_ms([&] {
            check_cuda(cudaMemcpy(matrix_.data(), context_.matrix.data(), matrix_.size() * sizeof(float), cudaMemcpyDeviceToDevice),
                       "restore matrix");
            check_cuda(cudaMemcpy(rhs_.data(), context_.rhs.data(), rhs_.size() * sizeof(float), cudaMemcpyDeviceToDevice),
                       "restore rhs");
        });
        const double factor = wall_ms([&] {
            if (generic_) {
                check_solver(cusolverDnXgeqrf(
                    solver_, params_, context_.m, context_.n, CUDA_R_32F, matrix_.data(), context_.m,
                    CUDA_R_32F, tau_.data(), CUDA_R_32F, generic_device_.data(), generic_device_bytes_,
                    generic_host_.data(), generic_host_bytes_, info_.data()), "Xgeqrf");
            } else {
                check_solver(cusolverDnSgeqrf(
                    solver_, context_.m, context_.n, matrix_.data(), context_.m, tau_.data(),
                    legacy_work_.data(), static_cast<int>(legacy_work_.size()), info_.data()), "Sgeqrf");
            }
        });
        verify_info(info_, "geqrf info");
        const double apply = wall_ms([&] {
            check_solver(cusolverDnSormqr(
                solver_, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, context_.m, 1, context_.n,
                matrix_.data(), context_.m, tau_.data(), rhs_.data(), context_.m,
                legacy_work_.data(), static_cast<int>(legacy_work_.size()), info_.data()), "Sormqr");
        });
        verify_info(info_, "ormqr info");
        const double triangular = wall_ms([&] {
            check_cublas(cublasStrsv(context_.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                     CUBLAS_DIAG_NON_UNIT, context_.n, matrix_.data(), context_.m,
                                     rhs_.data(), 1), "Strsv");
        });
        if (timings) {
            timings->restore_ms.push_back(restore);
            timings->factor_ms.push_back(factor);
            timings->apply_ms.push_back(apply);
            timings->triangular_ms.push_back(triangular);
            timings->solve_ms.push_back(factor + apply + triangular);
        }
    }

    const float* solution() const { return rhs_.data(); }

private:
    Context& context_;
    bool generic_ = false;
    cusolverDnHandle_t solver_ = nullptr;
    cusolverDnParams_t params_ = nullptr;
    DeviceBuffer<float> matrix_;
    DeviceBuffer<float> rhs_;
    DeviceBuffer<float> tau_;
    DeviceBuffer<int> info_;
    DeviceBuffer<float> legacy_work_;
    DeviceBuffer<unsigned char> generic_device_;
    std::vector<unsigned char> generic_host_;
    std::size_t generic_device_bytes_ = 0;
    std::size_t generic_host_bytes_ = 0;
};

__global__ void gather_upper_kernel(
    const float* input, int input_lda, float* output, int output_lda, int n
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) output[row + static_cast<std::size_t>(col) * output_lda] = row <= col
        ? input[row + static_cast<std::size_t>(col) * input_lda] : 0.0f;
}

class TsqrSolver {
public:
    TsqrSolver(Context& context, int blocks)
        : context_(context), blocks_(blocks), matrix_(context.matrix.size()), rhs_(context.m),
          stacked_(static_cast<std::size_t>(blocks) * context.n * context.n),
          stacked_rhs_(static_cast<std::size_t>(blocks) * context.n), upper_tau_(context.n), upper_info_(1) {
        if (blocks_ < 2 || context_.m / blocks_ < context_.n) throw std::runtime_error("invalid TSQR block count");
        offsets_.resize(blocks_ + 1);
        for (int block = 0; block <= blocks_; ++block) offsets_[block] = context_.m * block / blocks_;
        local_.resize(blocks_);
        for (int block = 0; block < blocks_; ++block) {
            auto& state = local_[block];
            state.rows = offsets_[block + 1] - offsets_[block];
            check_cuda(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking), "create TSQR stream");
            check_solver(cusolverDnCreate(&state.solver), "create TSQR solver");
            check_solver(cusolverDnSetStream(state.solver, state.stream), "set TSQR stream");
            state.tau.allocate(context_.n);
            state.info.allocate(1);
            int geqrf_work = 0;
            int ormqr_work = 0;
            float* block_matrix = matrix_.data() + offsets_[block];
            float* block_rhs = rhs_.data() + offsets_[block];
            check_solver(cusolverDnSgeqrf_bufferSize(
                state.solver, state.rows, context_.n, block_matrix, context_.m, &geqrf_work),
                "TSQR local geqrf buffer");
            check_solver(cusolverDnSormqr_bufferSize(
                state.solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, state.rows, 1, context_.n,
                block_matrix, context_.m, state.tau.data(), block_rhs, state.rows, &ormqr_work),
                "TSQR local ormqr buffer");
            state.work.allocate(static_cast<std::size_t>(std::max(geqrf_work, ormqr_work)));
        }
        check_solver(cusolverDnCreate(&upper_solver_), "create TSQR upper solver");
        int geqrf_work = 0;
        int ormqr_work = 0;
        const int upper_rows = blocks_ * context_.n;
        check_solver(cusolverDnSgeqrf_bufferSize(
            upper_solver_, upper_rows, context_.n, stacked_.data(), upper_rows, &geqrf_work),
            "TSQR upper geqrf buffer");
        check_solver(cusolverDnSormqr_bufferSize(
            upper_solver_, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, upper_rows, 1, context_.n,
            stacked_.data(), upper_rows, upper_tau_.data(), stacked_rhs_.data(), upper_rows,
            &ormqr_work), "TSQR upper ormqr buffer");
        upper_work_.allocate(static_cast<std::size_t>(std::max(geqrf_work, ormqr_work)));
    }

    ~TsqrSolver() {
        for (auto& state : local_) {
            if (state.solver) cusolverDnDestroy(state.solver);
            if (state.stream) cudaStreamDestroy(state.stream);
        }
        if (upper_solver_) cusolverDnDestroy(upper_solver_);
    }

    void run_once(Timings* timings) {
        const double restore = wall_ms([&] {
            check_cuda(cudaMemcpy(matrix_.data(), context_.matrix.data(), matrix_.size() * sizeof(float), cudaMemcpyDeviceToDevice),
                       "restore TSQR matrix");
            check_cuda(cudaMemcpy(rhs_.data(), context_.rhs.data(), rhs_.size() * sizeof(float), cudaMemcpyDeviceToDevice),
                       "restore TSQR rhs");
        });
        const double local_factor = wall_ms([&] {
            for (int block = 0; block < blocks_; ++block) {
                auto& state = local_[block];
                check_solver(cusolverDnSgeqrf(
                    state.solver, state.rows, context_.n, matrix_.data() + offsets_[block], context_.m,
                    state.tau.data(), state.work.data(), static_cast<int>(state.work.size()), state.info.data()),
                    "TSQR local geqrf");
            }
        });
        for (auto& state : local_) verify_info(state.info, "TSQR local geqrf info");
        const double local_apply = wall_ms([&] {
            for (int block = 0; block < blocks_; ++block) {
                auto& state = local_[block];
                check_solver(cusolverDnSormqr(
                    state.solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, state.rows, 1, context_.n,
                    matrix_.data() + offsets_[block], context_.m, state.tau.data(),
                    rhs_.data() + offsets_[block], state.rows, state.work.data(),
                    static_cast<int>(state.work.size()), state.info.data()), "TSQR local ormqr");
            }
        });
        for (auto& state : local_) verify_info(state.info, "TSQR local ormqr info");

        const double gather = wall_ms([&] {
            const dim3 block(16, 16);
            const dim3 grid((context_.n + block.x - 1) / block.x, (context_.n + block.y - 1) / block.y);
            const int upper_rows = blocks_ * context_.n;
            for (int index = 0; index < blocks_; ++index) {
                gather_upper_kernel<<<grid, block>>>(
                    matrix_.data() + offsets_[index], context_.m,
                    stacked_.data() + static_cast<std::size_t>(index) * context_.n,
                    upper_rows, context_.n);
                check_cuda(cudaMemcpyAsync(
                    stacked_rhs_.data() + static_cast<std::size_t>(index) * context_.n,
                    rhs_.data() + offsets_[index], static_cast<std::size_t>(context_.n) * sizeof(float),
                    cudaMemcpyDeviceToDevice), "gather TSQR rhs");
            }
            check_cuda(cudaGetLastError(), "gather TSQR upper triangles");
        });
        const int upper_rows = blocks_ * context_.n;
        const double upper_factor = wall_ms([&] {
            check_solver(cusolverDnSgeqrf(
                upper_solver_, upper_rows, context_.n, stacked_.data(), upper_rows, upper_tau_.data(),
                upper_work_.data(), static_cast<int>(upper_work_.size()), upper_info_.data()),
                "TSQR upper geqrf");
        });
        verify_info(upper_info_, "TSQR upper geqrf info");
        const double upper_apply = wall_ms([&] {
            check_solver(cusolverDnSormqr(
                upper_solver_, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, upper_rows, 1, context_.n,
                stacked_.data(), upper_rows, upper_tau_.data(), stacked_rhs_.data(), upper_rows,
                upper_work_.data(), static_cast<int>(upper_work_.size()), upper_info_.data()),
                "TSQR upper ormqr");
        });
        verify_info(upper_info_, "TSQR upper ormqr info");
        const double triangular = wall_ms([&] {
            check_cublas(cublasStrsv(context_.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                     CUBLAS_DIAG_NON_UNIT, context_.n, stacked_.data(), upper_rows,
                                     stacked_rhs_.data(), 1), "TSQR upper trsv");
        });
        if (timings) {
            timings->restore_ms.push_back(restore);
            timings->factor_ms.push_back(local_factor + gather + upper_factor);
            timings->apply_ms.push_back(local_apply + upper_apply);
            timings->triangular_ms.push_back(triangular);
            timings->solve_ms.push_back(local_factor + local_apply + gather + upper_factor + upper_apply + triangular);
        }
    }

    const float* solution() const { return stacked_rhs_.data(); }

private:
    struct LocalState {
        int rows = 0;
        cudaStream_t stream = nullptr;
        cusolverDnHandle_t solver = nullptr;
        DeviceBuffer<float> tau;
        DeviceBuffer<float> work;
        DeviceBuffer<int> info;
    };
    Context& context_;
    int blocks_ = 0;
    std::vector<int> offsets_;
    std::vector<LocalState> local_;
    DeviceBuffer<float> matrix_;
    DeviceBuffer<float> rhs_;
    DeviceBuffer<float> stacked_;
    DeviceBuffer<float> stacked_rhs_;
    DeviceBuffer<float> upper_tau_;
    DeviceBuffer<float> upper_work_;
    DeviceBuffer<int> upper_info_;
    cusolverDnHandle_t upper_solver_ = nullptr;
};

class NormalEquationSolver {
public:
    NormalEquationSolver(Context& context, bool cholesky_qr2, bool tf32)
        : context_(context), cholesky_qr2_(cholesky_qr2), tf32_(tf32),
          matrix_(cholesky_qr2 ? context.matrix.size() : 0),
          gram1_(static_cast<std::size_t>(context.n) * context.n),
          gram2_(cholesky_qr2 ? static_cast<std::size_t>(context.n) * context.n : 0),
          solution_(context.n), intermediate_(context.n), info_(1) {
        check_solver(cusolverDnCreate(&solver_), "normal equation cusolver create");
        check_cublas(cublasSetMathMode(context_.blas, tf32_ ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH),
                     "set normal equation math mode");
        int work1 = 0;
        check_solver(cusolverDnSpotrf_bufferSize(
            solver_, CUBLAS_FILL_MODE_UPPER, context_.n, gram1_.data(), context_.n, &work1),
            "normal equation potrf buffer");
        int work2 = 0;
        if (cholesky_qr2_) {
            check_solver(cusolverDnSpotrf_bufferSize(
                solver_, CUBLAS_FILL_MODE_UPPER, context_.n, gram2_.data(), context_.n, &work2),
                "CholeskyQR2 second potrf buffer");
        }
        work_.allocate(static_cast<std::size_t>(std::max(work1, work2)));
    }
    ~NormalEquationSolver() {
        cublasSetMathMode(context_.blas, CUBLAS_DEFAULT_MATH);
        cusolverDnDestroy(solver_);
    }

    void run_once(Timings* timings) {
        const float one = 1.0f;
        const float zero = 0.0f;
        double restore = 0.0;
        if (cholesky_qr2_) {
            restore = wall_ms([&] {
                check_cuda(cudaMemcpy(matrix_.data(), context_.matrix.data(), matrix_.size() * sizeof(float),
                                      cudaMemcpyDeviceToDevice), "restore CholeskyQR2 matrix");
            });
        }
        const double factor = wall_ms([&] {
            check_cublas(cublasSgemm(
                context_.blas, CUBLAS_OP_T, CUBLAS_OP_N, context_.n, context_.n, context_.m,
                &one, context_.matrix.data(), context_.m, context_.matrix.data(), context_.m,
                &zero, gram1_.data(), context_.n), "form A^T A");
            check_solver(cusolverDnSpotrf(
                solver_, CUBLAS_FILL_MODE_UPPER, context_.n, gram1_.data(), context_.n,
                work_.data(), static_cast<int>(work_.size()), info_.data()), "first potrf");
            if (cholesky_qr2_) {
                check_cublas(cublasStrsm(
                    context_.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, context_.m, context_.n, &one, gram1_.data(), context_.n,
                    matrix_.data(), context_.m), "first CholeskyQR right solve");
                check_cublas(cublasSgemm(
                    context_.blas, CUBLAS_OP_T, CUBLAS_OP_N, context_.n, context_.n, context_.m,
                    &one, matrix_.data(), context_.m, matrix_.data(), context_.m,
                    &zero, gram2_.data(), context_.n), "form Q1^T Q1");
                check_solver(cusolverDnSpotrf(
                    solver_, CUBLAS_FILL_MODE_UPPER, context_.n, gram2_.data(), context_.n,
                    work_.data(), static_cast<int>(work_.size()), info_.data()), "second potrf");
                check_cublas(cublasStrsm(
                    context_.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, context_.m, context_.n, &one, gram2_.data(), context_.n,
                    matrix_.data(), context_.m), "second CholeskyQR right solve");
            }
        });
        verify_info(info_, "normal equation potrf info");
        const double apply = wall_ms([&] {
            const float* basis = cholesky_qr2_ ? matrix_.data() : context_.matrix.data();
            check_cublas(cublasSgemv(
                context_.blas, CUBLAS_OP_T, context_.m, context_.n, &one, basis, context_.m,
                context_.rhs.data(), 1, &zero, solution_.data(), 1), "normal equation apply rhs");
        });
        const double triangular = wall_ms([&] {
            if (cholesky_qr2_) {
                check_cublas(cublasStrsv(
                    context_.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    context_.n, gram2_.data(), context_.n, solution_.data(), 1), "CholeskyQR2 R2 solve");
                check_cublas(cublasStrsv(
                    context_.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    context_.n, gram1_.data(), context_.n, solution_.data(), 1), "CholeskyQR2 R1 solve");
            } else {
                check_cublas(cublasStrsv(
                    context_.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    context_.n, gram1_.data(), context_.n, solution_.data(), 1), "normal equation R^T solve");
                check_cublas(cublasStrsv(
                    context_.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    context_.n, gram1_.data(), context_.n, solution_.data(), 1), "normal equation R solve");
            }
        });
        if (timings) {
            timings->restore_ms.push_back(restore);
            timings->factor_ms.push_back(factor);
            timings->apply_ms.push_back(apply);
            timings->triangular_ms.push_back(triangular);
            timings->solve_ms.push_back(factor + apply + triangular);
        }
    }

    const float* solution() const { return solution_.data(); }

private:
    Context& context_;
    bool cholesky_qr2_ = false;
    bool tf32_ = false;
    cusolverDnHandle_t solver_ = nullptr;
    DeviceBuffer<float> matrix_;
    DeviceBuffer<float> gram1_;
    DeviceBuffer<float> gram2_;
    DeviceBuffer<float> solution_;
    DeviceBuffer<float> intermediate_;
    DeviceBuffer<float> work_;
    DeviceBuffer<int> info_;
};

ErrorMetrics measure_error(Context& context, const float* solution) {
    check_cublas(cublasSetMathMode(context.blas, CUBLAS_PEDANTIC_MATH), "set error metric math mode");
    ErrorMetrics metrics;
    DeviceBuffer<float> residual(context.m);
    DeviceBuffer<float> physical_residual(context.data_rows);
    DeviceBuffer<float> normal(context.n);
    DeviceBuffer<float> difference(context.n);
    const float one = 1.0f;
    const float minus_one = -1.0f;
    const float zero = 0.0f;
    check_cuda(cudaMemcpy(residual.data(), context.rhs.data(), static_cast<std::size_t>(context.m) * sizeof(float),
                          cudaMemcpyDeviceToDevice), "copy residual rhs");
    check_cublas(cublasSscal(context.blas, context.m, &minus_one, residual.data(), 1), "negate residual rhs");
    check_cublas(cublasSgemv(context.blas, CUBLAS_OP_N, context.m, context.n, &one,
                             context.matrix.data(), context.m, solution, 1, &one, residual.data(), 1),
                 "compute augmented residual");
    float residual_norm = 0.0f;
    float solution_norm = 0.0f;
    check_cublas(cublasSnrm2(context.blas, context.m, residual.data(), 1, &residual_norm), "residual norm");
    check_cublas(cublasSnrm2(context.blas, context.n, solution, 1, &solution_norm), "solution norm");
    check_cublas(cublasSgemv(context.blas, CUBLAS_OP_T, context.m, context.n, &one,
                             context.matrix.data(), context.m, residual.data(), 1, &zero, normal.data(), 1),
                 "compute normal residual");
    float normal_norm = 0.0f;
    check_cublas(cublasSnrm2(context.blas, context.n, normal.data(), 1, &normal_norm), "normal residual norm");

    check_cuda(cudaMemcpy(physical_residual.data(), context.rhs.data(),
                          static_cast<std::size_t>(context.data_rows) * sizeof(float), cudaMemcpyDeviceToDevice),
               "copy physical rhs");
    check_cublas(cublasSscal(context.blas, context.data_rows, &minus_one, physical_residual.data(), 1),
                 "negate physical rhs");
    check_cublas(cublasSgemv(context.blas, CUBLAS_OP_N, context.data_rows, context.n, &one,
                             context.matrix.data(), context.m, solution, 1, &one, physical_residual.data(), 1),
                 "compute physical residual");
    float physical_norm = 0.0f;
    float physical_rhs_norm = 0.0f;
    check_cublas(cublasSnrm2(context.blas, context.data_rows, physical_residual.data(), 1, &physical_norm),
                 "physical residual norm");
    check_cublas(cublasSnrm2(context.blas, context.data_rows, context.rhs.data(), 1, &physical_rhs_norm),
                 "physical rhs norm");

    check_cuda(cudaMemcpy(difference.data(), solution, static_cast<std::size_t>(context.n) * sizeof(float),
                          cudaMemcpyDeviceToDevice), "copy coefficient difference");
    check_cublas(cublasSaxpy(context.blas, context.n, &minus_one, context.reference_solution.data(), 1,
                             difference.data(), 1), "coefficient difference");
    float difference_norm = 0.0f;
    float reference_norm = 0.0f;
    check_cublas(cublasSnrm2(context.blas, context.n, difference.data(), 1, &difference_norm),
                 "coefficient difference norm");
    check_cublas(cublasSnrm2(context.blas, context.n, context.reference_solution.data(), 1, &reference_norm),
                 "reference coefficient norm");
    check_cuda(cudaDeviceSynchronize(), "error metric synchronize");

    metrics.coefficient_relative = difference_norm / std::max<double>(reference_norm, 1.0e-30);
    metrics.augmented_residual_relative = residual_norm / std::max<double>(context.rhs_norm, 1.0e-30);
    metrics.physical_residual_relative = physical_norm / std::max<double>(physical_rhs_norm, 1.0e-30);
    metrics.backward_error = residual_norm /
        std::max<double>(context.matrix_norm * solution_norm + context.rhs_norm, 1.0e-30);
    metrics.normal_residual_relative = normal_norm /
        std::max<double>(context.matrix_norm * residual_norm, 1.0e-30);
    return metrics;
}

template <typename Solver>
std::pair<Timings, ErrorMetrics> benchmark_solver(
    Context& context, Solver& solver, int warmups, int repeats
) {
    for (int repeat = 0; repeat < warmups; ++repeat) solver.run_once(nullptr);
    Timings timings;
    for (int repeat = 0; repeat < repeats; ++repeat) solver.run_once(&timings);
    return {timings, measure_error(context, solver.solution())};
}

void print_array(const std::vector<double>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void print_result(
    const std::string& method,
    const Context& context,
    const Timings& timings,
    const ErrorMetrics& error,
    const cudaDeviceProp& properties
) {
    const double m = context.m;
    const double n = context.n;
    const double householder_flops = 2.0 * m * n * n - (2.0 / 3.0) * n * n * n;
    const double solve_p50 = percentile(timings.solve_ms, 0.5);
    std::cout << std::setprecision(12);
    std::cout << '{'
              << "\"method\":\"" << method << "\"," 
              << "\"gpu\":\"" << properties.name << "\"," 
              << "\"rows\":" << context.m << ','
              << "\"cols\":" << context.n << ','
              << "\"data_rows\":" << context.data_rows << ','
              << "\"householder_flops\":" << householder_flops << ','
              << "\"solve_ms_p50\":" << solve_p50 << ','
              << "\"solve_ms_p95\":" << percentile(timings.solve_ms, 0.95) << ','
              << "\"equivalent_tflops_p50\":" << householder_flops / (solve_p50 * 1.0e9) << ','
              << "\"factor_ms_p50\":" << percentile(timings.factor_ms, 0.5) << ','
              << "\"apply_ms_p50\":" << percentile(timings.apply_ms, 0.5) << ','
              << "\"triangular_ms_p50\":" << percentile(timings.triangular_ms, 0.5) << ','
              << "\"restore_ms_p50\":" << percentile(timings.restore_ms, 0.5) << ','
              << "\"coefficient_relative_error\":" << error.coefficient_relative << ','
              << "\"augmented_residual_relative\":" << error.augmented_residual_relative << ','
              << "\"physical_residual_relative\":" << error.physical_residual_relative << ','
              << "\"backward_error\":" << error.backward_error << ','
              << "\"normal_residual_relative\":" << error.normal_residual_relative << ','
              << "\"restore_ms\":";
    print_array(timings.restore_ms);
    std::cout << ",\"factor_ms\":";
    print_array(timings.factor_ms);
    std::cout << ",\"apply_ms\":";
    print_array(timings.apply_ms);
    std::cout << ",\"triangular_ms\":";
    print_array(timings.triangular_ms);
    std::cout << ",\"solve_ms\":";
    print_array(timings.solve_ms);
    std::cout << "}\n";
}

struct Arguments {
    std::string snapshot;
    std::string method = "legacy";
    int device = 0;
    int warmups = 1;
    int repeats = 5;
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) throw std::runtime_error("missing value for " + key);
            return argv[index];
        };
        if (key == "--snapshot") arguments.snapshot = value();
        else if (key == "--method") arguments.method = value();
        else if (key == "--device") arguments.device = std::stoi(value());
        else if (key == "--warmups") arguments.warmups = std::stoi(value());
        else if (key == "--repeats") arguments.repeats = std::stoi(value());
        else throw std::runtime_error("unknown argument: " + key);
    }
    if (arguments.snapshot.empty()) throw std::runtime_error("--snapshot is required");
    if (arguments.warmups < 0 || arguments.repeats <= 0) throw std::runtime_error("invalid repeat counts");
    return arguments;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        check_cuda(cudaSetDevice(arguments.device), "cudaSetDevice");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, arguments.device), "cudaGetDeviceProperties");
        Snapshot snapshot = read_snapshot(arguments.snapshot);
        Context context(snapshot);
        snapshot.matrix.clear();
        snapshot.matrix.shrink_to_fit();
        snapshot.rhs.clear();
        snapshot.rhs.shrink_to_fit();

        {
            HouseholderSolver reference(context, false);
            reference.run_once(nullptr);
            check_cuda(cudaMemcpy(context.reference_solution.data(), reference.solution(),
                                  static_cast<std::size_t>(context.n) * sizeof(float), cudaMemcpyDeviceToDevice),
                       "save reference solution");
        }

        Timings timings;
        ErrorMetrics error;
        if (arguments.method == "legacy" || arguments.method == "generic") {
            HouseholderSolver solver(context, arguments.method == "generic");
            std::tie(timings, error) = benchmark_solver(context, solver, arguments.warmups, arguments.repeats);
        } else if (arguments.method.rfind("tsqr", 0) == 0) {
            const int blocks = std::stoi(arguments.method.substr(4));
            TsqrSolver solver(context, blocks);
            std::tie(timings, error) = benchmark_solver(context, solver, arguments.warmups, arguments.repeats);
        } else if (arguments.method == "normal_fp32" || arguments.method == "normal_tf32" ||
                   arguments.method == "choleskyqr2") {
            NormalEquationSolver solver(
                context, arguments.method == "choleskyqr2", arguments.method == "normal_tf32");
            std::tie(timings, error) = benchmark_solver(context, solver, arguments.warmups, arguments.repeats);
        } else {
            throw std::runtime_error("unknown method: " + arguments.method);
        }
        print_result(arguments.method, context, timings, error, properties);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "psi_qr_benchmark: " << error.what() << '\n';
        return 1;
    }
}
