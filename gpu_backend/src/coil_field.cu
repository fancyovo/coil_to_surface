#include "coil_field.h"
#include "coil_field_internal.h"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr double MU0_OVER_4PI = 1.0e-7;
constexpr double TWOPI = 6.283185307179586476925286766559;
constexpr int WARP_SIZE = 32;
#ifndef SGPU_WARPS_PER_BLOCK
#define SGPU_WARPS_PER_BLOCK 8
#endif
#ifndef SGPU_SEG_TILE
#define SGPU_SEG_TILE 256
#endif
constexpr int WARPS_PER_BLOCK = SGPU_WARPS_PER_BLOCK;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int SEG_TILE = SGPU_SEG_TILE;
constexpr int MAX_PSI_DEGREE = 24;
constexpr int MAX_PSI_MTOR = 32;
constexpr int SURFACE_THREADS = 64;

thread_local std::string g_last_error;

struct CoilField {
    int device_id = 0;
    int n_segments = 0;
    double* d_x = nullptr;
    double* d_y = nullptr;
    double* d_z = nullptr;
    double* d_wx = nullptr;
    double* d_wy = nullptr;
    double* d_wz = nullptr;
    float* d_x_f = nullptr;
    float* d_y_f = nullptr;
    float* d_z_f = nullptr;
    float* d_wx_f = nullptr;
    float* d_wy_f = nullptr;
    float* d_wz_f = nullptr;
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
};

void set_error(const char* msg) { g_last_error = msg ? msg : ""; }
void set_error(const std::string& msg) { g_last_error = msg; }

int cuda_check(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        set_error(std::string(where) + ": " + cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int cublas_check(cublasStatus_t status, const char* where) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        set_error(std::string(where) + ": cublas status " + std::to_string(static_cast<int>(status)));
        return 1;
    }
    return 0;
}

int cusolver_check(cusolverStatus_t status, const char* where) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        set_error(std::string(where) + ": cusolver status " + std::to_string(static_cast<int>(status)));
        return 1;
    }
    return 0;
}

__device__ double warp_sum(double v) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

__device__ void eval_B_warp(
    double px, double py, double pz,
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    double* sh,
    double& bx_out,
    double& by_out,
    double& bz_out
) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    double* sx = sh;
    double* sy = sx + SEG_TILE;
    double* sz = sy + SEG_TILE;
    double* swx = sz + SEG_TILE;
    double* swy = swx + SEG_TILE;
    double* swz = swy + SEG_TILE;

    double bx = 0.0;
    double by = 0.0;
    double bz = 0.0;

    for (int base = 0; base < nseg; base += SEG_TILE) {
        int count = min(SEG_TILE, nseg - base);
        for (int j = tid; j < count; j += blockDim.x) {
            int idx = base + j;
            sx[j] = seg_x[idx];
            sy[j] = seg_y[idx];
            sz[j] = seg_z[idx];
            swx[j] = seg_wx[idx];
            swy[j] = seg_wy[idx];
            swz[j] = seg_wz[idx];
        }
        __syncthreads();

        for (int j = lane; j < count; j += WARP_SIZE) {
            double rx = px - sx[j];
            double ry = py - sy[j];
            double rz = pz - sz[j];
            double r2 = rx * rx + ry * ry + rz * rz + 1.0e-300;
            double invr = rsqrt(r2);
            double invr3 = invr * invr * invr;
            double wx = swx[j];
            double wy = swy[j];
            double wz = swz[j];
            bx += (wy * rz - wz * ry) * invr3;
            by += (wz * rx - wx * rz) * invr3;
            bz += (wx * ry - wy * rx) * invr3;
        }
        __syncthreads();
    }

    bx = warp_sum(bx) * MU0_OVER_4PI;
    by = warp_sum(by) * MU0_OVER_4PI;
    bz = warp_sum(bz) * MU0_OVER_4PI;
    bx = __shfl_sync(0xffffffffu, bx, 0);
    by = __shfl_sync(0xffffffffu, by, 0);
    bz = __shfl_sync(0xffffffffu, bz, 0);
    bx_out = bx;
    by_out = by;
    bz_out = bz;
}

__global__ void eval_B_kernel(
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ xyz,
    double* __restrict__ B,
    int npoints
) {
    extern __shared__ double sh[];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & 31;
    int point = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    double bx = 0.0, by = 0.0, bz = 0.0;
    if (point < npoints) {
        double px = xyz[3 * point + 0];
        double py = xyz[3 * point + 1];
        double pz = xyz[3 * point + 2];
        eval_B_warp(px, py, pz, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz);
        if (lane == 0) {
            B[3 * point + 0] = bx;
            B[3 * point + 1] = by;
            B[3 * point + 2] = bz;
        }
    } else {
        // Still participate in shared-memory synchronizations inside eval_B_warp.
        eval_B_warp(0.0, 0.0, 0.0, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz);
    }
}

__device__ void rhs_cyl(
    double R, double Z, double phi,
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    double* sh,
    double& dR,
    double& dZ
) {
    double cp = cos(phi);
    double sp = sin(phi);
    double bx, by, bz;
    eval_B_warp(R * cp, R * sp, Z, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz);
    double br = bx * cp + by * sp;
    double bphi = -bx * sp + by * cp;
    double denom = fabs(bphi) > 1.0e-14 ? bphi : copysign(1.0e-14, bphi == 0.0 ? 1.0 : bphi);
    dR = R * br / denom;
    dZ = R * bz / denom;
}

__global__ void trace_period_kernel(
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R0,
    const double* __restrict__ Z0,
    double* __restrict__ R1,
    double* __restrict__ Z1,
    int nlines,
    int nfp,
    int steps
) {
    extern __shared__ double sh[];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & 31;
    int line = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    double R = (line < nlines) ? R0[line] : 1.0;
    double Z = (line < nlines) ? Z0[line] : 0.0;
    double period = TWOPI / static_cast<double>(nfp);
    double h = period / static_cast<double>(steps);

    for (int s = 0; s < steps; ++s) {
        double phi = h * static_cast<double>(s);
        double k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
        rhs_cyl(R, Z, phi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k1r, k1z);
        rhs_cyl(R + 0.5 * h * k1r, Z + 0.5 * h * k1z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k2r, k2z);
        rhs_cyl(R + 0.5 * h * k2r, Z + 0.5 * h * k2z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k3r, k3z);
        rhs_cyl(R + h * k3r, Z + h * k3z, phi + h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k4r, k4z);
        R += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
        Z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    }
    if (line < nlines && lane == 0) {
        R1[line] = R;
        Z1[line] = Z;
    }
}

__device__ void eval_B_block(
    double px, double py, double pz,
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    double* sh,
    double& bx_out,
    double& by_out,
    double& bz_out
) {
    int tid = threadIdx.x;
    double* sx = sh;
    double* sy = sx + SEG_TILE;
    double* sz = sy + SEG_TILE;
    double* swx = sz + SEG_TILE;
    double* swy = swx + SEG_TILE;
    double* swz = swy + SEG_TILE;
    double* rb = swz + SEG_TILE;
    double* rby = rb + blockDim.x;
    double* rbz = rby + blockDim.x;

    double bx = 0.0, by = 0.0, bz = 0.0;
    for (int base = 0; base < nseg; base += SEG_TILE) {
        int count = min(SEG_TILE, nseg - base);
        for (int j = tid; j < count; j += blockDim.x) {
            int idx = base + j;
            sx[j] = seg_x[idx];
            sy[j] = seg_y[idx];
            sz[j] = seg_z[idx];
            swx[j] = seg_wx[idx];
            swy[j] = seg_wy[idx];
            swz[j] = seg_wz[idx];
        }
        __syncthreads();
        for (int j = tid; j < count; j += blockDim.x) {
            double rx = px - sx[j];
            double ry = py - sy[j];
            double rz = pz - sz[j];
            double r2 = rx * rx + ry * ry + rz * rz + 1.0e-300;
            double invr = rsqrt(r2);
            double invr3 = invr * invr * invr;
            double wx = swx[j];
            double wy = swy[j];
            double wz = swz[j];
            bx += (wy * rz - wz * ry) * invr3;
            by += (wz * rx - wx * rz) * invr3;
            bz += (wx * ry - wy * rx) * invr3;
        }
        __syncthreads();
    }
    rb[tid] = bx;
    rby[tid] = by;
    rbz[tid] = bz;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            rb[tid] += rb[tid + stride];
            rby[tid] += rby[tid + stride];
            rbz[tid] += rbz[tid + stride];
        }
        __syncthreads();
    }
    bx_out = rb[0] * MU0_OVER_4PI;
    by_out = rby[0] * MU0_OVER_4PI;
    bz_out = rbz[0] * MU0_OVER_4PI;
}

__device__ void rhs_cyl_block(
    double R, double Z, double phi,
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    double* sh,
    double& dR,
    double& dZ
) {
    double cp = cos(phi);
    double sp = sin(phi);
    double bx, by, bz;
    eval_B_block(R * cp, R * sp, Z, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz);
    double br = bx * cp + by * sp;
    double bphi = -bx * sp + by * cp;
    double denom = fabs(bphi) > 1.0e-14 ? bphi : copysign(1.0e-14, bphi == 0.0 ? 1.0 : bphi);
    dR = R * br / denom;
    dZ = R * bz / denom;
}

__device__ void eval_B_block_f32(
    float px, float py, float pz,
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    float* sh,
    float& bx_out,
    float& by_out,
    float& bz_out
) {
    int tid = threadIdx.x;
    float* sx = sh;
    float* sy = sx + SEG_TILE;
    float* sz = sy + SEG_TILE;
    float* swx = sz + SEG_TILE;
    float* swy = swx + SEG_TILE;
    float* swz = swy + SEG_TILE;
    float* rb = swz + SEG_TILE;
    float* rby = rb + blockDim.x;
    float* rbz = rby + blockDim.x;

    float bx = 0.0f, by = 0.0f, bz = 0.0f;
    for (int base = 0; base < nseg; base += SEG_TILE) {
        int count = min(SEG_TILE, nseg - base);
        for (int j = tid; j < count; j += blockDim.x) {
            int idx = base + j;
            sx[j] = seg_x[idx];
            sy[j] = seg_y[idx];
            sz[j] = seg_z[idx];
            swx[j] = seg_wx[idx];
            swy[j] = seg_wy[idx];
            swz[j] = seg_wz[idx];
        }
        __syncthreads();
        for (int j = tid; j < count; j += blockDim.x) {
            float rx = px - sx[j];
            float ry = py - sy[j];
            float rz = pz - sz[j];
            float r2 = rx * rx + ry * ry + rz * rz + 1.0e-30f;
            float invr = rsqrtf(r2);
            float invr3 = invr * invr * invr;
            float wx = swx[j];
            float wy = swy[j];
            float wz = swz[j];
            bx += (wy * rz - wz * ry) * invr3;
            by += (wz * rx - wx * rz) * invr3;
            bz += (wx * ry - wy * rx) * invr3;
        }
        __syncthreads();
    }
    rb[tid] = bx;
    rby[tid] = by;
    rbz[tid] = bz;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            rb[tid] += rb[tid + stride];
            rby[tid] += rby[tid + stride];
            rbz[tid] += rbz[tid + stride];
        }
        __syncthreads();
    }
    bx_out = rb[0] * static_cast<float>(MU0_OVER_4PI);
    by_out = rby[0] * static_cast<float>(MU0_OVER_4PI);
    bz_out = rbz[0] * static_cast<float>(MU0_OVER_4PI);
}

__device__ void rhs_cyl_block_bf32_state64(
    double R, double Z, double phi,
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    float* sh,
    double& dR,
    double& dZ
) {
    double cp = cos(phi);
    double sp = sin(phi);
    float bx, by, bz;
    eval_B_block_f32(
        static_cast<float>(R * cp), static_cast<float>(R * sp), static_cast<float>(Z),
        seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz
    );
    double br = static_cast<double>(bx) * cp + static_cast<double>(by) * sp;
    double bphi = -static_cast<double>(bx) * sp + static_cast<double>(by) * cp;
    double denom = fabs(bphi) > 1.0e-14 ? bphi : copysign(1.0e-14, bphi == 0.0 ? 1.0 : bphi);
    dR = R * br / denom;
    dZ = R * static_cast<double>(bz) / denom;
}

__device__ void rhs_cyl_block_f32(
    float R, float Z, float phi,
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    float* sh,
    float& dR,
    float& dZ
) {
    float cp = cosf(phi);
    float sp = sinf(phi);
    float bx, by, bz;
    eval_B_block_f32(R * cp, R * sp, Z, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz);
    float br = bx * cp + by * sp;
    float bphi = -bx * sp + by * cp;
    float denom = fabsf(bphi) > 1.0e-12f ? bphi : copysignf(1.0e-12f, bphi == 0.0f ? 1.0f : bphi);
    dR = R * br / denom;
    dZ = R * bz / denom;
}

__global__ void trace_period_blockline_kernel(
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R0,
    const double* __restrict__ Z0,
    double* __restrict__ R1,
    double* __restrict__ Z1,
    int nlines,
    int nfp,
    int steps
) {
    extern __shared__ double sh[];
    int line = blockIdx.x;
    double R = (line < nlines) ? R0[line] : 1.0;
    double Z = (line < nlines) ? Z0[line] : 0.0;
    double period = TWOPI / static_cast<double>(nfp);
    double h = period / static_cast<double>(steps);
    for (int s = 0; s < steps; ++s) {
        double phi = h * static_cast<double>(s);
        double k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
        rhs_cyl_block(R, Z, phi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k1r, k1z);
        rhs_cyl_block(R + 0.5 * h * k1r, Z + 0.5 * h * k1z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k2r, k2z);
        rhs_cyl_block(R + 0.5 * h * k2r, Z + 0.5 * h * k2z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k3r, k3z);
        rhs_cyl_block(R + h * k3r, Z + h * k3z, phi + h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k4r, k4z);
        R += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
        Z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    }
    if (line < nlines && threadIdx.x == 0) {
        R1[line] = R;
        Z1[line] = Z;
    }
}

__global__ void trace_period_blockline_bf32_state64_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R0,
    const double* __restrict__ Z0,
    double* __restrict__ R1,
    double* __restrict__ Z1,
    int nlines,
    int nfp,
    int steps
) {
    extern __shared__ unsigned char shmem_bf32[];
    float* sh = reinterpret_cast<float*>(shmem_bf32);
    int line = blockIdx.x;
    double R = (line < nlines) ? R0[line] : 1.0;
    double Z = (line < nlines) ? Z0[line] : 0.0;
    double period = TWOPI / static_cast<double>(nfp);
    double h = period / static_cast<double>(steps);
    for (int s = 0; s < steps; ++s) {
        double phi = h * static_cast<double>(s);
        double k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
        rhs_cyl_block_bf32_state64(R, Z, phi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k1r, k1z);
        rhs_cyl_block_bf32_state64(R + 0.5 * h * k1r, Z + 0.5 * h * k1z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k2r, k2z);
        rhs_cyl_block_bf32_state64(R + 0.5 * h * k2r, Z + 0.5 * h * k2z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k3r, k3z);
        rhs_cyl_block_bf32_state64(R + h * k3r, Z + h * k3z, phi + h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k4r, k4z);
        R += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
        Z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    }
    if (line < nlines && threadIdx.x == 0) {
        R1[line] = R;
        Z1[line] = Z;
    }
}

__global__ void trace_period_blockline_f32_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R0,
    const double* __restrict__ Z0,
    double* __restrict__ R1,
    double* __restrict__ Z1,
    int nlines,
    int nfp,
    int steps
) {
    extern __shared__ unsigned char shmem_f32[];
    float* sh = reinterpret_cast<float*>(shmem_f32);
    int line = blockIdx.x;
    float R = (line < nlines) ? static_cast<float>(R0[line]) : 1.0f;
    float Z = (line < nlines) ? static_cast<float>(Z0[line]) : 0.0f;
    float period = static_cast<float>(TWOPI) / static_cast<float>(nfp);
    float h = period / static_cast<float>(steps);
    for (int s = 0; s < steps; ++s) {
        float phi = h * static_cast<float>(s);
        float k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
        rhs_cyl_block_f32(R, Z, phi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k1r, k1z);
        rhs_cyl_block_f32(R + 0.5f * h * k1r, Z + 0.5f * h * k1z, phi + 0.5f * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k2r, k2z);
        rhs_cyl_block_f32(R + 0.5f * h * k2r, Z + 0.5f * h * k2z, phi + 0.5f * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k3r, k3z);
        rhs_cyl_block_f32(R + h * k3r, Z + h * k3z, phi + h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k4r, k4z);
        R += (h / 6.0f) * (k1r + 2.0f * k2r + 2.0f * k3r + k4r);
        Z += (h / 6.0f) * (k1z + 2.0f * k2z + 2.0f * k3z + k4z);
    }
    if (line < nlines && threadIdx.x == 0) {
        R1[line] = static_cast<double>(R);
        Z1[line] = static_cast<double>(Z);
    }
}

__global__ void trace_period_blockline_f32_state16_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R0,
    const double* __restrict__ Z0,
    double* __restrict__ R1,
    double* __restrict__ Z1,
    int nlines,
    int nfp,
    int steps
) {
    extern __shared__ unsigned char shmem_f16[];
    float* sh = reinterpret_cast<float*>(shmem_f16);
    int line = blockIdx.x;
    __half Rh = __float2half_rn((line < nlines) ? static_cast<float>(R0[line]) : 1.0f);
    __half Zh = __float2half_rn((line < nlines) ? static_cast<float>(Z0[line]) : 0.0f);
    float period = static_cast<float>(TWOPI) / static_cast<float>(nfp);
    float h = period / static_cast<float>(steps);
    for (int s = 0; s < steps; ++s) {
        float R = __half2float(Rh);
        float Z = __half2float(Zh);
        float phi = h * static_cast<float>(s);
        float k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
        rhs_cyl_block_f32(R, Z, phi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k1r, k1z);
        rhs_cyl_block_f32(R + 0.5f * h * k1r, Z + 0.5f * h * k1z, phi + 0.5f * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k2r, k2z);
        rhs_cyl_block_f32(R + 0.5f * h * k2r, Z + 0.5f * h * k2z, phi + 0.5f * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k3r, k3z);
        rhs_cyl_block_f32(R + h * k3r, Z + h * k3z, phi + h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k4r, k4z);
        R += (h / 6.0f) * (k1r + 2.0f * k2r + 2.0f * k3r + k4r);
        Z += (h / 6.0f) * (k1z + 2.0f * k2z + k3z * 2.0f + k4z);
        Rh = __float2half_rn(R);
        Zh = __float2half_rn(Z);
    }
    if (line < nlines && threadIdx.x == 0) {
        R1[line] = static_cast<double>(__half2float(Rh));
        Z1[line] = static_cast<double>(__half2float(Zh));
    }
}

void eval_fourier(const double* c, int order, double t, double& x, double& dxdt) {
    x = c[0];
    dxdt = 0.0;
    for (int m = 1; m <= order; ++m) {
        double s = sin(TWOPI * m * t);
        double co = cos(TWOPI * m * t);
        double sin_coef = c[2 * m - 1];
        double cos_coef = c[2 * m];
        x += sin_coef * s + cos_coef * co;
        dxdt += TWOPI * m * (sin_coef * co - cos_coef * s);
    }
}

void rotate_z(double angle, double x, double y, double& xo, double& yo) {
    double c = cos(angle);
    double s = sin(angle);
    xo = c * x - s * y;
    yo = s * x + c * y;
}

void append_segment(
    std::vector<double>& x,
    std::vector<double>& y,
    std::vector<double>& z,
    std::vector<double>& wx,
    std::vector<double>& wy,
    std::vector<double>& wz,
    double px,
    double py,
    double pz,
    double dlx,
    double dly,
    double dlz,
    double current
) {
    x.push_back(px);
    y.push_back(py);
    z.push_back(pz);
    wx.push_back(current * dlx);
    wy.push_back(current * dly);
    wz.push_back(current * dlz);
}

void generate_segments(
    const double* cx,
    const double* cy,
    const double* cz,
    const double* currents,
    int nbase,
    int ncoeff,
    int nfp,
    int segs_per_coil,
    std::vector<double>& x,
    std::vector<double>& y,
    std::vector<double>& z,
    std::vector<double>& wx,
    std::vector<double>& wy,
    std::vector<double>& wz
) {
    int order = (ncoeff - 1) / 2;
    x.clear(); y.clear(); z.clear(); wx.clear(); wy.clear(); wz.clear();
    x.reserve(static_cast<size_t>(nbase) * 2 * nfp * segs_per_coil);
    y.reserve(x.capacity()); z.reserve(x.capacity());
    wx.reserve(x.capacity()); wy.reserve(x.capacity()); wz.reserve(x.capacity());

    for (int b = 0; b < nbase; ++b) {
        const double* bx = cx + static_cast<size_t>(b) * ncoeff;
        const double* by = cy + static_cast<size_t>(b) * ncoeff;
        const double* bz = cz + static_cast<size_t>(b) * ncoeff;
        double current = currents[b];
        for (int s = 0; s < segs_per_coil; ++s) {
            double t = (static_cast<double>(s) + 0.5) / static_cast<double>(segs_per_coil);
            double px, py, pz, vx, vy, vz;
            eval_fourier(bx, order, t, px, vx);
            eval_fourier(by, order, t, py, vy);
            eval_fourier(bz, order, t, pz, vz);
            double scale = 1.0 / static_cast<double>(segs_per_coil);
            vx *= scale; vy *= scale; vz *= scale;
            for (int k = 0; k < nfp; ++k) {
                double angle = TWOPI * static_cast<double>(k) / static_cast<double>(nfp);
                double rx, ry, rdx, rdy;
                rotate_z(angle, px, py, rx, ry);
                rotate_z(angle, vx, vy, rdx, rdy);
                append_segment(x, y, z, wx, wy, wz, rx, ry, pz, rdx, rdy, vz, current);
                double mx = px, my = -py, mz = -pz;
                double mdx = vx, mdy = -vy, mdz = -vz;
                rotate_z(angle, mx, my, rx, ry);
                rotate_z(angle, mdx, mdy, rdx, rdy);
                append_segment(x, y, z, wx, wy, wz, rx, ry, mz, rdx, rdy, mdz, -current);
            }
        }
    }
}

template <typename T>
int copy_to_device(T** dst, const std::vector<T>& src, const char* name) {
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(dst), src.size() * sizeof(T)), name)) return 1;
    if (cuda_check(cudaMemcpy(*dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), name)) return 1;
    return 0;
}

std::vector<float> to_float_vector(const std::vector<double>& src) {
    std::vector<float> out;
    out.reserve(src.size());
    for (double v : src) out.push_back(static_cast<float>(v));
    return out;
}

int psi_mode_count(int poly_degree, int m_tor) {
    int modes = 0;
    for (int deg = 2; deg <= poly_degree; ++deg) {
        for (int ax = deg; ax >= 0; --ax) {
            int bz = deg - ax;
            for (int m = 0; m <= m_tor; ++m) {
                if (ax == 2 && bz == 0 && m == 0) continue;
                modes += (m == 0) ? 1 : 2;
            }
        }
    }
    return modes;
}

__device__ inline double periodic_interp_uniform(double phi, const double* values, int n, double period) {
    double p = fmod(phi, period);
    if (p < 0.0) p += period;
    double pos = p * static_cast<double>(n) / period;
    int i0 = static_cast<int>(floor(pos));
    double t = pos - static_cast<double>(i0);
    if (i0 >= n) i0 = 0;
    int i1 = (i0 + 1 == n) ? 0 : (i0 + 1);
    return values[i0] * (1.0 - t) + values[i1] * t;
}

__device__ inline float periodic_interp_uniform_f32(float phi, const float* values, int n, float period) {
    float p = fmodf(phi, period);
    if (p < 0.0f) p += period;
    float pos = p * static_cast<float>(n) / period;
    int i0 = static_cast<int>(floorf(pos));
    float t = pos - static_cast<float>(i0);
    if (i0 >= n) i0 = 0;
    int i1 = (i0 + 1 == n) ? 0 : (i0 + 1);
    return values[i0] * (1.0f - t) + values[i1] * t;
}

__global__ void trace_axis_samples_mixed_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    double R0,
    double Z0,
    int nfp,
    int substeps_per_sample,
    int n_samples,
    double* __restrict__ R_out,
    double* __restrict__ Z_out,
    double* __restrict__ R_phi_out,
    double* __restrict__ Z_phi_out
) {
    extern __shared__ unsigned char shmem_axis[];
    float* sh = reinterpret_cast<float*>(shmem_axis);
    double R = R0;
    double Z = Z0;
    const int total_steps = n_samples * substeps_per_sample;
    const double period = TWOPI / static_cast<double>(nfp);
    const double h = period / static_cast<double>(total_steps);
    for (int sample = 0; sample < n_samples; ++sample) {
        for (int substep = 0; substep < substeps_per_sample; ++substep) {
            const int step = sample * substeps_per_sample + substep;
            const double phi = h * static_cast<double>(step);
            double k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
            rhs_cyl_block_bf32_state64(R, Z, phi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k1r, k1z);
            if (substep == 0 && threadIdx.x == 0) {
                R_out[sample] = R;
                Z_out[sample] = Z;
                R_phi_out[sample] = k1r;
                Z_phi_out[sample] = k1z;
            }
            rhs_cyl_block_bf32_state64(R + 0.5 * h * k1r, Z + 0.5 * h * k1z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k2r, k2z);
            rhs_cyl_block_bf32_state64(R + 0.5 * h * k2r, Z + 0.5 * h * k2z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k3r, k3z);
            rhs_cyl_block_bf32_state64(R + h * k3r, Z + h * k3z, phi + h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k4r, k4z);
            R += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
            Z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
        }
    }
}

template <typename T>
__device__ T warp_sum_t(T value) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

template <typename T>
__global__ void eval_B_only_kernel(
    const T* __restrict__ seg_x,
    const T* __restrict__ seg_y,
    const T* __restrict__ seg_z,
    const T* __restrict__ seg_wx,
    const T* __restrict__ seg_wy,
    const T* __restrict__ seg_wz,
    int nseg,
    const T* __restrict__ xyz,
    T* __restrict__ B,
    int npoints
) {
    extern __shared__ unsigned char shared_raw[];
    T* sx = reinterpret_cast<T*>(shared_raw);
    T* sy = sx + SEG_TILE;
    T* sz = sy + SEG_TILE;
    T* swx = sz + SEG_TILE;
    T* swy = swx + SEG_TILE;
    T* swz = swy + SEG_TILE;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int point = blockIdx.x * WARPS_PER_BLOCK + tid / WARP_SIZE;
    bool valid = point < npoints;
    T px = valid ? xyz[3 * point + 0] : T(0);
    T py = valid ? xyz[3 * point + 1] : T(0);
    T pz = valid ? xyz[3 * point + 2] : T(0);
    T accum[3] = {};

    for (int base = 0; base < nseg; base += SEG_TILE) {
        int count = min(SEG_TILE, nseg - base);
        for (int j = tid; j < count; j += blockDim.x) {
            int index = base + j;
            sx[j] = seg_x[index];
            sy[j] = seg_y[index];
            sz[j] = seg_z[index];
            swx[j] = seg_wx[index];
            swy[j] = seg_wy[index];
            swz[j] = seg_wz[index];
        }
        __syncthreads();
        if (valid) {
            for (int j = lane; j < count; j += WARP_SIZE) {
                T rx = px - sx[j];
                T ry = py - sy[j];
                T rz = pz - sz[j];
                T invr = T(1) / sqrt(rx * rx + ry * ry + rz * rz + T(1.0e-30));
                T invr3 = invr * invr * invr;
                T wx = swx[j];
                T wy = swy[j];
                T wz = swz[j];
                accum[0] += (wy * rz - wz * ry) * invr3;
                accum[1] += (wz * rx - wx * rz) * invr3;
                accum[2] += (wx * ry - wy * rx) * invr3;
            }
        }
        __syncthreads();
    }
    for (int component = 0; component < 3; ++component) {
        accum[component] = warp_sum_t(accum[component]) * T(MU0_OVER_4PI);
    }
    if (valid && lane == 0) {
        B[3 * point + 0] = accum[0];
        B[3 * point + 1] = accum[1];
        B[3 * point + 2] = accum[2];
    }
}

template <typename T>
__global__ void eval_B_grad_kernel(
    const T* __restrict__ seg_x,
    const T* __restrict__ seg_y,
    const T* __restrict__ seg_z,
    const T* __restrict__ seg_wx,
    const T* __restrict__ seg_wy,
    const T* __restrict__ seg_wz,
    int nseg,
    const T* __restrict__ xyz,
    T* __restrict__ B,
    T* __restrict__ grad_B,
    int npoints
) {
    extern __shared__ unsigned char shared_raw[];
    T* sx = reinterpret_cast<T*>(shared_raw);
    T* sy = sx + SEG_TILE;
    T* sz = sy + SEG_TILE;
    T* swx = sz + SEG_TILE;
    T* swy = swx + SEG_TILE;
    T* swz = swy + SEG_TILE;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid / WARP_SIZE;
    int point = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    bool valid = point < npoints;
    T px = valid ? xyz[3 * point + 0] : T(0);
    T py = valid ? xyz[3 * point + 1] : T(0);
    T pz = valid ? xyz[3 * point + 2] : T(0);

    T accum[12] = {};
    for (int base = 0; base < nseg; base += SEG_TILE) {
        int count = min(SEG_TILE, nseg - base);
        for (int j = tid; j < count; j += blockDim.x) {
            int index = base + j;
            sx[j] = seg_x[index];
            sy[j] = seg_y[index];
            sz[j] = seg_z[index];
            swx[j] = seg_wx[index];
            swy[j] = seg_wy[index];
            swz[j] = seg_wz[index];
        }
        __syncthreads();

        if (valid) {
            for (int j = lane; j < count; j += WARP_SIZE) {
                T rx = px - sx[j];
                T ry = py - sy[j];
                T rz = pz - sz[j];
                T r2 = rx * rx + ry * ry + rz * rz + T(1.0e-30);
                T invr = T(1) / sqrt(r2);
                T invr2 = invr * invr;
                T invr3 = invr * invr2;
                T invr5 = invr3 * invr2;
                T wx = swx[j];
                T wy = swy[j];
                T wz = swz[j];
                T ux = wy * rz - wz * ry;
                T uy = wz * rx - wx * rz;
                T uz = wx * ry - wy * rx;

                accum[0] += ux * invr3;
                accum[1] += uy * invr3;
                accum[2] += uz * invr3;

                T common_x = T(-3) * rx * invr5;
                T common_y = T(-3) * ry * invr5;
                T common_z = T(-3) * rz * invr5;
                accum[3] += ux * common_x;
                accum[4] += -wz * invr3 + ux * common_y;
                accum[5] += wy * invr3 + ux * common_z;
                accum[6] += wz * invr3 + uy * common_x;
                accum[7] += uy * common_y;
                accum[8] += -wx * invr3 + uy * common_z;
                accum[9] += -wy * invr3 + uz * common_x;
                accum[10] += wx * invr3 + uz * common_y;
                accum[11] += uz * common_z;
            }
        }
        __syncthreads();
    }

    for (int component = 0; component < 12; ++component) {
        accum[component] = warp_sum_t(accum[component]) * T(MU0_OVER_4PI);
    }
    if (valid && lane == 0) {
        B[3 * point + 0] = accum[0];
        B[3 * point + 1] = accum[1];
        B[3 * point + 2] = accum[2];
        for (int component = 0; component < 9; ++component) {
            grad_B[9 * point + component] = accum[3 + component];
        }
    }
}

__global__ void B_grad_segment_vjp_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const float* __restrict__ xyz,
    const float* __restrict__ adj_B,
    const float* __restrict__ adj_grad_B,
    int npoints,
    float* __restrict__ adj_segment_position,
    float* __restrict__ adj_segment_weight
) {
    const int segment = blockIdx.x;
    if (segment >= nseg) return;
    const float sx = seg_x[segment];
    const float sy = seg_y[segment];
    const float sz = seg_z[segment];
    const float wx = seg_wx[segment];
    const float wy = seg_wy[segment];
    const float wz = seg_wz[segment];
    float accum[6] = {};
    for (int point = threadIdx.x; point < npoints; point += blockDim.x) {
        const float rx = xyz[3 * point] - sx;
        const float ry = xyz[3 * point + 1] - sy;
        const float rz = xyz[3 * point + 2] - sz;
        const float r2 = rx * rx + ry * ry + rz * rz + 1.0e-30f;
        const float invr = rsqrtf(r2);
        const float invr2 = invr * invr;
        const float invr3 = invr * invr2;
        const float invr5 = invr3 * invr2;
        const float invr7 = invr5 * invr2;
        const float ux = wy * rz - wz * ry;
        const float uy = wz * rx - wx * rz;
        const float uz = wx * ry - wy * rx;
        const float abx = adj_B[3 * point];
        const float aby = adj_B[3 * point + 1];
        const float abz = adj_B[3 * point + 2];
        const float* aJ = adj_grad_B + 9 * point;

        float au_x = MU0_OVER_4PI * invr3 * abx;
        float au_y = MU0_OVER_4PI * invr3 * aby;
        float au_z = MU0_OVER_4PI * invr3 * abz;
        float a_f = MU0_OVER_4PI * (abx * ux + aby * uy + abz * uz);

        const float k00 = 0.0f;
        const float k10 = wz;
        const float k20 = -wy;
        const float k01 = -wz;
        const float k11 = 0.0f;
        const float k21 = wx;
        const float k02 = wy;
        const float k12 = -wx;
        const float k22 = 0.0f;
        a_f += MU0_OVER_4PI * (
            aJ[0] * k00 + aJ[3] * k10 + aJ[6] * k20 +
            aJ[1] * k01 + aJ[4] * k11 + aJ[7] * k21 +
            aJ[2] * k02 + aJ[5] * k12 + aJ[8] * k22
        );
        const float aJ_r_x = aJ[0] * rx + aJ[1] * ry + aJ[2] * rz;
        const float aJ_r_y = aJ[3] * rx + aJ[4] * ry + aJ[5] * rz;
        const float aJ_r_z = aJ[6] * rx + aJ[7] * ry + aJ[8] * rz;
        const float factor_u = -3.0f * MU0_OVER_4PI * invr5;
        au_x += factor_u * aJ_r_x;
        au_y += factor_u * aJ_r_y;
        au_z += factor_u * aJ_r_z;
        const float aJT_u_x = aJ[0] * ux + aJ[3] * uy + aJ[6] * uz;
        const float aJT_u_y = aJ[1] * ux + aJ[4] * uy + aJ[7] * uz;
        const float aJT_u_z = aJ[2] * ux + aJ[5] * uy + aJ[8] * uz;
        float ar_x = factor_u * aJT_u_x;
        float ar_y = factor_u * aJT_u_y;
        float ar_z = factor_u * aJT_u_z;
        const float outer_contraction =
            ux * aJ_r_x + uy * aJ_r_y + uz * aJ_r_z;
        const float a_g = -3.0f * MU0_OVER_4PI * outer_contraction;

        const float aK10 = MU0_OVER_4PI * invr3 * aJ[3];
        const float aK20 = MU0_OVER_4PI * invr3 * aJ[6];
        const float aK01 = MU0_OVER_4PI * invr3 * aJ[1];
        const float aK21 = MU0_OVER_4PI * invr3 * aJ[7];
        const float aK02 = MU0_OVER_4PI * invr3 * aJ[2];
        const float aK12 = MU0_OVER_4PI * invr3 * aJ[5];
        float aw_x = aK21 - aK12;
        float aw_y = -aK20 + aK02;
        float aw_z = aK10 - aK01;

        aw_x += ry * au_z - rz * au_y;
        aw_y += rz * au_x - rx * au_z;
        aw_z += rx * au_y - ry * au_x;
        ar_x += au_y * wz - au_z * wy;
        ar_y += au_z * wx - au_x * wz;
        ar_z += au_x * wy - au_y * wx;
        const float radial_factor = -3.0f * a_f * invr5 - 5.0f * a_g * invr7;
        ar_x += radial_factor * rx;
        ar_y += radial_factor * ry;
        ar_z += radial_factor * rz;
        accum[0] -= ar_x;
        accum[1] -= ar_y;
        accum[2] -= ar_z;
        accum[3] += aw_x;
        accum[4] += aw_y;
        accum[5] += aw_z;
    }
    __shared__ float reduction[6 * 256];
    for (int component = 0; component < 6; ++component) {
        reduction[component * blockDim.x + threadIdx.x] = accum[component];
    }
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            for (int component = 0; component < 6; ++component) {
                reduction[component * blockDim.x + threadIdx.x] +=
                    reduction[component * blockDim.x + threadIdx.x + offset];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int component = 0; component < 3; ++component) {
            adj_segment_position[3 * segment + component] = reduction[component * blockDim.x];
            adj_segment_weight[3 * segment + component] = reduction[(component + 3) * blockDim.x];
        }
    }
}

__global__ void B_grad_point_vjp_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const float* __restrict__ xyz,
    const float* __restrict__ adj_B,
    const float* __restrict__ adj_grad_B,
    int npoints,
    float* __restrict__ adj_xyz
) {
    extern __shared__ float shared[];
    float* sx = shared;
    float* sy = sx + SEG_TILE;
    float* sz = sy + SEG_TILE;
    float* swx = sz + SEG_TILE;
    float* swy = swx + SEG_TILE;
    float* swz = swy + SEG_TILE;

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int point = blockIdx.x * WARPS_PER_BLOCK + tid / WARP_SIZE;
    const bool valid = point < npoints;
    const float px = valid ? xyz[3 * point] : 0.0f;
    const float py = valid ? xyz[3 * point + 1] : 0.0f;
    const float pz = valid ? xyz[3 * point + 2] : 0.0f;
    const float abx = valid ? adj_B[3 * point] : 0.0f;
    const float aby = valid ? adj_B[3 * point + 1] : 0.0f;
    const float abz = valid ? adj_B[3 * point + 2] : 0.0f;
    const float* aJ = valid ? adj_grad_B + 9 * point : nullptr;
    float accum_x = 0.0f;
    float accum_y = 0.0f;
    float accum_z = 0.0f;

    for (int base = 0; base < nseg; base += SEG_TILE) {
        const int count = min(SEG_TILE, nseg - base);
        for (int index = tid; index < count; index += blockDim.x) {
            sx[index] = seg_x[base + index];
            sy[index] = seg_y[base + index];
            sz[index] = seg_z[base + index];
            swx[index] = seg_wx[base + index];
            swy[index] = seg_wy[base + index];
            swz[index] = seg_wz[base + index];
        }
        __syncthreads();
        if (valid) {
            for (int index = lane; index < count; index += WARP_SIZE) {
                const float rx = px - sx[index];
                const float ry = py - sy[index];
                const float rz = pz - sz[index];
                const float wx = swx[index];
                const float wy = swy[index];
                const float wz = swz[index];
                const float r2 = rx * rx + ry * ry + rz * rz + 1.0e-30f;
                const float invr = rsqrtf(r2);
                const float invr2 = invr * invr;
                const float invr3 = invr * invr2;
                const float invr5 = invr3 * invr2;
                const float invr7 = invr5 * invr2;
                const float ux = wy * rz - wz * ry;
                const float uy = wz * rx - wx * rz;
                const float uz = wx * ry - wy * rx;

                float au_x = MU0_OVER_4PI * invr3 * abx;
                float au_y = MU0_OVER_4PI * invr3 * aby;
                float au_z = MU0_OVER_4PI * invr3 * abz;
                float adj_invr3 = MU0_OVER_4PI * (abx * ux + aby * uy + abz * uz);
                adj_invr3 += MU0_OVER_4PI * (
                    aJ[3] * wz - aJ[6] * wy - aJ[1] * wz +
                    aJ[7] * wx + aJ[2] * wy - aJ[5] * wx
                );
                const float aJ_r_x = aJ[0] * rx + aJ[1] * ry + aJ[2] * rz;
                const float aJ_r_y = aJ[3] * rx + aJ[4] * ry + aJ[5] * rz;
                const float aJ_r_z = aJ[6] * rx + aJ[7] * ry + aJ[8] * rz;
                const float factor_u = -3.0f * MU0_OVER_4PI * invr5;
                au_x += factor_u * aJ_r_x;
                au_y += factor_u * aJ_r_y;
                au_z += factor_u * aJ_r_z;
                float ar_x = factor_u * (aJ[0] * ux + aJ[3] * uy + aJ[6] * uz);
                float ar_y = factor_u * (aJ[1] * ux + aJ[4] * uy + aJ[7] * uz);
                float ar_z = factor_u * (aJ[2] * ux + aJ[5] * uy + aJ[8] * uz);
                const float outer = ux * aJ_r_x + uy * aJ_r_y + uz * aJ_r_z;
                const float adj_invr5 = -3.0f * MU0_OVER_4PI * outer;

                ar_x += au_y * wz - au_z * wy;
                ar_y += au_z * wx - au_x * wz;
                ar_z += au_x * wy - au_y * wx;
                const float radial =
                    -3.0f * adj_invr3 * invr5 - 5.0f * adj_invr5 * invr7;
                accum_x += ar_x + radial * rx;
                accum_y += ar_y + radial * ry;
                accum_z += ar_z + radial * rz;
            }
        }
        __syncthreads();
    }
    accum_x = warp_sum_t(accum_x);
    accum_y = warp_sum_t(accum_y);
    accum_z = warp_sum_t(accum_z);
    if (valid && lane == 0) {
        adj_xyz[3 * point] = accum_x;
        adj_xyz[3 * point + 1] = accum_y;
        adj_xyz[3 * point + 2] = accum_z;
    }
}

__device__ inline void periodic_hermite_uniform(
    double phi,
    const double* values,
    const double* derivatives,
    int n,
    double period,
    double& value,
    double& derivative
) {
    double p = fmod(phi, period);
    if (p < 0.0) p += period;
    double pos = p * static_cast<double>(n) / period;
    int i0 = static_cast<int>(floor(pos));
    double t = pos - static_cast<double>(i0);
    if (i0 >= n) i0 = 0;
    int i1 = (i0 + 1 == n) ? 0 : (i0 + 1);
    double h = period / static_cast<double>(n);
    double t2 = t * t;
    double t3 = t2 * t;
    double y0 = values[i0];
    double y1 = values[i1];
    double d0 = derivatives[i0];
    double d1 = derivatives[i1];
    value = (2.0 * t3 - 3.0 * t2 + 1.0) * y0
          + (t3 - 2.0 * t2 + t) * h * d0
          + (-2.0 * t3 + 3.0 * t2) * y1
          + (t3 - t2) * h * d1;
    derivative = (6.0 * t2 - 6.0 * t) * y0 / h
               + (3.0 * t2 - 4.0 * t + 1.0) * d0
               + (-6.0 * t2 + 6.0 * t) * y1 / h
               + (3.0 * t2 - 2.0 * t) * d1;
}

__device__ inline void periodic_hermite_uniform_f32(
    float phi,
    const float* values,
    const float* derivatives,
    int n,
    float period,
    float& value,
    float& derivative
) {
    float p = fmodf(phi, period);
    if (p < 0.0f) p += period;
    float pos = p * static_cast<float>(n) / period;
    int i0 = static_cast<int>(floorf(pos));
    float t = pos - static_cast<float>(i0);
    if (i0 >= n) i0 = 0;
    int i1 = (i0 + 1 == n) ? 0 : (i0 + 1);
    float h = period / static_cast<float>(n);
    float t2 = t * t;
    float t3 = t2 * t;
    float y0 = values[i0];
    float y1 = values[i1];
    float d0 = derivatives[i0];
    float d1 = derivatives[i1];
    value = (2.0f * t3 - 3.0f * t2 + 1.0f) * y0
          + (t3 - 2.0f * t2 + t) * h * d0
          + (-2.0f * t3 + 3.0f * t2) * y1
          + (t3 - t2) * h * d1;
    derivative = (6.0f * t2 - 6.0f * t) * y0 / h
               + (3.0f * t2 - 4.0f * t + 1.0f) * d0
               + (-6.0f * t2 + 6.0f * t) * y1 / h
               + (3.0f * t2 - 2.0f * t) * d1;
}

__global__ void convert_double_to_float_kernel(const double* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<float>(src[i]);
}

__global__ void convert_float_to_double_kernel(const float* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<double>(src[i]);
}

__global__ void scale_rhs_kernel(double* atb, const double* scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atb[i] /= scale[i];
}

__global__ void extract_scale_kernel(const double* ata, double* scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double d = ata[static_cast<size_t>(i) * n + i];
        scale[i] = sqrt(fmax(d, 1.0e-30));
    }
}

__global__ void scale_system_kernel(double* ata, const double* scale, int n, double ridge) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        double v = ata[static_cast<size_t>(i) * n + j] / (scale[i] * scale[j]);
        if (i == j) v += ridge;
        ata[static_cast<size_t>(i) * n + j] = v;
    }
}

__global__ void unscale_coeff_kernel(double* coeff, const double* scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) coeff[i] /= scale[i];
}

__global__ void unscale_coeff_kernel_f32(float* coeff, const double* scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) coeff[i] /= static_cast<float>(scale[i]);
}

template <typename T>
__global__ void axpy_negative_kernel(const T* x, const T* y, T* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] - y[i];
}

template <typename T>
__global__ void transpose_rowmajor_to_colmajor_kernel(const T* src, T* dst, int n_rows, int n_cols, int dst_lda) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n_rows && col < n_cols) {
        dst[static_cast<size_t>(col) * dst_lda + row] = src[static_cast<size_t>(row) * n_cols + col];
    }
}

__global__ void column_norms_kernel_f64(const double* mat, double* scale, int n_rows, int lda, int n_cols) {
    int col = blockIdx.x;
    if (col >= n_cols) return;
    __shared__ double sh[THREADS_PER_BLOCK];
    double sum = 0.0;
    const double* col_ptr = mat + static_cast<size_t>(col) * lda;
    for (int row = threadIdx.x; row < n_rows; row += blockDim.x) {
        double v = col_ptr[row];
        sum += v * v;
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) scale[col] = sqrt(fmax(sh[0], 1.0e-30));
}

__global__ void column_norms_kernel_f32(const float* mat, double* scale, int n_rows, int lda, int n_cols) {
    int col = blockIdx.x;
    if (col >= n_cols) return;
    __shared__ double sh[THREADS_PER_BLOCK];
    double sum = 0.0;
    const float* col_ptr = mat + static_cast<size_t>(col) * lda;
    for (int row = threadIdx.x; row < n_rows; row += blockDim.x) {
        double v = static_cast<double>(col_ptr[row]);
        sum += v * v;
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) scale[col] = sqrt(fmax(sh[0], 1.0e-30));
}

__global__ void scale_columns_kernel_f64(double* mat, const double* scale, int n_rows, int lda, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n_rows && col < n_cols) {
        mat[static_cast<size_t>(col) * lda + row] /= scale[col];
    }
}

__global__ void scale_columns_kernel_f32(float* mat, const double* scale, int n_rows, int lda, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n_rows && col < n_cols) {
        mat[static_cast<size_t>(col) * lda + row] /= static_cast<float>(scale[col]);
    }
}

template <typename T>
__global__ void set_ridge_tail_kernel(T* mat, int n_rows_data, int lda, int n_cols, T lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_cols * n_cols;
    if (idx < total) {
        int col = idx / n_cols;
        int tail_row = idx - col * n_cols;
        mat[static_cast<size_t>(col) * lda + (n_rows_data + tail_row)] = (tail_row == col) ? lambda : static_cast<T>(0);
    }
}

__global__ void psi_fill_matrix_kernel_f64(
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R,
    const double* __restrict__ Z,
    const double* __restrict__ phi,
    const double* __restrict__ axis_R,
    const double* __restrict__ axis_Z,
    const double* __restrict__ axis_R_phi,
    const double* __restrict__ axis_Z_phi,
    int n_axis,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int n_cols,
    int nfp,
    double a,
    int poly_degree,
    int m_tor,
    double* __restrict__ mat,
    double* __restrict__ rhs
) {
    extern __shared__ unsigned char shbuf[];
    int row = blockIdx.x;
    if (row >= gridDim.x) return;
    double* sh = reinterpret_cast<double*>(shbuf);
    int base_count = SEG_TILE * 6 + blockDim.x * 3;
    double* aux = sh + base_count;
    double* xpow = aux;
    double* zpow = xpow + (MAX_PSI_DEGREE + 1);
    double* cosv = zpow + (MAX_PSI_DEGREE + 1);
    double* sinv = cosv + (MAX_PSI_MTOR + 1);
    double* scalar = sinv + (MAX_PSI_MTOR + 1);

    double Ri = R[row];
    double Zi = Z[row];
    double phii = phi[row];

    double cp = cos(phii);
    double sp = sin(phii);
    double bx, by, bz;
    eval_B_block(Ri * cp, Ri * sp, Zi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz);

    if (threadIdx.x == 0) {
        double period = TWOPI / static_cast<double>(nfp);
        double ra, za, rap, zap;
        periodic_hermite_uniform(phii, axis_R, axis_R_phi, n_axis, period, ra, rap);
        periodic_hermite_uniform(phii, axis_Z, axis_Z_phi, n_axis, period, za, zap);
        double X = (Ri - ra) / a;
        double Zc = (Zi - za) / a;
        double br = bx * cp + by * sp;
        double bphi = -bx * sp + by * cp;
        double cphi = bphi / Ri;
        double cR = br - cphi * rap;
        double cZ = bz - cphi * zap;
        xpow[0] = 1.0;
        zpow[0] = 1.0;
        for (int k = 1; k <= poly_degree; ++k) {
            xpow[k] = xpow[k - 1] * X;
            zpow[k] = zpow[k - 1] * Zc;
        }
        cosv[0] = 1.0;
        sinv[0] = 0.0;
        for (int m = 1; m <= m_tor; ++m) {
            double arg = static_cast<double>(m * nfp) * phii;
            cosv[m] = cos(arg);
            sinv[m] = sin(arg);
        }
        scalar[0] = cR;
        scalar[1] = cZ;
        scalar[2] = cphi;
        scalar[3] = -2.0 * X * cR / a;
    }
    __syncthreads();

    if (threadIdx.x == 0) rhs[row] = scalar[3];
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        int ax = mode_a[col];
        int bz_exp = mode_b[col];
        int m = mode_m[col];
        int kind = mode_kind[col];
        double mono = xpow[ax] * zpow[bz_exp];
        double mono_x = (ax == 0) ? 0.0 : static_cast<double>(ax) * xpow[ax - 1] * zpow[bz_exp];
        double mono_z = (bz_exp == 0) ? 0.0 : static_cast<double>(bz_exp) * xpow[ax] * zpow[bz_exp - 1];
        double spatial = (scalar[0] * mono_x + scalar[1] * mono_z) / a;
        double trig = (m == 0) ? 1.0 : ((kind == 0) ? cosv[m] : sinv[m]);
        double trig_phi = 0.0;
        if (m != 0) {
            double fac = static_cast<double>(m * nfp);
            trig_phi = (kind == 0) ? (-fac * sinv[m]) : (fac * cosv[m]);
        }
        mat[static_cast<size_t>(row) * n_cols + col] = spatial * trig + scalar[2] * mono * trig_phi;
    }
}

__global__ void psi_fill_matrix_kernel_f32(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const float* __restrict__ R,
    const float* __restrict__ Z,
    const float* __restrict__ phi,
    const float* __restrict__ axis_R,
    const float* __restrict__ axis_Z,
    const float* __restrict__ axis_R_phi,
    const float* __restrict__ axis_Z_phi,
    int n_axis,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int n_cols,
    int nfp,
    float a,
    int poly_degree,
    int m_tor,
    float* __restrict__ mat,
    float* __restrict__ rhs
) {
    extern __shared__ unsigned char shbuf[];
    int row = blockIdx.x;
    if (row >= gridDim.x) return;
    float* sh = reinterpret_cast<float*>(shbuf);
    int base_count = SEG_TILE * 6 + blockDim.x * 3;
    float* aux = sh + base_count;
    float* xpow = aux;
    float* zpow = xpow + (MAX_PSI_DEGREE + 1);
    float* cosv = zpow + (MAX_PSI_DEGREE + 1);
    float* sinv = cosv + (MAX_PSI_MTOR + 1);
    float* scalar = sinv + (MAX_PSI_MTOR + 1);

    float Ri = R[row];
    float Zi = Z[row];
    float phii = phi[row];

    float cp = cosf(phii);
    float sp = sinf(phii);
    float bx, by, bz;
    eval_B_block_f32(Ri * cp, Ri * sp, Zi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz);

    if (threadIdx.x == 0) {
        float period = static_cast<float>(TWOPI) / static_cast<float>(nfp);
        float ra, za, rap, zap;
        periodic_hermite_uniform_f32(phii, axis_R, axis_R_phi, n_axis, period, ra, rap);
        periodic_hermite_uniform_f32(phii, axis_Z, axis_Z_phi, n_axis, period, za, zap);
        float X = (Ri - ra) / a;
        float Zc = (Zi - za) / a;
        float br = bx * cp + by * sp;
        float bphi = -bx * sp + by * cp;
        float cphi = bphi / Ri;
        float cR = br - cphi * rap;
        float cZ = bz - cphi * zap;
        xpow[0] = 1.0f;
        zpow[0] = 1.0f;
        for (int k = 1; k <= poly_degree; ++k) {
            xpow[k] = xpow[k - 1] * X;
            zpow[k] = zpow[k - 1] * Zc;
        }
        cosv[0] = 1.0f;
        sinv[0] = 0.0f;
        for (int m = 1; m <= m_tor; ++m) {
            float arg = static_cast<float>(m * nfp) * phii;
            cosv[m] = cosf(arg);
            sinv[m] = sinf(arg);
        }
        scalar[0] = cR;
        scalar[1] = cZ;
        scalar[2] = cphi;
        scalar[3] = -2.0f * X * cR / a;
    }
    __syncthreads();

    if (threadIdx.x == 0) rhs[row] = scalar[3];
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        int ax = mode_a[col];
        int bz_exp = mode_b[col];
        int m = mode_m[col];
        int kind = mode_kind[col];
        float mono = xpow[ax] * zpow[bz_exp];
        float mono_x = (ax == 0) ? 0.0f : static_cast<float>(ax) * xpow[ax - 1] * zpow[bz_exp];
        float mono_z = (bz_exp == 0) ? 0.0f : static_cast<float>(bz_exp) * xpow[ax] * zpow[bz_exp - 1];
        float spatial = (scalar[0] * mono_x + scalar[1] * mono_z) / a;
        float trig = (m == 0) ? 1.0f : ((kind == 0) ? cosv[m] : sinv[m]);
        float trig_phi = 0.0f;
        if (m != 0) {
            float fac = static_cast<float>(m * nfp);
            trig_phi = (kind == 0) ? (-fac * sinv[m]) : (fac * cosv[m]);
        }
        mat[static_cast<size_t>(row) * n_cols + col] = spatial * trig + scalar[2] * mono * trig_phi;
    }
}

__device__ inline void eval_ray_poly_and_deriv(
    const double* coeff_deg,
    int poly_degree,
    double u,
    double a_scale,
    double& psi,
    double& dpsi
) {
    psi = 0.0;
    dpsi = 0.0;
    if (poly_degree < 2) return;
    double u_pow_prev = u;
    double u_pow = u * u;
    psi = coeff_deg[2] * u_pow;
    dpsi = 2.0 * coeff_deg[2] * u_pow_prev / a_scale;
    for (int deg = 3; deg <= poly_degree; ++deg) {
        u_pow_prev = u_pow;
        u_pow *= u;
        psi += coeff_deg[deg] * u_pow;
        dpsi += static_cast<double>(deg) * coeff_deg[deg] * u_pow_prev / a_scale;
    }
}

__device__ inline double eval_ray_poly_only(
    const double* coeff_deg,
    int poly_degree,
    double u
) {
    if (poly_degree < 2) return 0.0;
    double u_pow = u * u;
    double psi = coeff_deg[2] * u_pow;
    for (int deg = 3; deg <= poly_degree; ++deg) {
        u_pow *= u;
        psi += coeff_deg[deg] * u_pow;
    }
    return psi;
}

__global__ void surface_build_ray_poly_kernel(
    const double* __restrict__ coeffs,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int n_coeff,
    int nfp,
    int poly_degree,
    int m_tor,
    int nphi,
    int ntheta,
    double* __restrict__ ray_coeff
) {
    extern __shared__ double sh[];
    double* cosv = sh;
    double* sinv = sh + (MAX_PSI_MTOR + 1);
    int iphi = blockIdx.x;
    double phi = (static_cast<double>(iphi) * TWOPI) / (static_cast<double>(nphi) * static_cast<double>(nfp));
    for (int m = threadIdx.x; m <= m_tor; m += blockDim.x) {
        if (m == 0) {
            cosv[m] = 1.0;
            sinv[m] = 0.0;
        } else {
            double arg = static_cast<double>(m * nfp) * phi;
            cosv[m] = cos(arg);
            sinv[m] = sin(arg);
        }
    }
    __syncthreads();

    for (int itheta = threadIdx.x; itheta < ntheta; itheta += blockDim.x) {
        double theta = (static_cast<double>(itheta) * TWOPI) / static_cast<double>(ntheta);
        double cth = cos(theta);
        double sth = sin(theta);
        double xpow[MAX_PSI_DEGREE + 1];
        double zpow[MAX_PSI_DEGREE + 1];
        double coeff_deg[MAX_PSI_DEGREE + 1];
        xpow[0] = 1.0;
        zpow[0] = 1.0;
        coeff_deg[0] = 0.0;
        coeff_deg[1] = 0.0;
        for (int deg = 2; deg <= poly_degree; ++deg) coeff_deg[deg] = 0.0;
        for (int deg = 1; deg <= poly_degree; ++deg) {
            xpow[deg] = xpow[deg - 1] * cth;
            zpow[deg] = zpow[deg - 1] * sth;
        }
        coeff_deg[2] = cth * cth;
        for (int k = 0; k < n_coeff; ++k) {
            int ax = mode_a[k];
            int bz = mode_b[k];
            int deg = ax + bz;
            int m = mode_m[k];
            double trig = (m == 0) ? 1.0 : ((mode_kind[k] == 0) ? cosv[m] : sinv[m]);
            coeff_deg[deg] += coeffs[k] * xpow[ax] * zpow[bz] * trig;
        }
        size_t base = (static_cast<size_t>(iphi) * ntheta + itheta) * static_cast<size_t>(poly_degree + 1);
        for (int deg = 0; deg <= poly_degree; ++deg) {
            ray_coeff[base + deg] = coeff_deg[deg];
        }
    }
}

__global__ void surface_newton_from_ray_poly_kernel(
    const double* __restrict__ ray_coeff,
    int poly_degree,
    const double* __restrict__ axis_R,
    const double* __restrict__ axis_Z,
    int n_axis,
    int nfp,
    double a_scale,
    int nphi,
    int ntheta,
    double psi_level,
    int maxiter,
    double tol,
    double max_radius_scale,
    double* __restrict__ xyz,
    double* __restrict__ radii
) {
    __shared__ double ra_sh;
    __shared__ double za_sh;
    __shared__ double phi_sh;
    int iphi = blockIdx.x;
    if (threadIdx.x == 0) {
        phi_sh = (static_cast<double>(iphi) * TWOPI) / (static_cast<double>(nphi) * static_cast<double>(nfp));
        double period = TWOPI / static_cast<double>(nfp);
        ra_sh = periodic_interp_uniform(phi_sh, axis_R, n_axis, period);
        za_sh = periodic_interp_uniform(phi_sh, axis_Z, n_axis, period);
    }
    __syncthreads();

    double max_radius = max_radius_scale * a_scale;
    double fallback = a_scale * sqrt(fmax(psi_level, 1.0e-16));
    double cp = cos(phi_sh);
    double sp = sin(phi_sh);
    for (int itheta = threadIdx.x; itheta < ntheta; itheta += blockDim.x) {
        size_t base = (static_cast<size_t>(iphi) * ntheta + itheta) * static_cast<size_t>(poly_degree + 1);
        const double* coeff_deg = ray_coeff + base;
        double q = coeff_deg[2];
        double rho = (q > 1.0e-10) ? (a_scale * sqrt(fmax(psi_level, 0.0) / q)) : fallback;
        rho = fmin(fmax(rho, 1.0e-12 * a_scale), max_radius);
        for (int iter = 0; iter < maxiter; ++iter) {
            double u = rho / a_scale;
            double psi = 0.0, dpsi = 0.0;
            eval_ray_poly_and_deriv(coeff_deg, poly_degree, u, a_scale, psi, dpsi);
            double f = psi - psi_level;
            if (fabs(f) <= tol) break;
            double denom = (fabs(dpsi) > 1.0e-14) ? dpsi : copysign(1.0e-14, dpsi == 0.0 ? 1.0 : dpsi);
            double lim = 0.45 * fmax(fabs(rho), 1.0e-8 * a_scale);
            double step = f / denom;
            if (step > lim) step = lim;
            if (step < -lim) step = -lim;
            double trial = fmin(fmax(rho - step, 1.0e-12 * a_scale), max_radius);
            double psi_trial = eval_ray_poly_only(coeff_deg, poly_degree, trial / a_scale);
            if (fabs(psi_trial - psi_level) <= fabs(f)) {
                rho = trial;
            } else {
                rho = 0.5 * (rho + trial);
            }
        }
        double theta = (static_cast<double>(itheta) * TWOPI) / static_cast<double>(ntheta);
        double cth = cos(theta);
        double sth = sin(theta);
        double R = ra_sh + rho * cth;
        double Z = za_sh + rho * sth;
        size_t idx = static_cast<size_t>(iphi) * ntheta + itheta;
        radii[idx] = rho;
        xyz[3 * idx + 0] = R * cp;
        xyz[3 * idx + 1] = R * sp;
        xyz[3 * idx + 2] = Z;
    }
}

struct GpuFitStats {
    double copy_in_s = 0.0;
    double assemble_s = 0.0;
    double linear_prep_s = 0.0;
    double solve_s = 0.0;
    double residual_s = 0.0;
    double copy_out_s = 0.0;
    double total_s = 0.0;
    double qr_transpose_s = 0.0;
    double qr_scale_s = 0.0;
    double qr_factor_s = 0.0;
    double qr_apply_qtb_s = 0.0;
    double qr_tri_s = 0.0;
};

} // namespace

extern "C" {

int sgpu_create_field(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    int segments_per_coil,
    int device_id,
    void** out_handle
) {
    try {
        if (!out_handle) {
            set_error("out_handle is null");
            return 1;
        }
        *out_handle = nullptr;
        if (n_base_coils <= 0 || n_coeff <= 0 || nfp <= 0 || segments_per_coil <= 0) {
            set_error("invalid create_field dimensions");
            return 1;
        }
        if (cuda_check(cudaSetDevice(device_id), "cudaSetDevice")) return 1;
        std::vector<double> x, y, z, wx, wy, wz;
        generate_segments(coeffs_x, coeffs_y, coeffs_z, currents_a, n_base_coils, n_coeff, nfp, segments_per_coil, x, y, z, wx, wy, wz);
        std::vector<float> xf = to_float_vector(x);
        std::vector<float> yf = to_float_vector(y);
        std::vector<float> zf = to_float_vector(z);
        std::vector<float> wxf = to_float_vector(wx);
        std::vector<float> wyf = to_float_vector(wy);
        std::vector<float> wzf = to_float_vector(wz);
        CoilField* f = new CoilField();
        f->device_id = device_id;
        f->n_segments = static_cast<int>(x.size());
        if (cublas_check(cublasCreate(&f->blas), "cublasCreate field") ||
            cusolver_check(cusolverDnCreate(&f->solver), "cusolverDnCreate field")) {
            sgpu_destroy_field(f);
            return 1;
        }
        if (copy_to_device(&f->d_x, x, "d_x") ||
            copy_to_device(&f->d_y, y, "d_y") ||
            copy_to_device(&f->d_z, z, "d_z") ||
            copy_to_device(&f->d_wx, wx, "d_wx") ||
            copy_to_device(&f->d_wy, wy, "d_wy") ||
            copy_to_device(&f->d_wz, wz, "d_wz") ||
            copy_to_device(&f->d_x_f, xf, "d_x_f") ||
            copy_to_device(&f->d_y_f, yf, "d_y_f") ||
            copy_to_device(&f->d_z_f, zf, "d_z_f") ||
            copy_to_device(&f->d_wx_f, wxf, "d_wx_f") ||
            copy_to_device(&f->d_wy_f, wyf, "d_wy_f") ||
            copy_to_device(&f->d_wz_f, wzf, "d_wz_f")) {
            sgpu_destroy_field(f);
            return 1;
        }
        *out_handle = f;
        set_error("");
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return 1;
    }
}

void sgpu_destroy_field(void* handle) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f) return;
    cudaSetDevice(f->device_id);
    if (f->blas) cublasDestroy(f->blas);
    if (f->solver) cusolverDnDestroy(f->solver);
    cudaFree(f->d_x);
    cudaFree(f->d_y);
    cudaFree(f->d_z);
    cudaFree(f->d_wx);
    cudaFree(f->d_wy);
    cudaFree(f->d_wz);
    cudaFree(f->d_x_f);
    cudaFree(f->d_y_f);
    cudaFree(f->d_z_f);
    cudaFree(f->d_wx_f);
    cudaFree(f->d_wy_f);
    cudaFree(f->d_wz_f);
    delete f;
}

int sgpu_segment_count(void* handle) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    return f ? f->n_segments : 0;
}

int sgpu_eval_B(void* handle, const double* xyz_host, double* B_host, int n_points) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !xyz_host || !B_host || n_points < 0) {
        set_error("invalid eval_B arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    double *d_xyz = nullptr, *d_B = nullptr;
    size_t xyz_bytes = static_cast<size_t>(n_points) * 3 * sizeof(double);
    if (cuda_check(cudaMalloc(&d_xyz, xyz_bytes), "cudaMalloc d_xyz")) return 1;
    if (cuda_check(cudaMalloc(&d_B, xyz_bytes), "cudaMalloc d_B")) { cudaFree(d_xyz); return 1; }
    if (cuda_check(cudaMemcpy(d_xyz, xyz_host, xyz_bytes, cudaMemcpyHostToDevice), "copy xyz")) { cudaFree(d_xyz); cudaFree(d_B); return 1; }
    int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(double);
    eval_B_kernel<<<blocks, THREADS_PER_BLOCK, shmem>>>(f->d_x, f->d_y, f->d_z, f->d_wx, f->d_wy, f->d_wz, f->n_segments, d_xyz, d_B, n_points);
    if (cuda_check(cudaGetLastError(), "eval_B kernel")) { cudaFree(d_xyz); cudaFree(d_B); return 1; }
    if (cuda_check(cudaDeviceSynchronize(), "eval_B sync")) { cudaFree(d_xyz); cudaFree(d_B); return 1; }
    if (cuda_check(cudaMemcpy(B_host, d_B, xyz_bytes, cudaMemcpyDeviceToHost), "copy B")) { cudaFree(d_xyz); cudaFree(d_B); return 1; }
    cudaFree(d_xyz);
    cudaFree(d_B);
    set_error("");
    return 0;
}

int sgpu_eval_B_f32(void* handle, const float* xyz_host, float* B_host, int n_points) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !xyz_host || !B_host || n_points < 0) {
        set_error("invalid eval_B_f32 arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    float *d_xyz = nullptr, *d_B = nullptr;
    size_t xyz_bytes = static_cast<size_t>(n_points) * 3 * sizeof(float);
    if (cuda_check(cudaMalloc(&d_xyz, xyz_bytes), "eval_B_f32 d_xyz") ||
        cuda_check(cudaMalloc(&d_B, xyz_bytes), "eval_B_f32 d_B")) {
        cudaFree(d_xyz); cudaFree(d_B);
        return 1;
    }
    if (cuda_check(cudaMemcpy(d_xyz, xyz_host, xyz_bytes, cudaMemcpyHostToDevice), "eval_B_f32 copy xyz")) {
        cudaFree(d_xyz); cudaFree(d_B);
        return 1;
    }
    int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    eval_B_only_kernel<float><<<blocks, THREADS_PER_BLOCK, shmem>>>(
        f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
        f->n_segments, d_xyz, d_B, n_points
    );
    if (cuda_check(cudaGetLastError(), "eval_B_f32 kernel") ||
        cuda_check(cudaDeviceSynchronize(), "eval_B_f32 sync") ||
        cuda_check(cudaMemcpy(B_host, d_B, xyz_bytes, cudaMemcpyDeviceToHost), "eval_B_f32 copy B")) {
        cudaFree(d_xyz); cudaFree(d_B);
        return 1;
    }
    cudaFree(d_xyz); cudaFree(d_B);
    set_error("");
    return 0;
}

int sgpu_eval_B_grad(void* handle, const double* xyz_host, double* B_host, double* grad_B_host, int n_points) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !xyz_host || !B_host || !grad_B_host || n_points < 0) {
        set_error("invalid eval_B_grad arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    double *d_xyz = nullptr, *d_B = nullptr, *d_grad_B = nullptr;
    size_t xyz_bytes = static_cast<size_t>(n_points) * 3 * sizeof(double);
    size_t grad_bytes = static_cast<size_t>(n_points) * 9 * sizeof(double);
    if (cuda_check(cudaMalloc(&d_xyz, xyz_bytes), "eval_B_grad d_xyz") ||
        cuda_check(cudaMalloc(&d_B, xyz_bytes), "eval_B_grad d_B") ||
        cuda_check(cudaMalloc(&d_grad_B, grad_bytes), "eval_B_grad d_grad_B")) {
        cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
        return 1;
    }
    if (cuda_check(cudaMemcpy(d_xyz, xyz_host, xyz_bytes, cudaMemcpyHostToDevice), "eval_B_grad copy xyz")) {
        cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
        return 1;
    }
    int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(double);
    eval_B_grad_kernel<double><<<blocks, THREADS_PER_BLOCK, shmem>>>(
        f->d_x, f->d_y, f->d_z, f->d_wx, f->d_wy, f->d_wz,
        f->n_segments, d_xyz, d_B, d_grad_B, n_points
    );
    if (cuda_check(cudaGetLastError(), "eval_B_grad kernel") ||
        cuda_check(cudaDeviceSynchronize(), "eval_B_grad sync") ||
        cuda_check(cudaMemcpy(B_host, d_B, xyz_bytes, cudaMemcpyDeviceToHost), "eval_B_grad copy B") ||
        cuda_check(cudaMemcpy(grad_B_host, d_grad_B, grad_bytes, cudaMemcpyDeviceToHost), "eval_B_grad copy grad")) {
        cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
        return 1;
    }
    cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
    set_error("");
    return 0;
}

int sgpu_eval_B_grad_f32(void* handle, const float* xyz_host, float* B_host, float* grad_B_host, int n_points) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !xyz_host || !B_host || !grad_B_host || n_points < 0) {
        set_error("invalid eval_B_grad_f32 arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    float *d_xyz = nullptr, *d_B = nullptr, *d_grad_B = nullptr;
    size_t xyz_bytes = static_cast<size_t>(n_points) * 3 * sizeof(float);
    size_t grad_bytes = static_cast<size_t>(n_points) * 9 * sizeof(float);
    if (cuda_check(cudaMalloc(&d_xyz, xyz_bytes), "eval_B_grad_f32 d_xyz") ||
        cuda_check(cudaMalloc(&d_B, xyz_bytes), "eval_B_grad_f32 d_B") ||
        cuda_check(cudaMalloc(&d_grad_B, grad_bytes), "eval_B_grad_f32 d_grad_B")) {
        cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
        return 1;
    }
    if (cuda_check(cudaMemcpy(d_xyz, xyz_host, xyz_bytes, cudaMemcpyHostToDevice), "eval_B_grad_f32 copy xyz")) {
        cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
        return 1;
    }
    int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    eval_B_grad_kernel<float><<<blocks, THREADS_PER_BLOCK, shmem>>>(
        f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
        f->n_segments, d_xyz, d_B, d_grad_B, n_points
    );
    if (cuda_check(cudaGetLastError(), "eval_B_grad_f32 kernel") ||
        cuda_check(cudaDeviceSynchronize(), "eval_B_grad_f32 sync") ||
        cuda_check(cudaMemcpy(B_host, d_B, xyz_bytes, cudaMemcpyDeviceToHost), "eval_B_grad_f32 copy B") ||
        cuda_check(cudaMemcpy(grad_B_host, d_grad_B, grad_bytes, cudaMemcpyDeviceToHost), "eval_B_grad_f32 copy grad")) {
        cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
        return 1;
    }
    cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
    set_error("");
    return 0;
}

int sgpu_eval_B_grad_point_vjp_f32(
    void* handle,
    const float* xyz_host,
    const float* adj_B_host,
    const float* adj_grad_B_host,
    float* adj_xyz_host,
    int n_points
) {
    auto* field = reinterpret_cast<CoilField*>(handle);
    if (!field || !xyz_host || !adj_B_host || !adj_grad_B_host ||
        !adj_xyz_host || n_points < 0) {
        set_error("invalid eval_B_grad_point_vjp_f32 arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "point VJP cudaSetDevice")) return 1;
    const size_t xyz_bytes = static_cast<size_t>(n_points) * 3 * sizeof(float);
    const size_t grad_bytes = static_cast<size_t>(n_points) * 9 * sizeof(float);
    float* d_xyz = nullptr;
    float* d_adj_B = nullptr;
    float* d_adj_grad_B = nullptr;
    float* d_adj_xyz = nullptr;
    auto cleanup = [&]() {
        cudaFree(d_xyz);
        cudaFree(d_adj_B);
        cudaFree(d_adj_grad_B);
        cudaFree(d_adj_xyz);
    };
    if (cuda_check(cudaMalloc(&d_xyz, xyz_bytes), "point VJP d_xyz") ||
        cuda_check(cudaMalloc(&d_adj_B, xyz_bytes), "point VJP d_adj_B") ||
        cuda_check(cudaMalloc(&d_adj_grad_B, grad_bytes), "point VJP d_adj_grad_B") ||
        cuda_check(cudaMalloc(&d_adj_xyz, xyz_bytes), "point VJP d_adj_xyz") ||
        cuda_check(cudaMemcpy(d_xyz, xyz_host, xyz_bytes, cudaMemcpyHostToDevice), "point VJP copy xyz") ||
        cuda_check(cudaMemcpy(d_adj_B, adj_B_host, xyz_bytes, cudaMemcpyHostToDevice), "point VJP copy adj_B") ||
        cuda_check(cudaMemcpy(d_adj_grad_B, adj_grad_B_host, grad_bytes, cudaMemcpyHostToDevice), "point VJP copy adj_grad_B")) {
        cleanup();
        return 1;
    }
    const int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    B_grad_point_vjp_kernel<<<blocks, THREADS_PER_BLOCK, shared_bytes>>>(
        field->d_x_f, field->d_y_f, field->d_z_f,
        field->d_wx_f, field->d_wy_f, field->d_wz_f,
        field->n_segments, d_xyz, d_adj_B, d_adj_grad_B, n_points, d_adj_xyz
    );
    if (cuda_check(cudaGetLastError(), "point VJP kernel") ||
        cuda_check(cudaDeviceSynchronize(), "point VJP sync") ||
        cuda_check(cudaMemcpy(adj_xyz_host, d_adj_xyz, xyz_bytes, cudaMemcpyDeviceToHost), "point VJP copy output")) {
        cleanup();
        return 1;
    }
    cleanup();
    set_error("");
    return 0;
}

int sgpu_normal_eq(
    void* handle,
    const double* mat_host,
    const double* rhs_host,
    double* ata_host,
    double* atb_host,
    int n_rows,
    int n_cols
) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !mat_host || !rhs_host || !ata_host || !atb_host || n_rows < 0 || n_cols <= 0) {
        set_error("invalid normal_eq arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    double *d_mat = nullptr, *d_rhs = nullptr, *d_ata = nullptr, *d_atb = nullptr;
    size_t mat_bytes = static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols) * sizeof(double);
    size_t rhs_bytes = static_cast<size_t>(n_rows) * sizeof(double);
    size_t ata_bytes = static_cast<size_t>(n_cols) * static_cast<size_t>(n_cols) * sizeof(double);
    size_t atb_bytes = static_cast<size_t>(n_cols) * sizeof(double);
    cublasHandle_t blas = nullptr;
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mat), mat_bytes), "normal_eq d_mat") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rhs), rhs_bytes), "normal_eq d_rhs") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ata), ata_bytes), "normal_eq d_ata") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_atb), atb_bytes), "normal_eq d_atb") ||
        cuda_check(cudaMemcpy(d_mat, mat_host, mat_bytes, cudaMemcpyHostToDevice), "normal_eq copy mat") ||
        cuda_check(cudaMemcpy(d_rhs, rhs_host, rhs_bytes, cudaMemcpyHostToDevice), "normal_eq copy rhs") ||
        cuda_check(cudaMemset(d_ata, 0, ata_bytes), "normal_eq memset ata") ||
        cuda_check(cudaMemset(d_atb, 0, atb_bytes), "normal_eq memset atb") ||
        cublas_check(cublasCreate(&blas), "cublasCreate")) {
        cudaFree(d_mat); cudaFree(d_rhs); cudaFree(d_ata); cudaFree(d_atb);
        if (blas) cublasDestroy(blas);
        return 1;
    }
    const double one = 1.0;
    const double zero = 0.0;
    // mat_host is C-order A[n_rows, n_cols]. The same memory interpreted as
    // column-major is B[n_cols, n_rows] = A^T, so B * B^T = A^T A.
    if (cublas_check(
            cublasDgemm(
                blas,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                n_cols,
                n_cols,
                n_rows,
                &one,
                d_mat,
                n_cols,
                d_mat,
                n_cols,
                &zero,
                d_ata,
                n_cols),
            "cublasDgemm normal_eq") ||
        cublas_check(
            cublasDgemv(
                blas,
                CUBLAS_OP_N,
                n_cols,
                n_rows,
                &one,
                d_mat,
                n_cols,
                d_rhs,
                1,
                &zero,
                d_atb,
                1),
            "cublasDgemv normal_eq") ||
        cuda_check(cudaMemcpy(ata_host, d_ata, ata_bytes, cudaMemcpyDeviceToHost), "normal_eq copy ata") ||
        cuda_check(cudaMemcpy(atb_host, d_atb, atb_bytes, cudaMemcpyDeviceToHost), "normal_eq copy atb")) {
        cublasDestroy(blas);
        cudaFree(d_mat); cudaFree(d_rhs); cudaFree(d_ata); cudaFree(d_atb);
        return 1;
    }
    cublasDestroy(blas);
    cudaFree(d_mat);
    cudaFree(d_rhs);
    cudaFree(d_ata);
    cudaFree(d_atb);
    set_error("");
    return 0;
}

int sgpu_normal_eq_f32(
    void* handle,
    const float* mat_host,
    const float* rhs_host,
    float* ata_host,
    float* atb_host,
    int n_rows,
    int n_cols
) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !mat_host || !rhs_host || !ata_host || !atb_host || n_rows < 0 || n_cols <= 0) {
        set_error("invalid normal_eq_f32 arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    float *d_mat = nullptr, *d_rhs = nullptr, *d_ata = nullptr, *d_atb = nullptr;
    size_t mat_bytes = static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols) * sizeof(float);
    size_t rhs_bytes = static_cast<size_t>(n_rows) * sizeof(float);
    size_t ata_bytes = static_cast<size_t>(n_cols) * static_cast<size_t>(n_cols) * sizeof(float);
    size_t atb_bytes = static_cast<size_t>(n_cols) * sizeof(float);
    cublasHandle_t blas = nullptr;
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mat), mat_bytes), "normal_eq_f32 d_mat") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rhs), rhs_bytes), "normal_eq_f32 d_rhs") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ata), ata_bytes), "normal_eq_f32 d_ata") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_atb), atb_bytes), "normal_eq_f32 d_atb") ||
        cuda_check(cudaMemcpy(d_mat, mat_host, mat_bytes, cudaMemcpyHostToDevice), "normal_eq_f32 copy mat") ||
        cuda_check(cudaMemcpy(d_rhs, rhs_host, rhs_bytes, cudaMemcpyHostToDevice), "normal_eq_f32 copy rhs") ||
        cuda_check(cudaMemset(d_ata, 0, ata_bytes), "normal_eq_f32 memset ata") ||
        cuda_check(cudaMemset(d_atb, 0, atb_bytes), "normal_eq_f32 memset atb") ||
        cublas_check(cublasCreate(&blas), "cublasCreate f32")) {
        cudaFree(d_mat); cudaFree(d_rhs); cudaFree(d_ata); cudaFree(d_atb);
        if (blas) cublasDestroy(blas);
        return 1;
    }
    const float one = 1.0f;
    const float zero = 0.0f;
    if (cublas_check(
            cublasSgemm(
                blas,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                n_cols,
                n_cols,
                n_rows,
                &one,
                d_mat,
                n_cols,
                d_mat,
                n_cols,
                &zero,
                d_ata,
                n_cols),
            "cublasSgemm normal_eq_f32") ||
        cublas_check(
            cublasSgemv(
                blas,
                CUBLAS_OP_N,
                n_cols,
                n_rows,
                &one,
                d_mat,
                n_cols,
                d_rhs,
                1,
                &zero,
                d_atb,
                1),
            "cublasSgemv normal_eq_f32") ||
        cuda_check(cudaMemcpy(ata_host, d_ata, ata_bytes, cudaMemcpyDeviceToHost), "normal_eq_f32 copy ata") ||
        cuda_check(cudaMemcpy(atb_host, d_atb, atb_bytes, cudaMemcpyDeviceToHost), "normal_eq_f32 copy atb")) {
        cublasDestroy(blas);
        cudaFree(d_mat); cudaFree(d_rhs); cudaFree(d_ata); cudaFree(d_atb);
        return 1;
    }
    cublasDestroy(blas);
    cudaFree(d_mat);
    cudaFree(d_rhs);
    cudaFree(d_ata);
    cudaFree(d_atb);
    set_error("");
    return 0;
}

int sgpu_fit_psi_fullgpu(
    void* handle,
    const double* R_host,
    const double* Z_host,
    const double* phi_host,
    int n_points,
    const double* axis_R_host,
    const double* axis_Z_host,
    const double* axis_R_phi_host,
    const double* axis_Z_phi_host,
    int n_axis,
    const int* mode_a_host,
    const int* mode_b_host,
    const int* mode_m_host,
    const int* mode_kind_host,
    int n_coeff,
    int nfp,
    double a,
    int poly_degree,
    int m_tor,
    double ridge,
    int solver_mode,
    int precision_mode,
    double* coeff_host,
    double* train_rms_out,
    double* stats_out,
    int stats_len
) {
    using clock = std::chrono::steady_clock;

    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !R_host || !Z_host || !phi_host || !axis_R_host || !axis_Z_host || !axis_R_phi_host || !axis_Z_phi_host ||
        !mode_a_host || !mode_b_host || !mode_m_host || !mode_kind_host || !coeff_host || !train_rms_out ||
        n_points <= 0 || n_axis <= 1 || nfp <= 0 || poly_degree < 2 || m_tor < 0 || n_coeff <= 0) {
        set_error("invalid fit_psi_fullgpu arguments");
        return 1;
    }
    if (poly_degree > MAX_PSI_DEGREE || m_tor > MAX_PSI_MTOR) {
        set_error("psi degree exceeds compiled GPU limits");
        return 1;
    }
    int expected = psi_mode_count(poly_degree, m_tor);
    if (n_coeff != expected) {
        set_error("n_coeff does not match poly_degree/m_tor");
        return 1;
    }
    if (solver_mode != 1 && solver_mode != 2) {
        set_error("solver_mode must be 1 (normal_eq) or 2 (qr)");
        return 1;
    }
    if (precision_mode != 1 && precision_mode != 2) {
        set_error("precision_mode must be 1 (fp64) or 2 (fp32)");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;

    GpuFitStats stats;
    auto t_total = clock::now();

    double *d_R = nullptr, *d_Z = nullptr, *d_phi = nullptr;
    double *d_axis_R = nullptr, *d_axis_Z = nullptr, *d_axis_R_phi = nullptr, *d_axis_Z_phi = nullptr;
    int *d_mode_a = nullptr, *d_mode_b = nullptr, *d_mode_m = nullptr, *d_mode_kind = nullptr;
    double *d_scale = nullptr, *d_info_rhs = nullptr, *d_ata_d = nullptr, *d_atb_d = nullptr;
    int *d_info = nullptr, *d_ipiv = nullptr;
    double train_rms = 0.0;
    int h_info = 0;

    float *d_R_f = nullptr, *d_Z_f = nullptr, *d_phi_f = nullptr;
    float *d_axis_R_f = nullptr, *d_axis_Z_f = nullptr, *d_axis_R_phi_f = nullptr, *d_axis_Z_phi_f = nullptr;
    float *d_mat_f = nullptr, *d_rhs_f = nullptr, *d_ata_f = nullptr, *d_atb_f = nullptr, *d_coeff_f = nullptr, *d_pred_f = nullptr;
    double *d_mat_d = nullptr, *d_rhs_d = nullptr, *d_pred_d = nullptr;
    double *d_qr_mat_d = nullptr, *d_qr_rhs_d = nullptr, *d_tau_d = nullptr, *d_work_d = nullptr;
    float *d_qr_mat_f = nullptr, *d_qr_rhs_f = nullptr, *d_tau_f = nullptr, *d_work_f = nullptr;
    std::vector<float> coeff_host_f;
    double* d_coeff_src_d = nullptr;
    float* d_coeff_src_f = nullptr;

    auto cleanup = [&]() {
        cudaFree(d_R); cudaFree(d_Z); cudaFree(d_phi);
        cudaFree(d_axis_R); cudaFree(d_axis_Z); cudaFree(d_axis_R_phi); cudaFree(d_axis_Z_phi);
        cudaFree(d_mode_a); cudaFree(d_mode_b); cudaFree(d_mode_m); cudaFree(d_mode_kind);
        cudaFree(d_scale); cudaFree(d_info_rhs); cudaFree(d_ata_d); cudaFree(d_atb_d); cudaFree(d_info); cudaFree(d_ipiv);
        cudaFree(d_R_f); cudaFree(d_Z_f); cudaFree(d_phi_f);
        cudaFree(d_axis_R_f); cudaFree(d_axis_Z_f); cudaFree(d_axis_R_phi_f); cudaFree(d_axis_Z_phi_f);
        cudaFree(d_mat_f); cudaFree(d_rhs_f); cudaFree(d_ata_f); cudaFree(d_atb_f); cudaFree(d_coeff_f); cudaFree(d_pred_f);
        cudaFree(d_mat_d); cudaFree(d_rhs_d); cudaFree(d_pred_d);
        cudaFree(d_qr_mat_d); cudaFree(d_qr_rhs_d); cudaFree(d_tau_d); cudaFree(d_work_d);
        cudaFree(d_qr_mat_f); cudaFree(d_qr_rhs_f); cudaFree(d_tau_f); cudaFree(d_work_f);
    };

    auto fail = [&](const char* msg) {
        set_error(msg);
        cleanup();
        return 1;
    };

    size_t point_bytes = static_cast<size_t>(n_points) * sizeof(double);
    size_t axis_bytes = static_cast<size_t>(n_axis) * sizeof(double);
    size_t mode_bytes = static_cast<size_t>(n_coeff) * sizeof(int);
    auto t = clock::now();
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R), point_bytes), "fit d_R") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z), point_bytes), "fit d_Z") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_phi), point_bytes), "fit d_phi") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_R), axis_bytes), "fit d_axis_R") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_Z), axis_bytes), "fit d_axis_Z") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_R_phi), axis_bytes), "fit d_axis_R_phi") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_Z_phi), axis_bytes), "fit d_axis_Z_phi") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_a), mode_bytes), "fit d_mode_a") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_b), mode_bytes), "fit d_mode_b") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_m), mode_bytes), "fit d_mode_m") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_kind), mode_bytes), "fit d_mode_kind") ||
        cuda_check(cudaMemcpy(d_R, R_host, point_bytes, cudaMemcpyHostToDevice), "fit copy R") ||
        cuda_check(cudaMemcpy(d_Z, Z_host, point_bytes, cudaMemcpyHostToDevice), "fit copy Z") ||
        cuda_check(cudaMemcpy(d_phi, phi_host, point_bytes, cudaMemcpyHostToDevice), "fit copy phi") ||
        cuda_check(cudaMemcpy(d_axis_R, axis_R_host, axis_bytes, cudaMemcpyHostToDevice), "fit copy axis_R") ||
        cuda_check(cudaMemcpy(d_axis_Z, axis_Z_host, axis_bytes, cudaMemcpyHostToDevice), "fit copy axis_Z") ||
        cuda_check(cudaMemcpy(d_axis_R_phi, axis_R_phi_host, axis_bytes, cudaMemcpyHostToDevice), "fit copy axis_R_phi") ||
        cuda_check(cudaMemcpy(d_axis_Z_phi, axis_Z_phi_host, axis_bytes, cudaMemcpyHostToDevice), "fit copy axis_Z_phi") ||
        cuda_check(cudaMemcpy(d_mode_a, mode_a_host, mode_bytes, cudaMemcpyHostToDevice), "fit copy mode_a") ||
        cuda_check(cudaMemcpy(d_mode_b, mode_b_host, mode_bytes, cudaMemcpyHostToDevice), "fit copy mode_b") ||
        cuda_check(cudaMemcpy(d_mode_m, mode_m_host, mode_bytes, cudaMemcpyHostToDevice), "fit copy mode_m") ||
        cuda_check(cudaMemcpy(d_mode_kind, mode_kind_host, mode_bytes, cudaMemcpyHostToDevice), "fit copy mode_kind")) {
        cleanup();
        return 1;
    }
    stats.copy_in_s = std::chrono::duration<double>(clock::now() - t).count();

    double rhs_norm2 = 0.0;
    size_t mat_elems = static_cast<size_t>(n_points) * static_cast<size_t>(n_coeff);
    size_t mat_bytes_d = mat_elems * sizeof(double);
    size_t mat_bytes_f = mat_elems * sizeof(float);
    int threads = THREADS_PER_BLOCK;
    int blocks1d_points = (n_points + threads - 1) / threads;
    int blocks1d_coeff = (n_coeff + threads - 1) / threads;
    int shared_d = static_cast<int>((SEG_TILE * 6 + threads * 3 + 2 * (MAX_PSI_DEGREE + 1) + 2 * (MAX_PSI_MTOR + 1) + 4) * sizeof(double));
    int shared_f = static_cast<int>((SEG_TILE * 6 + threads * 3 + 2 * (MAX_PSI_DEGREE + 1) + 2 * (MAX_PSI_MTOR + 1) + 4) * sizeof(float));

    t = clock::now();
    if (precision_mode == 1) {
        if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mat_d), mat_bytes_d), "fit d_mat_d") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rhs_d), point_bytes), "fit d_rhs_d")) {
            cleanup();
            return 1;
        }
        psi_fill_matrix_kernel_f64<<<n_points, threads, shared_d>>>(
            f->d_x, f->d_y, f->d_z, f->d_wx, f->d_wy, f->d_wz, f->n_segments,
            d_R, d_Z, d_phi, d_axis_R, d_axis_Z, d_axis_R_phi, d_axis_Z_phi, n_axis,
            d_mode_a, d_mode_b, d_mode_m, d_mode_kind, n_coeff, nfp, a, poly_degree, m_tor, d_mat_d, d_rhs_d
        );
        if (cuda_check(cudaGetLastError(), "fit psi_fill_matrix f64") ||
            cublas_check(cublasDdot(f->blas, n_points, d_rhs_d, 1, d_rhs_d, 1, &rhs_norm2), "fit rhs_norm2 d64") ||
            cuda_check(cudaDeviceSynchronize(), "fit assemble sync f64")) {
            cleanup();
            return 1;
        }
    } else {
        if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R_f), static_cast<size_t>(n_points) * sizeof(float)), "fit d_R_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z_f), static_cast<size_t>(n_points) * sizeof(float)), "fit d_Z_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_phi_f), static_cast<size_t>(n_points) * sizeof(float)), "fit d_phi_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_R_f), static_cast<size_t>(n_axis) * sizeof(float)), "fit d_axis_R_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_Z_f), static_cast<size_t>(n_axis) * sizeof(float)), "fit d_axis_Z_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_R_phi_f), static_cast<size_t>(n_axis) * sizeof(float)), "fit d_axis_R_phi_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_Z_phi_f), static_cast<size_t>(n_axis) * sizeof(float)), "fit d_axis_Z_phi_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mat_f), mat_bytes_f), "fit d_mat_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rhs_f), static_cast<size_t>(n_points) * sizeof(float)), "fit d_rhs_f")) {
            cleanup();
            return 1;
        }
        convert_double_to_float_kernel<<<blocks1d_points, threads>>>(d_R, d_R_f, n_points);
        convert_double_to_float_kernel<<<blocks1d_points, threads>>>(d_Z, d_Z_f, n_points);
        convert_double_to_float_kernel<<<blocks1d_points, threads>>>(d_phi, d_phi_f, n_points);
        convert_double_to_float_kernel<<<(n_axis + threads - 1) / threads, threads>>>(d_axis_R, d_axis_R_f, n_axis);
        convert_double_to_float_kernel<<<(n_axis + threads - 1) / threads, threads>>>(d_axis_Z, d_axis_Z_f, n_axis);
        convert_double_to_float_kernel<<<(n_axis + threads - 1) / threads, threads>>>(d_axis_R_phi, d_axis_R_phi_f, n_axis);
        convert_double_to_float_kernel<<<(n_axis + threads - 1) / threads, threads>>>(d_axis_Z_phi, d_axis_Z_phi_f, n_axis);
        psi_fill_matrix_kernel_f32<<<n_points, threads, shared_f>>>(
            f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f, f->n_segments,
            d_R_f, d_Z_f, d_phi_f, d_axis_R_f, d_axis_Z_f, d_axis_R_phi_f, d_axis_Z_phi_f, n_axis,
            d_mode_a, d_mode_b, d_mode_m, d_mode_kind, n_coeff, nfp, static_cast<float>(a), poly_degree, m_tor, d_mat_f, d_rhs_f
        );
        float rhs_norm2_f = 0.0f;
        if (cuda_check(cudaGetLastError(), "fit psi_fill_matrix f32") ||
            cublas_check(cublasSdot(f->blas, n_points, d_rhs_f, 1, d_rhs_f, 1, &rhs_norm2_f), "fit rhs_norm2 s32") ||
            cuda_check(cudaDeviceSynchronize(), "fit assemble sync f32")) {
            cleanup();
            return 1;
        }
        rhs_norm2 = static_cast<double>(rhs_norm2_f);
    }
    stats.assemble_s = std::chrono::duration<double>(clock::now() - t).count();

    if (solver_mode == 1) {
        t = clock::now();
        if (precision_mode == 1) {
            size_t ata_bytes = static_cast<size_t>(n_coeff) * static_cast<size_t>(n_coeff) * sizeof(double);
            size_t atb_bytes = static_cast<size_t>(n_coeff) * sizeof(double);
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ata_d), ata_bytes), "fit d_ata_d") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_atb_d), atb_bytes), "fit d_atb_d")) {
                cleanup();
                return 1;
            }
            const double one = 1.0, zero = 0.0;
            if (cublas_check(cublasDgemm(f->blas, CUBLAS_OP_N, CUBLAS_OP_T, n_coeff, n_coeff, n_points, &one, d_mat_d, n_coeff, d_mat_d, n_coeff, &zero, d_ata_d, n_coeff), "fit dgemm") ||
                cublas_check(cublasDgemv(f->blas, CUBLAS_OP_N, n_coeff, n_points, &one, d_mat_d, n_coeff, d_rhs_d, 1, &zero, d_atb_d, 1), "fit dgemv") ||
                cuda_check(cudaDeviceSynchronize(), "fit normal_eq sync d64")) {
                cleanup();
                return 1;
            }
        } else {
            size_t ata_bytes_f = static_cast<size_t>(n_coeff) * static_cast<size_t>(n_coeff) * sizeof(float);
            size_t atb_bytes_f = static_cast<size_t>(n_coeff) * sizeof(float);
            size_t ata_bytes_d = static_cast<size_t>(n_coeff) * static_cast<size_t>(n_coeff) * sizeof(double);
            size_t atb_bytes_d = static_cast<size_t>(n_coeff) * sizeof(double);
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ata_f), ata_bytes_f), "fit d_ata_f") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_atb_f), atb_bytes_f), "fit d_atb_f")) {
                cleanup();
                return 1;
            }
            const float one = 1.0f, zero = 0.0f;
            if (cublas_check(cublasSgemm(f->blas, CUBLAS_OP_N, CUBLAS_OP_T, n_coeff, n_coeff, n_points, &one, d_mat_f, n_coeff, d_mat_f, n_coeff, &zero, d_ata_f, n_coeff), "fit sgemm") ||
                cublas_check(cublasSgemv(f->blas, CUBLAS_OP_N, n_coeff, n_points, &one, d_mat_f, n_coeff, d_rhs_f, 1, &zero, d_atb_f, 1), "fit sgemv")) {
                cleanup();
                return 1;
            }
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ata_d), ata_bytes_d), "fit d_ata_d from f") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_atb_d), atb_bytes_d), "fit d_atb_d from f")) {
                cleanup();
                return 1;
            }
            convert_float_to_double_kernel<<<(n_coeff * n_coeff + threads - 1) / threads, threads>>>(d_ata_f, d_ata_d, n_coeff * n_coeff);
            convert_float_to_double_kernel<<<blocks1d_coeff, threads>>>(d_atb_f, d_atb_d, n_coeff);
            if (cuda_check(cudaGetLastError(), "fit convert ata/atb f32->f64") ||
                cuda_check(cudaDeviceSynchronize(), "fit normal_eq sync f32")) {
                cleanup();
                return 1;
            }
        }
        stats.linear_prep_s = std::chrono::duration<double>(clock::now() - t).count();

        t = clock::now();
        if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_scale), static_cast<size_t>(n_coeff) * sizeof(double)), "fit d_scale") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)), "fit d_info")) {
            cleanup();
            return 1;
        }
        extract_scale_kernel<<<blocks1d_coeff, threads>>>(d_ata_d, d_scale, n_coeff);
        dim3 block2(16, 16);
        dim3 grid2((n_coeff + block2.x - 1) / block2.x, (n_coeff + block2.y - 1) / block2.y);
        scale_system_kernel<<<grid2, block2>>>(d_ata_d, d_scale, n_coeff, ridge);
        scale_rhs_kernel<<<blocks1d_coeff, threads>>>(d_atb_d, d_scale, n_coeff);
        if (cuda_check(cudaGetLastError(), "fit scale system") || cuda_check(cudaDeviceSynchronize(), "fit scale sync")) {
            cleanup();
            return 1;
        }
        int lwork = 0;
        if (cusolver_check(cusolverDnDgetrf_bufferSize(f->solver, n_coeff, n_coeff, d_ata_d, n_coeff, &lwork), "fit getrf buffer") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_info_rhs), static_cast<size_t>(lwork) * sizeof(double)), "fit getrf work") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ipiv), static_cast<size_t>(n_coeff) * sizeof(int)), "fit getrf piv")) {
            cleanup();
            return 1;
        }
        if (cusolver_check(cusolverDnDgetrf(f->solver, n_coeff, n_coeff, d_ata_d, n_coeff, d_info_rhs, d_ipiv, d_info), "fit getrf") ||
            cuda_check(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "fit getrf info copy")) {
            cleanup();
            return 1;
        }
        if (h_info != 0) return fail("fit getrf failed");
        if (cusolver_check(cusolverDnDgetrs(f->solver, CUBLAS_OP_N, n_coeff, 1, d_ata_d, n_coeff, d_ipiv, d_atb_d, n_coeff, d_info), "fit getrs") ||
            cuda_check(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "fit getrs info copy")) {
            cleanup();
            return 1;
        }
        if (h_info != 0) return fail("fit getrs failed");
        unscale_coeff_kernel<<<blocks1d_coeff, threads>>>(d_atb_d, d_scale, n_coeff);
        if (cuda_check(cudaGetLastError(), "fit unscale coeff") || cuda_check(cudaDeviceSynchronize(), "fit solve sync")) {
            cleanup();
            return 1;
        }
        stats.solve_s = std::chrono::duration<double>(clock::now() - t).count();
        d_coeff_src_d = d_atb_d;
    } else {
        if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_scale), static_cast<size_t>(n_coeff) * sizeof(double)), "fit qr d_scale") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)), "fit qr d_info")) {
            cleanup();
            return 1;
        }
        int qr_rows = n_points + n_coeff;
        size_t rhs_qr_bytes_d = static_cast<size_t>(qr_rows) * sizeof(double);
        size_t rhs_qr_bytes_f = static_cast<size_t>(qr_rows) * sizeof(float);
        dim3 block_t(16, 16);
        dim3 grid_t((n_coeff + block_t.x - 1) / block_t.x, (n_points + block_t.y - 1) / block_t.y);
        dim3 block_s(16, 16);
        dim3 grid_s((n_coeff + block_s.x - 1) / block_s.x, (n_points + block_s.y - 1) / block_s.y);
        double ridge_lambda = sqrt(fmax(ridge, 0.0));

        auto t_qr = clock::now();
        if (precision_mode == 1) {
            size_t mat_qr_bytes_d = static_cast<size_t>(qr_rows) * static_cast<size_t>(n_coeff) * sizeof(double);
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_qr_mat_d), mat_qr_bytes_d), "fit qr d_qr_mat_d") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_qr_rhs_d), rhs_qr_bytes_d), "fit qr d_qr_rhs_d")) {
                cleanup();
                return 1;
            }
            transpose_rowmajor_to_colmajor_kernel<<<grid_t, block_t>>>(d_mat_d, d_qr_mat_d, n_points, n_coeff, qr_rows);
            if (cuda_check(cudaGetLastError(), "fit qr transpose d64") ||
                cuda_check(cudaMemcpy(d_qr_rhs_d, d_rhs_d, point_bytes, cudaMemcpyDeviceToDevice), "fit qr copy rhs d64") ||
                cuda_check(cudaMemset(d_qr_rhs_d + n_points, 0, static_cast<size_t>(n_coeff) * sizeof(double)), "fit qr zero rhs tail d64") ||
                cuda_check(cudaDeviceSynchronize(), "fit qr transpose sync d64")) {
                cleanup();
                return 1;
            }
        } else {
            size_t mat_qr_bytes_f = static_cast<size_t>(qr_rows) * static_cast<size_t>(n_coeff) * sizeof(float);
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_qr_mat_f), mat_qr_bytes_f), "fit qr d_qr_mat_f") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_qr_rhs_f), rhs_qr_bytes_f), "fit qr d_qr_rhs_f")) {
                cleanup();
                return 1;
            }
            transpose_rowmajor_to_colmajor_kernel<<<grid_t, block_t>>>(d_mat_f, d_qr_mat_f, n_points, n_coeff, qr_rows);
            if (cuda_check(cudaGetLastError(), "fit qr transpose f32") ||
                cuda_check(cudaMemcpy(d_qr_rhs_f, d_rhs_f, static_cast<size_t>(n_points) * sizeof(float), cudaMemcpyDeviceToDevice), "fit qr copy rhs f32") ||
                cuda_check(cudaMemset(d_qr_rhs_f + n_points, 0, static_cast<size_t>(n_coeff) * sizeof(float)), "fit qr zero rhs tail f32") ||
                cuda_check(cudaDeviceSynchronize(), "fit qr transpose sync f32")) {
                cleanup();
                return 1;
            }
        }
        stats.qr_transpose_s = std::chrono::duration<double>(clock::now() - t_qr).count();

        t_qr = clock::now();
        if (precision_mode == 1) {
            column_norms_kernel_f64<<<n_coeff, threads>>>(d_qr_mat_d, d_scale, n_points, qr_rows, n_coeff);
            scale_columns_kernel_f64<<<grid_s, block_s>>>(d_qr_mat_d, d_scale, n_points, qr_rows, n_coeff);
            set_ridge_tail_kernel<double><<<(n_coeff * n_coeff + threads - 1) / threads, threads>>>(d_qr_mat_d, n_points, qr_rows, n_coeff, ridge_lambda);
        } else {
            column_norms_kernel_f32<<<n_coeff, threads>>>(d_qr_mat_f, d_scale, n_points, qr_rows, n_coeff);
            scale_columns_kernel_f32<<<grid_s, block_s>>>(d_qr_mat_f, d_scale, n_points, qr_rows, n_coeff);
            set_ridge_tail_kernel<float><<<(n_coeff * n_coeff + threads - 1) / threads, threads>>>(d_qr_mat_f, n_points, qr_rows, n_coeff, static_cast<float>(ridge_lambda));
        }
        if (cuda_check(cudaGetLastError(), "fit qr scale/augment") ||
            cuda_check(cudaDeviceSynchronize(), "fit qr scale/augment sync")) {
            cleanup();
            return 1;
        }
        stats.qr_scale_s = std::chrono::duration<double>(clock::now() - t_qr).count();
        stats.linear_prep_s = stats.qr_transpose_s + stats.qr_scale_s;

        t_qr = clock::now();
        if (precision_mode == 1) {
            int lwork_geqrf = 0, lwork_ormqr = 0;
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_tau_d), static_cast<size_t>(n_coeff) * sizeof(double)), "fit qr d_tau_d") ||
                cusolver_check(cusolverDnDgeqrf_bufferSize(f->solver, qr_rows, n_coeff, d_qr_mat_d, qr_rows, &lwork_geqrf), "fit qr geqrf buffer d64") ||
                cusolver_check(cusolverDnDormqr_bufferSize(f->solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, n_coeff, d_qr_mat_d, qr_rows, d_tau_d, d_qr_rhs_d, qr_rows, &lwork_ormqr), "fit qr ormqr buffer d64")) {
                cleanup();
                return 1;
            }
            int lwork = (lwork_geqrf > lwork_ormqr) ? lwork_geqrf : lwork_ormqr;
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_work_d), static_cast<size_t>(lwork) * sizeof(double)), "fit qr d_work_d")) {
                cleanup();
                return 1;
            }
            if (cusolver_check(cusolverDnDgeqrf(f->solver, qr_rows, n_coeff, d_qr_mat_d, qr_rows, d_tau_d, d_work_d, lwork, d_info), "fit qr geqrf d64") ||
                cuda_check(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "fit qr geqrf info d64")) {
                cleanup();
                return 1;
            }
            if (h_info != 0) return fail("fit qr geqrf failed");
        } else {
            int lwork_geqrf = 0, lwork_ormqr = 0;
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_tau_f), static_cast<size_t>(n_coeff) * sizeof(float)), "fit qr d_tau_f") ||
                cusolver_check(cusolverDnSgeqrf_bufferSize(f->solver, qr_rows, n_coeff, d_qr_mat_f, qr_rows, &lwork_geqrf), "fit qr geqrf buffer f32") ||
                cusolver_check(cusolverDnSormqr_bufferSize(f->solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, n_coeff, d_qr_mat_f, qr_rows, d_tau_f, d_qr_rhs_f, qr_rows, &lwork_ormqr), "fit qr ormqr buffer f32")) {
                cleanup();
                return 1;
            }
            int lwork = (lwork_geqrf > lwork_ormqr) ? lwork_geqrf : lwork_ormqr;
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_work_f), static_cast<size_t>(lwork) * sizeof(float)), "fit qr d_work_f")) {
                cleanup();
                return 1;
            }
            if (cusolver_check(cusolverDnSgeqrf(f->solver, qr_rows, n_coeff, d_qr_mat_f, qr_rows, d_tau_f, d_work_f, lwork, d_info), "fit qr geqrf f32") ||
                cuda_check(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "fit qr geqrf info f32")) {
                cleanup();
                return 1;
            }
            if (h_info != 0) return fail("fit qr geqrf failed");
        }
        if (cuda_check(cudaDeviceSynchronize(), "fit qr geqrf sync")) {
            cleanup();
            return 1;
        }
        stats.qr_factor_s = std::chrono::duration<double>(clock::now() - t_qr).count();

        t_qr = clock::now();
        if (precision_mode == 1) {
            int lwork_ormqr = 0;
            if (cusolver_check(cusolverDnDormqr_bufferSize(f->solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, n_coeff, d_qr_mat_d, qr_rows, d_tau_d, d_qr_rhs_d, qr_rows, &lwork_ormqr), "fit qr ormqr buffer2 d64")) {
                cleanup();
                return 1;
            }
            cudaFree(d_work_d);
            d_work_d = nullptr;
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_work_d), static_cast<size_t>(lwork_ormqr) * sizeof(double)), "fit qr d_work_d ormqr")) {
                cleanup();
                return 1;
            }
            if (cusolver_check(cusolverDnDormqr(f->solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, n_coeff, d_qr_mat_d, qr_rows, d_tau_d, d_qr_rhs_d, qr_rows, d_work_d, lwork_ormqr, d_info), "fit qr ormqr d64") ||
                cuda_check(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "fit qr ormqr info d64")) {
                cleanup();
                return 1;
            }
            if (h_info != 0) return fail("fit qr ormqr failed");
        } else {
            int lwork_ormqr = 0;
            if (cusolver_check(cusolverDnSormqr_bufferSize(f->solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, n_coeff, d_qr_mat_f, qr_rows, d_tau_f, d_qr_rhs_f, qr_rows, &lwork_ormqr), "fit qr ormqr buffer2 f32")) {
                cleanup();
                return 1;
            }
            cudaFree(d_work_f);
            d_work_f = nullptr;
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_work_f), static_cast<size_t>(lwork_ormqr) * sizeof(float)), "fit qr d_work_f ormqr")) {
                cleanup();
                return 1;
            }
            if (cusolver_check(cusolverDnSormqr(f->solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, n_coeff, d_qr_mat_f, qr_rows, d_tau_f, d_qr_rhs_f, qr_rows, d_work_f, lwork_ormqr, d_info), "fit qr ormqr f32") ||
                cuda_check(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "fit qr ormqr info f32")) {
                cleanup();
                return 1;
            }
            if (h_info != 0) return fail("fit qr ormqr failed");
        }
        if (cuda_check(cudaDeviceSynchronize(), "fit qr ormqr sync")) {
            cleanup();
            return 1;
        }
        stats.qr_apply_qtb_s = std::chrono::duration<double>(clock::now() - t_qr).count();

        t_qr = clock::now();
        if (precision_mode == 1) {
            if (cublas_check(cublasDtrsv(f->blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n_coeff, d_qr_mat_d, qr_rows, d_qr_rhs_d, 1), "fit qr trsv d64")) {
                cleanup();
                return 1;
            }
            unscale_coeff_kernel<<<blocks1d_coeff, threads>>>(d_qr_rhs_d, d_scale, n_coeff);
        } else {
            if (cublas_check(cublasStrsv(f->blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n_coeff, d_qr_mat_f, qr_rows, d_qr_rhs_f, 1), "fit qr trsv f32")) {
                cleanup();
                return 1;
            }
            unscale_coeff_kernel_f32<<<blocks1d_coeff, threads>>>(d_qr_rhs_f, d_scale, n_coeff);
        }
        if (cuda_check(cudaGetLastError(), "fit qr unscale coeff") ||
            cuda_check(cudaDeviceSynchronize(), "fit qr trsv sync")) {
            cleanup();
            return 1;
        }
        stats.qr_tri_s = std::chrono::duration<double>(clock::now() - t_qr).count();
        stats.solve_s = stats.qr_factor_s + stats.qr_apply_qtb_s + stats.qr_tri_s;
        if (precision_mode == 1) d_coeff_src_d = d_qr_rhs_d;
        else d_coeff_src_f = d_qr_rhs_f;
    }

    t = clock::now();
    if (precision_mode == 1) {
        if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pred_d), point_bytes), "fit d_pred_d")) {
            cleanup();
            return 1;
        }
        const double one = 1.0, zero = 0.0;
        if (cublas_check(cublasDgemv(f->blas, CUBLAS_OP_T, n_coeff, n_points, &one, d_mat_d, n_coeff, d_coeff_src_d, 1, &zero, d_pred_d, 1), "fit residual gemv d64")) {
            cleanup();
            return 1;
        }
        axpy_negative_kernel<<<blocks1d_points, threads>>>(d_pred_d, d_rhs_d, d_pred_d, n_points);
        double resid2 = 0.0;
        if (cuda_check(cudaGetLastError(), "fit residual axpy d64") ||
            cublas_check(cublasDdot(f->blas, n_points, d_pred_d, 1, d_pred_d, 1, &resid2), "fit residual dot d64") ||
            cuda_check(cudaDeviceSynchronize(), "fit residual sync d64")) {
            cleanup();
            return 1;
        }
        train_rms = sqrt(fmax(resid2, 0.0) / static_cast<double>(n_points));
    } else {
        if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pred_f), static_cast<size_t>(n_points) * sizeof(float)), "fit d_pred_f")) {
            cleanup();
            return 1;
        }
        if (solver_mode == 1) {
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_coeff_f), static_cast<size_t>(n_coeff) * sizeof(float)), "fit d_coeff_f")) {
                cleanup();
                return 1;
            }
            convert_double_to_float_kernel<<<blocks1d_coeff, threads>>>(d_atb_d, d_coeff_f, n_coeff);
            d_coeff_src_f = d_coeff_f;
        }
        const float one = 1.0f, zero = 0.0f;
        if (cublas_check(cublasSgemv(f->blas, CUBLAS_OP_T, n_coeff, n_points, &one, d_mat_f, n_coeff, d_coeff_src_f, 1, &zero, d_pred_f, 1), "fit residual gemv f32")) {
            cleanup();
            return 1;
        }
        axpy_negative_kernel<<<blocks1d_points, threads>>>(d_pred_f, d_rhs_f, d_pred_f, n_points);
        float resid2_f = 0.0f;
        if (cuda_check(cudaGetLastError(), "fit residual axpy f32") ||
            cublas_check(cublasSdot(f->blas, n_points, d_pred_f, 1, d_pred_f, 1, &resid2_f), "fit residual dot f32") ||
            cuda_check(cudaDeviceSynchronize(), "fit residual sync f32")) {
            cleanup();
            return 1;
        }
        train_rms = sqrt(fmax(static_cast<double>(resid2_f), 0.0) / static_cast<double>(n_points));
    }
    stats.residual_s = std::chrono::duration<double>(clock::now() - t).count();

    t = clock::now();
    if (precision_mode == 1) {
        if (cuda_check(cudaMemcpy(coeff_host, d_coeff_src_d, static_cast<size_t>(n_coeff) * sizeof(double), cudaMemcpyDeviceToHost), "fit copy coeff d64")) {
            cleanup();
            return 1;
        }
    } else {
        coeff_host_f.resize(static_cast<size_t>(n_coeff));
        if (cuda_check(cudaMemcpy(coeff_host_f.data(), d_coeff_src_f, static_cast<size_t>(n_coeff) * sizeof(float), cudaMemcpyDeviceToHost), "fit copy coeff f32")) {
            cleanup();
            return 1;
        }
        for (int i = 0; i < n_coeff; ++i) coeff_host[i] = static_cast<double>(coeff_host_f[static_cast<size_t>(i)]);
    }
    *train_rms_out = train_rms;
    stats.copy_out_s = std::chrono::duration<double>(clock::now() - t).count();
    stats.total_s = std::chrono::duration<double>(clock::now() - t_total).count();
    if (stats_out && stats_len >= 12) {
        stats_out[0] = stats.copy_in_s;
        stats_out[1] = stats.assemble_s;
        stats_out[2] = stats.linear_prep_s;
        stats_out[3] = stats.solve_s;
        stats_out[4] = stats.residual_s;
        stats_out[5] = stats.copy_out_s;
        stats_out[6] = stats.total_s;
        stats_out[7] = stats.qr_transpose_s;
        stats_out[8] = stats.qr_scale_s;
        stats_out[9] = stats.qr_factor_s;
        stats_out[10] = stats.qr_apply_qtb_s;
        stats_out[11] = stats.qr_tri_s;
    }
    cleanup();
    set_error("");
    return 0;
}

int sgpu_surface_points_from_level(
    const double* coeff_host,
    const int* mode_a_host,
    const int* mode_b_host,
    const int* mode_m_host,
    const int* mode_kind_host,
    int n_coeff,
    int nfp,
    double a,
    int poly_degree,
    int m_tor,
    const double* axis_R_host,
    const double* axis_Z_host,
    int n_axis,
    int order,
    double psi_level,
    int maxiter,
    double tol,
    double max_radius_scale,
    int device_id,
    double* xyz_host,
    double* radii_host,
    double* stats_out,
    int stats_len
) {
    using clock = std::chrono::steady_clock;
    if (!coeff_host || !mode_a_host || !mode_b_host || !mode_m_host || !mode_kind_host ||
        !axis_R_host || !axis_Z_host || !xyz_host || !radii_host || n_coeff <= 0 || nfp <= 0 ||
        poly_degree < 2 || poly_degree > MAX_PSI_DEGREE || m_tor < 0 || m_tor > MAX_PSI_MTOR ||
        n_axis <= 1 || order < 1 || maxiter <= 0 || tol <= 0.0 || max_radius_scale <= 0.0) {
        set_error("invalid surface_points_from_level arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(device_id), "cudaSetDevice")) return 1;
    int nphi = 2 * order + 1;
    int ntheta = 2 * order + 1;
    size_t coeff_bytes = static_cast<size_t>(n_coeff) * sizeof(double);
    size_t mode_bytes = static_cast<size_t>(n_coeff) * sizeof(int);
    size_t axis_bytes = static_cast<size_t>(n_axis) * sizeof(double);
    size_t ray_coeff_bytes = static_cast<size_t>(nphi) * static_cast<size_t>(ntheta) * static_cast<size_t>(poly_degree + 1) * sizeof(double);
    size_t xyz_bytes = static_cast<size_t>(nphi) * static_cast<size_t>(ntheta) * 3 * sizeof(double);
    size_t radii_bytes = static_cast<size_t>(nphi) * static_cast<size_t>(ntheta) * sizeof(double);

    double *d_coeff = nullptr, *d_axis_R = nullptr, *d_axis_Z = nullptr, *d_ray_coeff = nullptr, *d_xyz = nullptr, *d_radii = nullptr;
    int *d_mode_a = nullptr, *d_mode_b = nullptr, *d_mode_m = nullptr, *d_mode_kind = nullptr;
    auto cleanup = [&]() {
        cudaFree(d_coeff);
        cudaFree(d_axis_R);
        cudaFree(d_axis_Z);
        cudaFree(d_ray_coeff);
        cudaFree(d_xyz);
        cudaFree(d_radii);
        cudaFree(d_mode_a);
        cudaFree(d_mode_b);
        cudaFree(d_mode_m);
        cudaFree(d_mode_kind);
    };

    double copy_in_s = 0.0, coeff_build_s = 0.0, newton_s = 0.0, copy_out_s = 0.0;
    auto t_total = clock::now();
    auto t = clock::now();
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_coeff), coeff_bytes), "surface d_coeff") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_R), axis_bytes), "surface d_axis_R") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_Z), axis_bytes), "surface d_axis_Z") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_a), mode_bytes), "surface d_mode_a") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_b), mode_bytes), "surface d_mode_b") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_m), mode_bytes), "surface d_mode_m") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_kind), mode_bytes), "surface d_mode_kind") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ray_coeff), ray_coeff_bytes), "surface d_ray_coeff") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_xyz), xyz_bytes), "surface d_xyz") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_radii), radii_bytes), "surface d_radii") ||
        cuda_check(cudaMemcpy(d_coeff, coeff_host, coeff_bytes, cudaMemcpyHostToDevice), "surface copy coeff") ||
        cuda_check(cudaMemcpy(d_axis_R, axis_R_host, axis_bytes, cudaMemcpyHostToDevice), "surface copy axis_R") ||
        cuda_check(cudaMemcpy(d_axis_Z, axis_Z_host, axis_bytes, cudaMemcpyHostToDevice), "surface copy axis_Z") ||
        cuda_check(cudaMemcpy(d_mode_a, mode_a_host, mode_bytes, cudaMemcpyHostToDevice), "surface copy mode_a") ||
        cuda_check(cudaMemcpy(d_mode_b, mode_b_host, mode_bytes, cudaMemcpyHostToDevice), "surface copy mode_b") ||
        cuda_check(cudaMemcpy(d_mode_m, mode_m_host, mode_bytes, cudaMemcpyHostToDevice), "surface copy mode_m") ||
        cuda_check(cudaMemcpy(d_mode_kind, mode_kind_host, mode_bytes, cudaMemcpyHostToDevice), "surface copy mode_kind")) {
        cleanup();
        return 1;
    }
    copy_in_s = std::chrono::duration<double>(clock::now() - t).count();

    t = clock::now();
    surface_build_ray_poly_kernel<<<nphi, SURFACE_THREADS, static_cast<size_t>(2 * (MAX_PSI_MTOR + 1)) * sizeof(double)>>>(
        d_coeff, d_mode_a, d_mode_b, d_mode_m, d_mode_kind, n_coeff, nfp, poly_degree, m_tor, nphi, ntheta, d_ray_coeff
    );
    if (cuda_check(cudaGetLastError(), "surface_build_ray_poly_kernel") ||
        cuda_check(cudaDeviceSynchronize(), "surface_build_ray_poly sync")) {
        cleanup();
        return 1;
    }
    coeff_build_s = std::chrono::duration<double>(clock::now() - t).count();

    t = clock::now();
    surface_newton_from_ray_poly_kernel<<<nphi, SURFACE_THREADS>>>(
        d_ray_coeff,
        poly_degree,
        d_axis_R,
        d_axis_Z,
        n_axis,
        nfp,
        a,
        nphi,
        ntheta,
        psi_level,
        maxiter,
        tol,
        max_radius_scale,
        d_xyz,
        d_radii
    );
    if (cuda_check(cudaGetLastError(), "surface_newton_from_ray_poly_kernel") ||
        cuda_check(cudaDeviceSynchronize(), "surface_newton_from_ray_poly sync")) {
        cleanup();
        return 1;
    }
    newton_s = std::chrono::duration<double>(clock::now() - t).count();

    t = clock::now();
    if (cuda_check(cudaMemcpy(xyz_host, d_xyz, xyz_bytes, cudaMemcpyDeviceToHost), "surface copy xyz") ||
        cuda_check(cudaMemcpy(radii_host, d_radii, radii_bytes, cudaMemcpyDeviceToHost), "surface copy radii")) {
        cleanup();
        return 1;
    }
    copy_out_s = std::chrono::duration<double>(clock::now() - t).count();

    if (stats_out && stats_len >= 5) {
        stats_out[0] = copy_in_s;
        stats_out[1] = coeff_build_s;
        stats_out[2] = newton_s;
        stats_out[3] = copy_out_s;
        stats_out[4] = std::chrono::duration<double>(clock::now() - t_total).count();
    }
    cleanup();
    set_error("");
    return 0;
}

int sgpu_trace_period(void* handle, const double* R0_host, const double* Z0_host, double* R1_host, double* Z1_host, int n_lines, int nfp, int steps) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !R0_host || !Z0_host || !R1_host || !Z1_host || n_lines < 0 || nfp <= 0 || steps <= 0) {
        set_error("invalid trace_period arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    double *d_R0 = nullptr, *d_Z0 = nullptr, *d_R1 = nullptr, *d_Z1 = nullptr;
    size_t bytes = static_cast<size_t>(n_lines) * sizeof(double);
    if (cuda_check(cudaMalloc(&d_R0, bytes), "cudaMalloc R0") ||
        cuda_check(cudaMalloc(&d_Z0, bytes), "cudaMalloc Z0") ||
        cuda_check(cudaMalloc(&d_R1, bytes), "cudaMalloc R1") ||
        cuda_check(cudaMalloc(&d_Z1, bytes), "cudaMalloc Z1")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    if (cuda_check(cudaMemcpy(d_R0, R0_host, bytes, cudaMemcpyHostToDevice), "copy R0") ||
        cuda_check(cudaMemcpy(d_Z0, Z0_host, bytes, cudaMemcpyHostToDevice), "copy Z0")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    int blocks = (n_lines + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(double);
    trace_period_kernel<<<blocks, THREADS_PER_BLOCK, shmem>>>(f->d_x, f->d_y, f->d_z, f->d_wx, f->d_wy, f->d_wz, f->n_segments, d_R0, d_Z0, d_R1, d_Z1, n_lines, nfp, steps);
    if (cuda_check(cudaGetLastError(), "trace_period kernel") ||
        cuda_check(cudaDeviceSynchronize(), "trace_period sync") ||
        cuda_check(cudaMemcpy(R1_host, d_R1, bytes, cudaMemcpyDeviceToHost), "copy R1") ||
        cuda_check(cudaMemcpy(Z1_host, d_Z1, bytes, cudaMemcpyDeviceToHost), "copy Z1")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
    set_error("");
    return 0;
}

int sgpu_trace_period_blockline(void* handle, const double* R0_host, const double* Z0_host, double* R1_host, double* Z1_host, int n_lines, int nfp, int steps, int threads_per_line) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !R0_host || !Z0_host || !R1_host || !Z1_host || n_lines < 0 || nfp <= 0 || steps <= 0) {
        set_error("invalid trace_period_blockline arguments");
        return 1;
    }
    if (!(threads_per_line == 128 || threads_per_line == 256 || threads_per_line == 512 || threads_per_line == 1024)) {
        set_error("threads_per_line must be 128, 256, 512, or 1024");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    double *d_R0 = nullptr, *d_Z0 = nullptr, *d_R1 = nullptr, *d_Z1 = nullptr;
    size_t bytes = static_cast<size_t>(n_lines) * sizeof(double);
    if (cuda_check(cudaMalloc(&d_R0, bytes), "cudaMalloc R0") ||
        cuda_check(cudaMalloc(&d_Z0, bytes), "cudaMalloc Z0") ||
        cuda_check(cudaMalloc(&d_R1, bytes), "cudaMalloc R1") ||
        cuda_check(cudaMalloc(&d_Z1, bytes), "cudaMalloc Z1")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    if (cuda_check(cudaMemcpy(d_R0, R0_host, bytes, cudaMemcpyHostToDevice), "copy R0") ||
        cuda_check(cudaMemcpy(d_Z0, Z0_host, bytes, cudaMemcpyHostToDevice), "copy Z0")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(double) + static_cast<size_t>(threads_per_line) * 3 * sizeof(double);
    trace_period_blockline_kernel<<<n_lines, threads_per_line, shmem>>>(f->d_x, f->d_y, f->d_z, f->d_wx, f->d_wy, f->d_wz, f->n_segments, d_R0, d_Z0, d_R1, d_Z1, n_lines, nfp, steps);
    if (cuda_check(cudaGetLastError(), "trace_period_blockline kernel") ||
        cuda_check(cudaDeviceSynchronize(), "trace_period_blockline sync") ||
        cuda_check(cudaMemcpy(R1_host, d_R1, bytes, cudaMemcpyDeviceToHost), "copy R1") ||
        cuda_check(cudaMemcpy(Z1_host, d_Z1, bytes, cudaMemcpyDeviceToHost), "copy Z1")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
    set_error("");
    return 0;
}

int sgpu_trace_period_blockline_mixed(void* handle, const double* R0_host, const double* Z0_host, double* R1_host, double* Z1_host, int n_lines, int nfp, int steps, int threads_per_line, int mode) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !R0_host || !Z0_host || !R1_host || !Z1_host || n_lines < 0 || nfp <= 0 || steps <= 0) {
        set_error("invalid trace_period_blockline_mixed arguments");
        return 1;
    }
    if (threads_per_line != 128 && threads_per_line != 256 && threads_per_line != 512 && threads_per_line != 1024) {
        set_error("threads_per_line must be 128, 256, 512, or 1024");
        return 1;
    }
    if (mode != 1 && mode != 2 && mode != 3) {
        set_error("mixed mode must be 1 (B fp32, state fp64), 2 (B/state fp32), or 3 (B fp32, state fp16)");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    double *d_R0 = nullptr, *d_Z0 = nullptr, *d_R1 = nullptr, *d_Z1 = nullptr;
    size_t bytes = static_cast<size_t>(n_lines) * sizeof(double);
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R0), bytes), "trace mixed d_R0") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z0), bytes), "trace mixed d_Z0") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R1), bytes), "trace mixed d_R1") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z1), bytes), "trace mixed d_Z1") ||
        cuda_check(cudaMemcpy(d_R0, R0_host, bytes, cudaMemcpyHostToDevice), "trace mixed copy R0") ||
        cuda_check(cudaMemcpy(d_Z0, Z0_host, bytes, cudaMemcpyHostToDevice), "trace mixed copy Z0")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    size_t shmem = static_cast<size_t>(SEG_TILE * 6 + threads_per_line * 3) * sizeof(float);
    if (mode == 1) {
        trace_period_blockline_bf32_state64_kernel<<<n_lines, threads_per_line, shmem>>>(
            f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
            f->n_segments, d_R0, d_Z0, d_R1, d_Z1, n_lines, nfp, steps
        );
    } else if (mode == 2) {
        trace_period_blockline_f32_kernel<<<n_lines, threads_per_line, shmem>>>(
            f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
            f->n_segments, d_R0, d_Z0, d_R1, d_Z1, n_lines, nfp, steps
        );
    } else {
        trace_period_blockline_f32_state16_kernel<<<n_lines, threads_per_line, shmem>>>(
            f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
            f->n_segments, d_R0, d_Z0, d_R1, d_Z1, n_lines, nfp, steps
        );
    }
    if (cuda_check(cudaGetLastError(), "trace_period_blockline_mixed kernel") ||
        cuda_check(cudaDeviceSynchronize(), "trace_period_blockline_mixed sync") ||
        cuda_check(cudaMemcpy(R1_host, d_R1, bytes, cudaMemcpyDeviceToHost), "trace mixed copy R1") ||
        cuda_check(cudaMemcpy(Z1_host, d_Z1, bytes, cudaMemcpyDeviceToHost), "trace mixed copy Z1")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
    set_error("");
    return 0;
}

int sgpu_trace_axis_samples(
    void* handle,
    double R0,
    double Z0,
    int nfp,
    int integration_steps,
    int n_samples,
    double* R_host,
    double* Z_host,
    double* R_phi_host,
    double* Z_phi_host
) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !R_host || !Z_host || !R_phi_host || !Z_phi_host ||
        nfp <= 0 || integration_steps <= 0 || n_samples <= 1) {
        set_error("invalid trace_axis_samples arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "trace axis cudaSetDevice")) return 1;
    double *d_R = nullptr, *d_Z = nullptr, *d_R_phi = nullptr, *d_Z_phi = nullptr;
    const size_t bytes = static_cast<size_t>(n_samples) * sizeof(double);
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R), bytes), "trace axis d_R") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z), bytes), "trace axis d_Z") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R_phi), bytes), "trace axis d_R_phi") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z_phi), bytes), "trace axis d_Z_phi")) {
        cudaFree(d_R); cudaFree(d_Z); cudaFree(d_R_phi); cudaFree(d_Z_phi);
        return 1;
    }
    const int substeps = (integration_steps + n_samples - 1) / n_samples;
    const int threads = 256;
    const size_t shmem = static_cast<size_t>(SEG_TILE * 6 + threads * 3) * sizeof(float);
    trace_axis_samples_mixed_kernel<<<1, threads, shmem>>>(
        f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
        f->n_segments, R0, Z0, nfp, substeps, n_samples,
        d_R, d_Z, d_R_phi, d_Z_phi
    );
    if (cuda_check(cudaGetLastError(), "trace axis kernel") ||
        cuda_check(cudaDeviceSynchronize(), "trace axis sync") ||
        cuda_check(cudaMemcpy(R_host, d_R, bytes, cudaMemcpyDeviceToHost), "trace axis copy R") ||
        cuda_check(cudaMemcpy(Z_host, d_Z, bytes, cudaMemcpyDeviceToHost), "trace axis copy Z") ||
        cuda_check(cudaMemcpy(R_phi_host, d_R_phi, bytes, cudaMemcpyDeviceToHost), "trace axis copy R_phi") ||
        cuda_check(cudaMemcpy(Z_phi_host, d_Z_phi, bytes, cudaMemcpyDeviceToHost), "trace axis copy Z_phi")) {
        cudaFree(d_R); cudaFree(d_Z); cudaFree(d_R_phi); cudaFree(d_Z_phi);
        return 1;
    }
    cudaFree(d_R); cudaFree(d_Z); cudaFree(d_R_phi); cudaFree(d_Z_phi);
    set_error("");
    return 0;
}

int sgpu_internal_eval_B_f32_device(
    void* handle,
    const float* xyz_device,
    float* B_device,
    int n_points
) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !xyz_device || !B_device || n_points < 0) {
        set_error("invalid internal eval_B_f32_device arguments");
        return 1;
    }
    const int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    eval_B_only_kernel<float><<<blocks, THREADS_PER_BLOCK, shmem>>>(
        f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
        f->n_segments, xyz_device, B_device, n_points
    );
    return cuda_check(cudaGetLastError(), "internal eval_B_f32_device kernel");
}

int sgpu_internal_eval_B_grad_f32_device(
    void* handle,
    const float* xyz_device,
    float* B_device,
    float* grad_B_device,
    int n_points
) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !xyz_device || !B_device || !grad_B_device || n_points < 0) {
        set_error("invalid internal eval_B_grad_f32_device arguments");
        return 1;
    }
    const int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    eval_B_grad_kernel<float><<<blocks, THREADS_PER_BLOCK, shmem>>>(
        f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
        f->n_segments, xyz_device, B_device, grad_B_device, n_points
    );
    return cuda_check(cudaGetLastError(), "internal eval_B_grad_f32_device kernel");
}

int sgpu_internal_B_grad_segment_vjp_f32_device(
    void* handle,
    const float* xyz_device,
    const float* adj_B_device,
    const float* adj_grad_B_device,
    int n_points,
    float* adj_segment_position_device,
    float* adj_segment_weight_device
) {
    auto* field = reinterpret_cast<CoilField*>(handle);
    if (!field || !xyz_device || !adj_B_device || !adj_grad_B_device ||
        n_points <= 0 || !adj_segment_position_device || !adj_segment_weight_device) {
        set_error("invalid internal B/grad(B) segment VJP arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "internal B/grad(B) VJP cudaSetDevice")) return 1;
    constexpr int threads = 256;
    B_grad_segment_vjp_kernel<<<field->n_segments, threads>>>(
        field->d_x_f, field->d_y_f, field->d_z_f,
        field->d_wx_f, field->d_wy_f, field->d_wz_f,
        field->n_segments, xyz_device, adj_B_device, adj_grad_B_device,
        n_points, adj_segment_position_device, adj_segment_weight_device
    );
    return cuda_check(cudaGetLastError(), "internal B/grad(B) segment VJP kernel");
}

int sgpu_internal_B_grad_point_vjp_f32_device(
    void* handle,
    const float* xyz_device,
    const float* adj_B_device,
    const float* adj_grad_B_device,
    int n_points,
    float* adj_xyz_device
) {
    auto* field = reinterpret_cast<CoilField*>(handle);
    if (!field || !xyz_device || !adj_B_device || !adj_grad_B_device ||
        n_points <= 0 || !adj_xyz_device) {
        set_error("invalid internal B/grad(B) point VJP arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "internal point VJP cudaSetDevice")) return 1;
    const int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    B_grad_point_vjp_kernel<<<blocks, THREADS_PER_BLOCK, shared_bytes>>>(
        field->d_x_f, field->d_y_f, field->d_z_f,
        field->d_wx_f, field->d_wy_f, field->d_wz_f,
        field->n_segments, xyz_device, adj_B_device, adj_grad_B_device,
        n_points, adj_xyz_device
    );
    return cuda_check(cudaGetLastError(), "internal B/grad(B) point VJP kernel");
}

void sgpu_internal_set_error(const char* message) {
    set_error(message);
}

const char* sgpu_last_error() {
    return g_last_error.c_str();
}

} // extern "C"
