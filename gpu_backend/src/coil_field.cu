#include "coil_field.h"
#include "nvtx_profile.h"
#include "coil_field_internal.h"
#include "psi_qr_snapshot.h"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <exception>
#include <fstream>
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

struct PsiWarmPreconditionerCache {
    bool capture_enabled = false;
    int coefficient_count = 0;
    std::vector<float> upper_factor;
    std::vector<double> center_scale;
};

thread_local PsiWarmPreconditionerCache g_psi_warm_preconditioner;

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

struct BatchCoilField {
    int device_id = 0;
    int query_count = 0;
    int n_segments = 0;
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_z = nullptr;
    float* d_wx = nullptr;
    float* d_wy = nullptr;
    float* d_wz = nullptr;
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

__device__ void eval_fourier_device(
    const double* coefficients,
    int order,
    double t,
    double& value,
    double& derivative
) {
    value = coefficients[0];
    derivative = 0.0;
    for (int mode = 1; mode <= order; ++mode) {
        double sine = 0.0;
        double cosine = 0.0;
        sincos(TWOPI * static_cast<double>(mode) * t, &sine, &cosine);
        const double sin_coefficient = coefficients[2 * mode - 1];
        const double cos_coefficient = coefficients[2 * mode];
        value += sin_coefficient * sine + cos_coefficient * cosine;
        derivative += TWOPI * static_cast<double>(mode) *
            (sin_coefficient * cosine - cos_coefficient * sine);
    }
}

__global__ void generate_segment_batch_kernel(
    const double* __restrict__ coeffs_x,
    const double* __restrict__ coeffs_y,
    const double* __restrict__ coeffs_z,
    const double* __restrict__ currents,
    int query_count,
    int nbase,
    int ncoeff,
    int nfp,
    int segments_per_coil,
    int segments_per_query,
    float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ z,
    float* __restrict__ wx,
    float* __restrict__ wy,
    float* __restrict__ wz
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(query_count) * segments_per_query;
    if (flat >= total) return;
    const int query = static_cast<int>(flat / segments_per_query);
    int local = static_cast<int>(flat - static_cast<size_t>(query) * segments_per_query);
    const int image = local % (2 * nfp);
    local /= 2 * nfp;
    const int segment = local % segments_per_coil;
    const int coil = local / segments_per_coil;
    const int period_index = image / 2;
    const bool reflected = (image & 1) != 0;
    const size_t coefficient_offset =
        (static_cast<size_t>(query) * nbase + coil) * ncoeff;
    const double t = (static_cast<double>(segment) + 0.5) /
        static_cast<double>(segments_per_coil);
    const int order = (ncoeff - 1) / 2;
    double px = 0.0, py = 0.0, pz = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    eval_fourier_device(coeffs_x + coefficient_offset, order, t, px, vx);
    eval_fourier_device(coeffs_y + coefficient_offset, order, t, py, vy);
    eval_fourier_device(coeffs_z + coefficient_offset, order, t, pz, vz);
    const double segment_scale = 1.0 / static_cast<double>(segments_per_coil);
    vx *= segment_scale;
    vy *= segment_scale;
    vz *= segment_scale;
    double current = currents[static_cast<size_t>(query) * nbase + coil];
    if (reflected) {
        py = -py;
        pz = -pz;
        vy = -vy;
        vz = -vz;
        current = -current;
    }
    const double angle = TWOPI * static_cast<double>(period_index) /
        static_cast<double>(nfp);
    double sine = 0.0;
    double cosine = 0.0;
    sincos(angle, &sine, &cosine);
    const double rx = cosine * px - sine * py;
    const double ry = sine * px + cosine * py;
    const double rvx = cosine * vx - sine * vy;
    const double rvy = sine * vx + cosine * vy;
    x[flat] = static_cast<float>(rx);
    y[flat] = static_cast<float>(ry);
    z[flat] = static_cast<float>(pz);
    wx[flat] = static_cast<float>(current * rvx);
    wy[flat] = static_cast<float>(current * rvy);
    wz[flat] = static_cast<float>(current * vz);
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

template <bool WITH_GRADIENT>
__global__ void eval_B_batch_f32_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const float* __restrict__ xyz,
    float* __restrict__ B,
    float* __restrict__ grad_B,
    int points_per_query,
    int blocks_per_query
) {
    extern __shared__ float shared[];
    float* sx = shared;
    float* sy = sx + SEG_TILE;
    float* sz = sy + SEG_TILE;
    float* swx = sz + SEG_TILE;
    float* swy = swx + SEG_TILE;
    float* swz = swy + SEG_TILE;

    const int query = blockIdx.x / blocks_per_query;
    const int query_block = blockIdx.x - query * blocks_per_query;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int point = query_block * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE;
    const bool valid = point < points_per_query;
    const size_t point_index = static_cast<size_t>(query) * points_per_query + point;
    const size_t segment_offset = static_cast<size_t>(query) * nseg;
    const float px = valid ? xyz[3 * point_index] : 0.0f;
    const float py = valid ? xyz[3 * point_index + 1] : 0.0f;
    const float pz = valid ? xyz[3 * point_index + 2] : 0.0f;
    float accum[12] = {};

    for (int base = 0; base < nseg; base += SEG_TILE) {
        const int count = min(SEG_TILE, nseg - base);
        for (int item = threadIdx.x; item < count; item += blockDim.x) {
            const size_t source = segment_offset + base + item;
            sx[item] = seg_x[source];
            sy[item] = seg_y[source];
            sz[item] = seg_z[source];
            swx[item] = seg_wx[source];
            swy[item] = seg_wy[source];
            swz[item] = seg_wz[source];
        }
        __syncthreads();
        if (valid) {
            for (int item = lane; item < count; item += WARP_SIZE) {
                const float rx = px - sx[item];
                const float ry = py - sy[item];
                const float rz = pz - sz[item];
                const float r2 = rx * rx + ry * ry + rz * rz + 1.0e-30f;
                const float invr = rsqrtf(r2);
                const float invr2 = invr * invr;
                const float invr3 = invr * invr2;
                const float wx_value = swx[item];
                const float wy_value = swy[item];
                const float wz_value = swz[item];
                const float ux = wy_value * rz - wz_value * ry;
                const float uy = wz_value * rx - wx_value * rz;
                const float uz = wx_value * ry - wy_value * rx;
                accum[0] += ux * invr3;
                accum[1] += uy * invr3;
                accum[2] += uz * invr3;
                if constexpr (WITH_GRADIENT) {
                    const float invr5 = invr3 * invr2;
                    const float common_x = -3.0f * rx * invr5;
                    const float common_y = -3.0f * ry * invr5;
                    const float common_z = -3.0f * rz * invr5;
                    accum[3] += ux * common_x;
                    accum[4] += -wz_value * invr3 + ux * common_y;
                    accum[5] += wy_value * invr3 + ux * common_z;
                    accum[6] += wz_value * invr3 + uy * common_x;
                    accum[7] += uy * common_y;
                    accum[8] += -wx_value * invr3 + uy * common_z;
                    accum[9] += -wy_value * invr3 + uz * common_x;
                    accum[10] += wx_value * invr3 + uz * common_y;
                    accum[11] += uz * common_z;
                }
            }
        }
        __syncthreads();
    }
    const int component_count = WITH_GRADIENT ? 12 : 3;
    for (int component = 0; component < component_count; ++component) {
        accum[component] = warp_sum_t(accum[component]) *
            static_cast<float>(MU0_OVER_4PI);
    }
    if (valid && lane == 0) {
        B[3 * point_index] = accum[0];
        B[3 * point_index + 1] = accum[1];
        B[3 * point_index + 2] = accum[2];
        if constexpr (WITH_GRADIENT) {
            for (int component = 0; component < 9; ++component) {
                grad_B[9 * point_index + component] = accum[3 + component];
            }
        }
    }
}

__global__ void trace_period_batch_mixed_kernel(
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
    int lines_per_query,
    int nfp,
    int steps
) {
    extern __shared__ float shared[];
    const int query = blockIdx.x / lines_per_query;
    const int line = blockIdx.x - query * lines_per_query;
    const size_t line_index = static_cast<size_t>(query) * lines_per_query + line;
    const size_t segment_offset = static_cast<size_t>(query) * nseg;
    double R = R0[line_index];
    double Z = Z0[line_index];
    const double period = TWOPI / static_cast<double>(nfp);
    const double h = period / static_cast<double>(steps);
    for (int step = 0; step < steps; ++step) {
        const double phi = h * static_cast<double>(step);
        double k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
        rhs_cyl_block_bf32_state64(
            R, Z, phi, seg_x + segment_offset, seg_y + segment_offset,
            seg_z + segment_offset, seg_wx + segment_offset,
            seg_wy + segment_offset, seg_wz + segment_offset, nseg, shared, k1r, k1z
        );
        rhs_cyl_block_bf32_state64(
            R + 0.5 * h * k1r, Z + 0.5 * h * k1z, phi + 0.5 * h,
            seg_x + segment_offset, seg_y + segment_offset, seg_z + segment_offset,
            seg_wx + segment_offset, seg_wy + segment_offset, seg_wz + segment_offset,
            nseg, shared, k2r, k2z
        );
        rhs_cyl_block_bf32_state64(
            R + 0.5 * h * k2r, Z + 0.5 * h * k2z, phi + 0.5 * h,
            seg_x + segment_offset, seg_y + segment_offset, seg_z + segment_offset,
            seg_wx + segment_offset, seg_wy + segment_offset, seg_wz + segment_offset,
            nseg, shared, k3r, k3z
        );
        rhs_cyl_block_bf32_state64(
            R + h * k3r, Z + h * k3z, phi + h,
            seg_x + segment_offset, seg_y + segment_offset, seg_z + segment_offset,
            seg_wx + segment_offset, seg_wy + segment_offset, seg_wz + segment_offset,
            nseg, shared, k4r, k4z
        );
        R += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
        Z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    }
    if (threadIdx.x == 0) {
        R1[line_index] = R;
        Z1[line_index] = Z;
    }
}

__global__ void trace_axis_samples_batch_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R0,
    const double* __restrict__ Z0,
    int nfp,
    int substeps_per_sample,
    int sample_count,
    double* __restrict__ R_out,
    double* __restrict__ Z_out,
    double* __restrict__ R_phi_out,
    double* __restrict__ Z_phi_out
) {
    extern __shared__ float shared[];
    const int query = blockIdx.x;
    const size_t segment_offset = static_cast<size_t>(query) * nseg;
    double R = R0[query];
    double Z = Z0[query];
    const int total_steps = sample_count * substeps_per_sample;
    const double period = TWOPI / static_cast<double>(nfp);
    const double h = period / static_cast<double>(total_steps);
    for (int sample = 0; sample < sample_count; ++sample) {
        for (int substep = 0; substep < substeps_per_sample; ++substep) {
            const int step = sample * substeps_per_sample + substep;
            const double phi = h * static_cast<double>(step);
            double k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
            rhs_cyl_block_bf32_state64(
                R, Z, phi, seg_x + segment_offset, seg_y + segment_offset,
                seg_z + segment_offset, seg_wx + segment_offset,
                seg_wy + segment_offset, seg_wz + segment_offset, nseg, shared, k1r, k1z
            );
            if (substep == 0 && threadIdx.x == 0) {
                const size_t output = static_cast<size_t>(query) * sample_count + sample;
                R_out[output] = R;
                Z_out[output] = Z;
                R_phi_out[output] = k1r;
                Z_phi_out[output] = k1z;
            }
            rhs_cyl_block_bf32_state64(
                R + 0.5 * h * k1r, Z + 0.5 * h * k1z, phi + 0.5 * h,
                seg_x + segment_offset, seg_y + segment_offset, seg_z + segment_offset,
                seg_wx + segment_offset, seg_wy + segment_offset, seg_wz + segment_offset,
                nseg, shared, k2r, k2z
            );
            rhs_cyl_block_bf32_state64(
                R + 0.5 * h * k2r, Z + 0.5 * h * k2z, phi + 0.5 * h,
                seg_x + segment_offset, seg_y + segment_offset, seg_z + segment_offset,
                seg_wx + segment_offset, seg_wy + segment_offset, seg_wz + segment_offset,
                nseg, shared, k3r, k3z
            );
            rhs_cyl_block_bf32_state64(
                R + h * k3r, Z + h * k3z, phi + h,
                seg_x + segment_offset, seg_y + segment_offset, seg_z + segment_offset,
                seg_wx + segment_offset, seg_wy + segment_offset, seg_wz + segment_offset,
                nseg, shared, k4r, k4z
            );
            R += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
            Z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
        }
    }
}

bool write_device_bytes(std::ofstream& stream, const void* device_data, std::size_t bytes) {
    constexpr std::size_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
    std::vector<unsigned char> host_buffer(std::min(bytes, chunk_bytes));
    const auto* source = static_cast<const unsigned char*>(device_data);
    for (std::size_t offset = 0; offset < bytes; offset += host_buffer.size()) {
        const std::size_t count = std::min(host_buffer.size(), bytes - offset);
        if (cuda_check(cudaMemcpy(host_buffer.data(), source + offset, count, cudaMemcpyDeviceToHost),
                       "psi QR snapshot copy to host")) {
            return false;
        }
        stream.write(reinterpret_cast<const char*>(host_buffer.data()), static_cast<std::streamsize>(count));
        if (!stream) {
            set_error("failed to write psi QR snapshot payload");
            return false;
        }
    }
    return true;
}

bool maybe_write_psi_qr_snapshot(
    const float* matrix,
    const float* rhs,
    const double* scale,
    int rows,
    int cols,
    int data_rows,
    double ridge
) {
    const char* path = std::getenv("SGPU_PSI_QR_SNAPSHOT");
    if (!path || path[0] == '\0') return true;

    sgpu::PsiQrSnapshotHeader header{};
    std::memcpy(header.magic, sgpu::kPsiQrSnapshotMagic, sizeof(header.magic));
    header.version = sgpu::kPsiQrSnapshotVersion;
    header.scalar_type = sgpu::kPsiQrScalarFloat32;
    header.layout = sgpu::kPsiQrLayoutColumnMajor;
    header.rows = static_cast<std::uint64_t>(rows);
    header.cols = static_cast<std::uint64_t>(cols);
    header.data_rows = static_cast<std::uint64_t>(data_rows);
    header.ridge = ridge;
    header.matrix_bytes = static_cast<std::uint64_t>(rows) * cols * sizeof(float);
    header.rhs_bytes = static_cast<std::uint64_t>(rows) * sizeof(float);
    header.scale_bytes = static_cast<std::uint64_t>(cols) * sizeof(double);

    const std::uint64_t expected_bytes = sizeof(header) + header.matrix_bytes +
        header.rhs_bytes + header.scale_bytes;
    std::ifstream existing(path, std::ios::binary | std::ios::ate);
    if (existing.good() && static_cast<std::uint64_t>(existing.tellg()) == expected_bytes) return true;
    existing.close();

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        set_error(std::string("failed to open psi QR snapshot: ") + path);
        return false;
    }
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!stream ||
        !write_device_bytes(stream, matrix, static_cast<std::size_t>(header.matrix_bytes)) ||
        !write_device_bytes(stream, rhs, static_cast<std::size_t>(header.rhs_bytes)) ||
        !write_device_bytes(stream, scale, static_cast<std::size_t>(header.scale_bytes))) {
        return false;
    }
    stream.close();
    if (!stream) {
        set_error("failed to finalize psi QR snapshot");
        return false;
    }
    return true;
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

__global__ void build_psi_common_basis_kernel(
    const float* __restrict__ normalized_R,
    const float* __restrict__ normalized_Z,
    const float* __restrict__ phi,
    int point_count,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int coefficient_count,
    int nfp,
    float radius_scale,
    float* __restrict__ derivative_R,
    float* __restrict__ derivative_Z,
    float* __restrict__ derivative_phi
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(point_count) * coefficient_count;
    if (flat >= count) return;
    const int column = static_cast<int>(flat / point_count);
    const int row = static_cast<int>(flat - static_cast<size_t>(column) * point_count);
    const int a = mode_a[column];
    const int b = mode_b[column];
    const int m = mode_m[column];
    const int kind = mode_kind[column];
    const float X = normalized_R[row];
    const float Y = normalized_Z[row];
    float x_power = 1.0f;
    float y_power = 1.0f;
    float x_derivative_power = 1.0f;
    float y_derivative_power = 1.0f;
    for (int power = 0; power < a; ++power) x_power *= X;
    for (int power = 0; power < b; ++power) y_power *= Y;
    for (int power = 1; power < a; ++power) x_derivative_power *= X;
    for (int power = 1; power < b; ++power) y_derivative_power *= Y;
    const float mono = x_power * y_power;
    const float mono_R = a == 0 ? 0.0f :
        static_cast<float>(a) * x_derivative_power * y_power / radius_scale;
    const float mono_Z = b == 0 ? 0.0f :
        static_cast<float>(b) * x_power * y_derivative_power / radius_scale;
    const float argument = static_cast<float>(m * nfp) * phi[row];
    const float cosine = m == 0 ? 1.0f : cosf(argument);
    const float sine = m == 0 ? 0.0f : sinf(argument);
    const float trig = m == 0 ? 1.0f : (kind == 0 ? cosine : sine);
    const float trig_phi = m == 0 ? 0.0f : static_cast<float>(m * nfp) *
        (kind == 0 ? -sine : cosine);
    derivative_R[flat] = mono_R * trig;
    derivative_Z[flat] = mono_Z * trig;
    derivative_phi[flat] = mono * trig_phi;
}

__global__ void build_psi_batch_features_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int segments_per_query,
    const float* __restrict__ normalized_R,
    const float* __restrict__ normalized_Z,
    const float* __restrict__ phi,
    int point_count,
    const float* __restrict__ axis_R,
    const float* __restrict__ axis_Z,
    const float* __restrict__ axis_R_phi,
    const float* __restrict__ axis_Z_phi,
    int axis_count,
    int nfp,
    float radius_scale,
    float* __restrict__ feature_R,
    float* __restrict__ feature_Z,
    float* __restrict__ feature_phi,
    float* __restrict__ rhs
) {
    extern __shared__ float shared[];
    const size_t flat = blockIdx.x;
    const int query = static_cast<int>(flat / point_count);
    const int row = static_cast<int>(flat - static_cast<size_t>(query) * point_count);
    const size_t segment_offset = static_cast<size_t>(query) * segments_per_query;
    const size_t axis_offset = static_cast<size_t>(query) * axis_count;
    const float phii = phi[row];
    const float period = static_cast<float>(TWOPI) / static_cast<float>(nfp);
    float axis_R_value = 0.0f, axis_Z_value = 0.0f;
    float axis_R_derivative = 0.0f, axis_Z_derivative = 0.0f;
    if (threadIdx.x == 0) {
        periodic_hermite_uniform_f32(
            phii, axis_R + axis_offset, axis_R_phi + axis_offset,
            axis_count, period, axis_R_value, axis_R_derivative
        );
        periodic_hermite_uniform_f32(
            phii, axis_Z + axis_offset, axis_Z_phi + axis_offset,
            axis_count, period, axis_Z_value, axis_Z_derivative
        );
        shared[SEG_TILE * 6 + blockDim.x * 3] = axis_R_value;
        shared[SEG_TILE * 6 + blockDim.x * 3 + 1] = axis_Z_value;
        shared[SEG_TILE * 6 + blockDim.x * 3 + 2] = axis_R_derivative;
        shared[SEG_TILE * 6 + blockDim.x * 3 + 3] = axis_Z_derivative;
    }
    __syncthreads();
    axis_R_value = shared[SEG_TILE * 6 + blockDim.x * 3];
    axis_Z_value = shared[SEG_TILE * 6 + blockDim.x * 3 + 1];
    axis_R_derivative = shared[SEG_TILE * 6 + blockDim.x * 3 + 2];
    axis_Z_derivative = shared[SEG_TILE * 6 + blockDim.x * 3 + 3];
    const float R = axis_R_value + radius_scale * normalized_R[row];
    const float Z = axis_Z_value + radius_scale * normalized_Z[row];
    const float cosine = cosf(phii);
    const float sine = sinf(phii);
    float bx = 0.0f, by = 0.0f, bz = 0.0f;
    eval_B_block_f32(
        R * cosine, R * sine, Z,
        seg_x + segment_offset, seg_y + segment_offset, seg_z + segment_offset,
        seg_wx + segment_offset, seg_wy + segment_offset, seg_wz + segment_offset,
        segments_per_query, shared, bx, by, bz
    );
    if (threadIdx.x == 0) {
        const float bR = bx * cosine + by * sine;
        const float bphi = -bx * sine + by * cosine;
        const float cphi = bphi / R;
        const float cR = bR - cphi * axis_R_derivative;
        const float cZ = bz - cphi * axis_Z_derivative;
        feature_R[flat] = cR;
        feature_Z[flat] = cZ;
        feature_phi[flat] = cphi;
        rhs[flat] = -2.0f * normalized_R[row] * cR / radius_scale;
    }
}

__global__ void combine_psi_feature_kernel(
    float* __restrict__ output,
    const float* __restrict__ feature,
    const float* __restrict__ image,
    size_t count,
    int initialize
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = feature[index] * image[index];
        output[index] = initialize ? value : output[index] + value;
    }
}

__global__ void weight_psi_residual_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ feature,
    float* __restrict__ weighted,
    size_t count
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) weighted[index] = residual[index] * feature[index];
}

__global__ void initialize_psi_batch_coefficients_kernel(
    const float* __restrict__ center_coefficients,
    const float* __restrict__ scale,
    int coefficient_count,
    int query_count,
    float* __restrict__ scaled_coefficients
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(coefficient_count) * query_count;
    if (index < count) {
        const int coefficient = static_cast<int>(index % coefficient_count);
        scaled_coefficients[index] = center_coefficients[coefficient] * scale[coefficient];
    }
}

__global__ void unscale_psi_batch_coefficients_kernel(
    const float* __restrict__ scaled,
    const float* __restrict__ scale,
    int coefficient_count,
    int query_count,
    float* __restrict__ unscaled
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(coefficient_count) * query_count;
    if (index < count) {
        const int coefficient = static_cast<int>(index % coefficient_count);
        unscaled[index] = scaled[index] / scale[coefficient];
    }
}

__global__ void initialize_psi_residual_kernel(
    const float* __restrict__ rhs,
    const float* __restrict__ prediction,
    size_t count,
    float* __restrict__ residual
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) residual[index] = rhs[index] - prediction[index];
}

__global__ void initialize_psi_ridge_residual_kernel(
    const float* __restrict__ scaled_coefficients,
    float ridge_lambda,
    size_t count,
    float* __restrict__ ridge_residual
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) ridge_residual[index] = -ridge_lambda * scaled_coefficients[index];
}

__global__ void scale_psi_batch_normal_kernel(
    float* __restrict__ normal,
    const float* __restrict__ scale,
    const float* __restrict__ ridge_residual,
    float ridge_lambda,
    int coefficient_count,
    int query_count
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(coefficient_count) * query_count;
    if (index < count) {
        const int coefficient = static_cast<int>(index % coefficient_count);
        normal[index] = normal[index] / scale[coefficient] +
            ridge_lambda * ridge_residual[index];
    }
}

__global__ void psi_batch_column_norm_kernel(
    const float* __restrict__ values,
    int rows,
    int columns,
    float* __restrict__ norms
) {
    const int column = blockIdx.x;
    float sum = 0.0f;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        const float value = values[static_cast<size_t>(column) * rows + row];
        sum += value * value;
    }
    extern __shared__ float reduction[];
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) reduction[threadIdx.x] += reduction[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0 && column < columns) norms[column] = reduction[0];
}

__global__ void psi_batch_delta_kernel(
    const float* __restrict__ image,
    int point_count,
    const float* __restrict__ ridge_image,
    int coefficient_count,
    int query_count,
    float* __restrict__ delta
) {
    const int query = blockIdx.x;
    float sum = 0.0f;
    for (int row = threadIdx.x; row < point_count; row += blockDim.x) {
        const float value = image[static_cast<size_t>(query) * point_count + row];
        sum += value * value;
    }
    for (int coefficient = threadIdx.x; coefficient < coefficient_count; coefficient += blockDim.x) {
        const float value = ridge_image[static_cast<size_t>(query) * coefficient_count + coefficient];
        sum += value * value;
    }
    extern __shared__ float reduction[];
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) reduction[threadIdx.x] += reduction[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0 && query < query_count) delta[query] = reduction[0];
}

__global__ void psi_batch_update_kernel(
    float* __restrict__ scaled_coefficients,
    float* __restrict__ residual,
    float* __restrict__ ridge_residual,
    const float* __restrict__ transformed_direction,
    const float* __restrict__ image,
    const float* __restrict__ ridge_image,
    const float* __restrict__ gamma,
    const float* __restrict__ delta,
    int coefficient_count,
    int point_count,
    int query_count
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t point_total = static_cast<size_t>(point_count) * query_count;
    const size_t coefficient_total = static_cast<size_t>(coefficient_count) * query_count;
    if (index < point_total) {
        const int query = static_cast<int>(index / point_count);
        const float alpha = gamma[query] / fmaxf(delta[query], 1.0e-30f);
        residual[index] -= alpha * image[index];
    }
    if (index < coefficient_total) {
        const int query = static_cast<int>(index / coefficient_count);
        const float alpha = gamma[query] / fmaxf(delta[query], 1.0e-30f);
        scaled_coefficients[index] += alpha * transformed_direction[index];
        ridge_residual[index] -= alpha * ridge_image[index];
    }
}

__global__ void psi_batch_direction_update_kernel(
    float* __restrict__ direction,
    const float* __restrict__ normal,
    const float* __restrict__ gamma_old,
    const float* __restrict__ gamma_new,
    int coefficient_count,
    int query_count
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(coefficient_count) * query_count;
    if (index < count) {
        const int query = static_cast<int>(index / coefficient_count);
        const float beta = gamma_new[query] / fmaxf(gamma_old[query], 1.0e-30f);
        direction[index] = normal[index] + beta * direction[index];
    }
}

__global__ void psi_batch_ridge_image_kernel(
    const float* __restrict__ direction,
    float ridge_lambda,
    size_t count,
    float* __restrict__ image
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) image[index] = ridge_lambda * direction[index];
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

__global__ void scale_coeff_kernel_f32(float* coeff, const double* scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) coeff[i] *= static_cast<float>(scale[i]);
}

__global__ void scale_coeff_float_kernel(float* coeff, const float* scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) coeff[i] *= scale[i];
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

int sgpu_create_field_batch_f32(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int query_count,
    int n_base_coils,
    int n_coeff,
    int nfp,
    int segments_per_coil,
    int device_id,
    void** out_handle
) {
    if (!out_handle) {
        set_error("batch field out_handle is null");
        return 1;
    }
    *out_handle = nullptr;
    if (!coeffs_x || !coeffs_y || !coeffs_z || !currents_a || query_count <= 0 ||
        n_base_coils <= 0 || n_coeff <= 0 || (n_coeff & 1) == 0 || nfp <= 0 ||
        segments_per_coil <= 0) {
        set_error("invalid batch field dimensions");
        return 1;
    }
    if (cuda_check(cudaSetDevice(device_id), "batch field cudaSetDevice")) return 1;
    BatchCoilField* field = new BatchCoilField();
    field->device_id = device_id;
    field->query_count = query_count;
    field->n_segments = n_base_coils * 2 * nfp * segments_per_coil;
    const size_t coefficient_count = static_cast<size_t>(query_count) * n_base_coils * n_coeff;
    const size_t current_count = static_cast<size_t>(query_count) * n_base_coils;
    const size_t segment_count = static_cast<size_t>(query_count) * field->n_segments;
    double* d_coeffs_x = nullptr;
    double* d_coeffs_y = nullptr;
    double* d_coeffs_z = nullptr;
    double* d_currents = nullptr;
    auto cleanup_inputs = [&] {
        cudaFree(d_coeffs_x);
        cudaFree(d_coeffs_y);
        cudaFree(d_coeffs_z);
        cudaFree(d_currents);
    };
    auto allocate = [](void** pointer, size_t bytes) {
        return cuda_check(cudaMalloc(pointer, bytes), "batch field allocation") == 0;
    };
    if (!allocate(reinterpret_cast<void**>(&d_coeffs_x), coefficient_count * sizeof(double)) ||
        !allocate(reinterpret_cast<void**>(&d_coeffs_y), coefficient_count * sizeof(double)) ||
        !allocate(reinterpret_cast<void**>(&d_coeffs_z), coefficient_count * sizeof(double)) ||
        !allocate(reinterpret_cast<void**>(&d_currents), current_count * sizeof(double)) ||
        !allocate(reinterpret_cast<void**>(&field->d_x), segment_count * sizeof(float)) ||
        !allocate(reinterpret_cast<void**>(&field->d_y), segment_count * sizeof(float)) ||
        !allocate(reinterpret_cast<void**>(&field->d_z), segment_count * sizeof(float)) ||
        !allocate(reinterpret_cast<void**>(&field->d_wx), segment_count * sizeof(float)) ||
        !allocate(reinterpret_cast<void**>(&field->d_wy), segment_count * sizeof(float)) ||
        !allocate(reinterpret_cast<void**>(&field->d_wz), segment_count * sizeof(float))) {
        cleanup_inputs();
        sgpu_destroy_field_batch(field);
        return 1;
    }
    const size_t coefficient_bytes = coefficient_count * sizeof(double);
    const size_t current_bytes = current_count * sizeof(double);
    if (cuda_check(cudaMemcpy(d_coeffs_x, coeffs_x, coefficient_bytes, cudaMemcpyHostToDevice), "batch copy coeffs_x") ||
        cuda_check(cudaMemcpy(d_coeffs_y, coeffs_y, coefficient_bytes, cudaMemcpyHostToDevice), "batch copy coeffs_y") ||
        cuda_check(cudaMemcpy(d_coeffs_z, coeffs_z, coefficient_bytes, cudaMemcpyHostToDevice), "batch copy coeffs_z") ||
        cuda_check(cudaMemcpy(d_currents, currents_a, current_bytes, cudaMemcpyHostToDevice), "batch copy currents")) {
        cleanup_inputs();
        sgpu_destroy_field_batch(field);
        return 1;
    }
    constexpr int threads = 256;
    const int blocks = static_cast<int>((segment_count + threads - 1) / threads);
    generate_segment_batch_kernel<<<blocks, threads>>>(
        d_coeffs_x, d_coeffs_y, d_coeffs_z, d_currents, query_count,
        n_base_coils, n_coeff, nfp, segments_per_coil, field->n_segments,
        field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz
    );
    const bool failed =
        cuda_check(cudaGetLastError(), "batch segment generation") ||
        cuda_check(cudaDeviceSynchronize(), "batch segment generation sync");
    cleanup_inputs();
    if (failed) {
        sgpu_destroy_field_batch(field);
        return 1;
    }
    *out_handle = field;
    set_error("");
    return 0;
}

void sgpu_destroy_field_batch(void* handle) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field) return;
    cudaSetDevice(field->device_id);
    cudaFree(field->d_x);
    cudaFree(field->d_y);
    cudaFree(field->d_z);
    cudaFree(field->d_wx);
    cudaFree(field->d_wy);
    cudaFree(field->d_wz);
    delete field;
}

int sgpu_batch_eval_B_f32(
    void* handle,
    const float* xyz_host,
    float* B_host,
    int points_per_query
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !xyz_host || !B_host || points_per_query <= 0) {
        set_error("invalid batch eval_B arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "batch eval_B cudaSetDevice")) return 1;
    const size_t point_count = static_cast<size_t>(field->query_count) * points_per_query;
    const size_t vector_bytes = point_count * 3 * sizeof(float);
    float* d_xyz = nullptr;
    float* d_B = nullptr;
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_xyz), vector_bytes), "batch eval_B xyz allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_B), vector_bytes), "batch eval_B output allocation")) {
        cudaFree(d_xyz);
        cudaFree(d_B);
        return 1;
    }
    const int blocks_per_query = (points_per_query + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    const bool failed =
        cuda_check(cudaMemcpy(d_xyz, xyz_host, vector_bytes, cudaMemcpyHostToDevice), "batch eval_B input copy") ||
        (eval_B_batch_f32_kernel<false><<<field->query_count * blocks_per_query, THREADS_PER_BLOCK, shared_bytes>>>(
            field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
            field->n_segments, d_xyz, d_B, nullptr, points_per_query, blocks_per_query
        ), false) ||
        cuda_check(cudaGetLastError(), "batch eval_B kernel") ||
        cuda_check(cudaDeviceSynchronize(), "batch eval_B sync") ||
        cuda_check(cudaMemcpy(B_host, d_B, vector_bytes, cudaMemcpyDeviceToHost), "batch eval_B output copy");
    cudaFree(d_xyz);
    cudaFree(d_B);
    if (failed) return 1;
    set_error("");
    return 0;
}

int sgpu_internal_batch_query_count(void* handle) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    return field ? field->query_count : 0;
}

int sgpu_internal_batch_eval_B_f32_device(
    void* handle,
    const float* xyz_device,
    float* B_device,
    int points_per_query
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !xyz_device || !B_device || points_per_query <= 0) {
        set_error("invalid device batch eval_B arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "device batch eval_B cudaSetDevice")) return 1;
    const int blocks_per_query = (points_per_query + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    eval_B_batch_f32_kernel<false><<<
        field->query_count * blocks_per_query, THREADS_PER_BLOCK, shared_bytes
    >>>(
        field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
        field->n_segments, xyz_device, B_device, nullptr,
        points_per_query, blocks_per_query
    );
    return cuda_check(cudaGetLastError(), "device batch eval_B kernel");
}

int sgpu_internal_batch_eval_B_grad_f32_device(
    void* handle,
    const float* xyz_device,
    float* B_device,
    float* grad_B_device,
    int points_per_query
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !xyz_device || !B_device || !grad_B_device || points_per_query <= 0) {
        set_error("invalid device batch eval_B_grad arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "device batch eval_B_grad cudaSetDevice")) return 1;
    const int blocks_per_query = (points_per_query + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    eval_B_batch_f32_kernel<true><<<
        field->query_count * blocks_per_query, THREADS_PER_BLOCK, shared_bytes
    >>>(
        field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
        field->n_segments, xyz_device, B_device, grad_B_device,
        points_per_query, blocks_per_query
    );
    return cuda_check(cudaGetLastError(), "device batch eval_B_grad kernel");
}

int sgpu_batch_eval_B_grad_f32(
    void* handle,
    const float* xyz_host,
    float* B_host,
    float* grad_B_host,
    int points_per_query
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !xyz_host || !B_host || !grad_B_host || points_per_query <= 0) {
        set_error("invalid batch eval_B_grad arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "batch eval_B_grad cudaSetDevice")) return 1;
    const size_t point_count = static_cast<size_t>(field->query_count) * points_per_query;
    const size_t vector_bytes = point_count * 3 * sizeof(float);
    const size_t gradient_bytes = point_count * 9 * sizeof(float);
    float* d_xyz = nullptr;
    float* d_B = nullptr;
    float* d_grad_B = nullptr;
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_xyz), vector_bytes), "batch eval_B_grad xyz allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_B), vector_bytes), "batch eval_B_grad B allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_grad_B), gradient_bytes), "batch eval_B_grad gradient allocation")) {
        cudaFree(d_xyz);
        cudaFree(d_B);
        cudaFree(d_grad_B);
        return 1;
    }
    const int blocks_per_query = (points_per_query + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    const bool failed =
        cuda_check(cudaMemcpy(d_xyz, xyz_host, vector_bytes, cudaMemcpyHostToDevice), "batch eval_B_grad input copy") ||
        (eval_B_batch_f32_kernel<true><<<field->query_count * blocks_per_query, THREADS_PER_BLOCK, shared_bytes>>>(
            field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
            field->n_segments, d_xyz, d_B, d_grad_B, points_per_query, blocks_per_query
        ), false) ||
        cuda_check(cudaGetLastError(), "batch eval_B_grad kernel") ||
        cuda_check(cudaDeviceSynchronize(), "batch eval_B_grad sync") ||
        cuda_check(cudaMemcpy(B_host, d_B, vector_bytes, cudaMemcpyDeviceToHost), "batch eval_B_grad B copy") ||
        cuda_check(cudaMemcpy(grad_B_host, d_grad_B, gradient_bytes, cudaMemcpyDeviceToHost), "batch eval_B_grad gradient copy");
    cudaFree(d_xyz);
    cudaFree(d_B);
    cudaFree(d_grad_B);
    if (failed) return 1;
    set_error("");
    return 0;
}

int sgpu_batch_trace_period_mixed(
    void* handle,
    const double* R0_host,
    const double* Z0_host,
    double* R1_host,
    double* Z1_host,
    int lines_per_query,
    int nfp,
    int steps
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !R0_host || !Z0_host || !R1_host || !Z1_host ||
        lines_per_query <= 0 || nfp <= 0 || steps <= 0) {
        set_error("invalid batch trace-period arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "batch trace cudaSetDevice")) return 1;
    const size_t line_count = static_cast<size_t>(field->query_count) * lines_per_query;
    const size_t bytes = line_count * sizeof(double);
    double* d_R0 = nullptr;
    double* d_Z0 = nullptr;
    double* d_R1 = nullptr;
    double* d_Z1 = nullptr;
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R0), bytes), "batch trace R0 allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z0), bytes), "batch trace Z0 allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R1), bytes), "batch trace R1 allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z1), bytes), "batch trace Z1 allocation")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    constexpr int threads = 256;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE * 6 + threads * 3) * sizeof(float);
    const bool failed =
        cuda_check(cudaMemcpy(d_R0, R0_host, bytes, cudaMemcpyHostToDevice), "batch trace R0 copy") ||
        cuda_check(cudaMemcpy(d_Z0, Z0_host, bytes, cudaMemcpyHostToDevice), "batch trace Z0 copy") ||
        (trace_period_batch_mixed_kernel<<<static_cast<int>(line_count), threads, shared_bytes>>>(
            field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
            field->n_segments, d_R0, d_Z0, d_R1, d_Z1, lines_per_query, nfp, steps
        ), false) ||
        cuda_check(cudaGetLastError(), "batch trace kernel") ||
        cuda_check(cudaDeviceSynchronize(), "batch trace sync") ||
        cuda_check(cudaMemcpy(R1_host, d_R1, bytes, cudaMemcpyDeviceToHost), "batch trace R1 copy") ||
        cuda_check(cudaMemcpy(Z1_host, d_Z1, bytes, cudaMemcpyDeviceToHost), "batch trace Z1 copy");
    cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
    if (failed) return 1;
    set_error("");
    return 0;
}

int sgpu_batch_trace_axis_samples(
    void* handle,
    const double* R0_host,
    const double* Z0_host,
    int nfp,
    int integration_steps,
    int sample_count,
    double* R_host,
    double* Z_host,
    double* R_phi_host,
    double* Z_phi_host
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !R0_host || !Z0_host || !R_host || !Z_host || !R_phi_host ||
        !Z_phi_host || nfp <= 0 || integration_steps <= 0 || sample_count <= 0 ||
        integration_steps % sample_count != 0) {
        set_error("invalid batch axis-sample arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "batch axis cudaSetDevice")) return 1;
    const size_t input_bytes = static_cast<size_t>(field->query_count) * sizeof(double);
    const size_t output_count = static_cast<size_t>(field->query_count) * sample_count;
    const size_t output_bytes = output_count * sizeof(double);
    double* d_R0 = nullptr;
    double* d_Z0 = nullptr;
    double* d_R = nullptr;
    double* d_Z = nullptr;
    double* d_R_phi = nullptr;
    double* d_Z_phi = nullptr;
    auto cleanup = [&] {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R); cudaFree(d_Z);
        cudaFree(d_R_phi); cudaFree(d_Z_phi);
    };
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R0), input_bytes), "batch axis R0 allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z0), input_bytes), "batch axis Z0 allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R), output_bytes), "batch axis R allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z), output_bytes), "batch axis Z allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R_phi), output_bytes), "batch axis R_phi allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z_phi), output_bytes), "batch axis Z_phi allocation")) {
        cleanup();
        return 1;
    }
    constexpr int threads = 256;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE * 6 + threads * 3) * sizeof(float);
    const bool failed =
        cuda_check(cudaMemcpy(d_R0, R0_host, input_bytes, cudaMemcpyHostToDevice), "batch axis R0 copy") ||
        cuda_check(cudaMemcpy(d_Z0, Z0_host, input_bytes, cudaMemcpyHostToDevice), "batch axis Z0 copy") ||
        (trace_axis_samples_batch_kernel<<<field->query_count, threads, shared_bytes>>>(
            field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
            field->n_segments, d_R0, d_Z0, nfp, integration_steps / sample_count,
            sample_count, d_R, d_Z, d_R_phi, d_Z_phi
        ), false) ||
        cuda_check(cudaGetLastError(), "batch axis kernel") ||
        cuda_check(cudaDeviceSynchronize(), "batch axis sync") ||
        cuda_check(cudaMemcpy(R_host, d_R, output_bytes, cudaMemcpyDeviceToHost), "batch axis R copy") ||
        cuda_check(cudaMemcpy(Z_host, d_Z, output_bytes, cudaMemcpyDeviceToHost), "batch axis Z copy") ||
        cuda_check(cudaMemcpy(R_phi_host, d_R_phi, output_bytes, cudaMemcpyDeviceToHost), "batch axis R_phi copy") ||
        cuda_check(cudaMemcpy(Z_phi_host, d_Z_phi, output_bytes, cudaMemcpyDeviceToHost), "batch axis Z_phi copy");
    cleanup();
    if (failed) return 1;
    set_error("");
    return 0;
}

int sgpu_batch_refine_axis_hint(
    void* handle,
    const double* hint_R_host,
    const double* hint_Z_host,
    int nfp,
    int trace_steps,
    int newton_iterations,
    double finite_difference_step,
    double maximum_newton_step,
    double residual_tolerance,
    double hint_max_distance,
    double* axis_R_host,
    double* axis_Z_host,
    double* residual_host,
    double* topology_trace_host,
    double* topology_det_host,
    unsigned char* valid_host
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !hint_R_host || !hint_Z_host || nfp <= 0 || trace_steps <= 0 ||
        newton_iterations < 0 || !(finite_difference_step > 0.0) ||
        !(maximum_newton_step > 0.0) || !(residual_tolerance > 0.0) ||
        !(hint_max_distance > 0.0) || !axis_R_host || !axis_Z_host ||
        !residual_host || !topology_trace_host || !topology_det_host || !valid_host) {
        set_error("invalid batch axis-refinement arguments");
        return 1;
    }
    const int query_count = field->query_count;
    std::vector<double> R(hint_R_host, hint_R_host + query_count);
    std::vector<double> Z(hint_Z_host, hint_Z_host + query_count);
    std::vector<double> R_end(query_count), Z_end(query_count);
    std::vector<double> fR(query_count), fZ(query_count), residual(query_count);
    if (sgpu_batch_trace_period_mixed(
            field, R.data(), Z.data(), R_end.data(), Z_end.data(), 1, nfp, trace_steps)) {
        return 1;
    }
    for (int query = 0; query < query_count; ++query) {
        fR[query] = R_end[query] - R[query];
        fZ[query] = Z_end[query] - Z[query];
        residual[query] = std::hypot(fR[query], fZ[query]);
    }
    constexpr int derivative_lines = 4;
    constexpr int line_search_lines = 4;
    const double line_search_scale[line_search_lines] = {1.0, 0.5, 0.25, 0.125};
    for (int iteration = 0; iteration < newton_iterations; ++iteration) {
        bool any_active = false;
        std::vector<double> eval_R(static_cast<size_t>(query_count) * derivative_lines);
        std::vector<double> eval_Z(eval_R.size());
        for (int query = 0; query < query_count; ++query) {
            any_active = any_active || residual[query] > residual_tolerance;
            const size_t base = static_cast<size_t>(query) * derivative_lines;
            eval_R[base] = R[query] + finite_difference_step;
            eval_Z[base] = Z[query];
            eval_R[base + 1] = R[query] - finite_difference_step;
            eval_Z[base + 1] = Z[query];
            eval_R[base + 2] = R[query];
            eval_Z[base + 2] = Z[query] + finite_difference_step;
            eval_R[base + 3] = R[query];
            eval_Z[base + 3] = Z[query] - finite_difference_step;
        }
        if (!any_active) break;
        std::vector<double> eval_R_end(eval_R.size()), eval_Z_end(eval_Z.size());
        if (sgpu_batch_trace_period_mixed(
                field, eval_R.data(), eval_Z.data(), eval_R_end.data(), eval_Z_end.data(),
                derivative_lines, nfp, trace_steps)) {
            return 1;
        }
        std::vector<double> step_R(query_count, 0.0), step_Z(query_count, 0.0);
        for (int query = 0; query < query_count; ++query) {
            if (!(residual[query] > residual_tolerance)) continue;
            const size_t base = static_cast<size_t>(query) * derivative_lines;
            const double residual_R_plus = eval_R_end[base] - eval_R[base];
            const double residual_R_minus = eval_R_end[base + 1] - eval_R[base + 1];
            const double residual_Z_r_plus = eval_Z_end[base] - eval_Z[base];
            const double residual_Z_r_minus = eval_Z_end[base + 1] - eval_Z[base + 1];
            const double residual_R_z_plus = eval_R_end[base + 2] - eval_R[base + 2];
            const double residual_R_z_minus = eval_R_end[base + 3] - eval_R[base + 3];
            const double residual_Z_plus = eval_Z_end[base + 2] - eval_Z[base + 2];
            const double residual_Z_minus = eval_Z_end[base + 3] - eval_Z[base + 3];
            const double inverse_denominator = 0.5 / finite_difference_step;
            const double j11 = (residual_R_plus - residual_R_minus) * inverse_denominator;
            const double j21 = (residual_Z_r_plus - residual_Z_r_minus) * inverse_denominator;
            const double j12 = (residual_R_z_plus - residual_R_z_minus) * inverse_denominator;
            const double j22 = (residual_Z_plus - residual_Z_minus) * inverse_denominator;
            const double determinant = j11 * j22 - j12 * j21;
            if (std::abs(determinant) <= 1.0e-14) continue;
            step_R[query] = (-fR[query] * j22 + j12 * fZ[query]) / determinant;
            step_Z[query] = (j21 * fR[query] - j11 * fZ[query]) / determinant;
            const double norm = std::hypot(step_R[query], step_Z[query]);
            const double scale = std::min(1.0, maximum_newton_step / std::max(norm, 1.0e-300));
            step_R[query] *= scale;
            step_Z[query] *= scale;
        }
        std::vector<double> trial_R(static_cast<size_t>(query_count) * line_search_lines);
        std::vector<double> trial_Z(trial_R.size());
        for (int query = 0; query < query_count; ++query) {
            const size_t base = static_cast<size_t>(query) * line_search_lines;
            for (int line = 0; line < line_search_lines; ++line) {
                trial_R[base + line] = R[query] + line_search_scale[line] * step_R[query];
                trial_Z[base + line] = Z[query] + line_search_scale[line] * step_Z[query];
            }
        }
        std::vector<double> trial_R_end(trial_R.size()), trial_Z_end(trial_Z.size());
        if (sgpu_batch_trace_period_mixed(
                field, trial_R.data(), trial_Z.data(), trial_R_end.data(), trial_Z_end.data(),
                line_search_lines, nfp, trace_steps)) {
            return 1;
        }
        for (int query = 0; query < query_count; ++query) {
            const size_t base = static_cast<size_t>(query) * line_search_lines;
            for (int line = 0; line < line_search_lines; ++line) {
                const double candidate_fR = trial_R_end[base + line] - trial_R[base + line];
                const double candidate_fZ = trial_Z_end[base + line] - trial_Z[base + line];
                const double candidate_residual = std::hypot(candidate_fR, candidate_fZ);
                if (std::isfinite(candidate_residual) && candidate_residual < residual[query]) {
                    R[query] = trial_R[base + line];
                    Z[query] = trial_Z[base + line];
                    fR[query] = candidate_fR;
                    fZ[query] = candidate_fZ;
                    residual[query] = candidate_residual;
                    break;
                }
            }
        }
    }

    std::vector<double> topology_R(static_cast<size_t>(query_count) * derivative_lines);
    std::vector<double> topology_Z(topology_R.size());
    for (int query = 0; query < query_count; ++query) {
        const size_t base = static_cast<size_t>(query) * derivative_lines;
        topology_R[base] = R[query] + finite_difference_step;
        topology_Z[base] = Z[query];
        topology_R[base + 1] = R[query] - finite_difference_step;
        topology_Z[base + 1] = Z[query];
        topology_R[base + 2] = R[query];
        topology_Z[base + 2] = Z[query] + finite_difference_step;
        topology_R[base + 3] = R[query];
        topology_Z[base + 3] = Z[query] - finite_difference_step;
    }
    std::vector<double> topology_R_end(topology_R.size()), topology_Z_end(topology_Z.size());
    if (sgpu_batch_trace_period_mixed(
            field, topology_R.data(), topology_Z.data(), topology_R_end.data(),
            topology_Z_end.data(), derivative_lines, nfp, trace_steps)) {
        return 1;
    }
    for (int query = 0; query < query_count; ++query) {
        const size_t base = static_cast<size_t>(query) * derivative_lines;
        const double inverse_denominator = 0.5 / finite_difference_step;
        const double a = (topology_R_end[base] - topology_R_end[base + 1]) * inverse_denominator;
        const double c = (topology_Z_end[base] - topology_Z_end[base + 1]) * inverse_denominator;
        const double b = (topology_R_end[base + 2] - topology_R_end[base + 3]) * inverse_denominator;
        const double d = (topology_Z_end[base + 2] - topology_Z_end[base + 3]) * inverse_denominator;
        const double trace = a + d;
        const double determinant = a * d - b * c;
        const bool elliptic = determinant > 0.0 && std::isfinite(trace) &&
            std::isfinite(determinant) && std::abs(trace / std::sqrt(determinant)) < 2.0;
        const double hint_distance = std::hypot(R[query] - hint_R_host[query], Z[query] - hint_Z_host[query]);
        axis_R_host[query] = R[query];
        axis_Z_host[query] = Z[query];
        residual_host[query] = residual[query];
        topology_trace_host[query] = trace;
        topology_det_host[query] = determinant;
        valid_host[query] = residual[query] <= residual_tolerance && elliptic &&
            hint_distance <= hint_max_distance;
    }
    set_error("");
    return 0;
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
    SGPU_NVTX_RANGE("field.eval_B.fp32");
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

int sgpu_set_psi_warm_preconditioner_capture(int enabled) {
    g_psi_warm_preconditioner.capture_enabled = enabled != 0;
    if (enabled) {
        g_psi_warm_preconditioner.coefficient_count = 0;
        g_psi_warm_preconditioner.upper_factor.clear();
        g_psi_warm_preconditioner.center_scale.clear();
    }
    set_error("");
    return 0;
}

int sgpu_has_psi_warm_preconditioner(int coefficient_count) {
    return coefficient_count > 0 &&
        g_psi_warm_preconditioner.coefficient_count == coefficient_count &&
        g_psi_warm_preconditioner.upper_factor.size() ==
            static_cast<size_t>(coefficient_count) * coefficient_count &&
        g_psi_warm_preconditioner.center_scale.size() == static_cast<size_t>(coefficient_count);
}

void sgpu_clear_psi_warm_preconditioner() {
    g_psi_warm_preconditioner = PsiWarmPreconditionerCache{};
}

int sgpu_fit_psi_batch_pcgls_f32(
    void* batch_field_handle,
    const double* axis_R_host,
    const double* axis_Z_host,
    const double* axis_R_phi_host,
    const double* axis_Z_phi_host,
    int axis_count,
    const int* mode_a_host,
    const int* mode_b_host,
    const int* mode_m_host,
    const int* mode_kind_host,
    int coefficient_count,
    int nfp,
    double radius_scale,
    int radial_grid,
    int vertical_grid,
    int phi_grid,
    double rho_min,
    double ridge,
    int iterations,
    const double* center_coefficients_host,
    double* coefficients_host,
    double* train_rms_host,
    double* stats_out,
    int stats_len
) {
    using clock = std::chrono::steady_clock;
    const auto total_started = clock::now();
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(batch_field_handle);
    if (!field || !axis_R_host || !axis_Z_host || !axis_R_phi_host || !axis_Z_phi_host ||
        axis_count <= 1 || !mode_a_host || !mode_b_host || !mode_m_host || !mode_kind_host ||
        coefficient_count <= 0 || nfp <= 0 || !(radius_scale > 0.0) || radial_grid <= 1 ||
        vertical_grid <= 1 || phi_grid <= 0 || rho_min < 0.0 || ridge < 0.0 ||
        iterations < 0 || !center_coefficients_host || !coefficients_host || !train_rms_host) {
        set_error("invalid batch psi PCGLS arguments");
        return 1;
    }
    if (!sgpu_has_psi_warm_preconditioner(coefficient_count)) {
        set_error("batch psi PCGLS requires a captured center QR preconditioner");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "batch psi cudaSetDevice")) return 1;
    const int query_count = field->query_count;
    std::vector<float> normalized_R;
    std::vector<float> normalized_Z;
    std::vector<float> phi;
    normalized_R.reserve(static_cast<size_t>(radial_grid) * vertical_grid * phi_grid);
    normalized_Z.reserve(normalized_R.capacity());
    phi.reserve(normalized_R.capacity());
    for (int radial = 0; radial < radial_grid; ++radial) {
        const double dR = -radius_scale + 2.0 * radius_scale * radial /
            static_cast<double>(radial_grid - 1);
        for (int vertical = 0; vertical < vertical_grid; ++vertical) {
            const double dZ = -radius_scale + 2.0 * radius_scale * vertical /
                static_cast<double>(vertical_grid - 1);
            const double radius = std::hypot(dR, dZ);
            if (radius < rho_min || radius > radius_scale) continue;
            for (int toroidal = 0; toroidal < phi_grid; ++toroidal) {
                normalized_R.push_back(static_cast<float>(dR / radius_scale));
                normalized_Z.push_back(static_cast<float>(dZ / radius_scale));
                phi.push_back(static_cast<float>(
                    TWOPI * toroidal / static_cast<double>(nfp * phi_grid)
                ));
            }
        }
    }
    const int point_count = static_cast<int>(normalized_R.size());
    if (point_count <= 0) {
        set_error("batch psi grid contains no points");
        return 1;
    }
    const size_t point_query_count = static_cast<size_t>(point_count) * query_count;
    const size_t coefficient_query_count = static_cast<size_t>(coefficient_count) * query_count;
    const size_t basis_count = static_cast<size_t>(point_count) * coefficient_count;
    const size_t axis_query_count = static_cast<size_t>(axis_count) * query_count;
    std::vector<float> axis_R(axis_query_count), axis_Z(axis_query_count);
    std::vector<float> axis_R_phi(axis_query_count), axis_Z_phi(axis_query_count);
    for (size_t index = 0; index < axis_query_count; ++index) {
        axis_R[index] = static_cast<float>(axis_R_host[index]);
        axis_Z[index] = static_cast<float>(axis_Z_host[index]);
        axis_R_phi[index] = static_cast<float>(axis_R_phi_host[index]);
        axis_Z_phi[index] = static_cast<float>(axis_Z_phi_host[index]);
    }
    std::vector<float> center_coefficients(coefficient_count);
    std::vector<float> center_scale(coefficient_count);
    for (int coefficient = 0; coefficient < coefficient_count; ++coefficient) {
        center_coefficients[coefficient] = static_cast<float>(center_coefficients_host[coefficient]);
        center_scale[coefficient] = static_cast<float>(
            g_psi_warm_preconditioner.center_scale[static_cast<size_t>(coefficient)]
        );
    }

    float *d_normalized_R = nullptr, *d_normalized_Z = nullptr, *d_phi = nullptr;
    float *d_axis_R = nullptr, *d_axis_Z = nullptr, *d_axis_R_phi = nullptr, *d_axis_Z_phi = nullptr;
    int *d_mode_a = nullptr, *d_mode_b = nullptr, *d_mode_m = nullptr, *d_mode_kind = nullptr;
    float *d_basis_R = nullptr, *d_basis_Z = nullptr, *d_basis_phi = nullptr;
    float *d_feature_R = nullptr, *d_feature_Z = nullptr, *d_feature_phi = nullptr, *d_rhs = nullptr;
    float *d_center_coefficients = nullptr, *d_scale = nullptr, *d_center_factor = nullptr;
    float *d_scaled_coefficients = nullptr, *d_unscaled = nullptr, *d_residual = nullptr;
    float *d_ridge_residual = nullptr, *d_normal = nullptr, *d_direction = nullptr;
    float *d_transformed_direction = nullptr, *d_image = nullptr, *d_temporary = nullptr;
    float *d_weighted = nullptr, *d_ridge_image = nullptr, *d_gamma = nullptr;
    float *d_gamma_new = nullptr, *d_delta = nullptr;
    cublasHandle_t blas = nullptr;
    auto cleanup = [&] {
        if (blas) cublasDestroy(blas);
        cudaFree(d_normalized_R); cudaFree(d_normalized_Z); cudaFree(d_phi);
        cudaFree(d_axis_R); cudaFree(d_axis_Z); cudaFree(d_axis_R_phi); cudaFree(d_axis_Z_phi);
        cudaFree(d_mode_a); cudaFree(d_mode_b); cudaFree(d_mode_m); cudaFree(d_mode_kind);
        cudaFree(d_basis_R); cudaFree(d_basis_Z); cudaFree(d_basis_phi);
        cudaFree(d_feature_R); cudaFree(d_feature_Z); cudaFree(d_feature_phi); cudaFree(d_rhs);
        cudaFree(d_center_coefficients); cudaFree(d_scale); cudaFree(d_center_factor);
        cudaFree(d_scaled_coefficients); cudaFree(d_unscaled); cudaFree(d_residual);
        cudaFree(d_ridge_residual); cudaFree(d_normal); cudaFree(d_direction);
        cudaFree(d_transformed_direction); cudaFree(d_image); cudaFree(d_temporary);
        cudaFree(d_weighted); cudaFree(d_ridge_image); cudaFree(d_gamma);
        cudaFree(d_gamma_new); cudaFree(d_delta);
    };
    auto allocate = [&](void** pointer, size_t bytes, const char* label) {
        if (cuda_check(cudaMalloc(pointer, bytes), label)) return false;
        return true;
    };
    const size_t common_point_bytes = static_cast<size_t>(point_count) * sizeof(float);
    const size_t axis_bytes = axis_query_count * sizeof(float);
    const size_t mode_bytes = static_cast<size_t>(coefficient_count) * sizeof(int);
    const size_t basis_bytes = basis_count * sizeof(float);
    const size_t point_query_bytes = point_query_count * sizeof(float);
    const size_t coefficient_query_bytes = coefficient_query_count * sizeof(float);
    const size_t coefficient_bytes = static_cast<size_t>(coefficient_count) * sizeof(float);
    const size_t factor_bytes = static_cast<size_t>(coefficient_count) * coefficient_count * sizeof(float);
    if (!allocate(reinterpret_cast<void**>(&d_normalized_R), common_point_bytes, "batch psi normalized R") ||
        !allocate(reinterpret_cast<void**>(&d_normalized_Z), common_point_bytes, "batch psi normalized Z") ||
        !allocate(reinterpret_cast<void**>(&d_phi), common_point_bytes, "batch psi phi") ||
        !allocate(reinterpret_cast<void**>(&d_axis_R), axis_bytes, "batch psi axis R") ||
        !allocate(reinterpret_cast<void**>(&d_axis_Z), axis_bytes, "batch psi axis Z") ||
        !allocate(reinterpret_cast<void**>(&d_axis_R_phi), axis_bytes, "batch psi axis R phi") ||
        !allocate(reinterpret_cast<void**>(&d_axis_Z_phi), axis_bytes, "batch psi axis Z phi") ||
        !allocate(reinterpret_cast<void**>(&d_mode_a), mode_bytes, "batch psi mode a") ||
        !allocate(reinterpret_cast<void**>(&d_mode_b), mode_bytes, "batch psi mode b") ||
        !allocate(reinterpret_cast<void**>(&d_mode_m), mode_bytes, "batch psi mode m") ||
        !allocate(reinterpret_cast<void**>(&d_mode_kind), mode_bytes, "batch psi mode kind") ||
        !allocate(reinterpret_cast<void**>(&d_basis_R), basis_bytes, "batch psi basis R") ||
        !allocate(reinterpret_cast<void**>(&d_basis_Z), basis_bytes, "batch psi basis Z") ||
        !allocate(reinterpret_cast<void**>(&d_basis_phi), basis_bytes, "batch psi basis phi") ||
        !allocate(reinterpret_cast<void**>(&d_feature_R), point_query_bytes, "batch psi feature R") ||
        !allocate(reinterpret_cast<void**>(&d_feature_Z), point_query_bytes, "batch psi feature Z") ||
        !allocate(reinterpret_cast<void**>(&d_feature_phi), point_query_bytes, "batch psi feature phi") ||
        !allocate(reinterpret_cast<void**>(&d_rhs), point_query_bytes, "batch psi rhs") ||
        !allocate(reinterpret_cast<void**>(&d_center_coefficients), coefficient_bytes, "batch psi center coeff") ||
        !allocate(reinterpret_cast<void**>(&d_scale), coefficient_bytes, "batch psi scale") ||
        !allocate(reinterpret_cast<void**>(&d_center_factor), factor_bytes, "batch psi center factor") ||
        !allocate(reinterpret_cast<void**>(&d_scaled_coefficients), coefficient_query_bytes, "batch psi coefficients") ||
        !allocate(reinterpret_cast<void**>(&d_unscaled), coefficient_query_bytes, "batch psi unscaled") ||
        !allocate(reinterpret_cast<void**>(&d_residual), point_query_bytes, "batch psi residual") ||
        !allocate(reinterpret_cast<void**>(&d_ridge_residual), coefficient_query_bytes, "batch psi ridge residual") ||
        !allocate(reinterpret_cast<void**>(&d_normal), coefficient_query_bytes, "batch psi normal") ||
        !allocate(reinterpret_cast<void**>(&d_direction), coefficient_query_bytes, "batch psi direction") ||
        !allocate(reinterpret_cast<void**>(&d_transformed_direction), coefficient_query_bytes, "batch psi transformed direction") ||
        !allocate(reinterpret_cast<void**>(&d_image), point_query_bytes, "batch psi image") ||
        !allocate(reinterpret_cast<void**>(&d_temporary), point_query_bytes, "batch psi temporary") ||
        !allocate(reinterpret_cast<void**>(&d_weighted), point_query_bytes, "batch psi weighted") ||
        !allocate(reinterpret_cast<void**>(&d_ridge_image), coefficient_query_bytes, "batch psi ridge image") ||
        !allocate(reinterpret_cast<void**>(&d_gamma), static_cast<size_t>(query_count) * sizeof(float), "batch psi gamma") ||
        !allocate(reinterpret_cast<void**>(&d_gamma_new), static_cast<size_t>(query_count) * sizeof(float), "batch psi gamma new") ||
        !allocate(reinterpret_cast<void**>(&d_delta), static_cast<size_t>(query_count) * sizeof(float), "batch psi delta")) {
        cleanup();
        return 1;
    }
    if (cublas_check(cublasCreate(&blas), "batch psi cublasCreate")) {
        cleanup();
        return 1;
    }
    cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH);
    auto copy_host = [&](void* target, const void* source, size_t bytes, const char* label) {
        return cuda_check(cudaMemcpy(target, source, bytes, cudaMemcpyHostToDevice), label) == 0;
    };
    if (!copy_host(d_normalized_R, normalized_R.data(), common_point_bytes, "batch psi copy normalized R") ||
        !copy_host(d_normalized_Z, normalized_Z.data(), common_point_bytes, "batch psi copy normalized Z") ||
        !copy_host(d_phi, phi.data(), common_point_bytes, "batch psi copy phi") ||
        !copy_host(d_axis_R, axis_R.data(), axis_bytes, "batch psi copy axis R") ||
        !copy_host(d_axis_Z, axis_Z.data(), axis_bytes, "batch psi copy axis Z") ||
        !copy_host(d_axis_R_phi, axis_R_phi.data(), axis_bytes, "batch psi copy axis R phi") ||
        !copy_host(d_axis_Z_phi, axis_Z_phi.data(), axis_bytes, "batch psi copy axis Z phi") ||
        !copy_host(d_mode_a, mode_a_host, mode_bytes, "batch psi copy mode a") ||
        !copy_host(d_mode_b, mode_b_host, mode_bytes, "batch psi copy mode b") ||
        !copy_host(d_mode_m, mode_m_host, mode_bytes, "batch psi copy mode m") ||
        !copy_host(d_mode_kind, mode_kind_host, mode_bytes, "batch psi copy mode kind") ||
        !copy_host(d_center_coefficients, center_coefficients.data(), coefficient_bytes, "batch psi copy center coeff") ||
        !copy_host(d_scale, center_scale.data(), coefficient_bytes, "batch psi copy scale") ||
        !copy_host(d_center_factor, g_psi_warm_preconditioner.upper_factor.data(), factor_bytes, "batch psi copy factor")) {
        cleanup();
        return 1;
    }

    constexpr int threads = 256;
    const auto basis_started = clock::now();
    const int basis_blocks = static_cast<int>((basis_count + threads - 1) / threads);
    build_psi_common_basis_kernel<<<basis_blocks, threads>>>(
        d_normalized_R, d_normalized_Z, d_phi, point_count,
        d_mode_a, d_mode_b, d_mode_m, d_mode_kind, coefficient_count,
        nfp, static_cast<float>(radius_scale), d_basis_R, d_basis_Z, d_basis_phi
    );
    if (cuda_check(cudaGetLastError(), "batch psi basis kernel") ||
        cuda_check(cudaDeviceSynchronize(), "batch psi basis sync")) {
        cleanup();
        return 1;
    }
    const double basis_seconds = std::chrono::duration<double>(clock::now() - basis_started).count();

    const auto feature_started = clock::now();
    const size_t feature_shared = static_cast<size_t>(SEG_TILE * 6 + threads * 3 + 4) * sizeof(float);
    build_psi_batch_features_kernel<<<static_cast<int>(point_query_count), threads, feature_shared>>>(
        field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
        field->n_segments, d_normalized_R, d_normalized_Z, d_phi, point_count,
        d_axis_R, d_axis_Z, d_axis_R_phi, d_axis_Z_phi, axis_count, nfp,
        static_cast<float>(radius_scale), d_feature_R, d_feature_Z, d_feature_phi, d_rhs
    );
    if (cuda_check(cudaGetLastError(), "batch psi feature kernel") ||
        cuda_check(cudaDeviceSynchronize(), "batch psi feature sync")) {
        cleanup();
        return 1;
    }
    const double feature_seconds = std::chrono::duration<double>(clock::now() - feature_started).count();

    const float one = 1.0f;
    const float zero = 0.0f;
    auto apply_operator = [&](const float* coefficients, float* output) {
        const float* bases[] = {d_basis_R, d_basis_Z, d_basis_phi};
        const float* features[] = {d_feature_R, d_feature_Z, d_feature_phi};
        for (int component = 0; component < 3; ++component) {
            if (cublas_check(cublasSgemm(
                    blas, CUBLAS_OP_N, CUBLAS_OP_N,
                    point_count, query_count, coefficient_count,
                    &one, bases[component], point_count, coefficients, coefficient_count,
                    &zero, d_temporary, point_count), "batch psi forward GEMM")) {
                return false;
            }
            combine_psi_feature_kernel<<<
                static_cast<int>((point_query_count + threads - 1) / threads), threads
            >>>(output, features[component], d_temporary, point_query_count, component == 0);
            if (cuda_check(cudaGetLastError(), "batch psi forward combine")) return false;
        }
        return true;
    };
    auto compute_normal = [&](float* output) {
        const float* bases[] = {d_basis_R, d_basis_Z, d_basis_phi};
        const float* features[] = {d_feature_R, d_feature_Z, d_feature_phi};
        for (int component = 0; component < 3; ++component) {
            weight_psi_residual_kernel<<<
                static_cast<int>((point_query_count + threads - 1) / threads), threads
            >>>(d_residual, features[component], d_weighted, point_query_count);
            if (cuda_check(cudaGetLastError(), "batch psi residual weighting")) return false;
            const float beta = component == 0 ? 0.0f : 1.0f;
            if (cublas_check(cublasSgemm(
                    blas, CUBLAS_OP_T, CUBLAS_OP_N,
                    coefficient_count, query_count, point_count,
                    &one, bases[component], point_count, d_weighted, point_count,
                    &beta, output, coefficient_count), "batch psi transpose GEMM")) {
                return false;
            }
        }
        scale_psi_batch_normal_kernel<<<
            static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
        >>>(output, d_scale, d_ridge_residual, static_cast<float>(std::sqrt(ridge)),
            coefficient_count, query_count);
        return cuda_check(cudaGetLastError(), "batch psi normal scaling") == 0;
    };

    const auto solve_started = clock::now();
    initialize_psi_batch_coefficients_kernel<<<
        static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
    >>>(d_center_coefficients, d_scale, coefficient_count, query_count, d_scaled_coefficients);
    unscale_psi_batch_coefficients_kernel<<<
        static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
    >>>(d_scaled_coefficients, d_scale, coefficient_count, query_count, d_unscaled);
    if (cuda_check(cudaGetLastError(), "batch psi coefficient initialization") ||
        !apply_operator(d_unscaled, d_image)) {
        cleanup();
        return 1;
    }
    initialize_psi_residual_kernel<<<
        static_cast<int>((point_query_count + threads - 1) / threads), threads
    >>>(d_rhs, d_image, point_query_count, d_residual);
    const float ridge_lambda = static_cast<float>(std::sqrt(ridge));
    initialize_psi_ridge_residual_kernel<<<
        static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
    >>>(d_scaled_coefficients, ridge_lambda, coefficient_query_count, d_ridge_residual);
    if (cuda_check(cudaGetLastError(), "batch psi residual initialization") ||
        !compute_normal(d_normal) ||
        cublas_check(cublasStrsm(
            blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
            CUBLAS_DIAG_NON_UNIT, coefficient_count, query_count, &one,
            d_center_factor, coefficient_count, d_normal, coefficient_count
        ), "batch psi center R^-T") ||
        cublas_check(cublasScopy(
            blas, static_cast<int>(coefficient_query_count), d_normal, 1, d_direction, 1
        ), "batch psi initial direction")) {
        cleanup();
        return 1;
    }
    psi_batch_column_norm_kernel<<<query_count, threads, threads * sizeof(float)>>>(
        d_normal, coefficient_count, query_count, d_gamma
    );
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (cublas_check(cublasScopy(
                blas, static_cast<int>(coefficient_query_count), d_direction, 1,
                d_transformed_direction, 1), "batch psi copy direction") ||
            cublas_check(cublasStrsm(
                blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, coefficient_count, query_count, &one,
                d_center_factor, coefficient_count, d_transformed_direction, coefficient_count
            ), "batch psi center R^-1")) {
            cleanup();
            return 1;
        }
        unscale_psi_batch_coefficients_kernel<<<
            static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
        >>>(d_transformed_direction, d_scale, coefficient_count, query_count, d_unscaled);
        if (cuda_check(cudaGetLastError(), "batch psi direction unscale") ||
            !apply_operator(d_unscaled, d_image)) {
            cleanup();
            return 1;
        }
        psi_batch_ridge_image_kernel<<<
            static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
        >>>(d_transformed_direction, ridge_lambda, coefficient_query_count, d_ridge_image);
        psi_batch_delta_kernel<<<query_count, threads, threads * sizeof(float)>>>(
            d_image, point_count, d_ridge_image, coefficient_count, query_count, d_delta
        );
        psi_batch_update_kernel<<<
            static_cast<int>((std::max(point_query_count, coefficient_query_count) + threads - 1) / threads),
            threads
        >>>(d_scaled_coefficients, d_residual, d_ridge_residual,
            d_transformed_direction, d_image, d_ridge_image, d_gamma, d_delta,
            coefficient_count, point_count, query_count);
        if (cuda_check(cudaGetLastError(), "batch psi PCGLS update") ||
            !compute_normal(d_normal) ||
            cublas_check(cublasStrsm(
                blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT, coefficient_count, query_count, &one,
                d_center_factor, coefficient_count, d_normal, coefficient_count
            ), "batch psi next center R^-T")) {
            cleanup();
            return 1;
        }
        psi_batch_column_norm_kernel<<<query_count, threads, threads * sizeof(float)>>>(
            d_normal, coefficient_count, query_count, d_gamma_new
        );
        psi_batch_direction_update_kernel<<<
            static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
        >>>(d_direction, d_normal, d_gamma, d_gamma_new, coefficient_count, query_count);
        if (cuda_check(cudaGetLastError(), "batch psi direction update") ||
            cuda_check(cudaMemcpy(
                d_gamma, d_gamma_new, static_cast<size_t>(query_count) * sizeof(float),
                cudaMemcpyDeviceToDevice), "batch psi gamma update")) {
            cleanup();
            return 1;
        }
    }
    unscale_psi_batch_coefficients_kernel<<<
        static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
    >>>(d_scaled_coefficients, d_scale, coefficient_count, query_count, d_unscaled);
    psi_batch_column_norm_kernel<<<query_count, threads, threads * sizeof(float)>>>(
        d_residual, point_count, query_count, d_gamma
    );
    if (cuda_check(cudaGetLastError(), "batch psi final kernels") ||
        cuda_check(cudaDeviceSynchronize(), "batch psi solve sync")) {
        cleanup();
        return 1;
    }
    const double solve_seconds = std::chrono::duration<double>(clock::now() - solve_started).count();
    std::vector<float> coefficient_output(coefficient_query_count);
    std::vector<float> residual_norm2(query_count);
    if (cuda_check(cudaMemcpy(
            coefficient_output.data(), d_unscaled, coefficient_query_bytes,
            cudaMemcpyDeviceToHost), "batch psi copy coefficients") ||
        cuda_check(cudaMemcpy(
            residual_norm2.data(), d_gamma, static_cast<size_t>(query_count) * sizeof(float),
            cudaMemcpyDeviceToHost), "batch psi copy residuals")) {
        cleanup();
        return 1;
    }
    for (size_t index = 0; index < coefficient_query_count; ++index) {
        coefficients_host[index] = static_cast<double>(coefficient_output[index]);
    }
    for (int query = 0; query < query_count; ++query) {
        train_rms_host[query] = std::sqrt(
            std::max(static_cast<double>(residual_norm2[query]), 0.0) / point_count
        );
    }
    const double total_seconds = std::chrono::duration<double>(clock::now() - total_started).count();
    if (stats_out && stats_len >= 6) {
        stats_out[0] = point_count;
        stats_out[1] = basis_seconds;
        stats_out[2] = feature_seconds;
        stats_out[3] = solve_seconds;
        stats_out[4] = total_seconds;
        stats_out[5] = static_cast<double>(basis_bytes) * 3.0 / (1024.0 * 1024.0 * 1024.0);
    }
    cleanup();
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
    SGPU_NVTX_RANGE("psi.fullgpu.total");
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
    const bool warm_preconditioned = solver_mode >= 2000;
    const bool warm_solver = solver_mode >= 1000;
    const int warm_iterations = warm_solver
        ? solver_mode - (warm_preconditioned ? 2000 : 1000)
        : 0;
    if (solver_mode != 1 && solver_mode != 2 && !warm_solver) {
        set_error("solver_mode must be 1 (normal_eq), 2 (qr), or an experimental warm mode");
        return 1;
    }
    if (precision_mode != 1 && precision_mode != 2) {
        set_error("precision_mode must be 1 (fp64) or 2 (fp32)");
        return 1;
    }
    if (warm_solver && precision_mode != 2) {
        set_error("experimental warm psi solve requires fp32");
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
    float *d_warm_residual_f = nullptr, *d_warm_normal_f = nullptr;
    float *d_warm_direction_f = nullptr, *d_warm_image_f = nullptr;
    float *d_warm_transformed_direction_f = nullptr, *d_warm_scale_ratio_f = nullptr;
    float *d_warm_center_factor_f = nullptr;
    std::vector<float> coeff_host_f;
    double* d_coeff_src_d = nullptr;
    float* d_coeff_src_f = nullptr;

    auto cleanup = [&]() {
        SGPU_NVTX_RANGE("psi.fullgpu.cleanup");
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
        cudaFree(d_warm_residual_f); cudaFree(d_warm_normal_f);
        cudaFree(d_warm_direction_f); cudaFree(d_warm_image_f);
        cudaFree(d_warm_transformed_direction_f); cudaFree(d_warm_scale_ratio_f);
        cudaFree(d_warm_center_factor_f);
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
    {
        SGPU_NVTX_RANGE("psi.fullgpu.copy_in");
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
    {
    SGPU_NVTX_RANGE("psi.fullgpu.assemble");
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
    }

    if (solver_mode == 1) {
        SgpuNvtxRange solver_range(precision_mode == 1
            ? "psi.fullgpu.normal_eq.fp64" : "psi.fullgpu.normal_eq.fp32");
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
        SgpuNvtxRange solver_range(precision_mode == 1
            ? "psi.fullgpu.qr.fp64" : "psi.fullgpu.qr.fp32");
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
        if (warm_solver) {
            column_norms_kernel_f32<<<n_coeff, threads>>>(d_qr_mat_f, d_scale, n_points, qr_rows, n_coeff);
            scale_columns_kernel_f32<<<grid_s, block_s>>>(d_qr_mat_f, d_scale, n_points, qr_rows, n_coeff);
            set_ridge_tail_kernel<float><<<(n_coeff * n_coeff + threads - 1) / threads, threads>>>(
                d_qr_mat_f, n_points, qr_rows, n_coeff, static_cast<float>(ridge_lambda));
            if (cuda_check(cudaGetLastError(), "fit warm scale/augment") ||
                cuda_check(cudaDeviceSynchronize(), "fit warm scale/augment sync")) {
                cleanup();
                return 1;
            }
            stats.qr_scale_s = std::chrono::duration<double>(clock::now() - t_qr).count();
            stats.linear_prep_s = stats.qr_transpose_s + stats.qr_scale_s;
            t_qr = clock::now();
            coeff_host_f.resize(static_cast<size_t>(n_coeff));
            for (int index = 0; index < n_coeff; ++index) {
                coeff_host_f[static_cast<size_t>(index)] = static_cast<float>(coeff_host[index]);
            }
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_coeff_f), static_cast<size_t>(n_coeff) * sizeof(float)), "fit warm d_coeff_f") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_warm_residual_f), rhs_qr_bytes_f), "fit warm residual") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_warm_normal_f), static_cast<size_t>(n_coeff) * sizeof(float)), "fit warm normal") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_warm_direction_f), static_cast<size_t>(n_coeff) * sizeof(float)), "fit warm direction") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_warm_image_f), rhs_qr_bytes_f), "fit warm image") ||
                cuda_check(cudaMemcpy(d_coeff_f, coeff_host_f.data(), static_cast<size_t>(n_coeff) * sizeof(float), cudaMemcpyHostToDevice), "fit warm copy initial")) {
                cleanup();
                return 1;
            }
            if (warm_preconditioned) {
                if (!sgpu_has_psi_warm_preconditioner(n_coeff)) {
                    return fail("fit warm center QR preconditioner is unavailable");
                }
                std::vector<double> endpoint_scale(static_cast<size_t>(n_coeff));
                std::vector<float> scale_ratio(static_cast<size_t>(n_coeff));
                if (cuda_check(cudaMemcpy(
                        endpoint_scale.data(), d_scale,
                        static_cast<size_t>(n_coeff) * sizeof(double), cudaMemcpyDeviceToHost
                    ), "fit warm copy endpoint scale")) {
                    cleanup();
                    return 1;
                }
                for (int index = 0; index < n_coeff; ++index) {
                    scale_ratio[static_cast<size_t>(index)] = static_cast<float>(
                        endpoint_scale[static_cast<size_t>(index)] /
                        g_psi_warm_preconditioner.center_scale[static_cast<size_t>(index)]
                    );
                }
                const size_t factor_bytes = static_cast<size_t>(n_coeff) * n_coeff * sizeof(float);
                if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_warm_transformed_direction_f), static_cast<size_t>(n_coeff) * sizeof(float)), "fit warm transformed direction") ||
                    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_warm_scale_ratio_f), static_cast<size_t>(n_coeff) * sizeof(float)), "fit warm scale ratio") ||
                    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_warm_center_factor_f), factor_bytes), "fit warm center factor") ||
                    cuda_check(cudaMemcpy(d_warm_scale_ratio_f, scale_ratio.data(), static_cast<size_t>(n_coeff) * sizeof(float), cudaMemcpyHostToDevice), "fit warm upload scale ratio") ||
                    cuda_check(cudaMemcpy(d_warm_center_factor_f, g_psi_warm_preconditioner.upper_factor.data(), factor_bytes, cudaMemcpyHostToDevice), "fit warm upload center factor")) {
                    cleanup();
                    return 1;
                }
            }
            scale_coeff_kernel_f32<<<blocks1d_coeff, threads>>>(d_coeff_f, d_scale, n_coeff);
            const float one = 1.0f, minus_one = -1.0f, zero = 0.0f;
            if (cuda_check(cudaGetLastError(), "fit warm scale initial") ||
                cuda_check(cudaMemcpy(d_warm_residual_f, d_qr_rhs_f, rhs_qr_bytes_f, cudaMemcpyDeviceToDevice), "fit warm copy rhs") ||
                cublas_check(cublasSgemv(f->blas, CUBLAS_OP_N, qr_rows, n_coeff, &minus_one,
                    d_qr_mat_f, qr_rows, d_coeff_f, 1, &one, d_warm_residual_f, 1), "fit warm initial residual") ||
                cublas_check(cublasSgemv(f->blas, CUBLAS_OP_T, qr_rows, n_coeff, &one,
                    d_qr_mat_f, qr_rows, d_warm_residual_f, 1, &zero, d_warm_normal_f, 1), "fit warm initial normal")) {
                cleanup();
                return 1;
            }
            if (warm_preconditioned) {
                scale_coeff_float_kernel<<<blocks1d_coeff, threads>>>(
                    d_warm_normal_f, d_warm_scale_ratio_f, n_coeff
                );
                if (cuda_check(cudaGetLastError(), "fit warm precondition normal scale") ||
                    cublas_check(cublasStrsv(
                        f->blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                        CUBLAS_DIAG_NON_UNIT, n_coeff, d_warm_center_factor_f,
                        n_coeff, d_warm_normal_f, 1
                    ), "fit warm center R^-T")) {
                    cleanup();
                    return 1;
                }
            }
            if (cublas_check(cublasScopy(
                    f->blas, n_coeff, d_warm_normal_f, 1,
                    d_warm_direction_f, 1
                ), "fit warm initial direction")) {
                cleanup();
                return 1;
            }
            float gamma = 0.0f;
            if (cublas_check(cublasSdot(f->blas, n_coeff, d_warm_normal_f, 1, d_warm_normal_f, 1, &gamma), "fit warm initial gamma")) {
                cleanup();
                return 1;
            }
            for (int iteration = 0; iteration < warm_iterations; ++iteration) {
                if (!(gamma > 0.0f) || !std::isfinite(gamma)) break;
                float* search_direction = d_warm_direction_f;
                if (warm_preconditioned) {
                    if (cublas_check(cublasScopy(
                            f->blas, n_coeff, d_warm_direction_f, 1,
                            d_warm_transformed_direction_f, 1
                        ), "fit warm copy preconditioned direction") ||
                        cublas_check(cublasStrsv(
                            f->blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                            CUBLAS_DIAG_NON_UNIT, n_coeff, d_warm_center_factor_f,
                            n_coeff, d_warm_transformed_direction_f, 1
                        ), "fit warm center R^-1")) {
                        cleanup();
                        return 1;
                    }
                    scale_coeff_float_kernel<<<blocks1d_coeff, threads>>>(
                        d_warm_transformed_direction_f, d_warm_scale_ratio_f, n_coeff
                    );
                    if (cuda_check(cudaGetLastError(), "fit warm precondition direction scale")) {
                        cleanup();
                        return 1;
                    }
                    search_direction = d_warm_transformed_direction_f;
                }
                if (cublas_check(cublasSgemv(f->blas, CUBLAS_OP_N, qr_rows, n_coeff, &one,
                        d_qr_mat_f, qr_rows, search_direction, 1, &zero, d_warm_image_f, 1), "fit warm A p")) {
                    cleanup();
                    return 1;
                }
                float delta = 0.0f;
                if (cublas_check(cublasSdot(f->blas, qr_rows, d_warm_image_f, 1, d_warm_image_f, 1, &delta), "fit warm delta")) {
                    cleanup();
                    return 1;
                }
                if (!(delta > 0.0f) || !std::isfinite(delta)) break;
                const float alpha = gamma / delta;
                const float minus_alpha = -alpha;
                if (cublas_check(cublasSaxpy(f->blas, n_coeff, &alpha, search_direction, 1, d_coeff_f, 1), "fit warm solution update") ||
                    cublas_check(cublasSaxpy(f->blas, qr_rows, &minus_alpha, d_warm_image_f, 1, d_warm_residual_f, 1), "fit warm residual update") ||
                    cublas_check(cublasSgemv(f->blas, CUBLAS_OP_T, qr_rows, n_coeff, &one,
                        d_qr_mat_f, qr_rows, d_warm_residual_f, 1, &zero, d_warm_normal_f, 1), "fit warm A^T r")) {
                    cleanup();
                    return 1;
                }
                if (warm_preconditioned) {
                    scale_coeff_float_kernel<<<blocks1d_coeff, threads>>>(
                        d_warm_normal_f, d_warm_scale_ratio_f, n_coeff
                    );
                    if (cuda_check(cudaGetLastError(), "fit warm precondition next normal scale") ||
                        cublas_check(cublasStrsv(
                            f->blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                            CUBLAS_DIAG_NON_UNIT, n_coeff, d_warm_center_factor_f,
                            n_coeff, d_warm_normal_f, 1
                        ), "fit warm next center R^-T")) {
                        cleanup();
                        return 1;
                    }
                }
                float gamma_new = 0.0f;
                if (cublas_check(cublasSdot(f->blas, n_coeff, d_warm_normal_f, 1, d_warm_normal_f, 1, &gamma_new), "fit warm gamma")) {
                    cleanup();
                    return 1;
                }
                if (!(gamma_new > 0.0f) || !std::isfinite(gamma_new)) break;
                const float beta = gamma_new / gamma;
                if (cublas_check(cublasSscal(f->blas, n_coeff, &beta, d_warm_direction_f, 1), "fit warm direction scale") ||
                    cublas_check(cublasSaxpy(f->blas, n_coeff, &one, d_warm_normal_f, 1, d_warm_direction_f, 1), "fit warm direction update")) {
                    cleanup();
                    return 1;
                }
                gamma = gamma_new;
            }
            unscale_coeff_kernel_f32<<<blocks1d_coeff, threads>>>(d_coeff_f, d_scale, n_coeff);
            if (cuda_check(cudaGetLastError(), "fit warm unscale") ||
                cuda_check(cudaDeviceSynchronize(), "fit warm solve sync")) {
                cleanup();
                return 1;
            }
            stats.qr_factor_s = 0.0;
            stats.qr_apply_qtb_s = std::chrono::duration<double>(clock::now() - t_qr).count();
            stats.qr_tri_s = 0.0;
            stats.solve_s = stats.qr_apply_qtb_s;
            d_coeff_src_f = d_coeff_f;
        } else if (precision_mode == 1) {
            column_norms_kernel_f64<<<n_coeff, threads>>>(d_qr_mat_d, d_scale, n_points, qr_rows, n_coeff);
            scale_columns_kernel_f64<<<grid_s, block_s>>>(d_qr_mat_d, d_scale, n_points, qr_rows, n_coeff);
            set_ridge_tail_kernel<double><<<(n_coeff * n_coeff + threads - 1) / threads, threads>>>(d_qr_mat_d, n_points, qr_rows, n_coeff, ridge_lambda);
        } else {
            column_norms_kernel_f32<<<n_coeff, threads>>>(d_qr_mat_f, d_scale, n_points, qr_rows, n_coeff);
            scale_columns_kernel_f32<<<grid_s, block_s>>>(d_qr_mat_f, d_scale, n_points, qr_rows, n_coeff);
            set_ridge_tail_kernel<float><<<(n_coeff * n_coeff + threads - 1) / threads, threads>>>(d_qr_mat_f, n_points, qr_rows, n_coeff, static_cast<float>(ridge_lambda));
        }
        if (!warm_solver) {
            if (cuda_check(cudaGetLastError(), "fit qr scale/augment") ||
                cuda_check(cudaDeviceSynchronize(), "fit qr scale/augment sync")) {
                cleanup();
                return 1;
            }
            stats.qr_scale_s = std::chrono::duration<double>(clock::now() - t_qr).count();
            stats.linear_prep_s = stats.qr_transpose_s + stats.qr_scale_s;
        }

        if (precision_mode == 2 &&
            !maybe_write_psi_qr_snapshot(
                d_qr_mat_f, d_qr_rhs_f, d_scale, qr_rows, n_coeff, n_points, ridge
            )) {
            cleanup();
            return 1;
        }

        t_qr = clock::now();
        if (!warm_solver && precision_mode == 1) {
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
        } else if (!warm_solver) {
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
        if (!warm_solver) {
            if (cuda_check(cudaDeviceSynchronize(), "fit qr geqrf sync")) {
                cleanup();
                return 1;
            }
            stats.qr_factor_s = std::chrono::duration<double>(clock::now() - t_qr).count();
            if (precision_mode == 2 && solver_mode == 2 &&
                g_psi_warm_preconditioner.capture_enabled) {
                g_psi_warm_preconditioner.coefficient_count = n_coeff;
                g_psi_warm_preconditioner.upper_factor.resize(
                    static_cast<size_t>(n_coeff) * n_coeff
                );
                g_psi_warm_preconditioner.center_scale.resize(static_cast<size_t>(n_coeff));
                if (cuda_check(cudaMemcpy2D(
                        g_psi_warm_preconditioner.upper_factor.data(),
                        static_cast<size_t>(n_coeff) * sizeof(float),
                        d_qr_mat_f, static_cast<size_t>(qr_rows) * sizeof(float),
                        static_cast<size_t>(n_coeff) * sizeof(float), n_coeff,
                        cudaMemcpyDeviceToHost
                    ), "fit capture center R") ||
                    cuda_check(cudaMemcpy(
                        g_psi_warm_preconditioner.center_scale.data(), d_scale,
                        static_cast<size_t>(n_coeff) * sizeof(double), cudaMemcpyDeviceToHost
                    ), "fit capture center scale")) {
                    cleanup();
                    return 1;
                }
            }
        }

        t_qr = clock::now();
        if (!warm_solver && precision_mode == 1) {
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
        } else if (!warm_solver) {
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
        if (!warm_solver) {
            if (cuda_check(cudaDeviceSynchronize(), "fit qr ormqr sync")) {
                cleanup();
                return 1;
            }
            stats.qr_apply_qtb_s = std::chrono::duration<double>(clock::now() - t_qr).count();
        }

        t_qr = clock::now();
        if (!warm_solver && precision_mode == 1) {
            if (cublas_check(cublasDtrsv(f->blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n_coeff, d_qr_mat_d, qr_rows, d_qr_rhs_d, 1), "fit qr trsv d64")) {
                cleanup();
                return 1;
            }
            unscale_coeff_kernel<<<blocks1d_coeff, threads>>>(d_qr_rhs_d, d_scale, n_coeff);
        } else if (!warm_solver) {
            if (cublas_check(cublasStrsv(f->blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n_coeff, d_qr_mat_f, qr_rows, d_qr_rhs_f, 1), "fit qr trsv f32")) {
                cleanup();
                return 1;
            }
            unscale_coeff_kernel_f32<<<blocks1d_coeff, threads>>>(d_qr_rhs_f, d_scale, n_coeff);
        }
        if (!warm_solver) {
            if (cuda_check(cudaGetLastError(), "fit qr unscale coeff") ||
                cuda_check(cudaDeviceSynchronize(), "fit qr trsv sync")) {
                cleanup();
                return 1;
            }
            stats.qr_tri_s = std::chrono::duration<double>(clock::now() - t_qr).count();
            stats.solve_s = stats.qr_factor_s + stats.qr_apply_qtb_s + stats.qr_tri_s;
        }
        if (!warm_solver) {
            if (precision_mode == 1) d_coeff_src_d = d_qr_rhs_d;
            else d_coeff_src_f = d_qr_rhs_f;
        }
    }

    t = clock::now();
    {
    SGPU_NVTX_RANGE("psi.fullgpu.residual");
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
    }

    t = clock::now();
    {
    SGPU_NVTX_RANGE("psi.fullgpu.copy_out");
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
    }
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
    SGPU_NVTX_RANGE("trace.period.fp64");
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
    SgpuNvtxRange trace_range(mode == 1
        ? "trace.period.Bfp32_statefp64"
        : mode == 2
            ? "trace.period.fp32"
            : "trace.period.Bfp32_statefp16");
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
    SGPU_NVTX_RANGE("trace.axis_samples.mixed");
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
