#include "coil_field.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
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

const char* sgpu_last_error() {
    return g_last_error.c_str();
}

} // extern "C"
