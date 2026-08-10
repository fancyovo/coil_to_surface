#!/usr/bin/env bash
#SBATCH --job-name=build-magma-qr
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build-magma-qr-%j.out
#SBATCH --error=logs/build-magma-qr-%j.err

set -euo pipefail

project=${PROJECT:?PROJECT is required}
source_tar=${MAGMA_TAR:?MAGMA_TAR is required}
work_root=${MAGMA_WORK_ROOT:?MAGMA_WORK_ROOT is required}
cuda_root=${CUDA_ROOT:-/public/app/cuda/13.0}
mkl_root=${MKL_ROOT:-/public/app/intel/oneapi/mkl/2026.0}

mkdir -p "$project/logs" "$work_root"
cd "$work_root"
export PATH="$cuda_root/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_root/lib64:$mkl_root/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -d magma-2.10.0 ]]; then
    tar -xzf "$source_tar"
fi

cmake -S magma-2.10.0 -B magma-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
    -DGPU_TARGET=sm_120 \
    -DMAGMA_ENABLE_CUDA=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DLAPACK_LIBRARIES="$mkl_root/lib/libmkl_rt.so" \
    -DCMAKE_INSTALL_PREFIX="$work_root/magma-install"
cmake --build magma-build -j "$SLURM_CPUS_PER_TASK"
cmake --install magma-build

cmake -S "$project/gpu_backend" -B "$project/build/qr_bench_magma" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DSGPU_BUILD_QR_BENCHMARK=ON \
    -DSGPU_MAGMA_ROOT="$work_root/magma-install"
cmake --build "$project/build/qr_bench_magma" -j "$SLURM_CPUS_PER_TASK"

printf '{"status":"ok","job_id":"%s","magma_root":"%s"}\n' \
    "$SLURM_JOB_ID" "$work_root/magma-install" > "$work_root/build_done.json"
