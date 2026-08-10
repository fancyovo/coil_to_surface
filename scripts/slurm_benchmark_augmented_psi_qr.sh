#!/usr/bin/env bash
#SBATCH --job-name=bench-psi-qr-aug
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/bench-psi-qr-aug-%j.out
#SBATCH --error=logs/bench-psi-qr-aug-%j.err

set -euo pipefail

project=${PROJECT:?PROJECT is required}
snapshot=${SNAPSHOT:?SNAPSHOT is required}
output=${OUTPUT:?OUTPUT is required}
cuda_root=${CUDA_ROOT:-/public/app/cuda/13.0}
build_dir=${BUILD_DIR:-$project/build/qr_bench}

cleanup() {
    status=$?
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits > "${output%.json}.gpu_postflight.csv" 2>/dev/null || true
    ps -u "$USER" -o pid=,ppid=,stat=,comm= | awk '$3 ~ /^Z/ {print}' \
        > "${output%.json}.zombies_postflight.txt" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$project/logs" "$(dirname "$output")"
cd "$project"
export PATH="$cuda_root/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_root/lib64:${LD_LIBRARY_PATH:-}"

cmake -S gpu_backend -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DSGPU_BUILD_QR_BENCHMARK=ON
cmake --build "$build_dir" -j4 --target psi_qr_benchmark

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "allocated GPU is not idle before QR benchmark" >&2
    exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "${output%.json}.gpu_preflight.csv"

"$build_dir/psi_qr_benchmark" \
    --snapshot "$snapshot" --method augmented_rhs --device 0 \
    --warmups 1 --repeats 5 > "$output"
