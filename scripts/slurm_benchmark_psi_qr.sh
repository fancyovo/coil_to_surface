#!/usr/bin/env bash
#SBATCH --job-name=bench-psi-qr
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench-psi-qr-%j.out
#SBATCH --error=logs/bench-psi-qr-%j.err

set -euo pipefail

project=${PROJECT:?PROJECT is required}
snapshot=${SNAPSHOT:?SNAPSHOT is required}
output=${OUTPUT:?OUTPUT is required}
method=${METHOD:?METHOD is required}
cuda_root=${CUDA_ROOT:-/public/app/cuda/13.0}
warmups=${WARMUPS:-1}
repeats=${REPEATS:-5}
benchmark_bin=${BENCHMARK_BIN:-$project/build/qr_bench/psi_qr_benchmark}
magma_root=${MAGMA_ROOT:-}

mkdir -p "$project/logs" "$(dirname "$output")"
cd "$project"
export PATH="$cuda_root/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
if [[ -n "$magma_root" ]]; then
    export LD_LIBRARY_PATH="$magma_root/lib:/public/app/intel/oneapi/mkl/2026.0/lib:$LD_LIBRARY_PATH"
fi

gpu_state_before=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)
if [[ "$gpu_state_before" -ne 0 ]]; then
    echo "allocated GPU is not idle before QR benchmark" >&2
    exit 1
fi

"$benchmark_bin" \
    --snapshot "$snapshot" --method "$method" --device 0 \
    --warmups "$warmups" --repeats "$repeats" > "$output"

if pgrep -u "$USER" -f 'psi_qr_benchmark' >/dev/null; then
    echo "QR benchmark process remains after completion" >&2
    exit 1
fi
