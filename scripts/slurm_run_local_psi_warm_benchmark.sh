#!/usr/bin/env bash
#SBATCH --job-name=psi-warm-bench
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT="${PROJECT:-$HOME/local_surface_evaluator/runtimes/local_score_d1b140f}"
RUN_ROOT="${RUN_ROOT:-$HOME/local_surface_evaluator/runs/local_psi_warm_start_20260811_v2}"
BUILD_DIR="${BUILD_DIR:-$PROJECT/build_psi_warm_start}"
CUDA_ROOT="${CUDA_ROOT:-/public/app/cuda/13.0}"
ITERATIONS="${ITERATIONS:-0 1 2 4 8 16 32 64}"
METHOD_PREFIX="${METHOD_PREFIX:-warmcgls}"
BENCHMARK_FILE="${BENCHMARK_FILE:-$RUN_ROOT/benchmark.jsonl}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$RUN_ROOT/analysis}"

export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$ANALYSIS_DIR"
test -f "$RUN_ROOT/snapshots/center.bin"
test -f "$RUN_ROOT/snapshots/manifest.json"

gpu_csv() {
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits
}

gpu_csv > "$RUN_ROOT/gpu_benchmark_preflight.csv"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    echo "Allocated GPU already has a compute process" >&2
    exit 2
fi

cmake -S "$PROJECT/gpu_backend" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
    -DSGPU_BUILD_QR_BENCHMARK=ON
cmake --build "$BUILD_DIR" --target psi_qr_benchmark -j4

BENCHMARK="$BENCHMARK_FILE"
if [[ "${REUSE_BENCHMARK:-0}" == "1" ]]; then
    test -s "$BENCHMARK"
else
    : > "$BENCHMARK"
    for endpoint in "$RUN_ROOT"/snapshots/direction_*_minus.bin "$RUN_ROOT"/snapshots/direction_*_plus.bin; do
        test -f "$endpoint"
        for iteration in $ITERATIONS; do
            "$BUILD_DIR/psi_qr_benchmark" \
                --snapshot "$endpoint" \
                --warm-snapshot "$RUN_ROOT/snapshots/center.bin" \
                --method "${METHOD_PREFIX}${iteration}" \
                --repeats 1 >> "$BENCHMARK"
        done
    done
fi

python "$PROJECT/scripts/analyze_local_psi_warm_start.py" \
    --run-root "$RUN_ROOT" \
    --benchmark "$BENCHMARK" \
    --output-dir "$ANALYSIS_DIR"

gpu_csv > "$RUN_ROOT/gpu_benchmark_postflight.csv"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    echo "GPU compute process remains after benchmark" >&2
    exit 3
fi
if ps -u "$USER" -o stat=,pid=,ppid=,comm= | awk '$1 ~ /^Z/ {found=1} END {exit !found}'; then
    echo "Zombie process remains after benchmark" >&2
    exit 4
fi

touch "$RUN_ROOT/benchmark_completed.ok"
