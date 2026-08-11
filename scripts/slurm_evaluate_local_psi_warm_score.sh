#!/usr/bin/env bash
#SBATCH --job-name=psi-warm-score
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
CANDIDATE_DIR="${CANDIDATE_DIR:-$HOME/local_surface_evaluator/runs/local_score_gradient_full300_20260811/candidates}"
RUN_ROOT="${RUN_ROOT:-$HOME/local_surface_evaluator/runs/local_psi_warm_score_20260811}"
BUILD_DIR="${BUILD_DIR:-$PROJECT/build_psi_warm_score}"
CUDA_ROOT="${CUDA_ROOT:-/public/app/cuda/13.0}"
ITERATIONS="${ITERATIONS:-0,2,4,8,16}"

export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"

test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$RUN_ROOT/gpu_preflight.csv"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    echo "Allocated GPU already has a compute process" >&2
    exit 2
fi

cmake -S "$PROJECT/gpu_backend" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc"
cmake --build "$BUILD_DIR" --target stellarator_gpu -j4

python "$PROJECT/scripts/evaluate_local_psi_warm_score.py" \
    --candidate-dir "$CANDIDATE_DIR" \
    --lib "$BUILD_DIR/libstellarator_gpu.so" \
    --output-dir "$RUN_ROOT/results" \
    --scale 0.005 \
    --iterations "$ITERATIONS"

git -C "$PROJECT" rev-parse HEAD > "$RUN_ROOT/git_head.txt"
sha256sum "$BUILD_DIR/libstellarator_gpu.so" > "$RUN_ROOT/library.sha256"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$RUN_ROOT/gpu_postflight.csv"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    echo "GPU compute process remains after warm-score evaluation" >&2
    exit 3
fi
if ps -u "$USER" -o stat=,pid=,ppid=,comm= | awk '$1 ~ /^Z/ {found=1} END {exit !found}'; then
    echo "Zombie process remains after warm-score evaluation" >&2
    exit 4
fi
touch "$RUN_ROOT/completed.ok"
