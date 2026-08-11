#!/usr/bin/env bash
#SBATCH --job-name=psi-pcgls-300
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=192G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT="${PROJECT:-$HOME/local_surface_evaluator/runtimes/local_score_d1b140f}"
CANDIDATE_DIR="${CANDIDATE_DIR:-$HOME/local_surface_evaluator/runs/local_score_gradient_full300_20260811/candidates}"
RUN_ROOT="${RUN_ROOT:-$HOME/local_surface_evaluator/runs/local_psi_pcgls_full300_20260811}"
BUILD_DIR="${BUILD_DIR:-$PROJECT/build_psi_warm_score}"
CUDA_ROOT="${CUDA_ROOT:-/public/app/cuda/13.0}"

export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"

test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT/results" "$RUN_ROOT/analysis"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$RUN_ROOT/gpu_preflight.csv"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    echo "An allocated GPU already has a compute process" >&2
    exit 2
fi

cmake -S "$PROJECT/gpu_backend" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc"
cmake --build "$BUILD_DIR" --target stellarator_gpu -j8

pids=()
for rank in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$rank" python "$PROJECT/scripts/evaluate_local_psi_pcgls_full300_shard.py" \
        --candidate-dir "$CANDIDATE_DIR" \
        --lib "$BUILD_DIR/libstellarator_gpu.so" \
        --output "$RUN_ROOT/results/rank_${rank}.jsonl" \
        --shard-index "$rank" \
        --shard-count 4 \
        --scale 0.005 \
        --iterations 2,4 > "$RUN_ROOT/results/rank_${rank}.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

python "$PROJECT/scripts/analyze_local_psi_pcgls_full300.py" \
    --input-dir "$RUN_ROOT/results" \
    --output-dir "$RUN_ROOT/analysis" \
    --scale 0.005

git -C "$PROJECT" rev-parse HEAD > "$RUN_ROOT/git_head.txt"
sha256sum "$BUILD_DIR/libstellarator_gpu.so" > "$RUN_ROOT/library.sha256"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$RUN_ROOT/gpu_postflight.csv"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    echo "GPU compute process remains after full300 evaluation" >&2
    exit 3
fi
if ps -u "$USER" -o stat=,pid=,ppid=,comm= | awk '$1 ~ /^Z/ {found=1} END {exit !found}'; then
    echo "Zombie process remains after full300 evaluation" >&2
    exit 4
fi
touch "$RUN_ROOT/completed.ok"
