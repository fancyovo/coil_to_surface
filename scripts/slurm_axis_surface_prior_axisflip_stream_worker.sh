#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_LIB_SHA:?EXPECTED_LIB_SHA is required}"
: "${EXPECTED_CHECKPOINT_SHA:?EXPECTED_CHECKPOINT_SHA is required}"
: "${WORKER_OFFSET:?WORKER_OFFSET is required}"

worker_index=$((WORKER_OFFSET + SLURM_ARRAY_TASK_ID))
cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(sha256sum "$SCORE_LIB" | awk '{print $1}')" = "$EXPECTED_LIB_SHA"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA"
source "$HOME/coil/.venv/bin/activate"
mkdir -p "$RUN_ROOT/workers" "$PROJECT/logs"

cleanup() {
  status=$?
  trap - EXIT INT TERM
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits \
    > "$RUN_ROOT/worker_${worker_index}_gpu_postflight.csv" 2>/dev/null || true
  ps -eo stat=,pid=,ppid=,cmd= | awk '$1 ~ /^Z/ {print}' \
    > "$RUN_ROOT/worker_${worker_index}_zombies.txt" || true
  exit "$status"
}
trap cleanup EXIT INT TERM

mapfile -t compute_processes < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#compute_processes[@]} )); then
  printf 'allocated GPU is not idle: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$RUN_ROOT/worker_${worker_index}_gpu_preflight.csv"

export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python "$PROJECT/scripts/run_axis_surface_prior_axisflip_stream.py" \
  --run-root "$RUN_ROOT" \
  --protocol-path "$RUN_ROOT/protocol.json" \
  --checkpoint "$CHECKPOINT" \
  --score-lib "$SCORE_LIB" \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-lib-sha "$EXPECTED_LIB_SHA" \
  --expected-checkpoint-sha "$EXPECTED_CHECKPOINT_SHA" \
  --worker-index "$worker_index" \
  --worker-count 6 \
  --seed "${PRIOR_SEED:-20260905}" \
  --device 0 \
  --discovery-wall-s "${DISCOVERY_WALL_S:-14400}" \
  --hard-wall-s "${HARD_WALL_S:-17700}" \
  --iterations 200
