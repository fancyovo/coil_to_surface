#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_LIB_SHA:?EXPECTED_LIB_SHA is required}"
: "${SHARD_OFFSET:?SHARD_OFFSET is required}"
: "${PROTOCOL_ID:?PROTOCOL_ID is required}"

cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(sha256sum "$SCORE_LIB" | awk '{print $1}')" = "$EXPECTED_LIB_SHA"
mkdir -p "$RUN_ROOT" "$PROJECT/logs"
shard_count="${SHARD_COUNT:-6}"
shard_index=$((SHARD_OFFSET + SLURM_ARRAY_TASK_ID))

cleanup() {
  status=$?
  trap - EXIT INT TERM
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits \
    > "$RUN_ROOT/shard_${shard_index}_gpu_postflight.csv" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

mapfile -t compute_processes < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} )); then
  printf 'allocated GPU is not idle: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$RUN_ROOT/shard_${shard_index}_gpu_preflight.csv"

source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python "$PROJECT/scripts/sample_axis_surface_prior_v3.py" \
  --output-dir "$RUN_ROOT" \
  --lib "$SCORE_LIB" \
  --total-count "${TOTAL_COUNT:-3600}" \
  --shard-index "$shard_index" \
  --shard-count "$shard_count" \
  --seed "${PRIOR_SEED:-20260903}" \
  --protocol-id "$PROTOCOL_ID" \
  --progress-every 10
