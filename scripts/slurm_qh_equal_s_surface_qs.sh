#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${EXPERIMENT_ROOT:?EXPERIMENT_ROOT is required}"
: "${GPU_LIB:?GPU_LIB is required}"

eval_env=${EVAL_ENV:-$HOME/local_surface_evaluator/.venv-desc016-py312}
shard_count=${SHARD_COUNT:-4}
shard_index=${SLURM_ARRAY_TASK_ID:-0}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  mapfile -t children < <(jobs -pr)
  if (( ${#children[@]} )); then
    kill "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
  fi
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits > "$EXPERIMENT_ROOT/equal_s_qs_postflight_${shard_index}.csv" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

mapfile -t compute_processes < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#compute_processes[@]} )); then
  printf 'allocated GPU is not idle: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits > "$EXPERIMENT_ROOT/equal_s_qs_preflight_${shard_index}.csv"

source "$eval_env/bin/activate"
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PROJECT:$eval_env/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "$PROJECT/scripts/run_qh_equal_s_surface_qs_shard.py" \
  --experiment-root "$EXPERIMENT_ROOT" \
  --gpu-lib "$GPU_LIB" \
  --device 0 \
  --shard-index "$shard_index" \
  --shard-count "$shard_count" \
  --n-phi "${N_PHI:-96}" \
  --n-theta "${N_THETA:-96}" \
  --output-name "${OUTPUT_NAME:-equal_s_qs_summary.json}" \
  ${OVERWRITE:+--overwrite}
