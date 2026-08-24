#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=23:30:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${WORKER_OFFSET:?WORKER_OFFSET is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_LIB_SHA:?EXPECTED_LIB_SHA is required}"
: "${EXPECTED_CHECKPOINT_SHA:?EXPECTED_CHECKPOINT_SHA is required}"

worker_index=$((WORKER_OFFSET + SLURM_ARRAY_TASK_ID))
checkpoint=${CHECKPOINT:-$HOME/local_surface_evaluator/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}
score_lib=${SCORE_LIB:-$HOME/local_surface_evaluator/runtimes/local_score_d1b140f/gpu_backend/build_query_batch/libstellarator_gpu.so}

cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(sha256sum "$score_lib" | awk '{print $1}')" = "$EXPECTED_LIB_SHA"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA"

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${child:-}" ]] && kill -0 "$child" 2>/dev/null; then
    kill -TERM -- "-$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits \
    > "$RUN_ROOT/worker_${worker_index}_gpu_postflight.csv" 2>/dev/null || true
  ps -eo stat=,pid=,ppid=,cmd= | awk '$1 ~ /^Z/ {print}' \
    > "$RUN_ROOT/worker_${worker_index}_zombies.txt" || true
  exit "$status"
}
trap cleanup EXIT INT TERM

mapfile -t compute_processes < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} )); then
  printf 'allocated GPU is not idle: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$RUN_ROOT/worker_${worker_index}_gpu_preflight.csv"

source "$HOME/coil/.venv/bin/activate"
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

child=""
extra_args=()
if [[ "${ALLOW_PARTIAL:-0}" == "1" ]]; then
  extra_args+=(--allow-partial)
fi
setsid python "$PROJECT/scripts/run_qh_data_space_trajectory_rerun.py" \
  --run-root "$RUN_ROOT" \
  --worker-index "$worker_index" \
  --device 0 \
  --max-wall-s "${MAX_WALL_S:-81000}" \
  --max-new-cases "${MAX_NEW_CASES:-0}" \
  --max-attempts "${MAX_ATTEMPTS:-0}" \
  --iterations "${ITERATIONS:-200}" \
  "${extra_args[@]}" &
child=$!
wait "$child"
child=""
