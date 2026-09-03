#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_CHECKPOINT_SHA:?EXPECTED_CHECKPOINT_SHA is required}"
: "${WORKER_OFFSET:?WORKER_OFFSET is required}"

worker_index=$((WORKER_OFFSET + SLURM_ARRAY_TASK_ID))
score_lib="$RUN_ROOT/build_curvature_r04/libstellarator_gpu.so"
expected_lib_sha=$(tr -d '[:space:]' < "$RUN_ROOT/score_library.sha256")
cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(sha256sum "$score_lib" | awk '{print $1}')" = "$expected_lib_sha"
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

export PYTHONPATH="$PROJECT:$PROJECT/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$cuda_wheel_lib:/public/app/cuda/13.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python "$PROJECT/scripts/run_axis_surface_prior_axisflip_stream.py" \
  --run-root "$RUN_ROOT" \
  --protocol-path "$RUN_ROOT/protocol.json" \
  --protocol-id qh-axis-surface-contour-compact-flexible-axisflip-r012-curvature-r04-adam200-64d-abi11-v1 \
  --checkpoint "$CHECKPOINT" \
  --score-lib "$score_lib" \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-lib-sha "$expected_lib_sha" \
  --expected-checkpoint-sha "$EXPECTED_CHECKPOINT_SHA" \
  --worker-index "$worker_index" \
  --worker-count 6 \
  --seed 20260906 \
  --device 0 \
  --discovery-wall-s 25200 \
  --hard-wall-s 28200 \
  --iterations 200 \
  --max-completed-cases 2 \
  --legality-audit-cases-per-worker 64 \
  --fixed-nfp 8 \
  --fixed-n-base-coils 3 \
  --minor-radius-m 0.12
