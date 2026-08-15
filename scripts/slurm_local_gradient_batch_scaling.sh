#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-gradient-scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
asset_root="${ASSET_ROOT:-$HOME/local_surface_evaluator}"
checkpoint="${FLOW_CHECKPOINT:-$asset_root/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
lib="${SCORE_LIB:?SCORE_LIB must name the validated query-batch score library}"
initial_case="${INITIAL_CASE:-$project/reports/assets/qh_score_adam_start_panel_29960/start_10.json}"
run_root="${RUN_ROOT:?RUN_ROOT is required}"
gpu_selector="${CUDA_VISIBLE_DEVICES:-}"
child=""

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$child" ]]; then
    pkill -TERM -P "$child" 2>/dev/null || true
    kill "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  if [[ -n "$gpu_selector" && -d "$run_root" ]]; then
    nvidia-smi --id="$gpu_selector" \
      --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits > "$run_root/gpu_postflight.csv" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$run_root"
cd "$project"
test -f "$checkpoint"
test -f "$lib"
test -f "$initial_case"
: "${gpu_selector:?CUDA_VISIBLE_DEVICES is required}"

source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS=',' read -r utilization memory_used; do
    utilization="${utilization// /}"
    memory_used="${memory_used// /}"
    if (( utilization != 0 || memory_used > 16 )); then idle=0; fi
  done < <(
    nvidia-smi --id="$gpu_selector" \
      --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
  )
  if nvidia-smi --id="$gpu_selector" \
      --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    idle=0
  fi
  if (( idle )); then
    ((idle_streak += 1))
    if (( idle_streak >= 3 )); then break; fi
  else
    idle_streak=0
  fi
  sleep 2
done
if (( idle_streak < 3 )); then
  echo "allocated GPU did not reach three consecutive idle probes" >&2
  exit 42
fi
nvidia-smi --id="$gpu_selector" \
  --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$run_root/gpu_preflight.csv"

python "$project/scripts/benchmark_local_gradient_batch_scaling.py" \
  --checkpoint "$checkpoint" \
  --initial-case "$initial_case" \
  --lib "$lib" \
  --output "$run_root/summary.json" \
  --endpoint-counts "${ENDPOINT_COUNTS:-2,4,8,16,32,64,128,256,600}" \
  --repeats "${REPEATS:-3}" \
  --perturbation "${PERTURBATION:-0.005}" \
  --seed "${SEED:-2026081501}" \
  --device 0 &
child=$!
wait "$child"
child=""

if nvidia-smi --id="$gpu_selector" \
    --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "GPU compute process remains after benchmark" >&2
  exit 3
fi
if ps -u "$USER" -o stat=,pid=,ppid=,comm= | \
    awk '$1 ~ /^Z/ {found=1} END {exit !found}'; then
  echo "Zombie process remains after benchmark" >&2
  exit 4
fi
touch "$run_root/completed.ok"
