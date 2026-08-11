#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-local-fullgrad-adam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:RTX5090:2
#SBATCH --mem=96G
#SBATCH --time=07:50:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
asset_root="${ASSET_ROOT:-$HOME/local_surface_evaluator}"
checkpoint="${FLOW_CHECKPOINT:-$asset_root/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
lib="${SCORE_LIB:?SCORE_LIB must name the validated experimental score library}"
initial_case="${INITIAL_CASE:-$project/reports/assets/qh_score_adam_start_panel_29960/start_10.json}"
run_root="${RUN_ROOT:-$project/runs/qh_local_full_gradient_adam/${SLURM_JOB_ID}}"
iterations="${ITERATIONS:-2000}"
max_wall_s="${MAX_WALL_S:-27300}"
resume="${RESUME:-0}"
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
export MPLBACKEND=Agg
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
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
  echo "allocated GPUs did not reach three consecutive idle probes" >&2
  exit 42
fi
nvidia-smi --id="$gpu_selector" \
  --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$run_root/gpu_preflight.csv"

resume_args=()
if [[ "$resume" == "1" ]]; then resume_args+=(--resume); fi

python "$project/scripts/optimize_flow_prior_local_full_gradient_adam.py" \
  --checkpoint "$checkpoint" \
  --initial-case "$initial_case" \
  --lib "$lib" \
  --out-dir "$run_root" \
  --iterations "$iterations" \
  --max-wall-s "$max_wall_s" \
  --flow-steps "${FLOW_STEPS:-128}" \
  --perturbation "${PERTURBATION:-0.005}" \
  --learning-rate "${LEARNING_RATE:-0.01}" \
  --beta1 "${BETA1:-0.7}" \
  --beta2 "${BETA2:-0.999}" \
  --psi-iterations "${PSI_ITERATIONS:-4}" \
  --alpha-iterations "${ALPHA_ITERATIONS:-4}" \
  --surface-theta-count "${SURFACE_THETA_COUNT:-128}" \
  --iota-degree "${IOTA_DEGREE:-3}" \
  --plot-every "${PLOT_EVERY:-5}" \
  "${resume_args[@]}" &
child=$!
wait "$child"
child=""
