#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=flow-adam-64d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Compatibility launcher for the older multi-GPU implementation. Its defaults
# are kept identical to the current 309-trajectory protocol; the canonical
# single-GPU entry point is scripts/optimize_flow_latent.py.

project="${PROJECT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
asset_root="${ASSET_ROOT:-$HOME/local_surface_evaluator}"
checkpoint="${FLOW_CHECKPOINT:-$asset_root/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
lib="${SCORE_LIB:-$project/gpu_backend/build_native_score/libstellarator_gpu.so}"
expected_lib_sha="${EXPECTED_SCORE_LIB_SHA:-565c32073b145d97a1f2244705fb06e4b3458ce798cd74d0c97ee4e0129dc729}"
run_root="${RUN_ROOT:-$project/runs/qh_flow_standard_adam/${SLURM_JOB_ID}}"
iterations="${ITERATIONS:-200}"
max_wall_s="${MAX_WALL_S:-1500}"
learning_rate="${LEARNING_RATE:-0.02}"
perturbation="${PERTURBATION:-0.005}"
directions="${DIRECTIONS:-64}"
direction_bank_size="${DIRECTION_BANK_SIZE:-$directions}"
reuse_update_direction_after="${REUSE_UPDATE_DIRECTION_AFTER:-0}"
gradient_estimator="${GRADIENT_ESTIMATOR:-central}"
flow_steps="${FLOW_STEPS:-128}"
flow_pipeline="${FLOW_PIPELINE:-1}"
score_gpus="${SCORE_GPUS:-0,1,2,3}"
score_gpus="${score_gpus//:/,}"
beta1="${BETA1:-0.7}"
beta2="${BETA2:-0.999}"
robust_direction_filter="${ROBUST_DIRECTION_FILTER:-1}"
reject_invalid_center="${REJECT_INVALID_CENTER:-1}"
invalid_center_backtracking="${INVALID_CENTER_BACKTRACKING:-0.5,0.25,0.125}"
direction_outlier_ratio="${DIRECTION_OUTLIER_RATIO:-8.0}"
direction_outlier_mad_factor="${DIRECTION_OUTLIER_MAD_FACTOR:-8.0}"
seed="${SEED:-2026073004}"
initial_case="${INITIAL_CASE:-}"
nfp="${NFP:-4}"
n_base_coils="${N_BASE_COILS:-3}"
score_surface_mode="${SCORE_SURFACE_MODE:-continuous}"
surface_confidence_periods="${SURFACE_CONFIDENCE_PERIODS:-1}"
surface_theta_count="${SURFACE_THETA_COUNT:-128}"
surface_trace_steps="${SURFACE_TRACE_STEPS:-400}"
surface_flux_bisection_iters="${SURFACE_FLUX_BISECTION_ITERS:-6}"
iota_degree="${IOTA_DEGREE:-3}"
axis_continuation="${AXIS_CONTINUATION:-1}"
axis_hint_verification="${AXIS_HINT_VERIFICATION:-mixed}"
resume="${RESUME:-0}"
gpu_selector="${CUDA_VISIBLE_DEVICES:-}"
children=()

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if (( ${#children[@]} )); then
    pkill -TERM -P "${children[0]}" 2>/dev/null || true
    kill "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
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
test "$(sha256sum "$lib" | awk '{print $1}')" = "$expected_lib_sha"
initial_args=()
if [[ -n "$initial_case" ]]; then
  test -f "$initial_case"
  initial_args+=(--initial-case "$initial_case")
fi
robust_gradient_args=(
  --direction-outlier-ratio "$direction_outlier_ratio"
  --direction-outlier-mad-factor "$direction_outlier_mad_factor"
)
if [[ "$robust_direction_filter" == "1" ]]; then
  robust_gradient_args+=(--robust-direction-filter)
else
  robust_gradient_args+=(--no-robust-direction-filter)
fi
if [[ "$reject_invalid_center" == "1" ]]; then
  robust_gradient_args+=(
    --reject-invalid-center
    --invalid-center-backtracking "$invalid_center_backtracking"
  )
else
  robust_gradient_args+=(--no-reject-invalid-center)
fi
score_mode_args=(
  --score-surface-mode "$score_surface_mode"
  --surface-confidence-periods "$surface_confidence_periods"
  --surface-theta-count "$surface_theta_count"
  --surface-trace-steps "$surface_trace_steps"
  --surface-flux-bisection-iters "$surface_flux_bisection_iters"
  --iota-degree "$iota_degree"
  --axis-hint-verification "$axis_hint_verification"
)
gradient_args=(
  --directions "$directions"
  --direction-bank-size "$direction_bank_size"
  --reuse-update-direction-after "$reuse_update_direction_after"
  --gradient-estimator "$gradient_estimator"
  --flow-steps "$flow_steps"
  --gpus "$score_gpus"
)
if [[ "$flow_pipeline" == "1" ]]; then
  gradient_args+=(--flow-pipeline)
else
  gradient_args+=(--no-flow-pipeline)
fi
if [[ "$axis_continuation" == "1" ]]; then
  score_mode_args+=(--axis-continuation)
else
  score_mode_args+=(--no-axis-continuation)
fi
resume_args=()
if [[ "$resume" == "1" ]]; then
  resume_args+=(--resume)
fi
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

python "$project/scripts/optimize_flow_prior_standard_adam.py" \
  --checkpoint "$checkpoint" \
  --lib "$lib" \
  --out-dir "$run_root" \
  --nfp "$nfp" \
  --n-base-coils "$n_base_coils" \
  --iterations "$iterations" \
  --max-wall-s "$max_wall_s" \
  --learning-rate "$learning_rate" \
  --perturbation "$perturbation" \
  --beta1 "$beta1" \
  --beta2 "$beta2" \
  --seed "$seed" \
  "${gradient_args[@]}" \
  "${score_mode_args[@]}" \
  "${robust_gradient_args[@]}" \
  "${resume_args[@]}" \
  "${initial_args[@]}" &
children+=("$!")
wait "${children[0]}"
children=()
