#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-flow-latent-64d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=96G
#SBATCH --time=07:50:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
asset_root="${ASSET_ROOT:-$HOME/local_surface_evaluator}"
checkpoint="${FLOW_CHECKPOINT:-$asset_root/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
lib="${SCORE_LIB:?SCORE_LIB must name the validated experimental score library}"
expected_lib_sha="${EXPECTED_SCORE_LIB_SHA:-565c32073b145d97a1f2244705fb06e4b3458ce798cd74d0c97ee4e0129dc729}"
gradient_lib="${GRADIENT_LIB:-$lib}"
expected_gradient_lib_sha="${EXPECTED_GRADIENT_LIB_SHA:-$expected_lib_sha}"
initial_case="${INITIAL_CASE:?INITIAL_CASE must name selected_start.json from screen_flow_starts.py}"
run_root="${RUN_ROOT:-$project/runs/qh_flow_latent/${SLURM_JOB_ID}}"
iterations="${ITERATIONS:-200}"
max_wall_s="${MAX_WALL_S:-7200}"
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
if [[ -n "${EXPECTED_COMMIT:-}" ]]; then
  test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
fi
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$checkpoint"
test -f "$lib"
test "$(sha256sum "$lib" | awk '{print $1}')" = "$expected_lib_sha"
test -f "$gradient_lib"
test "$(sha256sum "$gradient_lib" | awk '{print $1}')" = "$expected_gradient_lib_sha"
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
pipeline_args=(--flow-pipeline)
if [[ "${FLOW_PIPELINE:-1}" != "1" ]]; then pipeline_args=(--no-flow-pipeline); fi

python "$project/scripts/optimize_flow_latent.py" \
  --checkpoint "$checkpoint" \
  --initial-case "$initial_case" \
  --lib "$lib" \
  --gradient-lib "$gradient_lib" \
  --out-dir "$run_root" \
  --nfp "${NFP:-4}" \
  --n-base-coils "${N_BASE_COILS:-3}" \
  --iterations "$iterations" \
  --max-wall-s "$max_wall_s" \
  --flow-steps "${FLOW_STEPS:-128}" \
  --parameter-space "${PARAMETER_SPACE:-latent}" \
  --perturbation "${PERTURBATION:-0.005}" \
  --gradient-mode "${GRADIENT_MODE:-random-orthogonal}" \
  --random-directions "${RANDOM_DIRECTIONS:-64}" \
  --seed "${SEED:-20260812}" \
  --optimizer "${OPTIMIZER:-adam}" \
  --learning-rate "${LEARNING_RATE:-0.02}" \
  --beta1 "${BETA1:-0.7}" \
  --beta2 "${BETA2:-0.999}" \
  --flow-device 0 \
  --score-device 0 \
  --psi-iterations "${PSI_ITERATIONS:-4}" \
  --alpha-iterations "${ALPHA_ITERATIONS:-4}" \
  --formal-surface-theta-count "${FORMAL_SURFACE_THETA_COUNT:-128}" \
  --local-surface-theta-count "${LOCAL_SURFACE_THETA_COUNT:-64}" \
  --iota-degree "${IOTA_DEGREE:-3}" \
  --bfgs-initial-trust-rms "${BFGS_INITIAL_TRUST_RMS:-0.01}" \
  --bfgs-min-trust-rms "${BFGS_MIN_TRUST_RMS:-0.00001}" \
  --bfgs-max-trust-rms "${BFGS_MAX_TRUST_RMS:-0.05}" \
  --bfgs-trust-growth "${BFGS_TRUST_GROWTH:-1.2}" \
  --bfgs-trust-shrink "${BFGS_TRUST_SHRINK:-0.5}" \
  --bfgs-min-improvement "${BFGS_MIN_IMPROVEMENT:-0.0}" \
  --plot-every "${PLOT_EVERY:-5}" \
  "${pipeline_args[@]}" \
  "${resume_args[@]}" &
child=$!
wait "$child"
child=""
