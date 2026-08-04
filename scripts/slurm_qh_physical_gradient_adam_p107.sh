#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-g2-adam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator_worktrees/qh-blackbox-gradient}"
checkpoint="${FLOW_CHECKPOINT:-$HOME/local_surface_evaluator/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
gradient_lib="${GRADIENT_LIB:-$project/gpu_backend/build_gradient_sm120/libstellarator_gpu.so}"
initial_case="${INITIAL_CASE:-$project/reports/assets/qh_score_adam_start_panel_29960/start_10.json}"
output="${OUTPUT_DIR:?OUTPUT_DIR is required}"
rk4_steps="${RK4_STEPS:?RK4_STEPS is required}"
learning_rate="${LEARNING_RATE:?LEARNING_RATE is required}"
iterations="${ITERATIONS:-200}"
expected_checkpoint_sha="${EXPECTED_CHECKPOINT_SHA:-39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f}"
expected_gradient_sha="${EXPECTED_GRADIENT_SHA:-fdf142aad0f0e0739c61cf61fde9d7195a688079a26016a9c010571d9440f8d4}"
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
  if [[ -d "$output" ]]; then
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits > "$output/gpu_postflight.csv" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$project/logs" "$output"
cd "$project"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_checkpoint_sha"
test "$(sha256sum "$gradient_lib" | awk '{print $1}')" = "$expected_gradient_sha"
test -f "$initial_case"
: "${gpu_selector:?CUDA_VISIBLE_DEVICES is required}"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project:$project/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
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
  echo "allocated GPU did not reach three consecutive idle probes" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$output/gpu_preflight.csv"

python scripts/optimize_qh_physical_gradient_adam.py \
  --checkpoint "$checkpoint" \
  --gradient-lib "$gradient_lib" \
  --initial-case "$initial_case" \
  --out-dir "$output" \
  --nfp 4 \
  --iterations "$iterations" \
  --rk4-steps "$rk4_steps" \
  --learning-rate "$learning_rate" \
  --beta1 0.5 \
  --beta2 0.999 \
  --backtrack-fractions "0.5,0.25,0.125" \
  --compile-flow-model &
children+=("$!")
wait "${children[0]}"
children=()

test -s "$output/summary.json"
