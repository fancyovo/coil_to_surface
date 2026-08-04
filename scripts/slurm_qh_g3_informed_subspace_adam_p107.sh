#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-g3-sub
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator_worktrees/qh-blackbox-gradient}"
checkpoint="${FLOW_CHECKPOINT:-$HOME/local_surface_evaluator/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
expected_checkpoint_sha="${EXPECTED_CHECKPOINT_SHA:-39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f}"
gradient_lib="${GRADIENT_LIB:-$project/gpu_backend/build_g4_oracle/libstellarator_gpu.so}"
expected_gradient_sha="${EXPECTED_GRADIENT_SHA:-071f67925a8eca6cfc702ed6b8380f4677bb4c36711e6aae032b7dd227bdc88c}"
initial_case="${INITIAL_CASE:-$project/reports/assets/qh_score_adam_start_panel_29960/start_10.json}"
output="${OUTPUT_DIR:-$project/runs/qh_g3_informed_subspace_adam_${SLURM_JOB_ID}}"
iterations="${ITERATIONS:-50}"
rk4_steps="${RK4_STEPS:-64}"
random_directions="${RANDOM_DIRECTIONS:-4}"
perturbation="${PERTURBATION:-0.005}"
learning_rate="${LEARNING_RATE:-0.01}"
proposal_mode="${PROPOSAL_MODE:-adam}"
projected_step_rms="${PROJECTED_STEP_RMS:-$perturbation}"
beta1="${BETA1:-0.5}"
beta2="${BETA2:-0.999}"
seed="${SEED:-2026080504}"
gpus="${GPUS:-0,1,2,3}"
child=""

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$child" ]]; then
    pkill -TERM -P "$child" 2>/dev/null || true
    kill "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
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
for path in "$checkpoint" "$gradient_lib" "$initial_case"; do test -f "$path"; done
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_checkpoint_sha"
test "$(sha256sum "$gradient_lib" | awk '{print $1}')" = "$expected_gradient_sha"

source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project:$project/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS=',' read -r utilization memory_used; do
    utilization="${utilization// /}"
    memory_used="${memory_used// /}"
    # Some idle 5090 drivers retain a stale nonzero utilization sample after
    # the previous context exits. Memory and the compute-process table are the
    # authoritative ownership checks.
    if (( memory_used > 16 )); then idle=0; fi
  done < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
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
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$output/gpu_preflight.csv"

python scripts/optimize_qh_g3_informed_subspace_adam.py \
  --checkpoint "$checkpoint" \
  --gradient-lib "$gradient_lib" \
  --initial-case "$initial_case" \
  --out-dir "$output" \
  --nfp 4 \
  --iterations "$iterations" \
  --rk4-steps "$rk4_steps" \
  --random-directions "$random_directions" \
  --perturbation "$perturbation" \
  --learning-rate "$learning_rate" \
  --proposal-mode "$proposal_mode" \
  --projected-step-rms "$projected_step_rms" \
  --beta1 "$beta1" \
  --beta2 "$beta2" \
  --seed "$seed" \
  --gpus "$gpus" &
child="$!"
wait "$child"
child=""
test -s "$output/summary.json"
