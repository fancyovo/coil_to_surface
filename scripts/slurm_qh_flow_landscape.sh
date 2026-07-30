#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-flow-landscape
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator}"
data="${QH_DATA:-$HOME/local_surface_evaluator_data/quasr_qh_flow_v1}"
checkpoint="${FLOW_CHECKPOINT:-$project/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
lib="${SCORE_LIB:-$project/gpu_backend/build_native_score/libstellarator_gpu.so}"
output="${OUTPUT_DIR:-$project/runs/qh_flow_landscape_${SLURM_JOB_ID}}"
source_ids="${SOURCE_IDS:-1446077,1826200,2419096}"
directions="${DIRECTIONS:-4}"
closure_steps="${CLOSURE_STEPS:-32,64,128,256}"
alphas="${ALPHAS:--0.24,-0.225,-0.21,-0.195,-0.18,-0.165,-0.15,-0.135,-0.12,-0.105,-0.09,-0.075,-0.06,-0.045,-0.03,-0.015,0,0.015,0.03,0.045,0.06,0.075,0.09,0.105,0.12,0.135,0.15,0.165,0.18,0.195,0.21,0.225,0.24}"
world_size=4
children=()

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if (( ${#children[@]} )); then
    kill "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
  fi
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader > "$output/gpu_postflight.csv" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$output"
cd "$project"
test -f "$data/manifest.json"
test -f "$checkpoint"
test -f "$lib"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS= read -r memory_used; do
    memory_used="${memory_used// /}"
    if (( memory_used > 16 )); then idle=0; fi
  done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
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
  echo "allocated GPUs retained memory or compute processes during idle probes" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader > "$output/gpu_preflight.csv"

python "$project/scripts/qh_flow_landscape.py" \
  --data-dir "$data" --checkpoint "$checkpoint" --output-dir "$output" \
  --lib "$lib" --source-ids "$source_ids" --directions "$directions" \
  --alphas="$alphas" --closure-steps "$closure_steps" --prepare-only

for rank in 0 1 2 3; do
  python "$project/scripts/qh_flow_landscape.py" \
    --data-dir "$data" --checkpoint "$checkpoint" --output-dir "$output" \
    --lib "$lib" --source-ids "$source_ids" --directions "$directions" \
    --alphas="$alphas" --closure-steps "$closure_steps" \
    --rank "$rank" --world-size "$world_size" \
    > "$output/rank_$(printf '%02d' "$rank").log" 2>&1 &
  children+=("$!")
done
for child in "${children[@]}"; do wait "$child"; done
children=()

python "$project/scripts/qh_flow_landscape.py" \
  --data-dir "$data" --checkpoint "$checkpoint" --output-dir "$output" \
  --lib "$lib" --source-ids "$source_ids" --directions "$directions" \
  --alphas="$alphas" --closure-steps "$closure_steps" --analyze-only
