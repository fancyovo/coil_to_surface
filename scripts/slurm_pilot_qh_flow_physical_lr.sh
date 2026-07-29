#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-flow-lr-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo="${QH_FLOW_REPO:-$HOME/local_surface_evaluator}"
data="${QH_FLOW_DATA:-$HOME/local_surface_evaluator_data/quasr_qh_flow_v1}"
root="${QH_FLOW_PILOT_ROOT:-$repo/runs/qh_flow_physical_lr_${SLURM_JOB_ID}}"
steps="${QH_FLOW_PILOT_STEPS:-3000}"
batch_per_gpu="${QH_FLOW_BATCH_PER_GPU:-8192}"
learning_rates="${QH_FLOW_PILOT_LRS:-4e-4,8e-4,1.2e-3}"
children=()

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if (( ${#children[@]} )); then
    kill "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
  fi
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader > "$root/gpu_postflight.csv" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

cd "$repo"
test -f "$data/manifest.json"
mkdir -p "$root"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
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
    if (( memory_used > 16 )); then
      idle=0
    fi
  done < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    idle=0
  fi
  if (( idle )); then
    ((idle_streak += 1))
    if (( idle_streak >= 3 )); then
      break
    fi
  else
    idle_streak=0
  fi
  sleep 2
done
if (( idle_streak < 3 )); then
  echo "allocated GPUs retained memory or compute processes during idle probes" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader > "$root/gpu_preflight.csv"

IFS=',' read -r -a rates <<< "$learning_rates"
for learning_rate in "${rates[@]}"; do
  label="${learning_rate//./p}"
  label="${label//-/m}"
  output="$root/lr_$label"
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
    scripts/train_qh_flow.py \
    --data-dir "$data" \
    --output-dir "$output" \
    --steps "$steps" \
    --batch-per-gpu "$batch_per_gpu" \
    --learning-rate "$learning_rate" \
    --warmup-steps "${QH_FLOW_WARMUP_STEPS:-500}" \
    --lr-schedule constant \
    --geometry-relative-weight "${QH_FLOW_GEOMETRY_RELATIVE_WEIGHT:-0.05}" \
    --current-feature-weight "${QH_FLOW_CURRENT_FEATURE_WEIGHT:-1.0}" \
    --log-interval 20 \
    --validation-interval 200 \
    --sample-interval 500 \
    --sample-count 256 \
    --sample-steps 16 \
    --checkpoint-interval "$steps" &
  children=("$!")
  wait "${children[0]}"
  children=()
done
