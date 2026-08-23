#!/usr/bin/env bash
#SBATCH --job-name=summary1-modes
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/summary1-modes-%A_%a.out
#SBATCH --error=logs/summary1-modes-%A_%a.err
#SBATCH --array=0-3

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
trajectory_root="${TRAJECTORY_ROOT:?TRAJECTORY_ROOT is required}"
native_lib="${NATIVE_LIB:?NATIVE_LIB is required}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR is required}"
source "${VENV:-$HOME/coil/.venv}/bin/activate"
cd "$project"
mkdir -p logs "$output_dir"

gpu_record="$output_dir/gpu_preflight_${SLURM_ARRAY_TASK_ID}.csv"
gpu_post="$output_dir/gpu_postflight_${SLURM_ARRAY_TASK_ID}.csv"
zombie_post="$output_dir/zombies_postflight_${SLURM_ARRAY_TASK_ID}.txt"
assigned="${CUDA_VISIBLE_DEVICES%%,*}"
cleanup() {
  status=$?
  nvidia-smi -i "$assigned" --query-gpu=index,uuid,utilization.gpu,memory.used \
    --format=csv,noheader,nounits > "$gpu_post" 2>/dev/null || true
  ps -u "$USER" -o pid=,ppid=,stat=,comm= | awk '$3 ~ /^Z/ {print}' \
    > "$zombie_post" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

if nvidia-smi -i "$assigned" --query-compute-apps=pid \
    --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "allocated GPU is not idle before timing" >&2
  exit 42
fi
nvidia-smi -i "$assigned" --query-gpu=index,uuid,utilization.gpu,memory.used \
  --format=csv,noheader,nounits > "$gpu_record"
python scripts/benchmark_summary1_evaluator_modes.py \
  --trajectory-root "$trajectory_root" \
  --lib "$native_lib" \
  --output-dir "$output_dir" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-count "$SLURM_ARRAY_TASK_COUNT"
