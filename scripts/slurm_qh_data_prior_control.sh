#!/usr/bin/env bash
#SBATCH --job-name=qh-data-prior
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=1-04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
project="${PROJECT:?PROJECT is required}"
run_root="${RUN_ROOT:?RUN_ROOT is required}"
source "${VENV:-$HOME/coil/.venv}/bin/activate"
cd "$project"
mkdir -p logs "$run_root"

assigned="${CUDA_VISIBLE_DEVICES%%,*}"
gpu_pre="$run_root/gpu_preflight_${SLURM_ARRAY_TASK_ID}.csv"
gpu_post="$run_root/gpu_postflight_${SLURM_ARRAY_TASK_ID}.csv"
zombie_post="$run_root/zombies_postflight_${SLURM_ARRAY_TASK_ID}.txt"
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
  --format=csv,noheader,nounits > "$gpu_pre"

python scripts/generate_qh_data_prior_control.py \
  --run-root "$run_root" \
  --worker-index "$SLURM_ARRAY_TASK_ID" \
  --max-wall-s "${MAX_WALL_S:-97200}" \
  --max-new-cases "${MAX_NEW_CASES:-0}" \
  --device 0
