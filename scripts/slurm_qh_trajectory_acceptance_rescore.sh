#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh_traj_accept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
dataset_root="${DATASET_ROOT:?DATASET_ROOT is required}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR is required}"
data_dir="${DATA_DIR:-$HOME/local_surface_evaluator_data/quasr_qh_flow_v1}"
score_lib="${SCORE_LIB:?SCORE_LIB is required}"
expected_lib_sha="${EXPECTED_LIB_SHA:?EXPECTED_LIB_SHA is required}"
shard_offset="${SHARD_OFFSET:-0}"
shard_index=$((shard_offset + SLURM_ARRAY_TASK_ID))

cd "$project"
mkdir -p logs "$output_dir"
if [[ "$(sha256sum "$score_lib" | awk '{print $1}')" != "$expected_lib_sha" ]]; then
  echo "unexpected score-library hash" >&2
  exit 3
fi
source "$HOME/coil/.venv/bin/activate"

gpu_file="$output_dir/gpu_shard_${shard_index}"
idle_streak=0
for _ in {1..60}; do
  utilization="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
  memory_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "$utilization" == "0" && "$memory_used" -le 16 ]] && \
     ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    idle_streak=$((idle_streak + 1))
    if (( idle_streak >= 3 )); then
      break
    fi
  else
    idle_streak=0
  fi
  sleep 2
done
if (( idle_streak < 3 )); then
  echo "allocated GPU did not reach three consecutive idle probes" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader,nounits > "${gpu_file}_preflight.csv"

cleanup() {
  status=$?
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader,nounits > "${gpu_file}_postflight.csv" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

python scripts/score_qh_trajectory_acceptance_shard.py \
  --dataset-root "$dataset_root" \
  --data-dir "$data_dir" \
  --lib "$score_lib" \
  --output-dir "$output_dir" \
  --quasr-count "${QUASR_COUNT:-1024}" \
  --seed "${SEED:-20260813}" \
  --shard-index "$shard_index" \
  --shard-count "${SHARD_COUNT:-6}" \
  --device 0
