#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh_traj_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=22:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator_worktrees/trajectory-dataset-pilot}"
checkpoint="${CHECKPOINT:-$HOME/local_surface_evaluator/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
score_lib="${SCORE_LIB:-$HOME/local_surface_evaluator/runtimes/local_score_d1b140f/gpu_backend/build_query_batch/libstellarator_gpu.so}"
data_dir="${DATA_DIR:-$HOME/local_surface_evaluator_data/quasr_qh_flow_v1}"
dataset_root="${DATASET_ROOT:-$HOME/local_surface_evaluator_data/qh_screen32_adam200_v1_pilot_20260813}"
stream_offset="${STREAM_OFFSET:-0}"
stream_index=$((stream_offset + SLURM_ARRAY_TASK_ID))
stream_name="${STREAM_PREFIX:-p107}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
expected_commit="${EXPECTED_COMMIT:-}"
expected_lib_sha="${EXPECTED_LIB_SHA:-7834a88d5437ba9910c78bb0eb5483efc71134579624ee2cc74100297a5799a3}"

cd "$project"
mkdir -p logs "$dataset_root"
if [[ -n "$expected_commit" && "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected project commit" >&2
  exit 2
fi
if [[ "$(sha256sum "$score_lib" | awk '{print $1}')" != "$expected_lib_sha" ]]; then
  echo "unexpected score-library hash" >&2
  exit 3
fi
source "$HOME/coil/.venv/bin/activate"

echo "timestamp,gpu_index,name,memory_used_mib,utilization_gpu_percent" > "$dataset_root/${stream_name}_gpu_preflight.csv"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$dataset_root/${stream_name}_gpu_preflight.csv"

child=""
cleanup() {
  status=$?
  if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
    kill -TERM -- "-$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  echo "timestamp,gpu_index,name,memory_used_mib,utilization_gpu_percent" > "$dataset_root/${stream_name}_gpu_postflight.csv"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$dataset_root/${stream_name}_gpu_postflight.csv" || true
  exit "$status"
}
trap cleanup EXIT INT TERM

setsid python "$project/scripts/generate_qh_adam_trajectory_dataset.py" \
  --checkpoint "$checkpoint" \
  --lib "$score_lib" \
  --data-dir "$data_dir" \
  --dataset-root "$dataset_root" \
  --stream-index "$stream_index" \
  --stream-name "$stream_name" \
  --seed-base "${SEED_BASE:-20260813}" \
  --max-wall-s "${MAX_WALL_S:-75600}" \
  --max-trajectories "${MAX_TRAJECTORIES:-0}" \
  --candidate-count "${CANDIDATE_COUNT:-32}" \
  --iterations "${ITERATIONS:-200}" \
  --device 0 &
child=$!
wait "$child"
child=""
