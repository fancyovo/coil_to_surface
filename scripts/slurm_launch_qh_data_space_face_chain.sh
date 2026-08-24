#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --job-name=qh-data-face-launch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${GPU_LIB:?GPU_LIB is required}"

export DATASET_ROOT="$RUN_ROOT"
export EXPERIMENT_ROOT="$RUN_ROOT/face_qs_calibration"
export TRAJECTORY_SUMMARY="$RUN_ROOT/analysis/trajectory_summary.csv"
export TRAJECTORY_COUNT=96
export ITERATIONS=0,10,25,50,75,100,150,200
export INCLUDE_SCORE_BEST=1

cd "$PROJECT"
submission=$(bash scripts/submit_qh_trajectory_face_qs.sh)
printf '%s\n' "$submission" | tee "$RUN_ROOT/face_qs_submitted_jobs.txt"
analysis_job=$(printf '%s\n' "$submission" | awk -F= '$1 == "analysis" {print $2}')
if [[ -z "$analysis_job" ]]; then
  echo "failed to parse face analysis job id" >&2
  exit 4
fi
sbatch --test-only --dependency="afterok:$analysis_job" --export=ALL \
  scripts/slurm_launch_qh_equal_s_after_face.sh
equal_launcher=$(sbatch --parsable --dependency="afterok:$analysis_job" --export=ALL \
  scripts/slurm_launch_qh_equal_s_after_face.sh)
printf 'equal_s_launcher=%s\n' "$equal_launcher" \
  | tee -a "$RUN_ROOT/face_qs_submitted_jobs.txt"
