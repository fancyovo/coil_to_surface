#!/usr/bin/env bash

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${REFERENCE_ROOT:?REFERENCE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_LIB_SHA:?EXPECTED_LIB_SHA is required}"
: "${EXPECTED_CHECKPOINT_SHA:?EXPECTED_CHECKPOINT_SHA is required}"

cd "$PROJECT"
mkdir -p logs
test ! -e "$RUN_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet
export PROJECT REFERENCE_ROOT RUN_ROOT CHECKPOINT SCORE_LIB EXPECTED_COMMIT \
  EXPECTED_LIB_SHA EXPECTED_CHECKPOINT_SHA GPU_LIB="$SCORE_LIB"

sbatch --test-only --export=ALL scripts/slurm_prepare_qh_data_space_rerun.sh
prepare_job=$(sbatch --parsable --export=ALL scripts/slurm_prepare_qh_data_space_rerun.sh)

sbatch --test-only --dependency="afterok:$prepare_job" --account=competition \
  --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=4 \
  --gres=gpu:RTX5090:1 --array=0-3 --job-name=qh-data-rerun-p107 \
  --export=ALL,WORKER_OFFSET=0 scripts/slurm_run_qh_data_space_rerun.sh
p107_job=$(sbatch --parsable --dependency="afterok:$prepare_job" --account=competition \
  --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=4 \
  --gres=gpu:RTX5090:1 --array=0-3 --job-name=qh-data-rerun-p107 \
  --export=ALL,WORKER_OFFSET=0 scripts/slurm_run_qh_data_space_rerun.sh)

sbatch --test-only --dependency="afterok:$prepare_job" --account=stu \
  --partition=Students --qos=qos_stu_medium_2gpu --cpus-per-task=12 \
  --gres=gpu:RTX5090:1 --array=0-1 --job-name=qh-data-rerun-stu \
  --export=ALL,WORKER_OFFSET=4 scripts/slurm_run_qh_data_space_rerun.sh
student_job=$(sbatch --parsable --dependency="afterok:$prepare_job" --account=stu \
  --partition=Students --qos=qos_stu_medium_2gpu --cpus-per-task=12 \
  --gres=gpu:RTX5090:1 --array=0-1 --job-name=qh-data-rerun-stu \
  --export=ALL,WORKER_OFFSET=4 scripts/slurm_run_qh_data_space_rerun.sh)

worker_dependency="afterok:$p107_job:$student_job"
sbatch --test-only --dependency="$worker_dependency" --export=ALL \
  scripts/slurm_summarize_qh_data_space_rerun.sh
summary_job=$(sbatch --parsable --dependency="$worker_dependency" --export=ALL \
  scripts/slurm_summarize_qh_data_space_rerun.sh)

sbatch --test-only --dependency="afterok:$summary_job" --export=ALL \
  scripts/slurm_launch_qh_data_space_face_chain.sh
face_launcher=$(sbatch --parsable --dependency="afterok:$summary_job" --export=ALL \
  scripts/slurm_launch_qh_data_space_face_chain.sh)

cat <<EOF
prepare=$prepare_job
optimizer_p107=$p107_job
optimizer_students=$student_job
summary=$summary_job
face_launcher=$face_launcher
monitor: squeue -u \$USER
EOF
