#!/usr/bin/env bash

set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${EXPERIMENT_ROOT:?EXPERIMENT_ROOT is required}"
: "${GPU_LIB:?GPU_LIB is required}"

cd "$PROJECT"
mkdir -p logs
test ! -e "$EXPERIMENT_ROOT"
git diff --quiet
git diff --cached --quiet

TRAJECTORY_COUNT=${TRAJECTORY_COUNT:-96}
ITERATIONS=${ITERATIONS:-0,10,25,50,75,100,150,200}
INCLUDE_SCORE_BEST=${INCLUDE_SCORE_BEST:-0}
export PROJECT DATASET_ROOT EXPERIMENT_ROOT GPU_LIB TRAJECTORY_COUNT ITERATIONS INCLUDE_SCORE_BEST

sbatch --test-only --export=ALL scripts/slurm_qh_face_qs_select.sh
select_job=$(sbatch --parsable --export=ALL scripts/slurm_qh_face_qs_select.sh)

sbatch --test-only --dependency="afterok:$select_job" --account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=4 --gres=gpu:RTX5090:1 --array=0-3 --job-name=qh-face-prep-p107 --export=ALL,SHARD_OFFSET=0,SHARD_COUNT=6 scripts/slurm_qh_face_qs_gpu_prepare.sh
prep_p107=$(sbatch --parsable --dependency="afterok:$select_job" --account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=4 --gres=gpu:RTX5090:1 --array=0-3 --job-name=qh-face-prep-p107 --export=ALL,SHARD_OFFSET=0,SHARD_COUNT=6 scripts/slurm_qh_face_qs_gpu_prepare.sh)

sbatch --test-only --dependency="afterok:$select_job" --account=stu --partition=Students --qos=qos_stu_medium_2gpu --cpus-per-task=12 --gres=gpu:RTX5090:1 --array=0-1 --job-name=qh-face-prep-stu --export=ALL,SHARD_OFFSET=4,SHARD_COUNT=6 scripts/slurm_qh_face_qs_gpu_prepare.sh
prep_stu=$(sbatch --parsable --dependency="afterok:$select_job" --account=stu --partition=Students --qos=qos_stu_medium_2gpu --cpus-per-task=12 --gres=gpu:RTX5090:1 --array=0-1 --job-name=qh-face-prep-stu --export=ALL,SHARD_OFFSET=4,SHARD_COUNT=6 scripts/slurm_qh_face_qs_gpu_prepare.sh)

prepare_dependency="afterok:$prep_p107:$prep_stu"
sbatch --test-only --dependency="$prepare_dependency" --account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=16 --job-name=qh-face-cpu-p107 --export=ALL,RESIDUE_START=0,RESIDUE_COUNT=2,POOL_NAME=p107 scripts/slurm_qh_face_qs_cpu_pool.sh
cpu_p107=$(sbatch --parsable --dependency="$prepare_dependency" --account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=16 --job-name=qh-face-cpu-p107 --export=ALL,RESIDUE_START=0,RESIDUE_COUNT=2,POOL_NAME=p107 scripts/slurm_qh_face_qs_cpu_pool.sh)

sbatch --test-only --dependency="$prepare_dependency" --account=stu --partition=Students --qos=qos_stu_medium_2gpu --cpus-per-task=24 --job-name=qh-face-cpu-stu --export=ALL,RESIDUE_START=2,RESIDUE_COUNT=3,POOL_NAME=students scripts/slurm_qh_face_qs_cpu_pool.sh
cpu_stu=$(sbatch --parsable --dependency="$prepare_dependency" --account=stu --partition=Students --qos=qos_stu_medium_2gpu --cpus-per-task=24 --job-name=qh-face-cpu-stu --export=ALL,RESIDUE_START=2,RESIDUE_COUNT=3,POOL_NAME=students scripts/slurm_qh_face_qs_cpu_pool.sh)

cpu_dependency="afterok:$cpu_p107:$cpu_stu"
sbatch --test-only --dependency="$cpu_dependency" --account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=4 --gres=gpu:RTX5090:1 --array=0-3 --job-name=qh-face-val-p107 --export=ALL,SHARD_OFFSET=0,SHARD_COUNT=6 scripts/slurm_qh_face_qs_gpu_validate.sh
validation_p107=$(sbatch --parsable --dependency="$cpu_dependency" --account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=4 --gres=gpu:RTX5090:1 --array=0-3 --job-name=qh-face-val-p107 --export=ALL,SHARD_OFFSET=0,SHARD_COUNT=6 scripts/slurm_qh_face_qs_gpu_validate.sh)
sbatch --test-only --dependency="$cpu_dependency" --account=stu --partition=Students --qos=qos_stu_medium_2gpu --cpus-per-task=12 --gres=gpu:RTX5090:1 --array=0-1 --job-name=qh-face-val-stu --export=ALL,SHARD_OFFSET=4,SHARD_COUNT=6 scripts/slurm_qh_face_qs_gpu_validate.sh
validation_stu=$(sbatch --parsable --dependency="$cpu_dependency" --account=stu --partition=Students --qos=qos_stu_medium_2gpu --cpus-per-task=12 --gres=gpu:RTX5090:1 --array=0-1 --job-name=qh-face-val-stu --export=ALL,SHARD_OFFSET=4,SHARD_COUNT=6 scripts/slurm_qh_face_qs_gpu_validate.sh)

sbatch --test-only --dependency="afterok:$validation_p107:$validation_stu" --export=ALL scripts/slurm_qh_face_qs_analyze.sh
analysis_job=$(sbatch --parsable --dependency="afterok:$validation_p107:$validation_stu" --export=ALL scripts/slurm_qh_face_qs_analyze.sh)

cat <<EOF
select=$select_job
prepare_p107=$prep_p107
prepare_students=$prep_stu
cpu_p107=$cpu_p107
cpu_students=$cpu_stu
validation_p107=$validation_p107
validation_students=$validation_stu
analysis=$analysis_job
monitor: squeue -u \$USER
EOF
