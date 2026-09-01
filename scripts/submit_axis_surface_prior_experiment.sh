#!/usr/bin/env bash

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_LIB_SHA:?EXPECTED_LIB_SHA is required}"

cd "$PROJECT"
test ! -e "$RUN_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(sha256sum "$SCORE_LIB" | awk '{print $1}')" = "$EXPECTED_LIB_SHA"
git diff --quiet
git diff --cached --quiet
mkdir -p "$RUN_ROOT" logs
cp evaluation/axis_surface_contour_prior_v1.json "$RUN_ROOT/protocol.json"
export PROJECT RUN_ROOT SCORE_LIB EXPECTED_COMMIT EXPECTED_LIB_SHA

p107=(--account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=4 --mem=24G --array=0-3 --job-name=axis-prior-p107 --export=ALL,SHARD_OFFSET=0)
students=(--account=stu --partition=Students --qos=qos_stu_medium_2gpu --cpus-per-task=12 --mem=48G --array=0-1 --job-name=axis-prior-stu --export=ALL,SHARD_OFFSET=4)

sbatch --test-only "${p107[@]}" scripts/slurm_axis_surface_prior_worker.sh
p107_job=$(sbatch --parsable "${p107[@]}" scripts/slurm_axis_surface_prior_worker.sh)
sbatch --test-only "${students[@]}" scripts/slurm_axis_surface_prior_worker.sh
student_job=$(sbatch --parsable "${students[@]}" scripts/slurm_axis_surface_prior_worker.sh)

dependency="afterok:$p107_job:$student_job"
sbatch --test-only --dependency="$dependency" --export=ALL scripts/slurm_analyze_axis_surface_prior.sh
analysis_job=$(sbatch --parsable --dependency="$dependency" --export=ALL scripts/slurm_analyze_axis_surface_prior.sh)

cat <<EOF
p107=$p107_job
students=$student_job
analysis=$analysis_job
run_root=$RUN_ROOT
EOF
