#!/usr/bin/env bash

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${INPUT_ROOT:?INPUT_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_LIB_SHA:?EXPECTED_LIB_SHA is required}"
: "${EXPECTED_CHECKPOINT_SHA:?EXPECTED_CHECKPOINT_SHA is required}"

protocol_id="qh-axis-surface-contour-compact-flexible-random-ok-adam200-64d-abi11-v1"
source_protocol_id="qh-axis-surface-contour-compact-flexible-score-abi11-v3"
sample_count=120
selection_seed=20260904

cd "$PROJECT"
test ! -e "$RUN_ROOT"
test -d "$INPUT_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(sha256sum "$SCORE_LIB" | awk '{print $1}')" = "$EXPECTED_LIB_SHA"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA"
git diff --quiet
git diff --cached --quiet
mkdir -p logs
export PROJECT INPUT_ROOT RUN_ROOT CHECKPOINT SCORE_LIB EXPECTED_COMMIT EXPECTED_LIB_SHA EXPECTED_CHECKPOINT_SHA
export SAMPLE_COUNT="$sample_count" SELECTION_SEED="$selection_seed"

p107=(--account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=4 --mem=32G --array=0-3 --job-name=axis-v3-adam-p107 --export=ALL,WORKER_OFFSET=0)
students=(--account=stu --partition=Students --qos=qos_stu_medium_2gpu --cpus-per-task=12 --mem=48G --array=0-1 --job-name=axis-v3-adam-stu --export=ALL,WORKER_OFFSET=4)
smoke=(--account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --job-name=axis-v3-adam-smoke --export=ALL)

sbatch --test-only --export=ALL scripts/slurm_prepare_axis_surface_prior_v3_adam200.sh
sbatch --test-only "${smoke[@]}" scripts/slurm_smoke_axis_surface_prior_adam200.sh
sbatch --test-only "${p107[@]}" scripts/slurm_axis_surface_prior_v3_adam200_worker.sh
sbatch --test-only "${students[@]}" scripts/slurm_axis_surface_prior_v3_adam200_worker.sh
sbatch --test-only --export=ALL scripts/slurm_analyze_axis_surface_prior_adam200.sh

mkdir -p "$RUN_ROOT"
cp evaluation/axis_surface_contour_prior_compact_flexible_adam200_abi11_v1.json "$RUN_ROOT/protocol.json"
prepare_job=$(sbatch --parsable --export=ALL scripts/slurm_prepare_axis_surface_prior_v3_adam200.sh)
smoke_job=$(sbatch --parsable --dependency="afterok:$prepare_job" "${smoke[@]}" scripts/slurm_smoke_axis_surface_prior_adam200.sh)
p107_job=$(sbatch --parsable --dependency="afterok:$smoke_job" "${p107[@]}" scripts/slurm_axis_surface_prior_v3_adam200_worker.sh)
student_job=$(sbatch --parsable --dependency="afterok:$smoke_job" "${students[@]}" scripts/slurm_axis_surface_prior_v3_adam200_worker.sh)
analysis_job=$(sbatch --parsable --dependency="afterany:$p107_job:$student_job" --export=ALL scripts/slurm_analyze_axis_surface_prior_adam200.sh)
submitted_at=$(date --iso-8601=seconds)
cat > "$RUN_ROOT/runtime_manifest.json" <<EOF
{
  "protocol_id": "$protocol_id",
  "status": "registered-experimental",
  "submitted_at": "$submitted_at",
  "code_commit": "$EXPECTED_COMMIT",
  "tracked_worktree_dirty": false,
  "input_root": "$INPUT_ROOT",
  "source_protocol_id": "$source_protocol_id",
  "sample_population": "native status == ok",
  "sample_count": $sample_count,
  "selection_seed": $selection_seed,
  "worker_count": 6,
  "samples_per_worker": 20,
  "score_library": "$SCORE_LIB",
  "score_library_abi": 11,
  "score_library_sha256": "$EXPECTED_LIB_SHA",
  "checkpoint": "$CHECKPOINT",
  "checkpoint_sha256": "$EXPECTED_CHECKPOINT_SHA",
  "worker_slurm_wall_limit_s": 25200,
  "worker_internal_wall_limit_s": 24600,
  "minimum_case_start_reserve_s": 4200,
  "optimizer": {
    "parameter_space": "exact-unclipped standardized coil coefficients",
    "iterations": 200,
    "directions": 64,
    "difference": "centered",
    "perturbation": 0.0025,
    "learning_rate": 0.01,
    "beta": [0.7, 0.999]
  },
  "jobs": {
    "prepare": "$prepare_job",
    "smoke": "$smoke_job",
    "p107": "$p107_job",
    "students": "$student_job",
    "analysis": "$analysis_job"
  }
}
EOF

cat <<EOF
prepare=$prepare_job
smoke=$smoke_job
p107=$p107_job
students=$student_job
analysis=$analysis_job
run_root=$RUN_ROOT
EOF
