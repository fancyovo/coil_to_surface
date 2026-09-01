#!/usr/bin/env bash

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_LIB_SHA:?EXPECTED_LIB_SHA is required}"

protocol_id="qh-axis-surface-contour-compact-flexible-score-abi11-v3"
total_count=3600
prior_seed=20260903
shard_count=6

cd "$PROJECT"
test ! -e "$RUN_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(sha256sum "$SCORE_LIB" | awk '{print $1}')" = "$EXPECTED_LIB_SHA"
git diff --quiet
git diff --cached --quiet
export PROJECT RUN_ROOT SCORE_LIB EXPECTED_COMMIT EXPECTED_LIB_SHA
export PROTOCOL_ID="$protocol_id" TOTAL_COUNT="$total_count" PRIOR_SEED="$prior_seed" SHARD_COUNT="$shard_count"

p107=(--account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=4 --mem=24G --array=0-3 --job-name=axis-v3-p107 --export=ALL,SHARD_OFFSET=0)
students=(--account=stu --partition=Students --qos=qos_stu_medium_2gpu --cpus-per-task=12 --mem=48G --array=0-1 --job-name=axis-v3-stu --export=ALL,SHARD_OFFSET=4)

sbatch --test-only "${p107[@]}" scripts/slurm_axis_surface_prior_v3_worker.sh
sbatch --test-only "${students[@]}" scripts/slurm_axis_surface_prior_v3_worker.sh
sbatch --test-only --export=ALL scripts/slurm_analyze_axis_surface_prior.sh

mkdir -p "$RUN_ROOT" logs
cp evaluation/axis_surface_contour_prior_compact_flexible_abi11_v3.json "$RUN_ROOT/protocol.json"
p107_job=$(sbatch --parsable "${p107[@]}" scripts/slurm_axis_surface_prior_v3_worker.sh)
student_job=$(sbatch --parsable "${students[@]}" scripts/slurm_axis_surface_prior_v3_worker.sh)
analysis_job=$(sbatch --parsable --dependency="afterok:$p107_job:$student_job" --export=ALL scripts/slurm_analyze_axis_surface_prior.sh)
submitted_at=$(date --iso-8601=seconds)
cat > "$RUN_ROOT/runtime_manifest.json" <<EOF
{
  "protocol_id": "$protocol_id",
  "status": "registered-experimental",
  "submitted_at": "$submitted_at",
  "code_commit": "$EXPECTED_COMMIT",
  "tracked_worktree_dirty": false,
  "score_library": "$SCORE_LIB",
  "score_library_abi": 11,
  "score_library_sha256": "$EXPECTED_LIB_SHA",
  "preset": "compact_flexible",
  "target": "QH",
  "total_count": $total_count,
  "seed": $prior_seed,
  "shard_count": $shard_count,
  "per_shard_count": 600,
  "slurm_wall_limit": "01:00:00",
  "optimizer": null,
  "full_physical_evaluation": false,
  "jobs": {
    "p107": "$p107_job",
    "students": "$student_job",
    "analysis": "$analysis_job"
  }
}
EOF

cat <<EOF
p107=$p107_job
students=$student_job
analysis=$analysis_job
run_root=$RUN_ROOT
EOF
