#!/usr/bin/env bash

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${SOURCE_SCORE_ROOT:?SOURCE_SCORE_ROOT is required}"
: "${SOURCE_ADAM_ROOT:?SOURCE_ADAM_ROOT is required}"
: "${AUDIT_ROOT:?AUDIT_ROOT is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_LIB_SHA:?EXPECTED_LIB_SHA is required}"

cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet
test -d "$SOURCE_SCORE_ROOT"
test -d "$SOURCE_ADAM_ROOT"
test ! -e "$AUDIT_ROOT"
test "$(sha256sum "$SCORE_LIB" | awk '{print $1}')" = "$EXPECTED_LIB_SHA"
mkdir -p "$AUDIT_ROOT" "$PROJECT/logs"

export PROJECT SOURCE_SCORE_ROOT SOURCE_ADAM_ROOT AUDIT_ROOT SCORE_LIB EXPECTED_COMMIT EXPECTED_LIB_SHA
population=(--account=stu --partition=Students --qos=qos_stu_default --cpus-per-task=2 --mem=8G --time=00:30:00 --job-name=axis-hand-pop --export=ALL)
mirror=(--account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=4 --mem=32G --time=01:00:00 --job-name=axis-hand-mirror --export=ALL)
sbatch --test-only "${population[@]}" scripts/slurm_axis_surface_prior_handedness_population.sh
sbatch --test-only "${mirror[@]}" scripts/slurm_axis_surface_prior_signed_mirror_audit.sh
population_job=$(sbatch --parsable "${population[@]}" scripts/slurm_axis_surface_prior_handedness_population.sh)
mirror_job=$(sbatch --parsable "${mirror[@]}" scripts/slurm_axis_surface_prior_signed_mirror_audit.sh)
submitted_at=$(date --iso-8601=seconds)

cat > "$AUDIT_ROOT/runtime_manifest.json" <<EOF
{
  "format": "axis_surface_prior_handedness_audit_runtime_v1",
  "submitted_at": "$submitted_at",
  "code_commit": "$EXPECTED_COMMIT",
  "tracked_worktree_dirty": false,
  "source_score_root": "$SOURCE_SCORE_ROOT",
  "source_adam_root": "$SOURCE_ADAM_ROOT",
  "score_library": "$SCORE_LIB",
  "score_library_sha256": "$EXPECTED_LIB_SHA",
  "jobs": {"population": "$population_job", "signed_mirror": "$mirror_job"}
}
EOF

printf 'population=%s\nsigned_mirror=%s\naudit_root=%s\n' "$population_job" "$mirror_job" "$AUDIT_ROOT"
