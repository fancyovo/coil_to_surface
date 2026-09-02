#!/usr/bin/env bash

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${SOURCE_RUN_ROOT:?SOURCE_RUN_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_LIB_SHA:?EXPECTED_LIB_SHA is required}"
: "${EXPECTED_CHECKPOINT_SHA:?EXPECTED_CHECKPOINT_SHA is required}"

protocol_id="qh-axis-surface-compact-v3-top2-negative-hand-continue-adam200-64d-abi11-v1"
protocol_file="evaluation/axis_surface_prior_compact_flexible_negative_hand_continuation_abi11_v1.json"

cd "$PROJECT"
test ! -e "$RUN_ROOT"
test -d "$SOURCE_RUN_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(sha256sum "$SCORE_LIB" | awk '{print $1}')" = "$EXPECTED_LIB_SHA"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA"
git diff --quiet
git diff --cached --quiet
mkdir -p logs
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"

python "$PROJECT/scripts/prepare_axis_surface_prior_continuation.py" \
  --source-run-root "$SOURCE_RUN_ROOT" \
  --run-root "$RUN_ROOT" \
  --case-id 1341 \
  --case-id 2832 \
  --protocol-id "$protocol_id" \
  --expected-commit "$EXPECTED_COMMIT" \
  --optimizer-seed-base 20290920 \
  --target-helicity-sign -1
cp "$protocol_file" "$RUN_ROOT/protocol.json"

export PROJECT RUN_ROOT EXPECTED_COMMIT EXPECTED_LIB_SHA EXPECTED_CHECKPOINT_SHA
export WORKER_MAX_WALL_S=13800 MINIMUM_CASE_RESERVE_S=600
smoke=(--account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=4 --mem=32G --time=00:20:00 --job-name=axis-v3-neg-smoke --export=ALL)
workers=(--account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=4 --mem=32G --time=04:00:00 --array=0-1 --job-name=axis-v3-neg-adam --export=ALL,WORKER_OFFSET=0)

sbatch --test-only "${smoke[@]}" scripts/slurm_smoke_axis_surface_prior_adam200.sh
sbatch --test-only "${workers[@]}" scripts/slurm_axis_surface_prior_v3_adam200_worker.sh
smoke_job=$(sbatch --parsable "${smoke[@]}" scripts/slurm_smoke_axis_surface_prior_adam200.sh)
worker_job=$(sbatch --parsable --dependency="afterok:$smoke_job" "${workers[@]}" scripts/slurm_axis_surface_prior_v3_adam200_worker.sh)
submitted_at=$(date --iso-8601=seconds)

cat > "$RUN_ROOT/runtime_manifest.json" <<EOF
{
  "protocol_id": "$protocol_id",
  "status": "registered-experimental",
  "submitted_at": "$submitted_at",
  "code_commit": "$EXPECTED_COMMIT",
  "tracked_worktree_dirty": false,
  "source_run_root": "$SOURCE_RUN_ROOT",
  "target_helicity": "(1,-nfp)",
  "score_library": "$SCORE_LIB",
  "score_library_abi": 11,
  "score_library_sha256": "$EXPECTED_LIB_SHA",
  "checkpoint": "$CHECKPOINT",
  "checkpoint_sha256": "$EXPECTED_CHECKPOINT_SHA",
  "parallel_workers": 2,
  "jobs": {"smoke": "$smoke_job", "workers": "$worker_job"}
}
EOF

printf 'smoke=%s\nworkers=%s\nrun_root=%s\n' "$smoke_job" "$worker_job" "$RUN_ROOT"
