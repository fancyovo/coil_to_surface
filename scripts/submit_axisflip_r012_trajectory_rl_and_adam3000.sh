#!/usr/bin/env bash

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${RL_RUN_ROOT:?RL_RUN_ROOT is required}"
: "${TEACHER_DATASET:?TEACHER_DATASET is required}"
: "${LONG_RUN_ROOT:?LONG_RUN_ROOT is required}"
: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_CHECKPOINT_SHA:?EXPECTED_CHECKPOINT_SHA is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"
: "${SCORE_LIB_MANIFEST:?SCORE_LIB_MANIFEST is required}"
: "${EXPECTED_SCORE_LIB_SHA:?EXPECTED_SCORE_LIB_SHA is required}"

cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet
test ! -e "$RL_RUN_ROOT"
test ! -e "$TEACHER_DATASET"
test ! -e "$LONG_RUN_ROOT"
test -d "$SOURCE_ROOT"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA"
test "$(sha256sum "$SCORE_LIB" | awk '{print $1}')" = "$EXPECTED_SCORE_LIB_SHA"
test -f "$SCORE_LIB_MANIFEST"
for case_id in 36 4; do
  case_padded="$(printf '%07d' "$case_id")"
  test -f "$SOURCE_ROOT/trajectories/axisflip_r012_case_${case_padded}/optimization/best.json"
  test -f "$SOURCE_ROOT/starts/case_${case_padded}.json"
done
mkdir -p "$PROJECT/logs" "$RL_RUN_ROOT" "$LONG_RUN_ROOT"
cp evaluation/axisflip_r012_trajectory_online_rl_r04_abi11_v1.json \
  "$RL_RUN_ROOT/protocol.json"
cp evaluation/axisflip_r012_top2_adam3000_r04_abi11_v1.json \
  "$LONG_RUN_ROOT/protocol.json"

teacher_exports="ALL,AXIS_RL_REPO=$PROJECT,AXIS_RL_DATASET=$TEACHER_DATASET,AXIS_RL_COMMIT=$EXPECTED_COMMIT"
rl_exports="ALL,AXIS_RL_REPO=$PROJECT,AXIS_RL_DATASET=$TEACHER_DATASET,AXIS_RL_RUN_ROOT=$RL_RUN_ROOT,AXIS_RL_COMMIT=$EXPECTED_COMMIT,AXIS_RL_SCORE_LIB=$SCORE_LIB,AXIS_RL_SCORE_LIB_MANIFEST=$SCORE_LIB_MANIFEST,AXIS_RL_SCORE_LIB_SHA=$EXPECTED_SCORE_LIB_SHA,AXIS_RL_OPTIMIZER_CHECKPOINT=$CHECKPOINT,AXIS_RL_OPTIMIZER_CHECKPOINT_SHA=$EXPECTED_CHECKPOINT_SHA"

sbatch --test-only --export="$teacher_exports" scripts/slurm_generate_axisflip_teacher_p107.sh
sbatch --test-only --export="$teacher_exports" scripts/slurm_generate_axisflip_teacher_students.sh
sbatch --test-only --export="$teacher_exports" scripts/slurm_finalize_axisflip_teacher.sh
sbatch --test-only --export="$rl_exports" scripts/slurm_axisflip_prior_online_rl_p107.sh

for case_id in 36 4; do
  case_run="$LONG_RUN_ROOT/case_$(printf '%07d' "$case_id")"
  long_exports="ALL,AXIS_R012_LONG_REPO=$PROJECT,AXIS_R012_LONG_SOURCE_ROOT=$SOURCE_ROOT,AXIS_R012_LONG_RUN_ROOT=$case_run,AXIS_R012_LONG_COMMIT=$EXPECTED_COMMIT,AXIS_R012_LONG_CASE_ID=$case_id,AXIS_R012_LONG_SEED=$((2046090400 + case_id)),AXIS_R012_LONG_CHECKPOINT=$CHECKPOINT,AXIS_R012_LONG_CHECKPOINT_SHA=$EXPECTED_CHECKPOINT_SHA,AXIS_R012_LONG_SCORE_LIB=$SCORE_LIB,AXIS_R012_LONG_SCORE_LIB_SHA=$EXPECTED_SCORE_LIB_SHA"
  sbatch --test-only --export="$long_exports" scripts/slurm_axisflip_r012_long_adam3000_students.sh
done

teacher_p107=$(sbatch --parsable --export="$teacher_exports" \
  scripts/slurm_generate_axisflip_teacher_p107.sh)
teacher_students=$(sbatch --parsable --export="$teacher_exports" \
  scripts/slurm_generate_axisflip_teacher_students.sh)
finalize=$(sbatch --parsable --dependency="afterok:$teacher_p107:$teacher_students" \
  --export="$teacher_exports" scripts/slurm_finalize_axisflip_teacher.sh)
rl=$(sbatch --parsable --dependency="afterok:$finalize" --export="$rl_exports" \
  scripts/slurm_axisflip_prior_online_rl_p107.sh)

long_jobs=()
for case_id in 36 4; do
  case_run="$LONG_RUN_ROOT/case_$(printf '%07d' "$case_id")"
  long_exports="ALL,AXIS_R012_LONG_REPO=$PROJECT,AXIS_R012_LONG_SOURCE_ROOT=$SOURCE_ROOT,AXIS_R012_LONG_RUN_ROOT=$case_run,AXIS_R012_LONG_COMMIT=$EXPECTED_COMMIT,AXIS_R012_LONG_CASE_ID=$case_id,AXIS_R012_LONG_SEED=$((2046090400 + case_id)),AXIS_R012_LONG_CHECKPOINT=$CHECKPOINT,AXIS_R012_LONG_CHECKPOINT_SHA=$EXPECTED_CHECKPOINT_SHA,AXIS_R012_LONG_SCORE_LIB=$SCORE_LIB,AXIS_R012_LONG_SCORE_LIB_SHA=$EXPECTED_SCORE_LIB_SHA"
  long_jobs+=("$(sbatch --parsable --export="$long_exports" \
    scripts/slurm_axisflip_r012_long_adam3000_students.sh)")
done

submitted_at="$(date --iso-8601=seconds)"
python - "$RL_RUN_ROOT/submission_manifest.json" "$submitted_at" "$EXPECTED_COMMIT" \
  "$teacher_p107" "$teacher_students" "$finalize" "$rl" \
  "${long_jobs[0]}" "${long_jobs[1]}" "$TEACHER_DATASET" "$LONG_RUN_ROOT" <<'PY'
import json
from pathlib import Path
import sys

(
    path, submitted_at, commit, teacher_p107, teacher_students, finalize, rl,
    long_36, long_4, dataset, long_root,
) = sys.argv[1:]
Path(path).write_text(json.dumps({
    "format": "axisflip_r012_trajectory_rl_submission_v1",
    "submitted_at": submitted_at,
    "code_commit": commit,
    "teacher_dataset": dataset,
    "long_run_root": long_root,
    "jobs": {
        "teacher_p107": teacher_p107,
        "teacher_students": teacher_students,
        "teacher_finalize": finalize,
        "rl_p107": rl,
        "adam3000_case_36": long_36,
        "adam3000_case_4": long_4,
    },
    "rl_limits": {"wall": "4-00:00:00", "round_limit": None},
}, indent=2) + "\n", encoding="utf-8")
PY

printf 'teacher_p107=%s\nteacher_students=%s\nfinalize=%s\nrl=%s\nadam3000_case_36=%s\nadam3000_case_4=%s\n' \
  "$teacher_p107" "$teacher_students" "$finalize" "$rl" \
  "${long_jobs[0]}" "${long_jobs[1]}"
