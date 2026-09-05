#!/usr/bin/env bash

set -euo pipefail

: "${SCORE_GRADIENT_REPO:?set SCORE_GRADIENT_REPO}"
: "${SCORE_GRADIENT_RUN_ROOT:?set SCORE_GRADIENT_RUN_ROOT}"
: "${SCORE_GRADIENT_Q0_CHECKPOINT:?set SCORE_GRADIENT_Q0_CHECKPOINT}"
: "${SCORE_GRADIENT_Q0_SHA:?set SCORE_GRADIENT_Q0_SHA}"
: "${SCORE_GRADIENT_OPTIMIZER_CHECKPOINT:?set SCORE_GRADIENT_OPTIMIZER_CHECKPOINT}"
: "${SCORE_GRADIENT_OPTIMIZER_SHA:?set SCORE_GRADIENT_OPTIMIZER_SHA}"
: "${SCORE_GRADIENT_SCORE_LIB:?set SCORE_GRADIENT_SCORE_LIB}"
: "${SCORE_GRADIENT_SCORE_MANIFEST:?set SCORE_GRADIENT_SCORE_MANIFEST}"
: "${SCORE_GRADIENT_SCORE_SHA:?set SCORE_GRADIENT_SCORE_SHA}"
: "${SCORE_GRADIENT_COMMIT:?set SCORE_GRADIENT_COMMIT}"

repo="$SCORE_GRADIENT_REPO"
run_root="$SCORE_GRADIENT_RUN_ROOT"
cd "$repo"
test ! -e "$run_root"
test "$(git rev-parse HEAD)" = "$SCORE_GRADIENT_COMMIT"
git diff --quiet
git diff --cached --quiet
test -f "$SCORE_GRADIENT_Q0_CHECKPOINT"
test "$(sha256sum "$SCORE_GRADIENT_Q0_CHECKPOINT" | awk '{print $1}')" = "$SCORE_GRADIENT_Q0_SHA"
test -f "$SCORE_GRADIENT_OPTIMIZER_CHECKPOINT"
test "$(sha256sum "$SCORE_GRADIENT_OPTIMIZER_CHECKPOINT" | awk '{print $1}')" = "$SCORE_GRADIENT_OPTIMIZER_SHA"
test -f "$SCORE_GRADIENT_SCORE_LIB"
test "$(sha256sum "$SCORE_GRADIENT_SCORE_LIB" | awk '{print $1}')" = "$SCORE_GRADIENT_SCORE_SHA"
test -f "$SCORE_GRADIENT_SCORE_MANIFEST"
mkdir -p "$(dirname "$run_root")"

export SCORE_GRADIENT_REPO SCORE_GRADIENT_RUN_ROOT SCORE_GRADIENT_Q0_CHECKPOINT SCORE_GRADIENT_Q0_SHA
export SCORE_GRADIENT_OPTIMIZER_CHECKPOINT SCORE_GRADIENT_OPTIMIZER_SHA
export SCORE_GRADIENT_SCORE_LIB SCORE_GRADIENT_SCORE_MANIFEST SCORE_GRADIENT_SCORE_SHA SCORE_GRADIENT_COMMIT
export SCORE_GRADIENT_PROTOCOL_ID="qh-axisflip-r012-score-gradient-replay10-ema10-rl-r04-abi11-v1"
export SCORE_GRADIENT_FORMAT="axisflip_r012_score_gradient_replay10_ema10_rl_r04_v1"
export SCORE_GRADIENT_FLOW_OPTIMIZER_STEPS_PER_ROUND=10
export SCORE_GRADIENT_EMA_LERP=0.1
export SCORE_GRADIENT_PROTOCOL_MANIFEST="$repo/evaluation/axisflip_r012_score_gradient_replay10_ema10_rl_r04_abi11_v1.json"

sbatch_args=(
  --export=ALL
  --job-name=score-grad-r10-e10
)
sbatch --test-only "${sbatch_args[@]}" scripts/slurm_score_gradient_flow_rl_students_replay10_ema10.sh
job_id="$(sbatch --parsable "${sbatch_args[@]}" scripts/slurm_score_gradient_flow_rl_students_replay10_ema10.sh)"
cat > "${run_root}.submission.txt" <<EOF
job_id=$job_id
protocol_id=qh-axisflip-r012-score-gradient-replay10-ema10-rl-r04-abi11-v1
samples_per_round=64
samples_per_rank=32
flow_optimizer_steps_per_round=10
ema_lerp=0.1
code_commit=$SCORE_GRADIENT_COMMIT
submitted_at=$(date --iso-8601=seconds)
EOF
printf 'job_id=%s\nrun_root=%s\n' "$job_id" "$run_root"
