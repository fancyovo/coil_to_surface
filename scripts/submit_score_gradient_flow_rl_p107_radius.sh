#!/usr/bin/env bash
set -euo pipefail
: "${SCORE_GRADIENT_REPO:?}"
: "${SCORE_GRADIENT_RUN_ROOT:?}"
: "${SCORE_GRADIENT_Q0_CHECKPOINT:?}"
: "${SCORE_GRADIENT_Q0_SHA:?}"
: "${SCORE_GRADIENT_OPTIMIZER_CHECKPOINT:?}"
: "${SCORE_GRADIENT_OPTIMIZER_SHA:?}"
: "${SCORE_GRADIENT_SCORE_LIB:?}"
: "${SCORE_GRADIENT_SCORE_MANIFEST:?}"
: "${SCORE_GRADIENT_SCORE_SHA:?}"
: "${SCORE_GRADIENT_COMMIT:?}"
: "${SCORE_GRADIENT_PRIOR_RADIUS_M:?}"
: "${SCORE_GRADIENT_PRIOR_RADIUS_RANGE_M:?}"
repo="$SCORE_GRADIENT_REPO"
radius="$SCORE_GRADIENT_PRIOR_RADIUS_M"
case "$radius" in
  0.15) protocol_id=qh-axisflip-r015-score-gradient-replay10-ema10-rl-r04-abi11-v1; format=axisflip_r015_score_gradient_replay10_ema10_rl_r04_v1; manifest="$repo/evaluation/axisflip_r015_score_gradient_replay10_ema10_rl_r04_abi11_v1.json" ;;
  0.20) protocol_id=qh-axisflip-r020-score-gradient-replay10-ema10-rl-r04-abi11-v1; format=axisflip_r020_score_gradient_replay10_ema10_rl_r04_v1; manifest="$repo/evaluation/axisflip_r020_score_gradient_replay10_ema10_rl_r04_abi11_v1.json" ;;
  *) echo "unsupported prior radius: $radius" >&2; exit 2 ;;
esac
test -f "$manifest"
cd "$repo"
test "$(git rev-parse HEAD)" = "$SCORE_GRADIENT_COMMIT"
git diff --quiet; git diff --cached --quiet
test ! -e "$SCORE_GRADIENT_RUN_ROOT"
export SCORE_GRADIENT_PROTOCOL_ID="$protocol_id" SCORE_GRADIENT_FORMAT="$format"
export SCORE_GRADIENT_PROTOCOL_MANIFEST="$manifest"
export SCORE_GRADIENT_FLOW_OPTIMIZER_STEPS_PER_ROUND=10 SCORE_GRADIENT_EMA_LERP=0.1
export SCORE_GRADIENT_REPO SCORE_GRADIENT_RUN_ROOT SCORE_GRADIENT_Q0_CHECKPOINT SCORE_GRADIENT_Q0_SHA
export SCORE_GRADIENT_OPTIMIZER_CHECKPOINT SCORE_GRADIENT_OPTIMIZER_SHA SCORE_GRADIENT_SCORE_LIB SCORE_GRADIENT_SCORE_MANIFEST SCORE_GRADIENT_SCORE_SHA SCORE_GRADIENT_COMMIT
export SCORE_GRADIENT_PRIOR_RADIUS_M SCORE_GRADIENT_PRIOR_RADIUS_RANGE_M
args=(--export=ALL --job-name="score-grad-r${radius/./}-e10")
sbatch --test-only "${args[@]}" scripts/slurm_score_gradient_flow_rl_p107_radius.sh
job_id="$(sbatch --parsable "${args[@]}" scripts/slurm_score_gradient_flow_rl_p107_radius.sh)"
cat > "${SCORE_GRADIENT_RUN_ROOT}.submission.txt" <<EOF
job_id=$job_id
protocol_id=$protocol_id
prior_radius_m=$radius
prior_radius_range_m=$SCORE_GRADIENT_PRIOR_RADIUS_RANGE_M
samples_per_round=64
samples_per_rank=32
flow_optimizer_steps_per_round=10
ema_lerp=0.1
code_commit=$SCORE_GRADIENT_COMMIT
submitted_at=$(date --iso-8601=seconds)
EOF
printf 'job_id=%s\nrun_root=%s\n' "$job_id" "$SCORE_GRADIENT_RUN_ROOT"
