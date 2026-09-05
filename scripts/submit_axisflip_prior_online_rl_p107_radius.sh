#!/usr/bin/env bash

set -euo pipefail

: "${AXIS_RL_REPO:?set AXIS_RL_REPO}"
: "${AXIS_RL_DATASET:?set AXIS_RL_DATASET}"
: "${AXIS_RL_RUN_ROOT:?set AXIS_RL_RUN_ROOT}"
: "${AXIS_RL_COMMIT:?set AXIS_RL_COMMIT}"
: "${AXIS_RL_SCORE_LIB:?set AXIS_RL_SCORE_LIB}"
: "${AXIS_RL_SCORE_LIB_MANIFEST:?set AXIS_RL_SCORE_LIB_MANIFEST}"
: "${AXIS_RL_SCORE_LIB_SHA:?set AXIS_RL_SCORE_LIB_SHA}"
: "${AXIS_RL_OPTIMIZER_CHECKPOINT:?set AXIS_RL_OPTIMIZER_CHECKPOINT}"
: "${AXIS_RL_OPTIMIZER_CHECKPOINT_SHA:?set AXIS_RL_OPTIMIZER_CHECKPOINT_SHA}"
: "${AXIS_RL_MINOR_RADIUS_M:?set AXIS_RL_MINOR_RADIUS_M}"
: "${AXIS_RL_SAMPLE_SEED:?set AXIS_RL_SAMPLE_SEED}"

case "$AXIS_RL_MINOR_RADIUS_M" in
  0.15)
    protocol_id="qh-axisflip-r015-distilled-online-adam20-trajectory-rwcfm-r04-abi11-v1"
    protocol_file="evaluation/axisflip_r015_trajectory_online_rl_r04_abi11_v1.json"
    ;;
  0.20)
    protocol_id="qh-axisflip-r020-distilled-online-adam20-trajectory-rwcfm-r04-abi11-v1"
    protocol_file="evaluation/axisflip_r020_trajectory_online_rl_r04_abi11_v1.json"
    ;;
  *) echo "unsupported radius: $AXIS_RL_MINOR_RADIUS_M" >&2; exit 2 ;;
esac
protocol_manifest="$AXIS_RL_REPO/$protocol_file"
test -f "$protocol_manifest"
cd "$AXIS_RL_REPO"
test "$(git rev-parse HEAD)" = "$AXIS_RL_COMMIT"
git diff --quiet
git diff --cached --quiet
test ! -e "$AXIS_RL_RUN_ROOT"
mkdir -p "$(dirname "$AXIS_RL_DATASET")" "$(dirname "$AXIS_RL_RUN_ROOT")"

export AXIS_RL_PROTOCOL_MANIFEST="$protocol_manifest"
export AXIS_RL_PROTOCOL_ID="$protocol_id"
sbatch_args=(--export=ALL --job-name="axisrl-r${AXIS_RL_MINOR_RADIUS_M#0.}.2g")
sbatch --test-only "${sbatch_args[@]}" scripts/slurm_axisflip_prior_online_rl_p107_2gpu.sh
job_id="$(sbatch --parsable "${sbatch_args[@]}" scripts/slurm_axisflip_prior_online_rl_p107_2gpu.sh)"
cat > "${AXIS_RL_RUN_ROOT}.submission.txt" <<EOF
job_id=$job_id
protocol_id=$protocol_id
minor_radius_m=$AXIS_RL_MINOR_RADIUS_M
workers=2
samples_per_worker=32
samples_per_round=64
gpus=2
cpus_per_task=8
code_commit=$AXIS_RL_COMMIT
submitted_at=$(date --iso-8601=seconds)
EOF
printf 'job_id=%s\nrun_root=%s\n' "$job_id" "$AXIS_RL_RUN_ROOT"
