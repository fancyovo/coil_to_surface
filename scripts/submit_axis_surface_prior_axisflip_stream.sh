#!/usr/bin/env bash

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_LIB_SHA:?EXPECTED_LIB_SHA is required}"
: "${EXPECTED_CHECKPOINT_SHA:?EXPECTED_CHECKPOINT_SHA is required}"

protocol_id="qh-axis-surface-contour-compact-flexible-axisflip-stream-adam200-64d-abi11-v2"
prior_seed=20260905
discovery_wall_s=14400
hard_wall_s=17700

cd "$PROJECT"
test ! -e "$RUN_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(sha256sum "$SCORE_LIB" | awk '{print $1}')" = "$EXPECTED_LIB_SHA"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA"
git diff --quiet
git diff --cached --quiet
mkdir -p logs
export PROJECT RUN_ROOT CHECKPOINT SCORE_LIB EXPECTED_COMMIT EXPECTED_LIB_SHA EXPECTED_CHECKPOINT_SHA
export PRIOR_SEED="$prior_seed" DISCOVERY_WALL_S="$discovery_wall_s" HARD_WALL_S="$hard_wall_s"

p107=(--account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=4 --mem=24G --array=0-3 --job-name=axisflip-adam-p107 --export=ALL,WORKER_OFFSET=0)
students=(--account=stu --partition=Students --qos=qos_stu_medium_2gpu --cpus-per-task=8 --mem=24G --array=0-1 --job-name=axisflip-adam-stu --export=ALL,WORKER_OFFSET=4)
smoke=(--account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --job-name=axisflip-adam-smoke --export=ALL)

sbatch --test-only "${smoke[@]}" scripts/slurm_smoke_axis_surface_prior_axisflip_stream.sh
sbatch --test-only "${p107[@]}" scripts/slurm_axis_surface_prior_axisflip_stream_worker.sh
sbatch --test-only "${students[@]}" scripts/slurm_axis_surface_prior_axisflip_stream_worker.sh
sbatch --test-only --export=ALL scripts/slurm_analyze_axis_surface_prior_axisflip_stream.sh

mkdir -p "$RUN_ROOT"
cp evaluation/axis_surface_contour_prior_compact_flexible_axisflip_stream_adam200_abi11_v2.json "$RUN_ROOT/protocol.json"
smoke_job=$(sbatch --parsable "${smoke[@]}" scripts/slurm_smoke_axis_surface_prior_axisflip_stream.sh)
p107_job=$(sbatch --parsable --dependency="afterok:$smoke_job" "${p107[@]}" scripts/slurm_axis_surface_prior_axisflip_stream_worker.sh)
student_job=$(sbatch --parsable --dependency="afterok:$smoke_job" "${students[@]}" scripts/slurm_axis_surface_prior_axisflip_stream_worker.sh)
analysis_job=$(sbatch --parsable --dependency="afterany:$p107_job:$student_job" --export=ALL scripts/slurm_analyze_axis_surface_prior_axisflip_stream.sh)
submitted_at=$(date --iso-8601=seconds)
cat > "$RUN_ROOT/runtime_manifest.json" <<EOF
{
  "protocol_id": "$protocol_id",
  "status": "registered-experimental",
  "submitted_at": "$submitted_at",
  "code_commit": "$EXPECTED_COMMIT",
  "tracked_worktree_dirty": false,
  "generator": {
    "baseline_format": "axis_surface_contour_prior_compact_flexible_v3",
    "format": "axis_surface_contour_prior_compact_flexible_axis_flip_v4",
    "only_explicit_input_change": "construction reference-axis vertical Fourier coefficients multiplied by -1",
    "construction_axis_chirality": -1,
    "preset": "compact_flexible",
    "seed": $prior_seed,
    "conditions": "all 26 registered (nfp,nc) conditions with nc<=4"
  },
  "screening": {
    "target_helicity": [1, "+nfp"],
    "criterion_for_adam200": "native ABI-11 status == ok",
    "execution": "one candidate at a time per worker; every valid candidate immediately enters Adam200",
    "axis_continuation": "screening axis_R/axis_Z are retained and required as strict optimizer step-0 hints",
    "pre_update_consistency_gate": "abs(strict-hint optimizer step-0 score - screening score) <= 0.1"
  },
  "worker_count": 6,
  "parallelization": "four P107 workers plus two Students workers; no cross-worker dependency",
  "serial_reason_within_worker": "screening determines whether that candidate has a valid Adam start and the same GPU runs its immediate optimizer",
  "discovery_soft_deadline_s": $discovery_wall_s,
  "finish_inflight_after_soft_deadline": true,
  "slurm_wall_limit_s": 18000,
  "worker_internal_hard_wall_s": $hard_wall_s,
  "score_library": "$SCORE_LIB",
  "score_library_abi": 11,
  "score_library_sha256": "$EXPECTED_LIB_SHA",
  "checkpoint": "$CHECKPOINT",
  "checkpoint_sha256": "$EXPECTED_CHECKPOINT_SHA",
  "optimizer": {
    "parameter_space": "exact-unclipped standardized coil coefficients",
    "flow_calls": 0,
    "iterations": 200,
    "directions": 64,
    "direction_policy": "fresh random orthogonal directions per update",
    "difference": "centered",
    "perturbation": 0.0025,
    "learning_rate": 0.01,
    "beta": [0.7, 0.999]
  },
  "jobs": {
    "smoke": "$smoke_job",
    "p107": "$p107_job",
    "students": "$student_job",
    "analysis": "$analysis_job"
  }
}
EOF

cat <<EOF
smoke=$smoke_job
p107=$p107_job
students=$student_job
analysis=$analysis_job
run_root=$RUN_ROOT
EOF
