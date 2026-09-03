#!/usr/bin/env bash

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_CHECKPOINT_SHA:?EXPECTED_CHECKPOINT_SHA is required}"

protocol_id="qh-axis-surface-contour-compact-flexible-axisflip-r012-curvature-r04-adam200-64d-abi11-v1"
cd "$PROJECT"
test ! -e "$RUN_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA"
git diff --quiet
git diff --cached --quiet
mkdir -p logs
export PROJECT RUN_ROOT CHECKPOINT EXPECTED_COMMIT EXPECTED_CHECKPOINT_SHA

build=(--account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --job-name=r012-r04-build --export=ALL)
p107=(--account=competition --partition=P107-RTX5090 --qos=qos_p107-rtx5090 --cpus-per-task=4 --mem=24G --array=0-3 --job-name=r012-r04-p107 --export=ALL,WORKER_OFFSET=0)
students=(--account=stu --partition=Students --qos=qos_stu_medium_2gpu --cpus-per-task=8 --mem=24G --array=0-1 --job-name=r012-r04-stu --export=ALL,WORKER_OFFSET=4)
analysis=(--job-name=r012-r04-analysis --export=ALL)

sbatch --test-only "${build[@]}" scripts/slurm_axisflip_r012_curvature_r04_build_smoke.sh
sbatch --test-only "${p107[@]}" scripts/slurm_axisflip_r012_curvature_r04_worker.sh
sbatch --test-only "${students[@]}" scripts/slurm_axisflip_r012_curvature_r04_worker.sh
sbatch --test-only "${analysis[@]}" scripts/slurm_analyze_axis_surface_prior_axisflip_stream.sh

mkdir -p "$RUN_ROOT"
cp evaluation/axis_surface_contour_prior_r012_curvature_r04_axisflip_adam200_abi11_v1.json "$RUN_ROOT/protocol.json"
build_job=$(sbatch --parsable "${build[@]}" scripts/slurm_axisflip_r012_curvature_r04_build_smoke.sh)
p107_job=$(sbatch --parsable --dependency="afterok:$build_job" "${p107[@]}" scripts/slurm_axisflip_r012_curvature_r04_worker.sh)
student_job=$(sbatch --parsable --dependency="afterok:$build_job" "${students[@]}" scripts/slurm_axisflip_r012_curvature_r04_worker.sh)
analysis_job=$(sbatch --parsable --dependency="afterany:$p107_job:$student_job" "${analysis[@]}" scripts/slurm_analyze_axis_surface_prior_axisflip_stream.sh)
submitted_at=$(date --iso-8601=seconds)
cat > "$RUN_ROOT/runtime_manifest.json" <<EOF
{
  "protocol_id": "$protocol_id",
  "status": "registered-experimental",
  "submitted_at": "$submitted_at",
  "code_commit": "$EXPECTED_COMMIT",
  "tracked_worktree_dirty": false,
  "condition": {"nfp": 8, "n_base_coils": 3},
  "generator": {
    "format": "axis_surface_contour_prior_compact_flexible_axis_flip_r012_v1",
    "minor_radius_center_m": 0.12,
    "minor_radius_range_m": [0.108, 0.132],
    "construction_axis_chirality": -1,
    "seed": 20260906
  },
  "score_library": {
    "path": "$RUN_ROOT/build_curvature_r04/libstellarator_gpu.so",
    "sha256_path": "$RUN_ROOT/score_library.sha256",
    "abi": 11,
    "coil_curvature_p95_scale_m_inv": 25.0,
    "coil_curvature_p95_radius_m": 0.04,
    "coil_curvature_max_scale_m_inv": 35.0
  },
  "checkpoint": "$CHECKPOINT",
  "checkpoint_sha256": "$EXPECTED_CHECKPOINT_SHA",
  "legality_audit": {"samples_per_worker": 64, "total_samples": 384},
  "adam200": {
    "completed_per_worker": 2,
    "completed_total": 12,
    "iterations": 200,
    "directions": 64,
    "difference": "centered",
    "perturbation": 0.0025,
    "learning_rate": 0.01,
    "beta": [0.7, 0.999]
  },
  "parallelization": "four independent P107 workers plus two independent Students workers",
  "jobs": {
    "build_and_smoke": "$build_job",
    "p107": "$p107_job",
    "students": "$student_job",
    "analysis": "$analysis_job"
  }
}
EOF

cat <<EOF
build_and_smoke=$build_job
p107=$p107_job
students=$student_job
analysis=$analysis_job
run_root=$RUN_ROOT
EOF
