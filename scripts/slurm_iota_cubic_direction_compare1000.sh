#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=iota-dir1000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:RTX5090:2
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# DEPRECATED HISTORICAL 2-DIRECTION PROTOCOL (2026-08-10).
# Preserved only to reconstruct the recorded direction-reuse comparison.
echo "deprecated historical 2-direction launcher; no new runs are permitted" >&2
exit 64

project="${PROJECT:?PROJECT is required}"
variant="${DIRECTION_VARIANT:?DIRECTION_VARIANT is required}"
lib="${SCORE_LIB:-$project/gpu_backend/build_iota_cubic_cuda13/libstellarator_gpu.so}"
expected_lib_sha="${EXPECTED_SCORE_LIB_SHA:?EXPECTED_SCORE_LIB_SHA is required}"
initial_case="$project/reports/assets/qh_score_adam_start_panel_29960/start_10.json"
# The remote Git worktree stores this JSON with LF line endings.
expected_initial_case_sha="33bd263c737d712ccf4d7f6376dc6c8009e3b308bd7d04d0ec30cc3a9c11d7e9"

case "$variant" in
  random)
    reuse_after=0
    ;;
  previous_update_after300)
    reuse_after=300
    ;;
  *)
    echo "unknown DIRECTION_VARIANT: $variant" >&2
    exit 2
    ;;
esac

iterations="${ITERATIONS_OVERRIDE:-1000}"
reuse_after="${REUSE_UPDATE_DIRECTION_AFTER_OVERRIDE:-$reuse_after}"

run_root="${RUN_ROOT:-$HOME/local_surface_evaluator_runs/iota_cubic_dir1000_${variant}_${SLURM_JOB_ID}}"
cd "$project"
test ! -e "$run_root"
test -f "$lib"
test -f "$initial_case"
test "$(sha256sum "$lib" | awk '{print $1}')" = "$expected_lib_sha"
test "$(sha256sum "$initial_case" | awk '{print $1}')" = "$expected_initial_case_sha"

export PROJECT="$project"
export SCORE_LIB="$lib"
export EXPECTED_SCORE_LIB_SHA="$expected_lib_sha"
export RUN_ROOT="$run_root"
export INITIAL_CASE="$initial_case"
export ITERATIONS="$iterations"
export MAX_WALL_S=10000
export NFP=4
export N_BASE_COILS=3
export DIRECTIONS=2
export DIRECTION_BANK_SIZE=2
export REUSE_UPDATE_DIRECTION_AFTER="$reuse_after"
export GRADIENT_ESTIMATOR=central
export FLOW_STEPS=128
export FLOW_PIPELINE=1
export SCORE_GPUS=0:1
export LEARNING_RATE=0.01
export PERTURBATION=0.005
export BETA1=0.7
export BETA2=0.999
export ROBUST_DIRECTION_FILTER=1
export REJECT_INVALID_CENTER=1
export INVALID_CENTER_BACKTRACKING=0.5,0.25,0.125
export DIRECTION_OUTLIER_RATIO=8.0
export DIRECTION_OUTLIER_MAD_FACTOR=8.0
export SEED=20260804
export SCORE_SURFACE_MODE=continuous
export SURFACE_CONFIDENCE_PERIODS=1
export SURFACE_THETA_COUNT=128
export SURFACE_TRACE_STEPS=400
export SURFACE_FLUX_BISECTION_ITERS=6
export IOTA_DEGREE=3
export AXIS_CONTINUATION=1
export AXIS_HINT_VERIFICATION=mixed
export RESUME=0

exec bash "$project/scripts/slurm_flow_prior_standard_adam.sh"
