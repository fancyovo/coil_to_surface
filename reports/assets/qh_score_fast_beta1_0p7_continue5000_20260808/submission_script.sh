#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=sf-b07-5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:RTX5090:2
#SBATCH --exclude=anode01,anode02
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/scc/pb24511935/local_surface_evaluator_worktrees/score-fast-continuation/logs/sf-b07-5000-%j.out
#SBATCH --error=/home/scc/pb24511935/local_surface_evaluator_worktrees/score-fast-continuation/logs/sf-b07-5000-%j.err

set -euo pipefail

project=/home/scc/pb24511935/local_surface_evaluator_worktrees/score-fast-continuation
source_run="$project/runs/score_fast_beta1_0p7_continue2000_20260806"
run_root="$project/runs/score_fast_beta1_0p7_continue5000_20260808"
prepare_root="${run_root}.prepare.${SLURM_JOB_ID}"
source_state_sha=f7d06d55069e8b8b4a264444de6d80455e9ff6e58c8f62a6df7a830fa9916fc4
score_lib="$project/gpu_backend/build_score_fast/libstellarator_gpu.so"
score_lib_sha=387495353bd4c8a3c2984fcfdb6625937da47da0efa2e578610d666c5a8a2f52
checkpoint="$HOME/local_surface_evaluator/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt"

test -d "$project"
test -d "$source_run"
test -f "$source_run/summary.json"
test -f "$source_run/state_latest.npz"
test -f "$score_lib"
test -f "$checkpoint"
test "$(sha256sum "$source_run/state_latest.npz" | awk '{print $1}')" = "$source_state_sha"
test "$(sha256sum "$score_lib" | awk '{print $1}')" = "$score_lib_sha"
if [[ -e "$run_root" || -e "$prepare_root" ]]; then
  echo "refusing to overwrite continuation destination: $run_root" >&2
  exit 2
fi

cp -a "$source_run" "$prepare_root"
mv "$prepare_root/summary.json" "$prepare_root/summary_step2000.json"
cp "$0" "$prepare_root/submission_script.sh"
mv "$prepare_root" "$run_root"

export PROJECT="$project"
export ASSET_ROOT="$HOME/local_surface_evaluator"
export FLOW_CHECKPOINT="$checkpoint"
export SCORE_LIB="$score_lib"
export EXPECTED_SCORE_LIB_SHA="$score_lib_sha"
export RUN_ROOT="$run_root"
export ITERATIONS=5000
export MAX_WALL_S=20400
export LEARNING_RATE=0.01
export PERTURBATION=0.005
export DIRECTIONS=2
export DIRECTION_BANK_SIZE=2
export GRADIENT_ESTIMATOR=central
export FLOW_STEPS=128
export FLOW_PIPELINE=1
export SCORE_GPUS=0:1
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
export AXIS_CONTINUATION=1
export RESUME=1

exec bash "$project/scripts/slurm_flow_prior_standard_adam.sh"
