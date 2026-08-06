#!/usr/bin/env bash
set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator_worktrees/score-fast-continuation}"
run_root="${RUN_ROOT:-$project/runs/score_fast_beta1_long_20260806}"
launcher="$project/scripts/slurm_flow_prior_standard_adam.sh"
initial_case="${INITIAL_CASE:-$project/reports/assets/qh_score_adam_start_panel_29960/start_10.json}"
score_lib="${SCORE_LIB:-$project/gpu_backend/build_score_fast/libstellarator_gpu.so}"
expected_lib_sha="${EXPECTED_SCORE_LIB_SHA:-387495353bd4c8a3c2984fcfdb6625937da47da0efa2e578610d666c5a8a2f52}"
submit_mode="${SUBMIT_MODE:-submit}"

test -d "$project"
test -f "$launcher"
test -f "$initial_case"
test -f "$score_lib"
mkdir -p "$run_root" "$project/logs"

configs=(
  "beta1_0p5 0.5 p107"
  "beta1_0p7 0.7 p107"
  "beta1_0p9 0.9 students"
)

submission_log="$run_root/submissions.tsv"
if [[ "$submit_mode" == "submit" && -e "$submission_log" ]]; then
  echo "refusing to overwrite existing submission log: $submission_log" >&2
  exit 2
fi
if [[ "$submit_mode" == "submit" ]]; then
  printf 'name\tjob_id\tpartition\tbeta1\n' > "$submission_log"
fi

for config in "${configs[@]}"; do
  read -r name beta1 channel <<< "$config"
  common=(
    --nodes=1
    --ntasks=1
    --cpus-per-task=8
    --gres=gpu:RTX5090:2
    --mem=48G
    --time=03:00:00
    --job-name="sf-long-${name}"
    --output="$project/logs/sf-long-${name}-%j.out"
    --error="$project/logs/sf-long-${name}-%j.err"
    --export="ALL,PROJECT=$project,RUN_ROOT=$run_root/$name,INITIAL_CASE=$initial_case,SCORE_LIB=$score_lib,EXPECTED_SCORE_LIB_SHA=$expected_lib_sha,ITERATIONS=600,MAX_WALL_S=10200,LEARNING_RATE=0.01,PERTURBATION=0.005,BETA1=$beta1,BETA2=0.999,SEED=20260804,DIRECTIONS=2,DIRECTION_BANK_SIZE=2,GRADIENT_ESTIMATOR=central,FLOW_STEPS=128,FLOW_PIPELINE=1,SCORE_GPUS=0,1,ROBUST_DIRECTION_FILTER=1,REJECT_INVALID_CENTER=1,SCORE_SURFACE_MODE=continuous,SURFACE_CONFIDENCE_PERIODS=1,SURFACE_THETA_COUNT=128,SURFACE_TRACE_STEPS=400,SURFACE_FLUX_BISECTION_ITERS=6,AXIS_CONTINUATION=1"
  )
  if [[ "$channel" == "p107" ]]; then
    resource=(
      --account=competition
      --partition=P107-RTX5090
      --qos=qos_p107-rtx5090
    )
  else
    resource=(
      --account=stu
      --partition=Students
      --qos=qos_stu_medium_2gpu
    )
  fi

  if [[ "$submit_mode" == "test" ]]; then
    sbatch --test-only "${resource[@]}" "${common[@]}" "$launcher"
    continue
  fi
  if [[ "$submit_mode" != "submit" ]]; then
    echo "SUBMIT_MODE must be test or submit" >&2
    exit 2
  fi
  job_id="$(sbatch --parsable "${resource[@]}" "${common[@]}" "$launcher")"
  printf '%s\t%s\t%s\t%s\n' "$name" "$job_id" "$channel" "$beta1" \
    | tee -a "$submission_log"
done
