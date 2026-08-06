#!/usr/bin/env bash
set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator_worktrees/score-fast-continuation}"
run_root="${RUN_ROOT:-$project/runs/score_fast_optimizer_matrix_20260806}"
launcher="$project/scripts/slurm_flow_prior_standard_adam.sh"
initial_case="${INITIAL_CASE:-$project/reports/assets/qh_score_adam_start_panel_29960/start_10.json}"
expected_lib_sha="${EXPECTED_SCORE_LIB_SHA:-387495353bd4c8a3c2984fcfdb6625937da47da0efa2e578610d666c5a8a2f52}"
score_lib="${SCORE_LIB:-$project/gpu_backend/build_score_fast/libstellarator_gpu.so}"
submit_mode="${SUBMIT_MODE:-submit}"

test -d "$project"
test -f "$launcher"
test -f "$initial_case"
test -f "$score_lib"
mkdir -p "$run_root" "$project/logs"

configs=(
  "rk4_256_central4 256 central 4 p107"
  "rk4_128_central4 128 central 4 p107"
  "rk4_064_central4 64 central 4 p107"
  "rk4_256_one_sided4 256 one-sided 4 p107"
  "rk4_064_central2 64 central 2 p107"
  "rk4_128_one_sided4 128 one-sided 4 students"
  "rk4_064_one_sided4 64 one-sided 4 students"
  "rk4_256_central2 256 central 2 students"
  "rk4_128_central2 128 central 2 students"
)

submission_log="$run_root/submissions.tsv"
if [[ "$submit_mode" == "submit" && -e "$submission_log" ]]; then
  echo "refusing to overwrite existing submission log: $submission_log" >&2
  exit 2
fi
if [[ "$submit_mode" == "submit" ]]; then
  printf 'name\tjob_id\tpartition\trk4_steps\testimator\tdirections\n' > "$submission_log"
fi

for config in "${configs[@]}"; do
  read -r name rk4 estimator directions channel <<< "$config"
  common=(
    --nodes=1
    --ntasks=1
    --cpus-per-task=4
    --gres=gpu:RTX5090:1
    --mem=24G
    --time=02:00:00
    --job-name="sf-${name}"
    --output="$project/logs/sf-${name}-%j.out"
    --error="$project/logs/sf-${name}-%j.err"
    --export="ALL,PROJECT=$project,RUN_ROOT=$run_root/$name,INITIAL_CASE=$initial_case,SCORE_LIB=$score_lib,EXPECTED_SCORE_LIB_SHA=$expected_lib_sha,ITERATIONS=200,MAX_WALL_S=6900,LEARNING_RATE=0.01,PERTURBATION=0.005,BETA1=0.5,BETA2=0.999,SEED=20260804,DIRECTIONS=$directions,DIRECTION_BANK_SIZE=4,GRADIENT_ESTIMATOR=$estimator,FLOW_STEPS=$rk4,FLOW_PIPELINE=1,SCORE_GPUS=0,ROBUST_DIRECTION_FILTER=1,REJECT_INVALID_CENTER=1,SCORE_SURFACE_MODE=continuous,SURFACE_CONFIDENCE_PERIODS=1,SURFACE_THETA_COUNT=128,SURFACE_TRACE_STEPS=400,SURFACE_FLUX_BISECTION_ITERS=6,AXIS_CONTINUATION=1"
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$job_id" "$channel" "$rk4" "$estimator" "$directions" \
    | tee -a "$submission_log"
done
