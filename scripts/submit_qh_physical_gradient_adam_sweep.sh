#!/usr/bin/env bash

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator_worktrees/qh-blackbox-gradient}"
run_root="${RUN_ROOT:-$project/runs/qh_physical_gradient_adam_start10_sweep_20260804}"
manifest="$run_root/submitted_jobs.tsv"
mkdir -p "$project/logs" "$run_root"
cd "$project"
if [[ -e "$manifest" ]]; then
  echo "submission manifest already exists: $manifest" >&2
  exit 2
fi
printf 'job_id\tpartition\trk4_steps\tlearning_rate\toutput_dir\n' > "$manifest"

submit_one() {
  local partition="$1"
  local steps="$2"
  local learning_rate="$3"
  local tag="${learning_rate//./p}"
  local output="$run_root/rk4_$(printf '%03d' "$steps")_lr_${tag}"
  local launcher
  if [[ "$partition" == "Students" ]]; then
    launcher="scripts/slurm_qh_physical_gradient_adam_students.sh"
  else
    launcher="scripts/slurm_qh_physical_gradient_adam_p107.sh"
  fi
  local job_id
  job_id="$(
    sbatch --parsable \
      --job-name="g2-s${steps}-l${tag}" \
      --export="ALL,RK4_STEPS=${steps},LEARNING_RATE=${learning_rate},ITERATIONS=200,OUTPUT_DIR=${output}" \
      "$launcher"
  )"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$job_id" "$partition" "$steps" "$learning_rate" "$output" >> "$manifest"
}

# The Students subset carries 768 RK4-step units; P107 carries 1472. This
# matches the available 2:4 GPU ratio while spreading all learning rates.
submit_one Students 64 0.003
submit_one Students 64 0.1
submit_one Students 128 0.01
submit_one Students 256 0.003
submit_one Students 256 0.05

submit_one P107 64 0.01
submit_one P107 64 0.03
submit_one P107 64 0.05
submit_one P107 128 0.003
submit_one P107 128 0.03
submit_one P107 128 0.05
submit_one P107 128 0.1
submit_one P107 256 0.01
submit_one P107 256 0.03
submit_one P107 256 0.1

cat "$manifest"
