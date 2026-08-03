#!/usr/bin/env bash

set -euo pipefail

: "${CASE_FILE:?CASE_FILE is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${A_VALUES:?A_VALUES is required, for example 0.04,0.05,0.06,0.08}"

project=${PROJECT:-$HOME/local_surface_evaluator}
gpu_lib=${GPU_LIB:-$project/gpu_backend/build_mixed/libstellarator_gpu.so}
eval_env=${EVAL_ENV:-$HOME/local_surface_evaluator/.venv-desc016-py312}
candidate_root=$OUTPUT_ROOT/source_psi_candidates

for path in "$project" "$gpu_lib" "$eval_env" "$CASE_FILE" "$OUTPUT_ROOT"; do
  resolved=$(realpath -m "$path")
  [[ $resolved == "$HOME"/* ]] || {
    printf 'path must stay under HOME: %s\n' "$resolved" >&2
    exit 2
  }
done
test -f "$gpu_lib"
test -f "$CASE_FILE"
test -d "$eval_env"
mkdir -p "$candidate_root" "$project/logs"
python3 "$project/evaluation/full_physical/preflight.py"

manifest=$OUTPUT_ROOT/source_psi_jobs.tsv
printf 'a\tjob_id\toutput_dir\n' > "$manifest"
IFS=',' read -r -a values <<< "$A_VALUES"
for value in "${values[@]}"; do
  value=${value//[[:space:]]/}
  [[ $value =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    printf 'invalid A_VALUE: %s\n' "$value" >&2
    exit 2
  }
  slug=${value//./p}
  output_dir=$candidate_root/a_$slug
  test ! -e "$output_dir"
  exports="ALL,PROJECT=$project,GPU_LIB=$gpu_lib,EVAL_ENV=$eval_env,CASE_FILE=$CASE_FILE,A_VALUE=$value,OUTPUT_DIR=$output_dir"
  (cd "$project" && sbatch --test-only --export="$exports" scripts/slurm_fit_source_psi.sh) >/dev/null
  job_id=$(cd "$project" && sbatch --parsable --export="$exports" scripts/slurm_fit_source_psi.sh)
  job_id=${job_id%%;*}
  printf '%s\t%s\t%s\n' "$value" "$job_id" "$output_dir" | tee -a "$manifest"
done

job_ids=$(tail -n +2 "$manifest" | cut -f2 | paste -sd, -)
printf 'monitor: squeue -j %s -o %s\n' "$job_ids" "'%.18i %.12T %.10M %.30j %R'"
