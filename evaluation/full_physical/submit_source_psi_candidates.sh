#!/usr/bin/env bash

set -euo pipefail

: "${CASE_FILE:?CASE_FILE is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${A_VALUES:?A_VALUES is required, for example 0.04,0.05,0.06,0.08}"

project=${PROJECT:-$HOME/local_surface_evaluator}
gpu_lib=${GPU_LIB:-$project/gpu_backend/build_mixed/libstellarator_gpu.so}
eval_env=${EVAL_ENV:-$HOME/local_surface_evaluator/.venv-desc016-py312}
serial_candidates=${SERIAL_CANDIDATES:-0}
serial_reason=${SERIAL_REASON:-}
candidate_pools_csv=${CANDIDATE_POOLS:-}
candidate_root=$OUTPUT_ROOT/source_psi_candidates

[[ $serial_candidates == 0 || $serial_candidates == 1 ]] || {
  printf 'SERIAL_CANDIDATES must be 0 or 1\n' >&2
  exit 2
}
if [[ $serial_candidates == 1 && -z ${serial_reason//[[:space:]]/} ]]; then
  printf 'SERIAL_REASON is required when SERIAL_CANDIDATES=1\n' >&2
  exit 2
fi
if [[ $serial_candidates == 0 && -n ${serial_reason//[[:space:]]/} ]]; then
  printf 'SERIAL_REASON must be empty when candidates are parallel\n' >&2
  exit 2
fi

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
PYTHONPATH="$project" python3 - "$gpu_lib" <<'PY'
import sys
from pathlib import Path

from scripts.run_qh_face_qs_gpu_prepare import validate_gpu_library

validate_gpu_library(Path(sys.argv[1]))
print(f"PASS: GPU library exports the required full-evaluation ABI symbols: {sys.argv[1]}")
PY

IFS=',' read -r -a values <<< "$A_VALUES"
for index in "${!values[@]}"; do
  value=${values[$index]//[[:space:]]/}
  [[ $value =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    printf 'invalid A_VALUE: %s\n' "$value" >&2
    exit 2
  }
  values[$index]=$value
done

candidate_pools=()
if [[ -n $candidate_pools_csv ]]; then
  IFS=',' read -r -a candidate_pools <<< "$candidate_pools_csv"
else
  for _ in "${values[@]}"; do candidate_pools+=(p107); done
fi
[[ ${#candidate_pools[@]} -eq ${#values[@]} ]] || {
  printf 'CANDIDATE_POOLS must contain one pool per A_VALUE\n' >&2
  exit 2
}
policy_args=()
for index in "${!candidate_pools[@]}"; do
  pool=${candidate_pools[$index]//[[:space:]]/}
  [[ $pool == p107 || $pool == students ]] || {
    printf 'invalid candidate pool: %s\n' "$pool" >&2
    exit 2
  }
  candidate_pools[$index]=$pool
  policy_args+=(--candidate-pool "$pool")
done
python3 "$project/evaluation/full_physical/write_submission_policy.py" \
  --output "$OUTPUT_ROOT/source_psi_submission_policy.json" \
  --stage source_psi_candidates \
  --serial "$serial_candidates" \
  --serial-reason "$serial_reason" \
  --candidate-count "${#values[@]}" \
  "${policy_args[@]}"

manifest=$OUTPUT_ROOT/source_psi_jobs.tsv
printf 'a\tpool\tjob_id\toutput_dir\n' > "$manifest"
previous_job=
for index in "${!values[@]}"; do
  value=${values[$index]}
  pool=${candidate_pools[$index]}
  slug=${value//./p}
  output_dir=$candidate_root/a_$slug
  test ! -e "$output_dir"
  exports="ALL,PROJECT=$project,GPU_LIB=$gpu_lib,EVAL_ENV=$eval_env,CASE_FILE=$CASE_FILE,A_VALUE=$value,OUTPUT_DIR=$output_dir"
  pool_args=()
  if [[ $pool == students ]]; then
    pool_args+=(--account=stu --partition=Students --qos=qos_stu_medium_2gpu)
  fi
  submit_args=(--parsable "${pool_args[@]}" --export="$exports")
  if [[ $serial_candidates == 1 && -n $previous_job ]]; then
    submit_args+=(--dependency="afterany:$previous_job")
  fi
  (cd "$project" && sbatch --test-only "${submit_args[@]:1}" scripts/slurm_fit_source_psi.sh) >/dev/null
  job_id=$(cd "$project" && sbatch "${submit_args[@]}" scripts/slurm_fit_source_psi.sh)
  job_id=${job_id%%;*}
  printf '%s\t%s\t%s\t%s\n' "$value" "$pool" "$job_id" "$output_dir" | tee -a "$manifest"
  previous_job=$job_id
done

job_ids=$(tail -n +2 "$manifest" | cut -f3 | paste -sd, -)
printf 'monitor: squeue -j %s -o %s\n' "$job_ids" "'%.18i %.12T %.10M %.30j %R'"
