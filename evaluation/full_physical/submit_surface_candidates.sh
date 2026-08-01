#!/usr/bin/env bash

set -euo pipefail

: "${CASE_FILE:?CASE_FILE is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${S_EDGES:?S_EDGES is required, for example 0.12,0.20,0.24}"

project=${PROJECT:-$HOME/local_surface_evaluator}
gpu_lib=${GPU_LIB:-$project/gpu_backend/build_mixed/libstellarator_gpu.so}
eval_env=${EVAL_ENV:-$project/.venv-desc016-py312}
serial_candidates=${SERIAL_CANDIDATES:-0}
candidate_cpus_per_task=${CANDIDATE_CPUS_PER_TASK:-4}
candidate_root=$OUTPUT_ROOT/candidates

[[ $serial_candidates == 0 || $serial_candidates == 1 ]] || {
    printf 'SERIAL_CANDIDATES must be 0 or 1\n' >&2
    exit 2
}
[[ $candidate_cpus_per_task =~ ^[1-9][0-9]*$ ]] || {
    printf 'CANDIDATE_CPUS_PER_TASK must be a positive integer\n' >&2
    exit 2
}

require_home_path() {
    local resolved
    resolved=$(realpath -m "$1")
    if [[ $resolved != "$HOME"/* ]]; then
        printf 'path must stay under HOME: %s\n' "$resolved" >&2
        exit 2
    fi
}

for path in "$project" "$gpu_lib" "$eval_env" "$CASE_FILE" "$RUN_DIR" "$OUTPUT_ROOT"; do
    require_home_path "$path"
done
test -f "$gpu_lib"
test -f "$CASE_FILE"
test -f "$RUN_DIR/psi_model.npz"
mkdir -p "$candidate_root" "$project/logs"
python3 "$project/evaluation/full_physical/preflight.py"

manifest=$OUTPUT_ROOT/candidate_jobs.tsv
printf 's_edge\tjob_id\toutput_dir\n' > "$manifest"
IFS=',' read -r -a edges <<< "$S_EDGES"
previous_job=

for edge in "${edges[@]}"; do
    edge=${edge//[[:space:]]/}
    [[ $edge =~ ^[0-9]+([.][0-9]+)?$ ]] || {
        printf 'invalid S_EDGE: %s\n' "$edge" >&2
        exit 2
    }
    slug=${edge//./p}
    output_dir=$candidate_root/s_$slug
    test ! -e "$output_dir"
    submit_args=(
        --parsable
        --cpus-per-task="$candidate_cpus_per_task"
        --export="ALL,PROJECT=$project,GPU_LIB=$gpu_lib,EVAL_ENV=$eval_env,CASE_FILE=$CASE_FILE,RUN_DIR=$RUN_DIR,S_EDGE=$edge,OUTPUT_DIR=$output_dir"
    )
    if [[ $serial_candidates == 1 && -n $previous_job ]]; then
        submit_args+=(--dependency="afterany:$previous_job")
    fi
    (cd "$project" && sbatch --test-only "${submit_args[@]:1}" scripts/slurm_alpha_nu_guarded_boozer.sh) >/dev/null
    job_id=$(cd "$project" && sbatch "${submit_args[@]}" scripts/slurm_alpha_nu_guarded_boozer.sh)
    job_id=${job_id%%;*}
    printf '%s\t%s\t%s\n' "$edge" "$job_id" "$output_dir" | tee -a "$manifest"
    previous_job=$job_id
done

job_ids=$(tail -n +2 "$manifest" | cut -f2 | paste -sd, -)
printf 'monitor: squeue -j %s -o %s\n' "$job_ids" "'%.18i %.12T %.10M %.30j %R'"
