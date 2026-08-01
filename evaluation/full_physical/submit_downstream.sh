#!/usr/bin/env bash

set -euo pipefail

: "${CASE_FILE:?CASE_FILE is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${DESC_BACKEND:?DESC_BACKEND must be explicitly set to cpu, cpu-p107, or gpu}"

project=${PROJECT:-$HOME/local_surface_evaluator}
eval_env=${EVAL_ENV:-$project/.venv-desc016-py312}
candidate_root=${CANDIDATE_ROOT:-$OUTPUT_ROOT/candidates}
selection_json=${SELECTION_JSON:-$OUTPUT_ROOT/selection.json}
full_output_dir=${FULL_OUTPUT_DIR:-$OUTPUT_ROOT/full}

submit_args=()
case "$DESC_BACKEND" in
    cpu)
        slurm_script=scripts/slurm_evaluate_saved_boozer_full_cpu.sh
        ;;
    cpu-p107)
        slurm_script=scripts/slurm_evaluate_saved_boozer_full_cpu.sh
        submit_args+=(
            --account=competition
            --partition=P107-RTX5090
            --qos=qos_p107-rtx5090
            --nodes=1
            --ntasks=1
            --cpus-per-task=16
            --mem=128G
            --time=00:30:00
            --job-name=boozer-full-cpu-p107
            --output=logs/%x-%j.out
            --error=logs/%x-%j.err
        )
        ;;
    gpu)
        slurm_script=scripts/slurm_evaluate_saved_boozer_full.sh
        ;;
    *) printf 'DESC_BACKEND must be cpu, cpu-p107, or gpu\n' >&2; exit 2 ;;
esac

for path in "$project" "$eval_env" "$CASE_FILE" "$OUTPUT_ROOT"; do
    resolved=$(realpath -m "$path")
    [[ $resolved == "$HOME"/* ]] || {
        printf 'path must stay under HOME: %s\n' "$resolved" >&2
        exit 2
    }
done
test ! -e "$full_output_dir"
python3 "$project/evaluation/full_physical/preflight.py"
surface_npz=$(python3 "$project/evaluation/full_physical/select_largest_standard_surface.py" --candidate-root "$candidate_root" --output "$selection_json")

cd "$project"
sbatch --test-only "${submit_args[@]}" --export="ALL,PROJECT=$project,EVAL_ENV=$eval_env,CASE_FILE=$CASE_FILE,SURFACE_NPZ=$surface_npz,OUTPUT_DIR=$full_output_dir" "$slurm_script" >/dev/null
job_id=$(sbatch --parsable "${submit_args[@]}" --export="ALL,PROJECT=$project,EVAL_ENV=$eval_env,CASE_FILE=$CASE_FILE,SURFACE_NPZ=$surface_npz,OUTPUT_DIR=$full_output_dir" "$slurm_script")
job_id=${job_id%%;*}
printf '%s\n' "$job_id" > "$OUTPUT_ROOT/downstream_job_id.txt"
printf 'job_id=%s\nsurface=%s\nmonitor: squeue -j %s -o %s\n' "$job_id" "$surface_npz" "$job_id" "'%.18i %.12T %.10M %.30j %R'"
