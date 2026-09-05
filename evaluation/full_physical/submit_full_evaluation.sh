#!/usr/bin/env bash

set -euo pipefail

: "${CASE_FILE:?CASE_FILE is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

project=${PROJECT:-$HOME/local_surface_evaluator}
gpu_lib=${GPU_LIB:-$project/gpu_backend/build_mixed/libstellarator_gpu.so}
eval_env=${EVAL_ENV:-$HOME/local_surface_evaluator/.venv-desc016-py312}
a_values=${A_VALUES:-0.04,0.05,0.06,0.08}
s_edges=${S_EDGES:-0.12,0.24,0.36,0.49,0.64,0.81,1.0}
pool=${FULL_EVAL_POOL:-p107}

for path in "$project" "$gpu_lib" "$eval_env" "$CASE_FILE" "$OUTPUT_ROOT"; do
    resolved=$(realpath -m "$path")
    [[ $resolved == "$HOME"/* ]] || {
        printf 'path must stay under HOME: %s\n' "$resolved" >&2
        exit 2
    }
done
test -d "$project"
test -d "$eval_env"
test -f "$gpu_lib"
test -f "$CASE_FILE"
test ! -e "$OUTPUT_ROOT"
test ! -e "$OUTPUT_ROOT.job_id"
mkdir -p "$(dirname "$OUTPUT_ROOT")"
mkdir -p "$project/logs"

python3 "$project/evaluation/full_physical/preflight.py"

case "$pool" in
    p107)
        candidate_cpus=4
        desc_cpus=16
        submit_args=(
            --account=competition
            --partition=P107-RTX5090
            --qos=qos_p107-rtx5090
            --cpus-per-task=16
            --gres=gpu:RTX5090:4
            --mem=128G
            --time=04:00:00
        )
        ;;
    students)
        candidate_cpus=4
        desc_cpus=24
        submit_args=(
            --account=stu
            --partition=Students
            --qos=qos_stu_medium_2gpu
            --cpus-per-task=24
            --gres=gpu:RTX5090:2
            --mem=128G
            --time=08:00:00
        )
        ;;
    students-default)
        candidate_cpus=4
        desc_cpus=4
        submit_args=(
            --account=stu
            --partition=Students
            --qos=qos_stu_default
            --cpus-per-task=4
            --gres=gpu:RTX5090:1
            --mem=16G
            --time=04:00:00
        )
        ;;
    *)
        printf 'FULL_EVAL_POOL must be p107, students, or students-default\n' >&2
        exit 2
        ;;
esac

cd "$project"
sbatch --test-only \
    "${submit_args[@]}" \
    --export="ALL,PROJECT=$project,GPU_LIB=$gpu_lib,EVAL_ENV=$eval_env,CANDIDATE_CPUS=$candidate_cpus,DESC_CPUS=$desc_cpus" \
    scripts/slurm_full_physical_evaluation.sh \
    "$CASE_FILE" "$OUTPUT_ROOT" "$a_values" "$s_edges"
job_id=$(sbatch --parsable \
    "${submit_args[@]}" \
    --export="ALL,PROJECT=$project,GPU_LIB=$gpu_lib,EVAL_ENV=$eval_env,CANDIDATE_CPUS=$candidate_cpus,DESC_CPUS=$desc_cpus" \
    scripts/slurm_full_physical_evaluation.sh \
    "$CASE_FILE" "$OUTPUT_ROOT" "$a_values" "$s_edges")
job_id=${job_id%%;*}
printf '%s\n' "$job_id" > "$OUTPUT_ROOT.job_id"
printf 'job_id=%s\noutput_root=%s\nmonitor: squeue -j %s -o %s\n' \
    "$job_id" "$OUTPUT_ROOT" "$job_id" "'%.18i %.12T %.10M %.30j %R'"
