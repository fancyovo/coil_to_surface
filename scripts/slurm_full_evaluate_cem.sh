#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=cem-full-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=anode02
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${CASE_FILE:?CASE_FILE must point to the selected CEM best.json under the project}"
: "${GPU_LIB:?GPU_LIB must point to the pinned native evaluator library}"
project=${PROJECT:-/home/scc/pb24511935/local_surface_evaluator}
output_dir=${OUTPUT_DIR:-$project/runs/cem_full_eval/${SLURM_JOB_ID}}

cleanup() {
    status=$?
    trap - EXIT INT TERM
    mapfile -t children < <(jobs -pr)
    if (( ${#children[@]} )); then
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits > "$output_dir/gpu_postflight.csv" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

cd "$project"
mkdir -p "$output_dir"
mapfile -t compute_processes < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
    printf 'the allocated GPU is not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
    exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$output_dir/gpu_preflight.csv"

eval_env=${EVAL_ENV:-$project/.venv-desc016-py312}

python3 "$project/evaluation/full_physical/run_full_evaluation.py" \
    --project "$project" \
    --case-file "$CASE_FILE" \
    --output-root "$output_dir" \
    --gpu-lib "$GPU_LIB" \
    --eval-env "$eval_env" \
    --a-values "${A_VALUES:-0.04,0.05,0.06,0.08}" \
    --s-edges "${S_EDGES:-0.12,0.24,0.36,0.49,0.64,0.81,1.0}" \
    --candidate-cpus "${CANDIDATE_CPUS:-4}" \
    --desc-cpus "${DESC_CPUS:-4}"
