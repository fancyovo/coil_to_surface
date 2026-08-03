#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=pcare-psi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${CASE_FILE:?CASE_FILE is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${SURFACE_NPZ:?SURFACE_NPZ is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

project=/home/scc/pb24511935/local_surface_evaluator
eval_env=${EVAL_ENV:-$project/.venv-desc016-py312}

cleanup() {
    status=$?
    trap - EXIT INT TERM
    mapfile -t children < <(jobs -pr)
    if (( ${#children[@]} )); then
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits > "$OUTPUT_DIR/gpu_postflight.csv" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

cd /
mkdir -p "$OUTPUT_DIR"
mapfile -t compute_processes < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
    printf 'the allocated GPU is not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
    exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$OUTPUT_DIR/gpu_preflight.csv"

export PYTHONPATH="$project:$eval_env/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

run_trace() {
    local label=$1
    local tmax=$2
    local tol=$3
    cd /
    python3 "$project/scripts/plot_poincare_validation.py" \
        --case-file "$CASE_FILE" \
        --surface-npz "$SURFACE_NPZ" \
        --psi-model "$RUN_DIR/psi_model.npz" \
        --output "$OUTPUT_DIR/${label}.png" \
        --mpol 12 \
        --ntor 12 \
        --nfieldlines 8 \
        --marker-size 6 \
        --tmax-fl "$tmax" \
        --tol "$tol" \
        > "$OUTPUT_DIR/${label}.log"
}

run_trace trace_16period_tol1e8 3.6e7 1e-8
run_trace trace_16period_tol1e11 3.6e7 1e-11
run_trace trace_long_tol1e11 2.0e8 1e-11
