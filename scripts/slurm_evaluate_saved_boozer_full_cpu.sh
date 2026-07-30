#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --job-name=boozer-full-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${CASE_FILE:?CASE_FILE is required}"
: "${SURFACE_NPZ:?SURFACE_NPZ is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

project="${PROJECT:-$HOME/local_surface_evaluator}"
eval_env=${EVAL_ENV:-$HOME/local_surface_evaluator/.venv-desc016-py312}
children=()

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if (( ${#children[@]} )); then
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$OUTPUT_DIR"
cd "$project"
source "$eval_env/bin/activate"
export PYTHONPATH="$project:$eval_env/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

surface_order_args=()
if [[ -n ${SURFACE_ORDER:-} ]]; then
    surface_order_args+=(--surface-order "$SURFACE_ORDER")
fi

python3 "$project/scripts/evaluate_saved_boozer_surface_full.py" \
    --case-file "$CASE_FILE" \
    --surface-npz "$SURFACE_NPZ" \
    --output-dir "$OUTPUT_DIR" \
    --poincare-nfieldlines "${POINCARE_NFIELDLINES:-8}" \
    --poincare-tmax "${POINCARE_TMAX:-2e7}" \
    --desc-maxiter "${DESC_MAXITER:-50}" \
    "${surface_order_args[@]}" &
children+=("$!")
wait "${children[0]}"
children=()
