#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --job-name=boozer-contours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${CASE_FILE:?CASE_FILE is required}"
: "${SURFACE_NPZ:?SURFACE_NPZ is required}"
: "${EQUILIBRIUM:?EQUILIBRIUM is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

project=${PROJECT:-$HOME/local_surface_evaluator}
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

for path in "$project" "$eval_env" "$CASE_FILE" "$SURFACE_NPZ" "$EQUILIBRIUM" "$OUTPUT_DIR"; do
    resolved=$(realpath -m "$path")
    [[ $resolved == "$HOME"/* ]] || {
        printf 'path must stay under HOME: %s\n' "$resolved" >&2
        exit 2
    }
done

cd "$project"
source "$eval_env/bin/activate"
export PYTHONPATH="$project:$eval_env/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

surface_order_args=()
if [[ -n ${SURFACE_ORDER:-} ]]; then
    surface_order_args+=(--surface-order "$SURFACE_ORDER")
fi

python3 "$project/scripts/replot_boozer_contours.py" \
    --case-file "$CASE_FILE" \
    --surface-npz "$SURFACE_NPZ" \
    --equilibrium "$EQUILIBRIUM" \
    --output-dir "$OUTPUT_DIR" \
    "${surface_order_args[@]}" &
children+=("$!")
wait "${children[0]}"
children=()
