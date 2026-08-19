#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${EXPERIMENT_ROOT:?EXPERIMENT_ROOT is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

eval_env=${EVAL_ENV:-$HOME/local_surface_evaluator/.venv-desc016-py312}
source "$eval_env/bin/activate"
export PYTHONPATH="$PROJECT:$eval_env/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

if [[ ${ANALYSIS_KIND:-formal} == convergence ]]; then
  python "$PROJECT/scripts/analyze_qh_equal_s_grid_convergence.py" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --output-dir "$OUTPUT_DIR"
else
  python "$PROJECT/scripts/analyze_qh_equal_s_surface_qs.py" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --equal-s-output-name "${OUTPUT_NAME:-equal_s_qs_summary.json}"
fi
