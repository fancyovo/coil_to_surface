#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --job-name=qh-face-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${EXPERIMENT_ROOT:?EXPERIMENT_ROOT is required}"
eval_env=${EVAL_ENV:-$HOME/local_surface_evaluator/.venv-desc016-py312}
source "$eval_env/bin/activate"
export PYTHONPATH="$PROJECT:$eval_env/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
python "$PROJECT/scripts/analyze_qh_trajectory_face_qs.py" --experiment-root "$EXPERIMENT_ROOT" --output-dir "$EXPERIMENT_ROOT/analysis"
