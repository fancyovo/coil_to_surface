#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --job-name=axis-prior-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"

cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
python "$PROJECT/scripts/analyze_axis_surface_prior.py" \
  --input-dir "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/analysis"
