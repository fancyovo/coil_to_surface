#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${SOURCE_SCORE_ROOT:?SOURCE_SCORE_ROOT is required}"
: "${SOURCE_ADAM_ROOT:?SOURCE_ADAM_ROOT is required}"
: "${AUDIT_ROOT:?AUDIT_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"

cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -d "$SOURCE_SCORE_ROOT"
test -d "$SOURCE_ADAM_ROOT"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
python "$PROJECT/scripts/analyze_axis_surface_prior_handedness.py" \
  --source-score-root "$SOURCE_SCORE_ROOT" \
  --adam-run-root "$SOURCE_ADAM_ROOT" \
  --output-dir "$AUDIT_ROOT/population" \
  --expected-commit "$EXPECTED_COMMIT"
