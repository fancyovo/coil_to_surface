#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${INPUT_ROOT:?INPUT_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_LIB_SHA:?EXPECTED_LIB_SHA is required}"
: "${EXPECTED_CHECKPOINT_SHA:?EXPECTED_CHECKPOINT_SHA is required}"

cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python "$PROJECT/scripts/prepare_axis_surface_prior_v3_adam200.py" \
  --input-dir "$INPUT_ROOT" \
  --run-root "$RUN_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --score-lib "$SCORE_LIB" \
  --sample-count "${SAMPLE_COUNT:-120}" \
  --worker-count 6 \
  --seed "${SELECTION_SEED:-20260904}" \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-lib-sha "$EXPECTED_LIB_SHA" \
  --expected-checkpoint-sha "$EXPECTED_CHECKPOINT_SHA"
