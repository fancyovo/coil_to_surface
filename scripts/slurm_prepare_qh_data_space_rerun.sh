#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --job-name=qh-data-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${REFERENCE_ROOT:?REFERENCE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"

source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
python "$PROJECT/scripts/prepare_qh_data_space_trajectory_rerun.py" \
  --reference-root "$REFERENCE_ROOT" \
  --run-root "$RUN_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --lib "$SCORE_LIB" \
  --worker-count 6
