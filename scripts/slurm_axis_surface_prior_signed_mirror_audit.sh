#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${SOURCE_ADAM_ROOT:?SOURCE_ADAM_ROOT is required}"
: "${AUDIT_ROOT:?AUDIT_ROOT is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_LIB_SHA:?EXPECTED_LIB_SHA is required}"

cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -d "$SOURCE_ADAM_ROOT"
test "$(sha256sum "$SCORE_LIB" | awk '{print $1}')" = "$EXPECTED_LIB_SHA"
mapfile -t compute_processes < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#compute_processes[@]} )); then
  printf 'allocated GPU is not idle: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python "$PROJECT/scripts/score_axis_surface_prior_signed_mirrors.py" \
  --case-file "$SOURCE_ADAM_ROOT/trajectories/axisv3_case_01341/optimization/best.json" \
  --case-file "$SOURCE_ADAM_ROOT/trajectories/axisv3_case_02832/optimization/best.json" \
  --lib "$SCORE_LIB" \
  --expected-lib-sha "$EXPECTED_LIB_SHA" \
  --expected-commit "$EXPECTED_COMMIT" \
  --output-dir "$AUDIT_ROOT/signed_mirror" \
  --device 0
