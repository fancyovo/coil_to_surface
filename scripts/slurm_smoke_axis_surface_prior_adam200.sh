#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"

cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
mapfile -t compute_processes < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#compute_processes[@]} )); then
  printf 'allocated GPU is not idle: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python "$PROJECT/scripts/smoke_axis_surface_prior_adam200.py" --run-root "$RUN_ROOT" --device 0
