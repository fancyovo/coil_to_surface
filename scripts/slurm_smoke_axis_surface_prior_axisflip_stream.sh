#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_LIB_SHA:?EXPECTED_LIB_SHA is required}"
: "${EXPECTED_CHECKPOINT_SHA:?EXPECTED_CHECKPOINT_SHA is required}"

cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
source "$HOME/coil/.venv/bin/activate"
mapfile -t compute_processes < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#compute_processes[@]} )); then
  printf 'allocated GPU is not idle: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
smoke_root="$RUN_ROOT/smoke_validation"
python "$PROJECT/scripts/run_axis_surface_prior_axisflip_stream.py" \
  --run-root "$smoke_root" \
  --protocol-path "$RUN_ROOT/protocol.json" \
  --checkpoint "$CHECKPOINT" \
  --score-lib "$SCORE_LIB" \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-lib-sha "$EXPECTED_LIB_SHA" \
  --expected-checkpoint-sha "$EXPECTED_CHECKPOINT_SHA" \
  --worker-index 0 \
  --worker-count 1 \
  --seed 30260905 \
  --device 0 \
  --discovery-wall-s 900 \
  --hard-wall-s 1180 \
  --iterations 3 \
  --max-valid-cases 1
python - "$smoke_root/workers/worker_00/done.json" <<'PY'
import json
from pathlib import Path
import sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
if result["valid_count"] != 1 or result["completed_adam_count"] != 1:
    raise SystemExit("axis-flip smoke did not complete exactly one valid trajectory")
if result["failed_valid_count"] != 0:
    raise SystemExit("axis-flip smoke recorded a valid-trajectory failure")
trajectory = next(
    (Path(sys.argv[1]).parents[2] / "trajectories").glob(
        "*/trajectory_manifest.json"
    )
)
manifest = json.load(open(trajectory, encoding="utf-8"))
gate = manifest.get("initial_consistency_gate")
if not isinstance(gate, dict) or gate.get("absolute_delta", 1.0) > 0.1:
    raise SystemExit("axis-flip smoke did not pass the pre-update consistency gate")
PY
