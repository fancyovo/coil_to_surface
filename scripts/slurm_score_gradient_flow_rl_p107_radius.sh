#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=score-grad-radius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:RTX5090:2
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SCORE_GRADIENT_REPO:?}"
: "${SCORE_GRADIENT_RUN_ROOT:?}"
: "${SCORE_GRADIENT_Q0_CHECKPOINT:?}"
: "${SCORE_GRADIENT_Q0_SHA:?}"
: "${SCORE_GRADIENT_OPTIMIZER_CHECKPOINT:?}"
: "${SCORE_GRADIENT_OPTIMIZER_SHA:?}"
: "${SCORE_GRADIENT_SCORE_LIB:?}"
: "${SCORE_GRADIENT_SCORE_MANIFEST:?}"
: "${SCORE_GRADIENT_SCORE_SHA:?}"
: "${SCORE_GRADIENT_COMMIT:?}"
: "${SCORE_GRADIENT_PROTOCOL_ID:?}"
: "${SCORE_GRADIENT_FORMAT:?}"
: "${SCORE_GRADIENT_PRIOR_RADIUS_M:?}"
: "${SCORE_GRADIENT_PRIOR_RADIUS_RANGE_M:?}"
: "${SCORE_GRADIENT_PROTOCOL_MANIFEST:?}"
: "${SCORE_GRADIENT_REFERENCE_MANIFEST:?}"
: "${SCORE_GRADIENT_REFERENCE_SHA:?}"

repo="$SCORE_GRADIENT_REPO"
run_root="$SCORE_GRADIENT_RUN_ROOT"
cd "$repo"
test "$(git rev-parse HEAD)" = "$SCORE_GRADIENT_COMMIT"
git diff --quiet
git diff --cached --quiet
test ! -e "$run_root"
for path in "$SCORE_GRADIENT_Q0_CHECKPOINT" "$SCORE_GRADIENT_OPTIMIZER_CHECKPOINT" "$SCORE_GRADIENT_SCORE_LIB" "$SCORE_GRADIENT_SCORE_MANIFEST" "$SCORE_GRADIENT_PROTOCOL_MANIFEST"; do test -f "$path"; done
mkdir -p "$repo/logs" "$(dirname "$run_root")"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$repo:$repo/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID MPLBACKEND=Agg
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python scripts/score_gradient_flow_rl.py prepare \
  --run-root "$run_root" --q0-checkpoint "$SCORE_GRADIENT_Q0_CHECKPOINT" \
  --expected-q0-sha "$SCORE_GRADIENT_Q0_SHA" \
  --optimizer-checkpoint "$SCORE_GRADIENT_OPTIMIZER_CHECKPOINT" \
  --expected-optimizer-sha "$SCORE_GRADIENT_OPTIMIZER_SHA" \
  --score-lib "$SCORE_GRADIENT_SCORE_LIB" \
  --score-library-manifest "$SCORE_GRADIENT_SCORE_MANIFEST" \
  --expected-score-lib-sha "$SCORE_GRADIENT_SCORE_SHA" \
  --expected-commit "$SCORE_GRADIENT_COMMIT" \
  --reference-manifest "$SCORE_GRADIENT_REFERENCE_MANIFEST" \
  --expected-reference-sha "$SCORE_GRADIENT_REFERENCE_SHA"
cp "$SCORE_GRADIENT_PROTOCOL_MANIFEST" "$run_root/protocol.json"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits > "$run_root/gpu_preflight.csv"
python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/score_gradient_flow_rl.py run --run-root "$run_root" --max-wall-s 342000 --reserve-s 3600
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits > "$run_root/gpu_postflight.csv"
