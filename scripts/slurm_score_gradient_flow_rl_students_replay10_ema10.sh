#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --job-name=score-grad-r10-e10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:RTX5090:2
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo="${SCORE_GRADIENT_REPO:?set SCORE_GRADIENT_REPO}"
run_root="${SCORE_GRADIENT_RUN_ROOT:?set SCORE_GRADIENT_RUN_ROOT}"
q0_checkpoint="${SCORE_GRADIENT_Q0_CHECKPOINT:?set SCORE_GRADIENT_Q0_CHECKPOINT}"
q0_sha="${SCORE_GRADIENT_Q0_SHA:?set SCORE_GRADIENT_Q0_SHA}"
optimizer_checkpoint="${SCORE_GRADIENT_OPTIMIZER_CHECKPOINT:?set SCORE_GRADIENT_OPTIMIZER_CHECKPOINT}"
optimizer_sha="${SCORE_GRADIENT_OPTIMIZER_SHA:?set SCORE_GRADIENT_OPTIMIZER_SHA}"
score_lib="${SCORE_GRADIENT_SCORE_LIB:?set SCORE_GRADIENT_SCORE_LIB}"
score_manifest="${SCORE_GRADIENT_SCORE_MANIFEST:?set SCORE_GRADIENT_SCORE_MANIFEST}"
score_sha="${SCORE_GRADIENT_SCORE_SHA:?set SCORE_GRADIENT_SCORE_SHA}"
commit="${SCORE_GRADIENT_COMMIT:?set SCORE_GRADIENT_COMMIT}"
protocol_id="${SCORE_GRADIENT_PROTOCOL_ID:?set SCORE_GRADIENT_PROTOCOL_ID}"
format="${SCORE_GRADIENT_FORMAT:?set SCORE_GRADIENT_FORMAT}"
flow_updates="${SCORE_GRADIENT_FLOW_OPTIMIZER_STEPS_PER_ROUND:?set SCORE_GRADIENT_FLOW_OPTIMIZER_STEPS_PER_ROUND}"
ema_lerp="${SCORE_GRADIENT_EMA_LERP:?set SCORE_GRADIENT_EMA_LERP}"
protocol_manifest="${SCORE_GRADIENT_PROTOCOL_MANIFEST:?set SCORE_GRADIENT_PROTOCOL_MANIFEST}"
test "$protocol_id" = "qh-axisflip-r012-score-gradient-replay10-ema10-rl-r04-abi11-v1"
test "$format" = "axisflip_r012_score_gradient_replay10_ema10_rl_r04_v1"
test "$flow_updates" = "10"
test "$ema_lerp" = "0.1"

cd "$repo"
mkdir -p "$repo/logs"
mkdir -p "$(dirname "$run_root")"
test -f "$q0_checkpoint"
test -f "$optimizer_checkpoint"
test -f "$score_lib"
test -f "$score_manifest"
test -f "$protocol_manifest"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$repo:$repo/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MPLBACKEND=Agg
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python scripts/score_gradient_flow_rl.py prepare \
  --run-root "$run_root" \
  --q0-checkpoint "$q0_checkpoint" \
  --expected-q0-sha "$q0_sha" \
  --optimizer-checkpoint "$optimizer_checkpoint" \
  --expected-optimizer-sha "$optimizer_sha" \
  --score-lib "$score_lib" \
  --score-library-manifest "$score_manifest" \
  --expected-score-lib-sha "$score_sha" \
  --expected-commit "$commit"
cp "$protocol_manifest" "$run_root/protocol.json"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$run_root/gpu_preflight.csv"

python -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/score_gradient_flow_rl.py run \
  --run-root "$run_root" \
  --max-wall-s 82800 \
  --reserve-s 3600

nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$run_root/gpu_postflight.csv"
