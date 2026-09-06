#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --job-name=score-valid-weighted
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:RTX5090:2
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
repo="${SCORE_GRADIENT_REPO:?}"
run_root="${SCORE_GRADIENT_RUN_ROOT:?}"
commit="${SCORE_GRADIENT_COMMIT:?}"
cd "$repo"
test "$(git rev-parse HEAD)" = "$commit"
git diff --quiet
git diff --cached --quiet
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$repo:$repo/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID MPLBACKEND=Agg
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SCORE_GRADIENT_PROTOCOL_ID=qh-axisflip-r012-valid-score-weighted-rl-r04-abi11-v1
export SCORE_GRADIENT_FORMAT=axisflip_r012_valid_score_weighted_rl_r04_v1
export SCORE_GRADIENT_FLOW_OPTIMIZER_STEPS_PER_ROUND=10 SCORE_GRADIENT_EMA_LERP=0.1
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m scripts.smoke_valid_score_weighting_ddp
python -m scripts.prepare_score_gradient_continuation \
  --protocol evaluation/axisflip_r012_valid_score_weighted_rl_r04_abi11_v1.json \
  --run-root "$run_root" --expected-commit "$commit"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$run_root/gpu_preflight.csv"
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/score_gradient_flow_rl.py run --run-root "$run_root" --max-wall-s 82800 --reserve-s 3600
