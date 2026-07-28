#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-flow-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo="${QH_FLOW_REPO:-$HOME/local_surface_evaluator}"
data="${QH_FLOW_DATA:-$HOME/local_surface_evaluator_data/quasr_qh_flow_v1}"
output="${QH_FLOW_OUTPUT:-$repo/runs/qh_flow_base_${SLURM_JOB_ID}}"
score_lib="${QH_FLOW_SCORE_LIB:-$repo/gpu_backend/build_native_score/libstellarator_gpu.so}"

cd "$repo"
test -f "$data/manifest.json"
test -f "$score_lib"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

python -m torch.distributed.run --standalone --nproc-per-node=4 scripts/train_qh_flow.py \
  --data-dir "$data" \
  --output-dir "$output" \
  --steps "${QH_FLOW_STEPS:-8000}" \
  --batch-per-gpu "${QH_FLOW_BATCH_PER_GPU:-1024}" \
  --log-interval 20 \
  --validation-interval 200 \
  --sample-interval 250 \
  --sample-count 256 \
  --sample-steps 16 \
  --checkpoint-interval 1000 \
  --score-lib "$score_lib" \
  --score-start-step "${QH_FLOW_SCORE_START_STEP:-2000}" \
  --score-interval "${QH_FLOW_SCORE_INTERVAL:-2000}" \
  --score-count "${QH_FLOW_SCORE_COUNT:-32}"
