#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-flow-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo="${QH_FLOW_REPO:-$HOME/local_surface_evaluator}"
data="${QH_FLOW_DATA:-$HOME/local_surface_evaluator_data/quasr_qh_flow_v1}"
output="${QH_FLOW_OUTPUT:-$repo/runs/qh_flow_pilot_${SLURM_JOB_ID}}"
batch="${QH_FLOW_BATCH_PER_GPU:-1024}"
steps="${QH_FLOW_STEPS:-300}"

cd "$repo"
test -f "$data/manifest.json"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

python -m torch.distributed.run --standalone --nproc-per-node=4 scripts/train_qh_flow.py \
  --data-dir "$data" \
  --output-dir "$output" \
  --steps "$steps" \
  --batch-per-gpu "$batch" \
  --log-interval 20 \
  --validation-interval 100 \
  --sample-interval 100 \
  --sample-count 128 \
  --sample-steps 8 \
  --checkpoint-interval "$steps"
