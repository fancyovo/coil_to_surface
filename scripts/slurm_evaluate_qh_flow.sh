#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-flow-eval
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
checkpoint="${QH_FLOW_CHECKPOINT:?set QH_FLOW_CHECKPOINT to the trained checkpoint}"
output="${QH_FLOW_OUTPUT:-$repo/runs/qh_flow_eval_${SLURM_JOB_ID}}"
score_lib="${QH_FLOW_SCORE_LIB:-$repo/gpu_backend/build_native_score/libstellarator_gpu.so}"

cd "$repo"
test -f "$data/manifest.json"
test -f "$checkpoint"
test -f "$score_lib"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS=',' read -r memory_used utilization; do
    memory_used="${memory_used// /}"
    utilization="${utilization// /}"
    if (( memory_used > 16 || utilization != 0 )); then
      idle=0
    fi
  done < <(
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
  )
  if (( idle )); then
    ((idle_streak += 1))
    if (( idle_streak >= 3 )); then
      break
    fi
  else
    idle_streak=0
  fi
  sleep 2
done
if (( idle_streak < 3 )); then
  echo "allocated GPUs did not remain idle for three consecutive probes" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

python -m torch.distributed.run --standalone --nproc-per-node=4 scripts/evaluate_qh_flow.py \
  --checkpoint "$checkpoint" \
  --data-dir "$data" \
  --output-dir "$output" \
  --lib "$score_lib" \
  --count "${QH_FLOW_EVAL_COUNT:-8192}" \
  --sample-steps "${QH_FLOW_SAMPLE_STEPS:-32}" \
  --sample-batch 512
