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
#SBATCH --time=12:00:00
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
idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS= read -r memory_used; do
    memory_used="${memory_used// /}"
    if (( memory_used > 16 )); then
      idle=0
    fi
  done < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    idle=0
  fi
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
  echo "allocated GPUs retained memory or compute processes during idle probes" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

resume_args=()
if [[ -n "${QH_FLOW_RESUME:-}" ]]; then
  test -f "$QH_FLOW_RESUME"
  resume_args=(--resume "$QH_FLOW_RESUME" --resume-model "${QH_FLOW_RESUME_MODEL:-model}")
  if [[ "${QH_FLOW_RESET_OPTIMIZER:-0}" == "1" ]]; then
    resume_args+=(--reset-optimizer)
  fi
fi

train_args=(
  --data-dir "$data"
  --output-dir "$output"
  --steps "${QH_FLOW_STEPS:-8000}"
  --batch-per-gpu "${QH_FLOW_BATCH_PER_GPU:-8192}"
  --learning-rate "${QH_FLOW_LEARNING_RATE:-2e-4}"
  --warmup-steps "${QH_FLOW_WARMUP_STEPS:-500}"
  --lr-schedule "${QH_FLOW_LR_SCHEDULE:-cosine}"
  --geometry-relative-weight "${QH_FLOW_GEOMETRY_RELATIVE_WEIGHT:-0.05}"
  --current-feature-weight "${QH_FLOW_CURRENT_FEATURE_WEIGHT:-1.0}"
  --log-interval 20
  --validation-interval "${QH_FLOW_VALIDATION_INTERVAL:-200}"
  --sample-interval "${QH_FLOW_SAMPLE_INTERVAL:-250}"
  --sample-count "${QH_FLOW_SAMPLE_COUNT:-256}"
  --sample-steps "${QH_FLOW_SAMPLE_STEPS:-16}"
  --checkpoint-interval "${QH_FLOW_CHECKPOINT_INTERVAL:-1000}"
  --score-lib "$score_lib"
  --score-start-step "${QH_FLOW_SCORE_START_STEP:-2000}"
  --score-interval "${QH_FLOW_SCORE_INTERVAL:-2000}"
  --score-count "${QH_FLOW_SCORE_COUNT:-32}"
  "${resume_args[@]}"
)

launch_attempts="${QH_FLOW_LAUNCH_ATTEMPTS:-3}"
if (( launch_attempts < 1 )); then
  echo "QH_FLOW_LAUNCH_ATTEMPTS must be positive" >&2
  exit 2
fi
for (( attempt = 1; attempt <= launch_attempts; attempt++ )); do
  if python -m torch.distributed.run --standalone --nproc-per-node=4 \
    scripts/train_qh_flow.py "${train_args[@]}"; then
    exit 0
  else
    status=$?
  fi
  if [[ -e "$output/metrics.jsonl" ]] || (( attempt == launch_attempts )); then
    exit "$status"
  fi
  echo "torchrun failed before metrics creation; retrying launch ($attempt/$launch_attempts)" >&2
  sleep 10
done
