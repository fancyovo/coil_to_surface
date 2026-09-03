#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --job-name=rlflow-lat64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

repo="${RL_LATENT_REPO:?set RL_LATENT_REPO}"
run_root="${RL_LATENT_RUN_ROOT:?set RL_LATENT_RUN_ROOT}"
checkpoint="${RL_LATENT_CHECKPOINT:?set RL_LATENT_CHECKPOINT}"
checkpoint_sha="${RL_LATENT_CHECKPOINT_SHA:?set RL_LATENT_CHECKPOINT_SHA}"
outer_round="${RL_LATENT_OUTER_ROUND:?set RL_LATENT_OUTER_ROUND}"
score_lib="${RL_LATENT_SCORE_LIB:?set RL_LATENT_SCORE_LIB}"
score_lib_sha="${RL_LATENT_SCORE_LIB_SHA:?set RL_LATENT_SCORE_LIB_SHA}"
commit="${RL_LATENT_COMMIT:?set RL_LATENT_COMMIT}"
start_count="${RL_LATENT_START_COUNT:-4}"
seed_base="${RL_LATENT_SEED_BASE:-202609031200}"
worker_index="${SLURM_ARRAY_TASK_ID:?array task id is required}"
gpu_selector="${CUDA_VISIBLE_DEVICES:-}"

cd "$repo"
mkdir -p "$run_root" "$repo/logs"
test -f "$checkpoint"
test -f "$score_lib"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$checkpoint_sha"
test "$(sha256sum "$score_lib" | awk '{print $1}')" = "$score_lib_sha"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$repo:$repo/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export CUDA_DEVICE_ORDER=PCI_BUS_ID
: "${gpu_selector:?CUDA_VISIBLE_DEVICES is required}"
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

gpu_record="$run_root/gpu_preflight_worker_${worker_index}.csv"
idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS=',' read -r utilization memory_used; do
    utilization="${utilization// /}"
    memory_used="${memory_used// /}"
    if (( utilization != 0 || memory_used > 16 )); then idle=0; fi
  done < <(
    nvidia-smi --id="$gpu_selector" \
      --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
  )
  if nvidia-smi --id="$gpu_selector" \
      --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    idle=0
  fi
  if (( idle )); then
    ((idle_streak += 1))
    if (( idle_streak >= 3 )); then break; fi
  else
    idle_streak=0
  fi
  sleep 2
done
if (( idle_streak < 3 )); then
  echo "allocated GPU did not reach three consecutive idle probes" >&2
  exit 42
fi
nvidia-smi --id="$gpu_selector" \
  --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$gpu_record"

python scripts/run_axisflip_rl_latent_adam200.py \
  --run-root "$run_root" \
  --checkpoint "$checkpoint" \
  --expected-checkpoint-sha "$checkpoint_sha" \
  --expected-outer-round "$outer_round" \
  --score-lib "$score_lib" \
  --expected-score-lib-sha "$score_lib_sha" \
  --expected-commit "$commit" \
  --worker-index "$worker_index" \
  --worker-count 2 \
  --start-count "$start_count" \
  --seed-base "$seed_base" \
  --per-case-max-wall-s 7200

nvidia-smi --id="$gpu_selector" \
  --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$run_root/gpu_postflight_worker_${worker_index}.csv"
