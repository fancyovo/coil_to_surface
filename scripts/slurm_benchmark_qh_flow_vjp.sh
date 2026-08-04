#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-flow-vjp-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator_worktrees/qh-blackbox-gradient}"
reference="${REFERENCE_DIR:-$project/runs/qh_blackbox_gradient_reference_31640}"
checkpoint="${FLOW_CHECKPOINT:-$HOME/local_surface_evaluator/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
gradient_lib="${GRADIENT_LIB:-$project/gpu_backend/build_gradient_sm120/libstellarator_gpu.so}"
output="${OUTPUT_DIR:-$project/runs/qh_flow_vjp_benchmark_${SLURM_JOB_ID}}"
centers=(main_nfp6_step0 main_nfp6_step200 main_nfp6_step400 cross_nfp4_step50)

mkdir -p "$project/logs" "$output"
cd "$project"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project:$project/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -f "$checkpoint"
test -f "$gradient_lib"
test -f "$reference/manifest.json"

idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS= read -r memory_used; do
    memory_used="${memory_used// /}"
    if (( memory_used > 16 )); then idle=0; fi
  done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
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
  echo "allocated GPUs retained memory or compute processes during idle probes" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader > "$output/gpu_preflight.csv"

pids=()
for rank in 0 1 2 3; do
  profile=()
  if (( rank == 1 )); then profile=(--profile); fi
  CUDA_VISIBLE_DEVICES="$rank" python scripts/benchmark_qh_flow_vjp.py \
    --reference-dir "$reference" \
    --checkpoint "$checkpoint" \
    --gradient-lib "$gradient_lib" \
    --output-dir "$output" \
    --center-id "${centers[$rank]}" \
    --steps 32,64,128,256 \
    --device cuda:0 \
    "${profile[@]}" \
    > "$output/worker_${rank}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if (( status != 0 )); then
  tail -n 80 "$output"/worker_*.log >&2 || true
  exit "$status"
fi

python scripts/benchmark_qh_flow_vjp.py \
  --reference-dir "$reference" \
  --output-dir "$output" \
  --analyze-only \
  --expected-centers 4

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader > "$output/gpu_postflight.csv"
