#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --job-name=qh-ref-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator_worktrees/qh-blackbox-gradient}"
expected_source_commit="${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required}"
checkpoint="${FLOW_CHECKPOINT:-$HOME/local_surface_evaluator/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
expected_checkpoint_sha="${EXPECTED_CHECKPOINT_SHA:-39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f}"
gradient_lib="${GRADIENT_LIB:-$project/gpu_backend/build_g4_oracle/libstellarator_gpu.so}"
expected_gradient_sha="${EXPECTED_GRADIENT_SHA:-071f67925a8eca6cfc702ed6b8380f4677bb4c36711e6aae032b7dd227bdc88c}"
initial_case="${INITIAL_CASE:-$project/reports/assets/qh_score_adam_start_panel_29960/start_10.json}"
output_root="${OUTPUT_ROOT:-$project/runs/qh_reference_direction_sweep_20260806}"
shard_index="${SHARD_INDEX:?SHARD_INDEX is required}"
iterations="${ITERATIONS:-100}"
seed="${SEED:-2026080601}"
allocated_gpus="${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"
child=""

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$child" ]]; then
    pkill -TERM -P "$child" 2>/dev/null || true
    kill "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  nvidia-smi --id="$allocated_gpus" \
    --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$output_root/shard_${shard_index}_gpu_postflight.csv" \
    2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$project/logs" "$output_root"
cd "$project"
test "$(git rev-parse HEAD)" = "$expected_source_commit"
for path in "$checkpoint" "$gradient_lib" "$initial_case"; do test -f "$path"; done
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_checkpoint_sha"
test "$(sha256sum "$gradient_lib" | awk '{print $1}')" = "$expected_gradient_sha"

source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project:$project/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS=',' read -r utilization memory_used; do
    utilization="${utilization// /}"
    memory_used="${memory_used// /}"
    if (( memory_used > 16 )); then idle=0; fi
  done < <(
    nvidia-smi --id="$allocated_gpus" \
      --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
  )
  if nvidia-smi --id="$allocated_gpus" \
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
nvidia-smi --id="$allocated_gpus" \
  --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$output_root/shard_${shard_index}_gpu_preflight.csv"

python scripts/run_qh_reference_direction_sweep_shard.py \
  --shard-index "$shard_index" \
  --shard-count 6 \
  --output-root "$output_root" \
  --checkpoint "$checkpoint" \
  --gradient-lib "$gradient_lib" \
  --initial-case "$initial_case" \
  --gpus "$allocated_gpus" \
  --iterations "$iterations" \
  --seed "$seed" &
child="$!"
wait "$child"
child=""
test "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$output_root/shard_$(printf '%02d' "$shard_index")_manifest.json")" = completed
