#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=query-batch-field
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
candidate_dir="${CANDIDATE_DIR:?CANDIDATE_DIR is required}"
run_root="${RUN_ROOT:?RUN_ROOT is required}"
query_count="${QUERY_COUNT:-600}"
surface_theta_count="${SURFACE_THETA_COUNT:-64}"
alpha_iterations="${ALPHA_ITERATIONS:-4}"
full_reference_dir="${FULL_REFERENCE_DIR:-}"
cuda_root="${CUDA_ROOT:-/public/app/cuda/13.0}"
build_dir="$project/gpu_backend/build_query_batch"

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -d "$run_root" ]]; then
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits > "$run_root/gpu_postflight.csv" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

cd "$project"
test -d "$candidate_dir"
test ! -e "$run_root"
mkdir -p "$run_root"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project/gpu_backend/python:$project${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$cuda_root/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_root/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS=',' read -r utilization memory_used; do
    utilization="${utilization// /}"
    memory_used="${memory_used// /}"
    if (( utilization != 0 || memory_used > 16 )); then idle=0; fi
  done < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
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
if (( idle_streak < 3 )); then exit 42; fi

nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$run_root/gpu_preflight.csv"
cmake -S "$project/gpu_backend" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
  -DCUDAToolkit_ROOT="$cuda_root" \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build "$build_dir" --target stellarator_gpu -j4
lib="$build_dir/libstellarator_gpu.so"
sha256sum "$lib" > "$run_root/library.sha256"
git rev-parse HEAD > "$run_root/git_head.txt"
command=(python "$project/scripts/benchmark_query_batch_field.py" \
  --candidate-dir "$candidate_dir" \
  --lib "$lib" \
  --output "$run_root/summary.json" \
  --query-count "$query_count" \
  --point-count 256 \
  --segments-per-coil 256 \
  --trace-steps 400 \
  --axis-integration-steps 960 \
  --axis-samples 240 \
  --surface-theta-count "$surface_theta_count" \
  --alpha-iterations "$alpha_iterations")
if [[ -n "$full_reference_dir" ]]; then
  command+=(--full-reference-dir "$full_reference_dir")
fi
"${command[@]}"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "GPU compute process remains after batch-field benchmark" >&2
  exit 3
fi
if ps -u "$USER" -o stat=,pid=,ppid=,comm= | awk '$1 ~ /^Z/ {found=1} END {exit !found}'; then
  echo "Zombie process remains after batch-field benchmark" >&2
  exit 4
fi
touch "$run_root/completed.ok"
