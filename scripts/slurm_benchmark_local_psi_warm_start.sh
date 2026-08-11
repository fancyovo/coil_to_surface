#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=psi-warm-local
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
candidate_dir="${CANDIDATE_DIR:?CANDIDATE_DIR is required}"
run_root="${RUN_ROOT:?RUN_ROOT is required}"
direction_count="${DIRECTION_COUNT:-4}"
build_dir="$project/gpu_backend/build_local_psi_warm_cuda13"

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
test -f "$candidate_dir/candidates.json"
test -f "$candidate_dir/candidates.npz"
test ! -e "$run_root"
mkdir -p "$run_root"

source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project/gpu_backend/python:$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

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
git rev-parse HEAD > "$run_root/git_head.txt"

cmake -S gpu_backend -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSGPU_BUILD_QR_BENCHMARK=ON
cmake --build "$build_dir" -j4
lib="$build_dir/libstellarator_gpu.so"
benchmark="$build_dir/psi_qr_benchmark"
sha256sum "$lib" "$benchmark" > "$run_root/binaries.sha256"

CUDA_VISIBLE_DEVICES=0 python scripts/freeze_local_psi_warm_snapshots.py \
  --candidate-dir "$candidate_dir" \
  --lib "$lib" \
  --output-dir "$run_root/snapshots" \
  --scale 0.005 \
  --direction-count "$direction_count" \
  > "$run_root/freeze_stdout.json"

center="$run_root/snapshots/center.bin"
results="$run_root/benchmark.jsonl"
: > "$results"
for endpoint in "$run_root"/snapshots/direction_*.bin; do
  endpoint_name="$(basename "$endpoint" .bin)"
  for iterations in 0 1 2 4 8 16 32 64; do
    "$benchmark" \
      --snapshot "$endpoint" \
      --warm-snapshot "$center" \
      --method "warmcgls${iterations}" \
      --device 0 \
      --warmups 1 \
      --repeats 3 \
      | python -c \
        'import json,sys; row=json.load(sys.stdin); row["endpoint"] = sys.argv[1]; print(json.dumps(row))' \
        "$endpoint_name" >> "$results"
  done
done

touch "$run_root/completed.ok"
