#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=local-score-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
checkpoint="${FLOW_CHECKPOINT:?FLOW_CHECKPOINT is required}"
run_root="${RUN_ROOT:?RUN_ROOT is required}"
expected_flow_sha="${EXPECTED_FLOW_SHA:?EXPECTED_FLOW_SHA is required}"
build_dir="$project/gpu_backend/build_local_score_gradient"

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
test ! -e "$run_root"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_flow_sha"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project/gpu_backend/python:$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

idle_streak=0
for _ in {1..60}; do
  utilization="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
  memory_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "$utilization" == 0 && "$memory_used" -le 16 ]] &&
      ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    ((idle_streak += 1))
    if (( idle_streak >= 3 )); then break; fi
  else
    idle_streak=0
  fi
  sleep 2
done
if (( idle_streak < 3 )); then exit 42; fi

mkdir -p "$run_root/results"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$run_root/gpu_preflight.csv"
cmake -S gpu_backend -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --target stellarator_gpu -j4
lib="$build_dir/libstellarator_gpu.so"
sha256sum "$lib" > "$run_root/score_lib_sha256.txt"

python scripts/prepare_local_score_gradient_candidates.py \
  --center "step120=$project/reports/assets/qh_iota_cubic_adam200_35864/trajectory/step_0120.json" \
  --checkpoint "$checkpoint" \
  --output-dir "$run_root/candidates" \
  --direction-mode random \
  --direction-count 2 \
  --scales 0.005 \
  --flow-steps 128
python scripts/evaluate_local_score_gradient_shard.py \
  --candidate-dir "$run_root/candidates" \
  --lib "$lib" \
  --output "$run_root/results/rank_0.jsonl" \
  --shard-index 0 \
  --shard-count 1
python scripts/analyze_local_score_gradient_calibration.py \
  --candidate-dir "$run_root/candidates" \
  --result-dir "$run_root/results" \
  --output-dir "$run_root/analysis"
touch "$run_root/completed.ok"
