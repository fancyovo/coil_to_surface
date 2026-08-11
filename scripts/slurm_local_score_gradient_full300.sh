#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=local-score-full300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
checkpoint="${FLOW_CHECKPOINT:?FLOW_CHECKPOINT is required}"
lib="${SCORE_LIB:?SCORE_LIB is required}"
run_root="${RUN_ROOT:?RUN_ROOT is required}"
expected_flow_sha="${EXPECTED_FLOW_SHA:?EXPECTED_FLOW_SHA is required}"
expected_lib_sha="${EXPECTED_SCORE_LIB_SHA:?EXPECTED_SCORE_LIB_SHA is required}"
children=()

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if (( ${#children[@]} )); then
    kill "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
  fi
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
test "$(sha256sum "$lib" | awk '{print $1}')" = "$expected_lib_sha"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project/gpu_backend/python:$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS=',' read -r utilization memory_used; do
    utilization="${utilization// /}"
    memory_used="${memory_used// /}"
    if (( utilization != 0 || memory_used > 16 )); then idle=0; fi
  done < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then idle=0; fi
  if (( idle )); then
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

CUDA_VISIBLE_DEVICES=0 python scripts/prepare_local_score_gradient_candidates.py \
  --center "step120=$project/reports/assets/qh_iota_cubic_adam200_35864/trajectory/step_0120.json" \
  --checkpoint "$checkpoint" \
  --output-dir "$run_root/candidates" \
  --direction-mode coordinate \
  --scales 0.005,0.01 \
  --flow-steps 128

for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$rank" python scripts/evaluate_local_score_gradient_shard.py \
    --candidate-dir "$run_root/candidates" \
    --lib "$lib" \
    --output "$run_root/results/rank_${rank}.jsonl" \
    --shard-index "$rank" \
    --shard-count 4 \
    --variants axis_qr16k &
  children+=("$!")
done
for child in "${children[@]}"; do wait "$child"; done
children=()

python scripts/analyze_local_score_gradient_calibration.py \
  --candidate-dir "$run_root/candidates" \
  --result-dir "$run_root/results" \
  --output-dir "$run_root/analysis" \
  --variants axis_qr16k
touch "$run_root/completed.ok"
