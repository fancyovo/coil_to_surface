#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --job-name=axisflip-a2k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo="${AXIS_LONG_REPO:?set AXIS_LONG_REPO}"
source_root="${AXIS_LONG_SOURCE_ROOT:?set AXIS_LONG_SOURCE_ROOT}"
run_root="${AXIS_LONG_RUN_ROOT:?set AXIS_LONG_RUN_ROOT}"
commit="${AXIS_LONG_COMMIT:?set AXIS_LONG_COMMIT}"
case_id="${AXIS_LONG_CASE_ID:?set AXIS_LONG_CASE_ID}"
nfp="${AXIS_LONG_NFP:?set AXIS_LONG_NFP}"
nc="${AXIS_LONG_NC:?set AXIS_LONG_NC}"
optimizer_seed="${AXIS_LONG_SEED:?set AXIS_LONG_SEED}"
checkpoint="${AXIS_LONG_CHECKPOINT:?set AXIS_LONG_CHECKPOINT}"
score_lib="${AXIS_LONG_SCORE_LIB:?set AXIS_LONG_SCORE_LIB}"
protocol_id="qh-axisflip-v4-representative-adam2000-64d-abi11-v1"
case_padded="$(printf '%07d' "$case_id")"
source_best="$source_root/trajectories/axisflip_case_${case_padded}/optimization/best.json"
source_start="$source_root/starts/case_${case_padded}.json"

cd "$repo"
mkdir -p "$run_root" "$repo/logs"
test -f "$source_best"
test -f "$source_start"
test -f "$checkpoint"
test -f "$score_lib"
export PYTHONPATH="$repo:$repo/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS= read -r memory_used; do
    memory_used="${memory_used// /}"
    if (( memory_used > 32 )); then idle=0; fi
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
  echo "allocated GPU was not idle before launch" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
  > "$run_root/gpu_preflight.csv"

start="$run_root/start.json"
python scripts/prepare_axisflip_long_start.py \
  --source-best "$source_best" \
  --source-start "$source_start" \
  --output "$start" \
  --protocol-id "$protocol_id" \
  --expected-commit "$commit" \
  --nfp "$nfp"

python scripts/optimize_flow_latent.py \
  --checkpoint "$checkpoint" \
  --initial-case "$start" \
  --lib "$score_lib" \
  --out-dir "$run_root/optimization" \
  --nfp "$nfp" \
  --n-base-coils "$nc" \
  --target-helicity-sign 1 \
  --iterations 2000 \
  --max-wall-s 42300 \
  --parameter-space data \
  --data-start-mode exact-unclipped \
  --recorded-initial-score-tolerance 0.1 \
  --perturbation 0.0025 \
  --gradient-mode random-orthogonal \
  --random-directions 64 \
  --seed "$optimizer_seed" \
  --optimizer adam \
  --learning-rate 0.01 \
  --beta1 0.7 \
  --beta2 0.999 \
  --flow-device 0 \
  --score-device 0 \
  --plot-every 0 \
  --progress-every 20 \
  --trajectory-every 0 \
  --state-every 50

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
  > "$run_root/gpu_postflight.csv"
echo "axis-flip Adam2000 complete: case=$case_id run_root=$run_root"
