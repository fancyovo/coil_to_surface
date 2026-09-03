#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --job-name=axis-shrink-a200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo="${AXIS_SHRINK_REPO:?set AXIS_SHRINK_REPO}"
start="${AXIS_SHRINK_START:?set AXIS_SHRINK_START}"
output="${AXIS_SHRINK_OUTPUT:?set AXIS_SHRINK_OUTPUT}"
checkpoint="${AXIS_SHRINK_CHECKPOINT:?set AXIS_SHRINK_CHECKPOINT}"
score_lib="${AXIS_SHRINK_SCORE_LIB:?set AXIS_SHRINK_SCORE_LIB}"
seed="${AXIS_SHRINK_SEED:?set AXIS_SHRINK_SEED}"

cd "$repo"
mkdir -p "$repo/logs" "$(dirname "$output")"
test -f "$start"
test -f "$checkpoint"
test -f "$score_lib"
test ! -e "$output"
export PYTHONPATH="$repo:$repo/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$output"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
  > "$output/gpu_preflight.csv"
python scripts/optimize_flow_latent.py \
  --checkpoint "$checkpoint" \
  --initial-case "$start" \
  --lib "$score_lib" \
  --out-dir "$output/optimization" \
  --nfp 6 \
  --n-base-coils 4 \
  --target-helicity-sign 1 \
  --iterations 200 \
  --max-wall-s 14100 \
  --parameter-space data \
  --data-start-mode exact-unclipped \
  --recorded-initial-score-tolerance 0.1 \
  --perturbation 0.0025 \
  --gradient-mode random-orthogonal \
  --random-directions 64 \
  --seed "$seed" \
  --optimizer adam \
  --learning-rate 0.01 \
  --beta1 0.7 \
  --beta2 0.999 \
  --flow-device 0 \
  --score-device 0 \
  --plot-every 0 \
  --progress-every 10 \
  --trajectory-every 0 \
  --state-every 50
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
  > "$output/gpu_postflight.csv"

