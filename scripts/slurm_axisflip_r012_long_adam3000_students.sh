#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --job-name=r012-a3k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo="${AXIS_R012_LONG_REPO:?set AXIS_R012_LONG_REPO}"
source_root="${AXIS_R012_LONG_SOURCE_ROOT:?set AXIS_R012_LONG_SOURCE_ROOT}"
run_root="${AXIS_R012_LONG_RUN_ROOT:?set AXIS_R012_LONG_RUN_ROOT}"
commit="${AXIS_R012_LONG_COMMIT:?set AXIS_R012_LONG_COMMIT}"
case_id="${AXIS_R012_LONG_CASE_ID:?set AXIS_R012_LONG_CASE_ID}"
optimizer_seed="${AXIS_R012_LONG_SEED:?set AXIS_R012_LONG_SEED}"
checkpoint="${AXIS_R012_LONG_CHECKPOINT:?set AXIS_R012_LONG_CHECKPOINT}"
checkpoint_sha="${AXIS_R012_LONG_CHECKPOINT_SHA:?set AXIS_R012_LONG_CHECKPOINT_SHA}"
score_lib="${AXIS_R012_LONG_SCORE_LIB:?set AXIS_R012_LONG_SCORE_LIB}"
score_lib_sha="${AXIS_R012_LONG_SCORE_LIB_SHA:?set AXIS_R012_LONG_SCORE_LIB_SHA}"
protocol_id="qh-axisflip-r012-top2-continue-adam3000-64d-r04-abi11-v1"
case_padded="$(printf '%07d' "$case_id")"
source_best="$source_root/trajectories/axisflip_r012_case_${case_padded}/optimization/best.json"
source_start="$source_root/starts/case_${case_padded}.json"

cd "$repo"
mkdir -p "$run_root" "$repo/logs"
test "$(git rev-parse HEAD)" = "$commit"
git diff --quiet
git diff --cached --quiet
test -f "$source_best"
test -f "$source_start"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$checkpoint_sha"
test "$(sha256sum "$score_lib" | awk '{print $1}')" = "$score_lib_sha"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$repo:$repo/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$cuda_wheel_lib:/public/app/cuda/13.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
  > "$run_root/gpu_preflight.csv"
start="$run_root/start.json"
python scripts/prepare_axisflip_long_start.py \
  --source-best "$source_best" \
  --source-start "$source_start" \
  --output "$start" \
  --protocol-id "$protocol_id" \
  --expected-commit "$commit" \
  --nfp 8 \
  --requested-iterations 3000

python scripts/optimize_flow_latent.py \
  --checkpoint "$checkpoint" \
  --expected-checkpoint-sha256 "$checkpoint_sha" \
  --initial-case "$start" \
  --lib "$score_lib" \
  --out-dir "$run_root/optimization" \
  --nfp 8 \
  --n-base-coils 3 \
  --target-helicity-sign 1 \
  --iterations 3000 \
  --max-wall-s 85800 \
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
echo "R012/R04 Adam3000 complete: case=$case_id run_root=$run_root"
