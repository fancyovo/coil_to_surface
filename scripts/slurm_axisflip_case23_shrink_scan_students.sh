#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --job-name=axis-shrink-scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo="${AXIS_SHRINK_REPO:?set AXIS_SHRINK_REPO}"
run_root="${AXIS_SHRINK_RUN_ROOT:?set AXIS_SHRINK_RUN_ROOT}"
commit="${AXIS_SHRINK_COMMIT:?set AXIS_SHRINK_COMMIT}"
source_best="${AXIS_SHRINK_SOURCE_BEST:?set AXIS_SHRINK_SOURCE_BEST}"
source_start="${AXIS_SHRINK_SOURCE_START:?set AXIS_SHRINK_SOURCE_START}"
axis_data="${AXIS_SHRINK_AXIS_DATA:?set AXIS_SHRINK_AXIS_DATA}"
surface_source="${AXIS_SHRINK_SURFACE:?set AXIS_SHRINK_SURFACE}"
checkpoint="${AXIS_SHRINK_CHECKPOINT:?set AXIS_SHRINK_CHECKPOINT}"
score_lib="${AXIS_SHRINK_SCORE_LIB:?set AXIS_SHRINK_SCORE_LIB}"
source_best_sha="${AXIS_SHRINK_SOURCE_BEST_SHA:?set AXIS_SHRINK_SOURCE_BEST_SHA}"
surface_sha="${AXIS_SHRINK_SURFACE_SHA:?set AXIS_SHRINK_SURFACE_SHA}"
checkpoint_sha="${AXIS_SHRINK_CHECKPOINT_SHA:?set AXIS_SHRINK_CHECKPOINT_SHA}"
score_lib_sha="${AXIS_SHRINK_SCORE_LIB_SHA:?set AXIS_SHRINK_SCORE_LIB_SHA}"
eval_env="${AXIS_SHRINK_EVAL_ENV:?set AXIS_SHRINK_EVAL_ENV}"
scales="${AXIS_SHRINK_SCALES:-1.0,0.8,0.65,0.55,0.50,0.45,0.40,0.35,0.30,0.25,0.20}"

cd "$repo"
mkdir -p "$repo/logs"
test ! -e "$run_root"
test -f "$source_best"
test -f "$source_start"
test -f "$axis_data"
test -f "$surface_source"
test -f "$checkpoint"
test -f "$score_lib"
test -x "$eval_env/bin/python3"
export PYTHONPATH="$repo:$repo/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$run_root"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
  > "$run_root/gpu_preflight.csv"
surface_mesh="$run_root/reference_surface_mesh.npz"
"$eval_env/bin/python3" scripts/sample_saved_boozer_surface_mesh.py \
  --surface "$surface_source" \
  --output "$surface_mesh"

python scripts/prepare_axisflip_case23_coil_shrink.py \
  --source-best "$source_best" \
  --source-start "$source_start" \
  --axis-data "$axis_data" \
  --surface-mesh "$surface_mesh" \
  --surface-source "$surface_source" \
  --checkpoint "$checkpoint" \
  --score-lib "$score_lib" \
  --protocol evaluation/axisflip_case23_coil_shrink_adam200_abi11_v1.json \
  --expected-commit "$commit" \
  --expected-score-lib-sha "$score_lib_sha" \
  --expected-checkpoint-sha "$checkpoint_sha" \
  --expected-source-best-sha "$source_best_sha" \
  --expected-surface-sha "$surface_sha" \
  --scales "$scales" \
  --output-dir "$run_root/scan"

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
  > "$run_root/gpu_postflight.csv"
