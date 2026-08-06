#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --job-name=qh-point-vjp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator_worktrees/qh-blackbox-gradient}"
lib="${GPU_LIB:-$project/gpu_backend/build_g4_point_vjp/libstellarator_gpu.so}"
output="${OUTPUT:-$project/runs/qh_field_point_vjp_${SLURM_JOB_ID}.json}"
expected_sha="${EXPECTED_LIB_SHA:-}"

cd "$project"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project:$project/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load cuda/13.0 2>/dev/null || true
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
export LD_LIBRARY_PATH="$cuda_wheel_lib:/public/app/cuda/13.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -f "$lib"
if [[ -n "$expected_sha" ]]; then
  test "$(sha256sum "$lib" | awk '{print $1}')" = "$expected_sha"
fi
python scripts/validate_field_gradient.py \
  --case examples/01.json \
  --gpu-lib "$lib" \
  --segments 256 \
  --points 128 \
  --device 0 \
  --output "$output"
test -s "$output"
