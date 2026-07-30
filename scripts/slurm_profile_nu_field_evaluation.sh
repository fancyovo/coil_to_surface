#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=nu-field-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${CASE_FILE:?CASE_FILE is required}"
: "${SURFACE_NPZ:?SURFACE_NPZ is required}"
: "${OUTPUT:?OUTPUT is required}"

project="$HOME/local_surface_evaluator"
output_dir="$(dirname "$OUTPUT")"
cleanup() {
  status=$?
  trap - EXIT INT TERM
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader > "$output_dir/gpu_postflight.csv" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$output_dir"
cd "$project"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "allocated GPU is not idle" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader > "$output_dir/gpu_preflight.csv"

eval_env="$project/.venv-desc016-py312"
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$project:$eval_env/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 "$project/scripts/profile_nu_field_evaluation.py" \
  --case-file "$CASE_FILE" \
  --surface-npz "$SURFACE_NPZ" \
  --output "$OUTPUT"
