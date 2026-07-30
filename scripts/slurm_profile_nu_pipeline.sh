#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=nu-pipeline-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${CASE_FILE:?CASE_FILE is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${ALPHA_DIR:?ALPHA_DIR is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

project="$HOME/local_surface_evaluator"
audit_dir="${OUTPUT_DIR}_job"
mkdir -p "$audit_dir"
cleanup() {
    status=$?
    trap - EXIT INT TERM
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
        --format=csv,noheader > "$audit_dir/gpu_postflight.csv" 2>/dev/null || true
    if [[ -d "$OUTPUT_DIR" ]]; then
        cp "$audit_dir"/gpu_*.csv "$OUTPUT_DIR"/ 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

cd "$project"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    echo "allocated GPU is not idle" >&2
    exit 42
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader > "$audit_dir/gpu_preflight.csv"

eval_env="$project/.venv-desc016-py312"
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$project:$eval_env/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

python3 "$project/scripts/diagnose_alpha_toroidal_correction.py" \
    --run-dir "$RUN_DIR" \
    --case-file "$CASE_FILE" \
    --alpha-dir "$ALPHA_DIR" \
    --alpha-fit alpha_fit_L12_M12_N16.npz \
    --output-dir "$OUTPUT_DIR" \
    --s-edge 0.25 \
    --rho-values 0.5 \
    --nu-orders 12 \
    --surface-order 12
