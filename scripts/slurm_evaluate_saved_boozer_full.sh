#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=boozer-full-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${CASE_FILE:?CASE_FILE is required}"
: "${SURFACE_NPZ:?SURFACE_NPZ is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

project="${PROJECT:-$HOME/local_surface_evaluator}"
eval_env=${EVAL_ENV:-$project/.venv-desc016-py312}

cleanup() {
    status=$?
    trap - EXIT INT TERM
    mapfile -t children < <(jobs -pr)
    if (( ${#children[@]} )); then
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits > "$OUTPUT_DIR/gpu_postflight.csv" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$OUTPUT_DIR"
cd "$project"
mapfile -t compute_processes < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
    printf 'the allocated GPU is not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
    exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$OUTPUT_DIR/gpu_preflight.csv"

export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$project:$eval_env/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

desc_gpu_args=()
if [[ ${REQUIRE_DESC_GPU:-1} == 1 ]]; then
    desc_gpu_args+=(--require-desc-gpu)
fi
surface_order_args=()
if [[ -n ${SURFACE_ORDER:-} ]]; then
    surface_order_args+=(--surface-order "$SURFACE_ORDER")
fi

python3 "$project/scripts/evaluate_saved_boozer_surface_full.py" \
    --case-file "$CASE_FILE" \
    --surface-npz "$SURFACE_NPZ" \
    --output-dir "$OUTPUT_DIR" \
    --poincare-nfieldlines "${POINCARE_NFIELDLINES:-8}" \
    --poincare-tmax "${POINCARE_TMAX:-2e7}" \
    --desc-maxiter "${DESC_MAXITER:-50}" \
    "${surface_order_args[@]}" \
    "${desc_gpu_args[@]}"
