#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=standard-bz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${CASE_FILE:?CASE_FILE is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${SURFACE_NPZ:?SURFACE_NPZ is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

eval_env=${EVAL_ENV:-$HOME/local_surface_evaluator/.venv-desc016-py312}
gpu_lib=${GPU_LIB:-$HOME/local_surface_evaluator/gpu_backend/build_mixed/libstellarator_gpu.so}
gpu_selector=${CUDA_VISIBLE_DEVICES:-}

cleanup() {
    status=$?
    trap - EXIT INT TERM
    mapfile -t children < <(jobs -pr)
    if (( ${#children[@]} )); then
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
    if [[ -n "$gpu_selector" && -d "$OUTPUT_DIR" ]]; then
        nvidia-smi --id="$gpu_selector" \
            --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits > "$OUTPUT_DIR/gpu_postflight.csv" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

for path in "$PROJECT" "$CASE_FILE" "$RUN_DIR" "$SURFACE_NPZ" "$OUTPUT_DIR" "$eval_env" "$gpu_lib"; do
    resolved=$(realpath -m "$path")
    [[ $resolved == "$HOME"/* ]] || {
        printf 'path must stay under HOME: %s\n' "$resolved" >&2
        exit 2
    }
done
test -f "$CASE_FILE"
test -d "$RUN_DIR"
test -f "$SURFACE_NPZ"
test -d "$eval_env"
test -f "$gpu_lib"
git -C "$PROJECT" diff --quiet
git -C "$PROJECT" diff --cached --quiet

if [[ -e "$OUTPUT_DIR" ]]; then
    printf 'refusing to overwrite existing output directory: %s\n' "$OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"
: "${gpu_selector:?CUDA_VISIBLE_DEVICES is required}"
mapfile -t compute_processes < <(
    nvidia-smi --id="$gpu_selector" \
        --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
    printf 'the allocated GPU is not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
    exit 42
fi
nvidia-smi --id="$gpu_selector" \
    --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$OUTPUT_DIR/gpu_preflight.csv"
git -C "$PROJECT" rev-parse HEAD > "$OUTPUT_DIR/code_commit.txt"

source "$eval_env/bin/activate"
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PROJECT:$eval_env/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

cd /
python3 "$PROJECT/scripts/solve_boozer_from_alpha_nu.py" \
    --case-file "$CASE_FILE" \
    --run-dir "$RUN_DIR" \
    --surface-npz "$SURFACE_NPZ" \
    --output-dir "$OUTPUT_DIR" \
    --ls-maxiter "${LS_MAXITER:-100}" \
    --newton-maxiter "${NEWTON_MAXITER:-30}" \
    --gpu-lib "$gpu_lib" \
    --gpu-device 0 \
    --validation-field-precision fp32
