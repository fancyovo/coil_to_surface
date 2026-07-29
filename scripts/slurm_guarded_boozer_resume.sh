#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=guarded-bz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${CASE_FILE:?CASE_FILE is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${SOURCE_DIR:?SOURCE_DIR is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

project=/home/scc/pb24511935/local_surface_evaluator
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

cd /
mkdir -p "$OUTPUT_DIR"
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
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

for rho in 0p5 0p8 1; do
    cd /
    python3 "$project/scripts/guarded_boozer_from_alpha_nu.py" \
        --case-file "$CASE_FILE" \
        --run-dir "$RUN_DIR" \
        --surface-npz "$SOURCE_DIR/alpha_nu/surfaces/rho_${rho}_alpha_nu.npz" \
        --output-dir "$OUTPUT_DIR/guarded_rho_${rho}"
done

cd /
python3 "$project/scripts/plot_poincare_validation.py" \
    --case-file "$CASE_FILE" \
    --surface-npz "$OUTPUT_DIR/guarded_rho_1/boozer_guarded.npz" \
    --output "$OUTPUT_DIR/poincare_guarded_boozer_rho1.png" \
    --mpol 12 \
    --ntor 12 \
    --nfieldlines 16 \
    --marker-size 5 \
    --tol 1e-11 \
    > "$OUTPUT_DIR/poincare_guarded_boozer_rho1.log"
