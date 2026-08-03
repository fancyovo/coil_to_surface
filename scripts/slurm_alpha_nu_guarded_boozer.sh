#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=alpha-nu-bz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${CASE_FILE:?CASE_FILE is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

project=${PROJECT:-/home/scc/pb24511935/local_surface_evaluator}
s_edge=${S_EDGE:-0.25}
alpha_train_points=${ALPHA_TRAIN_POINTS:-120000}
alpha_validation_points=${ALPHA_VALIDATION_POINTS:-60000}
alpha_grid_xy=${ALPHA_GRID_XY:-144}
alpha_sampling_backend=${ALPHA_SAMPLING_BACKEND:-gpu-ray}
alpha_min_candidate_valid_fraction=${ALPHA_MIN_CANDIDATE_VALID_FRACTION:-0.0}
eval_env=${EVAL_ENV:-$project/.venv-desc016-py312}
gpu_lib=${GPU_LIB:-$project/gpu_backend/build_mixed/libstellarator_gpu.so}
gpu_selector=${CUDA_VISIBLE_DEVICES:-}

[[ $alpha_sampling_backend == gpu-ray || $alpha_sampling_backend == legacy-cartesian ]] || {
    printf 'ALPHA_SAMPLING_BACKEND must be gpu-ray or legacy-cartesian\n' >&2
    exit 2
}
[[ $alpha_grid_xy =~ ^[1-9][0-9]*$ ]] || {
    printf 'ALPHA_GRID_XY must be a positive integer\n' >&2
    exit 2
}

cleanup() {
    status=$?
    trap - EXIT INT TERM
    mapfile -t children < <(jobs -pr)
    if (( ${#children[@]} )); then
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
    if [[ -n "$gpu_selector" ]]; then
        nvidia-smi --id="$gpu_selector" \
            --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits > "$OUTPUT_DIR/gpu_postflight.csv" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

cd "$project"
mkdir -p "$OUTPUT_DIR"
test -f "$gpu_lib"
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

alpha_dir="$OUTPUT_DIR/alpha"
nu_dir="$OUTPUT_DIR/alpha_nu"
cd /
python3 "$project/scripts/alpha_clebsch_ls_experiment.py" \
    --run-dir "$RUN_DIR" \
    --case-file "$CASE_FILE" \
    --s-edge "$s_edge" \
    --out-dir "$alpha_dir" \
    --orders 12:12:16 \
    --iota-degree 0 \
    --train-points "$alpha_train_points" \
    --validation-points "$alpha_validation_points" \
    --grid-xy "$alpha_grid_xy" \
    --minimum-candidate-valid-fraction "$alpha_min_candidate_valid_fraction" \
    --sampling-backend "$alpha_sampling_backend" \
    --precision fp32 \
    --gpu-lib "$gpu_lib" \
    --skip-fieldline-plot

cd /
python3 "$project/scripts/diagnose_alpha_toroidal_correction.py" \
    --run-dir "$RUN_DIR" \
    --case-file "$CASE_FILE" \
    --alpha-dir "$alpha_dir" \
    --alpha-fit alpha_fit_L12_M12_N16.npz \
    --output-dir "$nu_dir" \
    --s-edge "$s_edge" \
    --rho-values 0.5,0.8,1.0 \
    --nu-orders 12 \
    --surface-order 12 \
    --gpu-lib "$gpu_lib" \
    --gpu-device 0 \
    --field-precision fp32 \
    --save-surfaces

cd /
guard_status=0
python3 "$project/scripts/guarded_boozer_from_alpha_nu.py" \
    --case-file "$CASE_FILE" \
    --run-dir "$RUN_DIR" \
    --surface-npz "$nu_dir/surfaces/rho_1_alpha_nu.npz" \
    --output-dir "$OUTPUT_DIR/guarded_rho_1" \
    --gpu-lib "$gpu_lib" \
    --gpu-device 0 \
    --validation-field-precision fp32 || guard_status=$?
printf '%s\n' "$guard_status" > "$OUTPUT_DIR/guarded_exit_code.txt"
if (( guard_status != 0 && guard_status != 3 )); then
    printf 'guarded diagnostic failed unexpectedly with exit %s\n' "$guard_status" >&2
    exit "$guard_status"
fi

cd /
python3 "$project/scripts/solve_boozer_from_alpha_nu.py" \
    --case-file "$CASE_FILE" \
    --run-dir "$RUN_DIR" \
    --surface-npz "$nu_dir/surfaces/rho_1_alpha_nu.npz" \
    --output-dir "$OUTPUT_DIR/standard_rho_1" \
    --ls-maxiter "${LS_MAXITER:-100}" \
    --newton-maxiter "${NEWTON_MAXITER:-30}" \
    --gpu-lib "$gpu_lib" \
    --gpu-device 0 \
    --validation-field-precision fp32
