#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=score-eval-ncu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT must point to the plain score-eval-compression checkout}"
manifest="${PROFILE_MANIFEST:?PROFILE_MANIFEST must point to fixed high-quality cases}"
lib="${SCORE_LIB:?SCORE_LIB must point to the profiled native library}"
output_dir="${OUTPUT_DIR:-$HOME/local_surface_evaluator_runs/score_eval_ncu_${SLURM_JOB_ID}}"
ncu_bin="${NCU_BIN:-/public/app/cuda/13.0/bin/ncu}"

cleanup() {
    status=$?
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits > "$output_dir/gpu_postflight.csv" 2>/dev/null || true
    ps -u "$USER" -o pid=,ppid=,stat=,comm= | awk '$3 ~ /^Z/ {print}' \
        > "$output_dir/zombies_postflight.txt" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$output_dir"
cd "$project"
assigned="${CUDA_VISIBLE_DEVICES%%,*}"
if nvidia-smi -i "$assigned" --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "allocated GPU is not idle" >&2
    exit 42
fi
nvidia-smi -i "$assigned" --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$output_dir/gpu_preflight.csv"

module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
source "$HOME/coil/.venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

profile_kernel() {
    local name="$1"
    local expression="$2"
    local skip="$3"
    local roofline="$4"
    "$ncu_bin" --target-processes all --kernel-name-base function \
        --kernel-name "regex:${expression}" --launch-skip "$skip" --launch-count 1 \
        --section SpeedOfLight --section ComputeWorkloadAnalysis --section "$roofline" \
        --csv --log-file "$output_dir/${name}.csv" \
        python scripts/benchmark_score_eval_hinted.py \
            --manifest "$manifest" --lib "$lib" \
            --output "$output_dir/${name}_call.jsonl" --device 0 \
            --variants baseline --case-limit 1 --repeats 1 --warmups 1
}

profile_kernel axis_fp64 'trace_period_blockline_kernel' 1 \
    SpeedOfLight_HierarchicalDoubleRooflineChart
profile_kernel psi_assemble_fp32 'psi_fill_matrix_kernel_f32' 1 \
    SpeedOfLight_HierarchicalSingleRooflineChart
profile_kernel surface_mixed 'trace_period_blockline_bf32_state64_kernel' 3 \
    SpeedOfLight_HierarchicalSingleRooflineChart
profile_kernel axis_samples_mixed 'trace_axis_samples_mixed_kernel' 1 \
    SpeedOfLight_HierarchicalSingleRooflineChart

printf '{"status":"ok","job_id":"%s","output_dir":"%s"}\n' \
    "$SLURM_JOB_ID" "$output_dir" > "$output_dir/done.json"
