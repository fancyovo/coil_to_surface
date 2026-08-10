#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=axis-no-fp64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT must point to the score-eval-compression checkout}"
manifest="${MANIFEST:-$project/reports/assets/qh_psi_grid_reduction_20260810/cases.json}"
output_dir="${OUTPUT_DIR:-$HOME/local_surface_evaluator_runs/axis_no_fp64_${SLURM_JOB_ID}}"
build_dir="$project/gpu_backend/build_axis_no_fp64_cuda13"

cleanup() {
    status=$?
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits > "$output_dir/gpu_postflight.csv" 2>/dev/null || true
    ps -u "$USER" -o pid=,ppid=,stat=,comm= | awk '$3 ~ /^Z/ {print}' \
        > "$output_dir/zombies_postflight.txt" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$output_dir" "$project/logs"
cd "$project"

assigned="${CUDA_VISIBLE_DEVICES%%,*}"
if nvidia-smi -i "$assigned" --query-compute-apps=pid --format=csv,noheader,nounits |
        grep -q '[0-9]'; then
    echo "allocated GPU is not idle" >&2
    exit 42
fi
nvidia-smi -i "$assigned" --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$output_dir/gpu_preflight.csv"

module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
source "$HOME/coil/.venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cmake -S gpu_backend -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build "$build_dir" -j4
lib="$build_dir/libstellarator_gpu.so"

python scripts/benchmark_score_eval_hinted.py \
    --manifest "$manifest" --lib "$lib" --output "$output_dir/benchmark.jsonl" \
    --device 0 \
    --variants baseline axis_hint_no_fp64 \
        axis_hint_fp64_offset1e3 axis_hint_no_fp64_offset1e3 \
    --repeats 2 --warmups 1
python scripts/analyze_axis_hint_no_fp64.py \
    --input "$output_dir/benchmark.jsonl" --output-dir "$output_dir/analysis"

sha256sum "$lib" > "$output_dir/library.sha256"
git rev-parse HEAD > "$output_dir/git_head.txt"
git status --short > "$output_dir/git_status.txt"
printf '{"status":"ok","job_id":"%s","output_dir":"%s"}\n' \
    "$SLURM_JOB_ID" "$output_dir" > "$output_dir/done.json"
