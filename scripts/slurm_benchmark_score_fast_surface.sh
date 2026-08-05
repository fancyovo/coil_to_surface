#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=score-fast-cal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=anode02
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT must name the score-fast worktree}"
case_dir="${CASE_DIR:-$HOME/local_surface_evaluator_data/volume_score_2000/cases}"
metadata="${METADATA:-$HOME/local_surface_evaluator_data/volume_score_2000/metadata_selected.json}"
total_samples="${TOTAL_SAMPLES:-128}"
split="${SPLIT:-calibration}"
variant_profile="${VARIANT_PROFILE:-matrix}"
output_dir="${OUTPUT_DIR:-$project/runs/score_fast_continuation/calibration_${SLURM_JOB_ID}}"
build_dir="${BUILD_DIR:-gpu_backend/build_score_fast}"
children=()

cleanup() {
    status=$?
    if (( ${#children[@]} )); then
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits > "$output_dir/gpu_postflight.csv" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

cd "$project"
mkdir -p "$output_dir"
mapfile -t compute_processes < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
    printf 'allocated GPUs are not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
    exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$output_dir/gpu_preflight.csv"

module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
source "$HOME/coil/.venv/bin/activate"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
sha256sum "$build_dir/libstellarator_gpu.so" > "$output_dir/library.sha256"

started=$(date +%s.%N)
for worker in 0 1 2 3; do
    python scripts/benchmark_score_fast_surface.py \
        --case-dir "$case_dir" \
        --metadata "$metadata" \
        --lib "$build_dir/libstellarator_gpu.so" \
        --output "$output_dir/worker_${worker}.jsonl" \
        --split "$split" \
        --variant-profile "$variant_profile" \
        --total-limit "$total_samples" \
        --worker-index "$worker" \
        --worker-count 4 \
        --device "$worker" \
        > "$output_dir/worker_${worker}.summary.json" &
    children+=("$!")
done
status=0
for child in "${children[@]}"; do
    wait "$child" || status=$?
done
children=()
finished=$(date +%s.%N)
printf '{"split":"%s","total_samples":%d,"wall_started":%s,"wall_finished":%s}\n' \
    "$split" "$total_samples" "$started" "$finished" > "$output_dir/job.json"
exit "$status"
