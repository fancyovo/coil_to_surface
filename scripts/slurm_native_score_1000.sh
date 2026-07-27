#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=native-score-1000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --exclude=anode02
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project=/home/scc/pb24511935/local_surface_evaluator
case_dir=/home/scc/pb24511935/local_surface_evaluator_data/volume_score_2000/cases
metadata=/home/scc/pb24511935/local_surface_evaluator_data/volume_score_2000/metadata_selected.json
split=${SPLIT:-calibration}
total_samples=${TOTAL_SAMPLES:-1000}
workers=4
output_dir="$project/runs/native_score/${split}_${total_samples}_${SLURM_JOB_ID}"
children=()

cleanup() {
    if (( ${#children[@]} )); then
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

cd "$project"
module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
mapfile -t compute_processes < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
    printf 'an allocated GPU is not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
    exit 42
fi

source /home/scc/pb24511935/coil/.venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$output_dir"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$output_dir/gpu_preflight.csv"

started=$(date +%s.%N)
for ((worker=0; worker<workers; ++worker)); do
    python scripts/batch_native_score.py \
        --case-dir "$case_dir" \
        --metadata "$metadata" \
        --split "$split" \
        --output "$output_dir/worker_${worker}.jsonl" \
        --device "$worker" \
        --worker-index "$worker" \
        --worker-count "$workers" \
        --total-limit "$total_samples" \
        --warmup \
        > "$output_dir/worker_${worker}.summary.json" &
    children+=("$!")
done

status=0
for child in "${children[@]}"; do
    wait "$child" || status=$?
done
children=()
finished=$(date +%s.%N)
printf '{"workers":%d,"total_samples":%d,"split":"%s","wall_started":%s,"wall_finished":%s}\n' \
    "$workers" "$total_samples" "$split" "$started" "$finished" > "$output_dir/job.json"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$output_dir/gpu_postflight.csv"
exit "$status"
