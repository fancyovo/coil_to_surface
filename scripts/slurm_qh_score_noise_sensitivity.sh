#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-score-noise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project=/home/scc/pb24511935/local_surface_evaluator
data=${QH_DATA:-$project/../local_surface_evaluator_data/quasr_qh_flow_v1}
checkpoint=${QH_CHECKPOINT:-$project/runs/qh_flow_base_28494/checkpoints/step_00045000.pt}
lib=${SCORE_LIB:-$project/gpu_backend/build_native_score/libstellarator_gpu.so}
output=${OUTPUT_DIR:-$project/runs/qh_score_noise_${SLURM_JOB_ID}}
source_ids=${SOURCE_IDS:-1446077,1826200,2419096}
sigmas=${NOISE_SIGMAS:-0.001,0.003,0.01,0.03,0.1,0.3,1.0}
replicates=${REPLICATES:-12}
world_size=4
children=()

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if (( ${#children[@]} )); then
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used \
        --format=csv,noheader,nounits > "$output/gpu_postflight.csv" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

cd "$project"
module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
source /home/scc/pb24511935/coil/.venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

test -f "$data/manifest.json"
test -f "$checkpoint"
test -f "$lib"
mapfile -t compute_processes < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
    printf 'an allocated GPU is not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
    exit 42
fi

python "$project/scripts/qh_score_noise_sensitivity.py" \
    --data-dir "$data" \
    --checkpoint "$checkpoint" \
    --output-dir "$output" \
    --lib "$lib" \
    --source-ids "$source_ids" \
    --sigmas "$sigmas" \
    --replicates "$replicates" \
    --prepare-only
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used \
    --format=csv,noheader,nounits > "$output/gpu_preflight.csv"

for rank in 0 1 2 3; do
    python "$project/scripts/qh_score_noise_sensitivity.py" \
        --data-dir "$data" \
        --checkpoint "$checkpoint" \
        --output-dir "$output" \
        --lib "$lib" \
        --source-ids "$source_ids" \
        --sigmas "$sigmas" \
        --replicates "$replicates" \
        --rank "$rank" \
        --world-size "$world_size" \
        > "$output/rank_$(printf '%02d' "$rank").log" 2>&1 &
    children+=("$!")
done
for child in "${children[@]}"; do
    wait "$child"
done
children=()

cd /
python "$project/scripts/qh_score_noise_sensitivity.py" \
    --data-dir "$data" \
    --checkpoint "$checkpoint" \
    --output-dir "$output" \
    --lib "$lib" \
    --source-ids "$source_ids" \
    --sigmas "$sigmas" \
    --replicates "$replicates" \
    --analyze-only
