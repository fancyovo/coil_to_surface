#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=validate-score-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --exclude=anode02
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project=/home/scc/pb24511935/local_surface_evaluator
lib="$project/gpu_backend/build_native_score/libstellarator_gpu.so"
output_dir=${OUTPUT_DIR:-$project/runs/score_long_horizon_validation/${SLURM_JOB_ID}}

cleanup() {
    status=$?
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits > "$output_dir/gpu_postflight.csv" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

cd "$project"
module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$output_dir"

mapfile -t compute_processes < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
    printf 'allocated GPU is not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
    exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$output_dir/gpu_preflight.csv"

source /home/scc/pb24511935/coil/.venv/bin/activate
printf '{"git":"%s","library_sha256":"%s"}\n' \
    "$(git rev-parse HEAD)" "$(sha256sum "$lib" | cut -d' ' -f1)" > "$output_dir/manifest.json"

names=(
    known_good
    qp_cem_bad
    flow_long_bad
    quasr_qh_2407084
    quasr_qh_1446077
    quasr_qp_like_1551144
)
cases=(
    "$project/reports/assets/native_score_cem_validation/long_qh_cem/best_case.json"
    "$project/reports/assets/native_score_cem_validation/score_v2_three_seed/best_case.json"
    "$project/runs/qh_flow_full_eval_28546/cases/highest_score_id_001439.json"
    "/home/scc/pb24511935/local_surface_evaluator_data/volume_score_2000/cases/id_2407084.json"
    "/home/scc/pb24511935/local_surface_evaluator_data/volume_score_2000/cases/id_1446077.json"
    "/home/scc/pb24511935/local_surface_evaluator_data/volume_score_2000/cases/id_1551144.json"
)
for index in "${!names[@]}"; do
    python scripts/smoke_native_score.py \
        "${cases[$index]}" \
        --lib "$lib" \
        --device 0 \
        --output "$output_dir/${names[$index]}.json"
done
