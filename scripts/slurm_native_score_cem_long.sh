#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=native-cem-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=03:30:00
#SBATCH --exclude=anode02
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project=/home/scc/pb24511935/local_surface_evaluator
pca=/home/scc/pb24511935/coil/checkpoints/pca_ae/pca_ae.pt
lib="$project/gpu_backend/build_native_score/libstellarator_gpu.so"
run_root="$project/runs/native_score_cem_long/${SLURM_JOB_ID}"
targets=${TARGETS:-QH,QA}
iterations=${ITERATIONS:-32}
popsize=${POPSIZE:-128}
gpu_ids=${GPU_IDS:-0,0,1,1,2,2,3,3}
batch_timeout_s=${BATCH_TIMEOUT_S:-900}
children=()

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if (( ${#children[@]} )); then
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits > "$run_root/gpu_postflight.csv" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

cd "$project"
module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$run_root"
mapfile -t compute_processes < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
    printf 'an allocated GPU is not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
    exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$run_root/gpu_preflight.csv"

source /home/scc/pb24511935/coil/.venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

started=$(date +%s.%N)
printf '{"targets":"%s","iterations":%d,"popsize":%d,"elite":%d,"gpu_ids":"%s","started":%s}\n' \
    "$targets" "$iterations" "$popsize" "$((popsize / 4))" "$gpu_ids" "$started" \
    > "$run_root/job.json"

IFS=',' read -r -a target_list <<< "$targets"
for target in "${target_list[@]}"; do
    case "$target" in
        QH) seed=${SEED_QH:-2026072801} ;;
        QA) seed=${SEED_QA:-2026072802} ;;
        *) printf 'unsupported target: %s\n' "$target" >&2; exit 2 ;;
    esac
    python scripts/optimize_native_score_cem.py \
        --pca "$pca" \
        --lib "$lib" \
        --out-dir "$run_root/${target,,}" \
        --target "$target" \
        --nfp 3 \
        --n-base-coils 1 \
        --iterations "$iterations" \
        --popsize "$popsize" \
        --elite $((popsize / 4)) \
        --gpus "$gpu_ids" \
        --batch-timeout-s "$batch_timeout_s" \
        --seed "$seed" &
    children=("$!")
    wait "${children[0]}"
    children=()
done

finished=$(date +%s.%N)
printf '{"targets":"%s","iterations":%d,"popsize":%d,"elite":%d,"gpu_ids":"%s","started":%s,"finished":%s}\n' \
    "$targets" "$iterations" "$popsize" "$((popsize / 4))" "$gpu_ids" "$started" "$finished" \
    > "$run_root/job.json"
