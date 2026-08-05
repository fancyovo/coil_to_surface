#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=score-mode-cross
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
lib="${SCORE_LIB:?SCORE_LIB is required}"
expected_sha="${EXPECTED_SCORE_LIB_SHA:?EXPECTED_SCORE_LIB_SHA is required}"
old_case="${OLD_CASE:?OLD_CASE is required}"
new_case="${NEW_CASE:?NEW_CASE is required}"
output="${OUTPUT:?OUTPUT is required}"
gpu_selector="${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"

cleanup() {
    status=$?
    mkdir -p "$(dirname "$output")"
    nvidia-smi --id="$gpu_selector" \
        --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits > "${output%.json}_gpu_postflight.csv" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

cd "$project"
test -f "$lib"
test -f "$old_case"
test -f "$new_case"
test "$(sha256sum "$lib" | awk '{print $1}')" = "$expected_sha"
mkdir -p "$(dirname "$output")"

for _ in 1 2 3; do
    test "$(nvidia-smi --id="$gpu_selector" --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' ')" = "0"
    test "$(nvidia-smi --id="$gpu_selector" --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d ' ')" -le 16
    sleep 2
done
nvidia-smi --id="$gpu_selector" \
    --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "${output%.json}_gpu_preflight.csv"

module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
source "$HOME/coil/.venv/bin/activate"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/compare_native_score_modes.py \
    --lib "$lib" \
    --output "$output" \
    --device 0 \
    "$old_case" "$new_case"
