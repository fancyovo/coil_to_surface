#!/usr/bin/env bash
#SBATCH --job-name=vqs-concurrency
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/home/scc/pb24511935/volume_score_logs/%x-%j.out
#SBATCH --error=/home/scc/pb24511935/volume_score_logs/%x-%j.err

set -euo pipefail

source "$HOME/coil/.venv/bin/activate"
cd "$HOME/local_surface_evaluator"

dataset_dir="${DATASET_DIR:-$HOME/local_surface_evaluator_data/volume_score_2000}"
output_root="${OUTPUT_ROOT:-$HOME/local_surface_evaluator_runs/volume_score_concurrency}"
mkdir -p "$output_root" "$HOME/volume_score_logs"

for workers_per_gpu in 1 2 3; do
  python scripts/score_volume_qs_parallel.py \
    --dataset-dir "$dataset_dir" \
    --output-dir "$output_root/wpg_${workers_per_gpu}" \
    --split calibration \
    --gpu-count 4 \
    --workers-per-gpu "$workers_per_gpu" \
    --limit "${BENCHMARK_SAMPLES:-160}" \
    --fresh
done
