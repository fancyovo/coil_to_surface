#!/usr/bin/env bash
#SBATCH --job-name=vqs-1000
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/home/scc/pb24511935/volume_score_logs/%x-%j.out
#SBATCH --error=/home/scc/pb24511935/volume_score_logs/%x-%j.err

set -euo pipefail

source "$HOME/coil/.venv/bin/activate"
cd "$HOME/local_surface_evaluator"

dataset_dir="${DATASET_DIR:-$HOME/local_surface_evaluator_data/volume_score_2000}"
split="${SPLIT:-validation}"
workers_per_gpu="${WORKERS_PER_GPU:-2}"
output_dir="${OUTPUT_DIR:-$HOME/local_surface_evaluator_runs/volume_score_${split}_1000}"
mkdir -p "$output_dir" "$HOME/volume_score_logs"

python scripts/score_volume_qs_parallel.py \
  --dataset-dir "$dataset_dir" \
  --output-dir "$output_dir" \
  --split "$split" \
  --gpu-count 4 \
  --workers-per-gpu "$workers_per_gpu"

python scripts/analyze_volume_score.py \
  --batch-summary "$output_dir/batch_summary.json" \
  --output-dir "$output_dir/analysis"
