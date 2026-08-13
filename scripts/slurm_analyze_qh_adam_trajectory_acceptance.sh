#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh_traj_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
dataset_root="${DATASET_ROOT:?DATASET_ROOT is required}"
rescore_dir="${RESCORE_DIR:?RESCORE_DIR is required}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR is required}"

cd "$project"
source "$HOME/coil/.venv/bin/activate"
export MPLBACKEND=Agg
python scripts/analyze_qh_adam_trajectory_acceptance.py \
  "$dataset_root" "$rescore_dir" "$output_dir"
