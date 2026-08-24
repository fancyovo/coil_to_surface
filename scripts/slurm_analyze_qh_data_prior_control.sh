#!/usr/bin/env bash
#SBATCH --job-name=qh-data-prior-analysis
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
project="${PROJECT:?PROJECT is required}"
source "${VENV:-$HOME/coil/.venv}/bin/activate"
cd "$project"
mkdir -p logs
python scripts/analyze_qh_data_prior_control.py \
  --run-root "${RUN_ROOT:?RUN_ROOT is required}" \
  --output-dir "${OUTPUT_DIR:-${RUN_ROOT}/analysis}"
