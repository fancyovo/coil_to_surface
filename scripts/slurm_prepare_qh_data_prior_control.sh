#!/usr/bin/env bash
#SBATCH --job-name=qh-data-prior-prep
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
project="${PROJECT:?PROJECT is required}"
source "${VENV:-$HOME/coil/.venv}/bin/activate"
cd "$project"
mkdir -p logs
python scripts/prepare_qh_data_prior_control.py \
  --reference-root "${REFERENCE_ROOT:?REFERENCE_ROOT is required}" \
  --run-root "${RUN_ROOT:?RUN_ROOT is required}" \
  --checkpoint "${CHECKPOINT:?CHECKPOINT is required}" \
  --lib "${NATIVE_LIB:?NATIVE_LIB is required}" \
  --worker-count "${WORKER_COUNT:-4}" \
  --case-count "${CASE_COUNT:-0}" \
  --selection-seed "${SELECTION_SEED:-2026082403}"
