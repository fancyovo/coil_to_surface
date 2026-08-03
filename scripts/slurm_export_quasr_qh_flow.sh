#!/usr/bin/env bash
#SBATCH --partition=amd96c
#SBATCH --job-name=qh-flow-export
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo="${QH_FLOW_REPO:-$HOME/local_surface_evaluator}"
output="${QH_FLOW_EXPORT_OUTPUT:-$repo/runs/qh_flow_export_v1}"
metadata="${QH_FLOW_METADATA:-$HOME/stellarator_gpu_eval/quasr_private/QUASR_08072024_meta.csv}"
quasr_root="${QH_FLOW_QUASR_ROOT:-/data/zhouyebi/QUASR_08072024}"

cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 scripts/export_quasr_qh_flow.py \
  --quasr-root "$quasr_root" \
  --metadata "$metadata" \
  --output-dir "$output" \
  --workers "${SLURM_CPUS_PER_TASK:-16}" \
  --shard-size 8192
