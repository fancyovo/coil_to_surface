#!/usr/bin/env bash
#SBATCH --job-name=summary1-pair-analysis
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

pair_root="${PAIR_ROOT:?PAIR_ROOT is required}"
project="${PROJECT:?PROJECT is required}"
source "${VENV:-$HOME/coil/.venv}/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python "$project/scripts/analyze_summary1_flow_pairs.py" \
  --run-root "$pair_root" \
  --output-dir "$pair_root/analysis"

sha256sum \
  "$pair_root/analysis/summary.json" \
  "$pair_root/analysis/pair_metrics.csv" \
  "$pair_root/analysis/flow_vs_data_optimization.png" \
  > "$pair_root/analysis.sha256"
printf '{"status":"ok","job_id":"%s"}\n' "$SLURM_JOB_ID" \
  > "$pair_root/analysis_done.json"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "$pair_root/analysis_gpu_postflight.txt" || true
fi
ps -u "$USER" -o pid=,ppid=,stat=,comm= | awk '$3 ~ /^Z/ {print}' \
  > "$pair_root/analysis_zombies_postflight.txt"
