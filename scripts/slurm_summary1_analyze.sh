#!/usr/bin/env bash
#SBATCH --job-name=summary1-analyze
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

run_root="${RUN_ROOT:?RUN_ROOT is required}"
project="${PROJECT:-$run_root/source}"
source "${VENV:-$HOME/coil/.venv}/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python "$project/scripts/analyze_summary1_evaluator_modes.py" \
  --input-dir "$run_root/evaluator_modes/raw" \
  --output-dir "$run_root/evaluator_modes/analysis"
python "$project/scripts/analyze_summary1_flow_pairs.py" \
  --run-root "$run_root/flow_pairs" \
  --output-dir "$run_root/flow_pairs/analysis"

sha256sum \
  "$run_root/evaluator_modes/analysis/summary.json" \
  "$run_root/flow_pairs/analysis/summary.json" \
  > "$run_root/analysis.sha256"
printf '{"status":"ok","job_id":"%s"}\n' "$SLURM_JOB_ID" \
  > "$run_root/analysis_done.json"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "$run_root/logs/analysis_gpu_postflight.txt" || true
fi
ps -eo stat,pid,ppid,cmd | awk '$1 ~ /^Z/ {print}' \
  > "$run_root/logs/analysis_zombies_postflight.txt"
