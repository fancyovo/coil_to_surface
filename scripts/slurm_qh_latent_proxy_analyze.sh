#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --job-name=qh-proxy-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
run_root="${RUN_ROOT:?RUN_ROOT is required}"
control_scored="${CONTROL_SCORED:?CONTROL_SCORED is required}"
cd "$project"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg

python scripts/analyze_qh_latent_proxy_optimization.py \
  --optimized-scored "$run_root/scored_cases.jsonl" \
  --control-scored "$control_scored" \
  --selected-latents "$run_root/selected_optimized_latents.npz" \
  --output-dir "$run_root"
