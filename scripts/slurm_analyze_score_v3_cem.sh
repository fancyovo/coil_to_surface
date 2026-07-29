#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=cem-v3-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project=/home/scc/pb24511935/local_surface_evaluator
run_dir=${RUN_DIR:?RUN_DIR must point to a completed CEM target directory}
output_dir=${OUTPUT_DIR:-$run_dir/audit}

cd "$project"
source /home/scc/pb24511935/coil/.venv/bin/activate
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

python scripts/analyze_score_v3_cem_run.py \
    --summary "$run_dir/summary.json" \
    --candidates "$run_dir/candidates.jsonl" \
    --output-dir "$output_dir"
