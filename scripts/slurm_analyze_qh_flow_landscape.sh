#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --job-name=qh-landscape-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator}"
data="${QH_DATA:-$HOME/local_surface_evaluator_data/quasr_qh_flow_v1}"
checkpoint="${FLOW_CHECKPOINT:-$project/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
lib="${SCORE_LIB:-$project/gpu_backend/build_native_score/libstellarator_gpu.so}"
output="${OUTPUT_DIR:?OUTPUT_DIR must contain completed landscape score files}"

cd "$project"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

python "$project/scripts/qh_flow_landscape.py" \
  --data-dir "$data" --checkpoint "$checkpoint" --output-dir "$output" \
  --lib "$lib" --analyze-only
