#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-face-select
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${EXPERIMENT_ROOT:?EXPERIMENT_ROOT is required}"

trajectory_count=${TRAJECTORY_COUNT:-96}
iterations=${ITERATIONS:-0,10,25,50,75,100,150,200}
summary_csv=${TRAJECTORY_SUMMARY:-$PROJECT/reports/assets/qh_adam_trajectory_dataset_pilot_20260813/trajectory_summary.csv}
eval_env=${EVAL_ENV:-$HOME/local_surface_evaluator/.venv-desc016-py312}

source "$eval_env/bin/activate"
export PYTHONPATH="$PROJECT:$eval_env/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

extra_args=()
if [[ "${INCLUDE_SCORE_BEST:-0}" == "1" ]]; then
  extra_args+=(--include-score-best)
fi

python "$PROJECT/scripts/prepare_qh_trajectory_face_qs_cases.py" \
  --dataset-root "$DATASET_ROOT" \
  --trajectory-summary "$summary_csv" \
  --output-root "$EXPERIMENT_ROOT" \
  --trajectory-count "$trajectory_count" \
  --iterations "$iterations" \
  --fixed-probe-rho 0.8 \
  --source-a 0.05 \
  "${extra_args[@]}"
