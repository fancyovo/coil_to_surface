#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${EXPERIMENT_ROOT:?EXPERIMENT_ROOT is required}"
: "${RESIDUE_START:?RESIDUE_START is required}"
: "${RESIDUE_COUNT:?RESIDUE_COUNT is required}"
: "${POOL_NAME:?POOL_NAME is required}"

eval_env=${EVAL_ENV:-$HOME/local_surface_evaluator/.venv-desc016-py312}
source "$eval_env/bin/activate"
export PYTHONPATH="$PROJECT:$eval_env/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

python "$PROJECT/scripts/run_qh_face_qs_cpu_pool.py" \
  --experiment-root "$EXPERIMENT_ROOT" \
  --workers "$SLURM_CPUS_PER_TASK" \
  --residue-start "$RESIDUE_START" \
  --residue-count "$RESIDUE_COUNT" \
  --residue-modulus 5 \
  --pool-name "$POOL_NAME"
