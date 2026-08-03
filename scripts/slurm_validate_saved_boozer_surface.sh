#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --job-name=boozer-offgrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${CASE_FILE:?CASE_FILE must point to an evaluator case JSON}"
: "${SURFACE_NPZ:?SURFACE_NPZ must point to a saved Boozer surface}"
: "${OUTPUT:?OUTPUT must point to the validation JSON}"

project=/home/scc/pb24511935/local_surface_evaluator
eval_env=${EVAL_ENV:-$project/.venv-desc016-py312}

cd "$project"
source "$eval_env/bin/activate"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/validate_saved_boozer_surface.py \
    --case-file "$CASE_FILE" \
    --surface-npz "$SURFACE_NPZ" \
    --output "$OUTPUT"
