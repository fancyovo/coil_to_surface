#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --job-name=axisrl-teacher-stu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo="${AXIS_RL_REPO:?set AXIS_RL_REPO}"
dataset="${AXIS_RL_DATASET:?set AXIS_RL_DATASET}"
commit="${AXIS_RL_COMMIT:?set AXIS_RL_COMMIT}"
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/generate_axisflip_prior_dataset.py generate \
  --output-dir "$dataset" \
  --shard-name students_128000_200000 \
  --seed-start 204609168000 \
  --count 72000 \
  --workers 8 \
  --block-size 128 \
  --expected-commit "$commit"
