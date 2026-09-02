#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=axisrl-teacher-p107
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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
  --shard-name p107_000000_080000 \
  --seed-start 203609030000 \
  --count 80000 \
  --workers 16 \
  --block-size 128 \
  --expected-commit "$commit"
