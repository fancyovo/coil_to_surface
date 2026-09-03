#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --job-name=axisrl-teacher-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo="${AXIS_RL_REPO:?set AXIS_RL_REPO}"
dataset="${AXIS_RL_DATASET:?set AXIS_RL_DATASET}"
commit="${AXIS_RL_COMMIT:?set AXIS_RL_COMMIT}"
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"

python scripts/generate_axisflip_prior_dataset.py finalize \
  --output-dir "$dataset" \
  --shard-name p107_000000_128000 \
  --shard-name students_128000_200000 \
  --expected-total 200000 \
  --expected-commit "$commit"
