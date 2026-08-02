#!/usr/bin/env bash
set -euo pipefail

project="${PROJECT:-$(git rev-parse --show-toplevel)}"
exec sbatch \
  --chdir="$project" \
  --export="ALL,PROJECT=$project" \
  "$project/scripts/slurm_train_qh_score_regressor.sh"
