#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=quasr-qh-atlas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
data_dir="${DATA_DIR:?DATA_DIR is required}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR is required}"
expected_commit="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
source "${VENV:-$HOME/coil/.venv}/bin/activate"
cd "$project"
test "$(git rev-parse HEAD)" = "$expected_commit"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$data_dir/manifest.json"
test ! -e "$output_dir"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg

python scripts/analyze_quasr_qh_structure.py \
  --data-dir "$data_dir" \
  --output-dir "$output_dir" \
  --verify-hashes \
  --curve-samples "${CURVE_SAMPLES:-96}" \
  --fit-count "${FIT_COUNT:-5000}" \
  --silhouette-count "${SILHOUETTE_COUNT:-600}" \
  --gallery-group-count "${GALLERY_GROUP_COUNT:-8}"
