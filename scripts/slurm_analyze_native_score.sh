#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=analyze-native-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --exclude=anode02
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project=/home/scc/pb24511935/local_surface_evaluator
metadata=/home/scc/pb24511935/local_surface_evaluator_data/volume_score_2000/metadata_selected.json
: "${RUN_DIR:?RUN_DIR must name a run directory relative to the project}"

cd "$project"
source /home/scc/pb24511935/coil/.venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python scripts/analyze_native_score_batch.py \
    "$project/$RUN_DIR" \
    --metadata "$metadata" \
    --output-dir "$project/$RUN_DIR/analysis"
