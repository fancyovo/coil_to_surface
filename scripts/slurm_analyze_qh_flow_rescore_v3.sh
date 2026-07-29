#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-flow-v3-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project=/home/scc/pb24511935/local_surface_evaluator
input_dir=${INPUT_DIR:-$project/runs/qh_flow_eval_28546}
output_dir=${OUTPUT_DIR:?OUTPUT_DIR must contain completed score-v3 rank files}
lib=${SCORE_LIB:-$project/gpu_backend/build_native_score/libstellarator_gpu.so}

cd /
source /home/scc/pb24511935/coil/.venv/bin/activate
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

for rank in 0 1 2 3; do
    test -s "$output_dir/rescore_rank_$(printf '%02d' "$rank").jsonl"
    test -s "$output_dir/runtime_rank_$(printf '%02d' "$rank").json"
done

python "$project/scripts/rescore_qh_flow_saved.py" \
    --input-dir "$input_dir" \
    --output-dir "$output_dir" \
    --lib "$lib" \
    --analyze-only
