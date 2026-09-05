#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=full-physical-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${GPU_LIB:?GPU_LIB is required}"
: "${EVAL_ENV:?EVAL_ENV is required}"

if (( $# != 4 )); then
    printf 'usage: %s CASE_FILE OUTPUT_ROOT A_VALUES S_EDGES\n' "$0" >&2
    exit 2
fi

case_file=$1
output_root=$2
a_values=$3
s_edges=$4

cd "$PROJECT"
python3 "$PROJECT/evaluation/full_physical/run_full_evaluation.py" \
    --project "$PROJECT" \
    --case-file "$case_file" \
    --output-root "$output_root" \
    --gpu-lib "$GPU_LIB" \
    --eval-env "$EVAL_ENV" \
    --a-values "$a_values" \
    --s-edges "$s_edges" \
    --candidate-cpus "${CANDIDATE_CPUS:-4}" \
    --desc-cpus "${DESC_CPUS:-16}"
