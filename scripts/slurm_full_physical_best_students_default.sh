#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --job-name=full-best-stu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${PROJECT:?}"
: "${RL_RUN_ROOT:?}"
: "${EVAL_ROOT:?}"
: "${GPU_LIB:?}"
: "${EVAL_ENV:?}"
: "${EXPECTED_COMMIT:?}"
cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet
test ! -e "$EVAL_ROOT"
mkdir -p "$PROJECT/logs" "$(dirname "$EVAL_ROOT")"
source "$EVAL_ENV/bin/activate"
export PYTHONPATH="$PROJECT:$EVAL_ENV/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 evaluation/full_physical/preflight.py
selection_root="${EVAL_ROOT}.selection"
python3 scripts/select_highest_score_gradient_case.py --run-root "$RL_RUN_ROOT" --output-dir "$selection_root"
case_file="$selection_root/selected_case.json"
python3 evaluation/full_physical/run_full_evaluation.py \
  --project "$PROJECT" --case-file "$case_file" --output-root "$EVAL_ROOT" \
  --gpu-lib "$GPU_LIB" --eval-env "$EVAL_ENV" \
  --a-values "0.04,0.05,0.06,0.08" \
  --s-edges "0.12,0.24,0.36,0.49,0.64,0.81,1.0" \
  --candidate-cpus 4 --desc-cpus 4
