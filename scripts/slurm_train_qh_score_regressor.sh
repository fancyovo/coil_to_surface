#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-score-regressor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
corpus_root="${SCORE_CORPUS_ROOT:-$HOME/local_surface_evaluator_data/qh_iid_score_corpus_v1}"
dataset_root="${REGRESSION_DATASET_ROOT:-$HOME/local_surface_evaluator_data/qh_score_regression_snapshot_${SLURM_JOB_ID}}"
asset_root="${ASSET_ROOT:-$HOME/local_surface_evaluator}"
output="${REGRESSION_OUTPUT:-$asset_root/runs/qh_score_regression_proxy_${SLURM_JOB_ID}}"
score_sha="${EXPECTED_SCORE_SHA:-4bf7a12ea3dbdef9faf6de3ce4dc1840ecf48847ba795267500dd4179f730708}"
preflight="${output}.gpu_preflight.csv"
postflight="${output}.gpu_postflight.csv"
commit_file="${output}.code_commit.txt"
children=()

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if (( ${#children[@]} )); then
    pkill -TERM -P "${children[0]}" 2>/dev/null || true
    kill "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
  fi
  if [[ -d "$output" ]]; then
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits > "$postflight" 2>/dev/null || true
    mv "$preflight" "$output/gpu_preflight.csv" 2>/dev/null || true
    mv "$postflight" "$output/gpu_postflight.csv" 2>/dev/null || true
    mv "$commit_file" "$output/code_commit.txt" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

cd "$project"
test -d "$corpus_root/shards"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

if [[ "$(nvidia-smi -L | wc -l)" -ne 4 ]]; then
  echo "the training job must see exactly four allocated GPUs" >&2
  exit 41
fi
mapfile -t compute_processes < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
  printf 'allocated GPUs are not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi

mkdir -p "$(dirname "$dataset_root")" "$(dirname "$output")" logs
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$preflight"
git rev-parse HEAD > "$commit_file"

python scripts/prepare_qh_score_regression_dataset.py \
  --corpus-root "$corpus_root" \
  --output-dir "$dataset_root" \
  --score-library-sha256 "$score_sha" \
  --seed "${REGRESSION_SPLIT_SEED:-20260802}"

python -m torch.distributed.run --standalone --nproc-per-node=4 \
  scripts/train_qh_latent_score_regressor.py \
  --dataset-dir "$dataset_root" \
  --output-dir "$output" \
  --max-steps "${REGRESSION_MAX_STEPS:-40000}" \
  --batch-per-gpu "${REGRESSION_BATCH_PER_GPU:-2048}" \
  --eval-batch "${REGRESSION_EVAL_BATCH:-8192}" \
  --learning-rate "${REGRESSION_LEARNING_RATE:-3e-4}" \
  --validation-interval "${REGRESSION_VALIDATION_INTERVAL:-25}" \
  --plateau-validations "${REGRESSION_PLATEAU_VALIDATIONS:-16}" \
  --final-plateau-validations "${REGRESSION_FINAL_PLATEAU_VALIDATIONS:-32}" \
  --max-lr-reductions "${REGRESSION_MAX_LR_REDUCTIONS:-4}" &
children+=("$!")
wait "${children[0]}"
children=()
