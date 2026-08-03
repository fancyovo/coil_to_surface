#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-proxy-train
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
asset_root="${ASSET_ROOT:-$HOME/local_surface_evaluator}"
latent_dir="${LATENT_DIR:?LATENT_DIR is required}"
output="${PROXY_OUTPUT:-$asset_root/runs/qh_latent_proxy_training_${SLURM_JOB_ID}}"
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
test -f "$latent_dir/manifest.json"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mapfile -t compute_processes < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
  printf 'allocated GPUs are not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi
mkdir -p "$(dirname "$output")"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$preflight"
git rev-parse HEAD > "$commit_file"

launch_attempts="${PROXY_LAUNCH_ATTEMPTS:-3}"
for (( attempt = 1; attempt <= launch_attempts; attempt++ )); do
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
    scripts/train_qh_latent_proxy.py \
    --latent-dir "$latent_dir" \
    --output-dir "$output" \
    --max-steps "${PROXY_MAX_STEPS:-30000}" \
    --batch-per-gpu "${PROXY_BATCH_PER_GPU:-2048}" \
    --eval-batch "${PROXY_EVAL_BATCH:-8192}" \
    --learning-rate "${PROXY_LEARNING_RATE:-3e-4}" \
    --validation-interval "${PROXY_VALIDATION_INTERVAL:-100}" \
    --plateau-validations "${PROXY_PLATEAU_VALIDATIONS:-10}" \
    --final-plateau-validations "${PROXY_FINAL_PLATEAU_VALIDATIONS:-20}" \
    --max-lr-reductions "${PROXY_MAX_LR_REDUCTIONS:-3}" &
  children+=("$!")
  if wait "${children[0]}"; then
    children=()
    exit 0
  else
    status=$?
  fi
  children=()
  if [[ -e "$output/metrics.jsonl" ]] || [[ -d "$output" ]] || (( attempt == launch_attempts )); then
    exit "$status"
  fi
  echo "proxy torchrun failed before creating output; retrying launch ($attempt/$launch_attempts)" >&2
  sleep 10
done
