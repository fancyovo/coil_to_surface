#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-proxy-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
asset_root="${ASSET_ROOT:-$HOME/local_surface_evaluator}"
latent_dir="${LATENT_DIR:?LATENT_DIR is required}"
checkpoint="${PROXY_CHECKPOINT:?PROXY_CHECKPOINT is required}"
output="${PROXY_EVAL_OUTPUT:-$asset_root/runs/qh_latent_proxy_eval_${SLURM_JOB_ID}}"
preflight="${output}.gpu_preflight.csv"
postflight="${output}.gpu_postflight.csv"
commit_file="${output}.code_commit.txt"
child=""

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$child" ]]; then
    pkill -TERM -P "$child" 2>/dev/null || true
    kill "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
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

mkdir -p "$(dirname "$output")"
cd "$project"
test -f "$latent_dir/manifest.json"
test -f "$checkpoint"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

mapfile -t compute_processes < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
  printf 'allocated GPU is not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$preflight"
git rev-parse HEAD > "$commit_file"

python scripts/evaluate_qh_latent_proxy.py \
  --latent-dir "$latent_dir" \
  --checkpoint "$checkpoint" \
  --output-dir "$output" \
  --eval-batch "${PROXY_EVAL_BATCH:-8192}" &
child=$!
wait "$child"
child=""
