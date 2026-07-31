#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-latent-proxy
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
data="${QH_DATA:-$HOME/local_surface_evaluator_data/quasr_qh_flow_v1}"
flow_checkpoint="${FLOW_CHECKPOINT:-$asset_root/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
expected_flow_sha="${EXPECTED_FLOW_SHA:-39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f}"
run_root="${RUN_ROOT:-$asset_root/runs/qh_latent_proxy_${SLURM_JOB_ID}}"
children=()

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if (( ${#children[@]} )); then
    pkill -TERM -P "${children[0]}" 2>/dev/null || true
    kill "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
  fi
  if [[ -d "$run_root" ]]; then
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits > "$run_root/gpu_postflight.csv" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$run_root"
cd "$project"
test -f "$data/manifest.json"
test -f "$flow_checkpoint"
test "$(sha256sum "$flow_checkpoint" | awk '{print $1}')" = "$expected_flow_sha"
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
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$run_root/gpu_preflight.csv"
git rev-parse HEAD > "$run_root/code_commit.txt"

inverse_limit_args=()
if [[ -n "${INVERSE_LIMIT_PER_GROUP:-}" ]]; then
  inverse_limit_args+=(--limit-per-group "$INVERSE_LIMIT_PER_GROUP")
fi
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  scripts/invert_qh_flow_latents.py \
  --data-dir "$data" \
  --checkpoint "$flow_checkpoint" \
  --output-dir "$run_root/latents" \
  --steps "${INVERSE_STEPS:-256}" \
  --batch-size "${INVERSE_BATCH:-4096}" \
  --closure-count "${CLOSURE_COUNT:-8}" \
  "${inverse_limit_args[@]}" &
children+=("$!")
wait "${children[0]}"
children=()

python -m torch.distributed.run --standalone --nproc-per-node=4 \
  scripts/train_qh_latent_proxy.py \
  --latent-dir "$run_root/latents" \
  --output-dir "$run_root/training" \
  --max-steps "${PROXY_MAX_STEPS:-30000}" \
  --batch-per-gpu "${PROXY_BATCH_PER_GPU:-2048}" \
  --eval-batch "${PROXY_EVAL_BATCH:-8192}" \
  --learning-rate "${PROXY_LEARNING_RATE:-3e-4}" \
  --validation-interval "${PROXY_VALIDATION_INTERVAL:-100}" \
  --plateau-validations "${PROXY_PLATEAU_VALIDATIONS:-10}" \
  --final-plateau-validations "${PROXY_FINAL_PLATEAU_VALIDATIONS:-20}" \
  --max-lr-reductions "${PROXY_MAX_LR_REDUCTIONS:-3}" &
children+=("$!")
wait "${children[0]}"
children=()
