#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-proxy-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
asset_root="${ASSET_ROOT:-$HOME/local_surface_evaluator}"
proxy_checkpoint="${PROXY_CHECKPOINT:?PROXY_CHECKPOINT is required}"
calibration_summary="${PROXY_CALIBRATION_SUMMARY:?PROXY_CALIBRATION_SUMMARY is required}"
flow_checkpoint="${FLOW_CHECKPOINT:-$asset_root/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
lib="${SCORE_LIB:-$project/gpu_backend/build_native_score/libstellarator_gpu.so}"
expected_flow_sha="${EXPECTED_FLOW_SHA:-39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f}"
expected_lib_sha="${EXPECTED_SCORE_LIB_SHA:-0b7342db471788385931385c25ded8095c72cfb7fcea1e21376a0475dafaa427}"
run_root="${RUN_ROOT:-$asset_root/runs/qh_latent_proxy_score_${SLURM_JOB_ID}}"
preflight="${run_root}.gpu_preflight.csv"
postflight="${run_root}.gpu_postflight.csv"
commit_file="${run_root}.code_commit.txt"
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
      --format=csv,noheader,nounits > "$postflight" 2>/dev/null || true
    mv "$preflight" "$run_root/gpu_preflight.csv" 2>/dev/null || true
    mv "$postflight" "$run_root/gpu_postflight.csv" 2>/dev/null || true
    mv "$commit_file" "$run_root/code_commit.txt" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$(dirname "$run_root")"
cd "$project"
test -f "$proxy_checkpoint"
test -f "$calibration_summary"
test -f "$flow_checkpoint"
test -f "$lib"
test "$(sha256sum "$flow_checkpoint" | awk '{print $1}')" = "$expected_flow_sha"
test "$(sha256sum "$lib" | awk '{print $1}')" = "$expected_lib_sha"
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
  --format=csv,noheader,nounits > "$preflight"
git rev-parse HEAD > "$commit_file"

python scripts/evaluate_qh_latent_proxy_score.py \
  --mode prepare \
  --output-dir "$run_root" \
  --proxy-checkpoint "$proxy_checkpoint" \
  --calibration-summary "$calibration_summary" \
  --flow-checkpoint "$flow_checkpoint" \
  --pool-count "${PROXY_POOL_COUNT:-131072}" \
  --stratified-count "${PROXY_STRATIFIED_COUNT:-768}" \
  --iid-count "${PROXY_IID_COUNT:-256}" \
  --flow-steps "${FLOW_STEPS:-256}" &
children+=("$!")
wait "${children[0]}"
children=()
mv "$preflight" "$run_root/gpu_preflight.csv"
mv "$commit_file" "$run_root/code_commit.txt"

python scripts/evaluate_qh_latent_proxy_score.py \
  --mode score \
  --output-dir "$run_root" \
  --lib "$lib" \
  --gpu-ids 0,1,2,3 \
  --timeout-s "${SCORE_TIMEOUT_S:-6600}" &
children+=("$!")
wait "${children[0]}"
children=()
