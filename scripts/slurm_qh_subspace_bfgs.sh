#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-local-bfgs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
asset_root="${ASSET_ROOT:-$HOME/local_surface_evaluator}"
initial_case="${INITIAL_CASE:?INITIAL_CASE is required}"
adam_state="${ADAM_STATE:-}"
output="${OUT_DIR:?OUT_DIR is required}"
checkpoint="${FLOW_CHECKPOINT:-$asset_root/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
lib="${SCORE_LIB:-$project/gpu_backend/build_native_score/libstellarator_gpu.so}"
expected_flow_sha="${EXPECTED_FLOW_SHA:-39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f}"
expected_lib_sha="${EXPECTED_SCORE_LIB_SHA:-4bf7a12ea3dbdef9faf6de3ce4dc1840ecf48847ba795267500dd4179f730708}"
gpu_selector="${CUDA_VISIBLE_DEVICES:-}"
children=()

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if (( ${#children[@]} )); then
    pkill -TERM -P "${children[0]}" 2>/dev/null || true
    kill "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
  fi
  if [[ -n "$gpu_selector" && -d "$output" ]]; then
    nvidia-smi --id="$gpu_selector" \
      --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits > "$output/gpu_postflight.csv" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$output"
cd "$project"
test -f "$initial_case"
test -z "$adam_state" || test -f "$adam_state"
test -f "$checkpoint"
test -f "$lib"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_flow_sha"
test "$(sha256sum "$lib" | awk '{print $1}')" = "$expected_lib_sha"
: "${gpu_selector:?CUDA_VISIBLE_DEVICES is required}"
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
  nvidia-smi --id="$gpu_selector" --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
  printf 'allocated GPUs are not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi
nvidia-smi --id="$gpu_selector" \
  --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$output/gpu_preflight.csv"
git rev-parse HEAD > "$output/code_commit.txt"

state_args=()
if [[ -n "$adam_state" ]]; then
  state_args+=(--adam-state "$adam_state")
fi

cd /
python "$project/scripts/optimize_flow_prior_subspace_bfgs.py" \
  --checkpoint "$checkpoint" \
  --lib "$lib" \
  --initial-case "$initial_case" \
  --out-dir "$output" \
  --nfp "${NFP:-4}" \
  --n-base-coils "${N_BASE_COILS:-3}" \
  --iterations "${ITERATIONS:-12}" \
  --rank "${SUBSPACE_RANK:-4}" \
  --method "${METHOD:-bfgs}" \
  --perturbation "${PERTURBATION:-0.005}" \
  --min-perturbation "${MIN_PERTURBATION:-0.000625}" \
  --trust-radius "${TRUST_RADIUS:-0.002}" \
  --min-trust-radius "${MIN_TRUST_RADIUS:-0.00002}" \
  --max-trust-radius "${MAX_TRUST_RADIUS:-0.01}" \
  --max-wall-s "${MAX_WALL_S:-1500}" \
  --seed "${SEED:-2026080101}" \
  "${state_args[@]}" &
children+=("$!")
wait "${children[0]}"
children=()
