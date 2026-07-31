#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-score-start
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=00:35:00
#SBATCH --array=0-7%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
asset_root="${ASSET_ROOT:-$HOME/local_surface_evaluator}"
panel_dir="${PANEL_DIR:?PANEL_DIR is required}"
checkpoint="${FLOW_CHECKPOINT:-$asset_root/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
lib="${SCORE_LIB:-$project/gpu_backend/build_native_score/libstellarator_gpu.so}"
expected_flow_sha="${EXPECTED_FLOW_SHA:-39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f}"
expected_lib_sha="${EXPECTED_SCORE_LIB_SHA:-0b7342db471788385931385c25ded8095c72cfb7fcea1e21376a0475dafaa427}"
run_root="${RUN_ROOT:-$asset_root/runs/qh_score_adam_start_sweep_${SLURM_ARRAY_JOB_ID}}"
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
start_name="$(printf 'start_%02d' "$task_id")"
output="$run_root/$start_name"
initial_case="$panel_dir/$start_name.json"
iterations="${ITERATIONS:-40}"
max_wall_s="${MAX_WALL_S:-1950}"
learning_rate="${LEARNING_RATE:-0.003}"
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
seed_offset="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["flow_prior_start"]["direction_seed_offset"])' "$initial_case")"
seed="$(( ${DIRECTION_SEED_BASE:-20260804} + seed_offset ))"

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

python "$project/scripts/optimize_flow_prior_standard_adam.py" \
  --checkpoint "$checkpoint" \
  --lib "$lib" \
  --initial-case "$initial_case" \
  --out-dir "$output" \
  --iterations "$iterations" \
  --max-wall-s "$max_wall_s" \
  --learning-rate "$learning_rate" \
  --seed "$seed" &
children+=("$!")
wait "${children[0]}"
children=()
