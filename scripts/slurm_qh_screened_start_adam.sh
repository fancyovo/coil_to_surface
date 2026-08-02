#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-screen-adam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --array=0-7%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
asset_root="${ASSET_ROOT:-$HOME/local_surface_evaluator}"
checkpoint="${FLOW_CHECKPOINT:-$asset_root/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
lib="${SCORE_LIB:-$project/gpu_backend/build_native_score/libstellarator_gpu.so}"
expected_flow_sha="${EXPECTED_FLOW_SHA:-39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f}"
expected_lib_sha="${EXPECTED_SCORE_LIB_SHA:-4bf7a12ea3dbdef9faf6de3ce4dc1840ecf48847ba795267500dd4179f730708}"
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
candidate_seed="$(( ${SEED_BASE:-2026080200} + task_id ))"
optimizer_seed="$(( ${OPTIMIZER_SEED_BASE:-2026180200} + task_id ))"
candidate_count="${CANDIDATE_COUNT:-128}"
nfp="${NFP:-4}"
n_base_coils="${N_BASE_COILS:-3}"
iterations="${ITERATIONS:-50}"
run_root_base="${RUN_ROOT_BASE:-$asset_root/runs/qh_screened_start_adam_${SLURM_ARRAY_JOB_ID}}"
run_root="$run_root_base/seed_${candidate_seed}"
pool_root="$run_root/candidate_pool"
adam_root="$run_root/adam"
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
  if [[ -n "$gpu_selector" && -d "$run_root" ]]; then
    nvidia-smi --id="$gpu_selector" \
      --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits > "$run_root/gpu_postflight.csv" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$run_root"
cd "$project"
test ! -e "$pool_root"
test ! -e "$adam_root"
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
  nvidia-smi --id="$gpu_selector" \
    --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
  printf 'allocated GPUs are not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi
nvidia-smi --id="$gpu_selector" \
  --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$run_root/gpu_preflight.csv"
git rev-parse HEAD > "$run_root/code_commit.txt"
printf '%s\n' "$candidate_seed" > "$run_root/candidate_seed.txt"
printf '%s\n' "$optimizer_seed" > "$run_root/optimizer_seed.txt"

run_started_ns="$(date +%s%N)"
python scripts/evaluate_qh_random_score_pool.py \
  --mode prepare \
  --output-dir "$pool_root" \
  --flow-checkpoint "$checkpoint" \
  --count "$candidate_count" \
  --seed "$candidate_seed" \
  --nfp "$nfp" \
  --n-base-coils "$n_base_coils" \
  --flow-steps 256 \
  --flow-batch 128 &
children+=("$!")
wait "${children[0]}"
children=()

python scripts/evaluate_qh_random_score_pool.py \
  --mode score \
  --output-dir "$pool_root" \
  --lib "$lib" \
  --gpu-ids 0,1,2,3 \
  --timeout-s 3600 &
children+=("$!")
wait "${children[0]}"
children=()

python scripts/select_qh_screened_adam_start.py \
  --scored-cases "$pool_root/scored_cases.jsonl" \
  --random-latents "$pool_root/random_latents.npz" \
  --output-start "$run_root/selected_start.json" \
  --output-summary "$run_root/selection_summary.json" \
  --expected-count "$candidate_count" \
  --seed "$candidate_seed" \
  --nfp "$nfp" \
  --n-base-coils "$n_base_coils"
selection_finished_ns="$(date +%s%N)"

adam_started_ns="$(date +%s%N)"
python scripts/optimize_flow_prior_standard_adam.py \
  --checkpoint "$checkpoint" \
  --lib "$lib" \
  --out-dir "$adam_root" \
  --initial-case "$run_root/selected_start.json" \
  --nfp "$nfp" \
  --n-base-coils "$n_base_coils" \
  --iterations "$iterations" \
  --directions 4 \
  --flow-steps 256 \
  --max-wall-s 2700 \
  --learning-rate 0.01 \
  --perturbation 0.005 \
  --beta1 0.5 \
  --beta2 0.999 \
  --robust-direction-filter \
  --reject-invalid-center \
  --invalid-center-backtracking 0.5,0.25,0.125 \
  --gpus 0,1,2,3 \
  --seed "$optimizer_seed" &
children+=("$!")
wait "${children[0]}"
children=()
run_finished_ns="$(date +%s%N)"

python scripts/summarize_qh_screened_start_adam.py \
  --selection-summary "$run_root/selection_summary.json" \
  --adam-summary "$adam_root/summary.json" \
  --output "$run_root/experiment_summary.json" \
  --run-started-ns "$run_started_ns" \
  --selection-finished-ns "$selection_finished_ns" \
  --adam-started-ns "$adam_started_ns" \
  --run-finished-ns "$run_finished_ns" \
  --optimizer-seed "$optimizer_seed"
touch "$run_root/completed.ok"
