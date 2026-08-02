#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qs-gfix-calibration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
asset_root="${ASSET_ROOT:-$HOME/local_surface_evaluator}"
data_dir="${DATA_DIR:-$HOME/local_surface_evaluator_data/quasr_qh_flow_v1}"
checkpoint="${FLOW_CHECKPOINT:-$asset_root/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
lib="${SCORE_LIB:-$project/gpu_backend/build_native_score/libstellarator_gpu.so}"
run_root="${RUN_ROOT:-$project/runs/corrected_score_calibration_${SLURM_JOB_ID}}"
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

cd "$project"
mkdir -p "$run_root"
test -f "$checkpoint"
test -f "$lib"
test -f "$data_dir/manifest.json"
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
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
  printf 'allocated GPUs are not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$run_root/gpu_preflight.csv"

python scripts/evaluate_corrected_score_calibration.py \
  --data-dir "$data_dir" \
  --checkpoint "$checkpoint" \
  --lib "$lib" \
  --output-dir "$run_root/results" \
  --quasr-count "${QUASR_COUNT:-1024}" \
  --random-count "${RANDOM_COUNT:-1024}" \
  --seed "${SEED:-20260808}" \
  --flow-steps "${FLOW_STEPS:-256}" \
  --decode-batch "${DECODE_BATCH:-256}" \
  --gpus "${GPU_IDS:-0,0,1,1,2,2,3,3}" \
  --score-timeout-s "${SCORE_TIMEOUT_S:-7200}" \
  --known-case "score61=$project/reports/assets/qh_adam_topology_fixed_61p339_full_eval_20260801/evaluated_case.json" \
  --known-case "score63=$project/reports/assets/qh_adam_low_momentum_start10_200_30662/best.json" &
children+=("$!")
wait "${children[0]}"
children=()
