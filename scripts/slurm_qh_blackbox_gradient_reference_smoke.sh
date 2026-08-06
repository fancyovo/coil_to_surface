#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-grad-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator_worktrees/qh-blackbox-gradient}"
checkpoint="$HOME/local_surface_evaluator/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt"
lib="$HOME/local_surface_evaluator_worktrees/qh-volume-qs-g-fix/gpu_backend/build_native_score/libstellarator_gpu.so"
center="$project/reports/assets/qh_small_condition_adam_nfp6_nc2_20260803/adam/trajectory/step_0000.json"
output="${OUTPUT_DIR:-$project/runs/qh_blackbox_gradient_reference_smoke_${SLURM_JOB_ID}}"

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -d "$output" ]]; then
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
      --format=csv,noheader > "$output/gpu_postflight.csv" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$project/logs" "$output"
cd "$project"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python scripts/qh_blackbox_gradient_reference.py \
  --output-dir "$output" --checkpoint "$checkpoint" --lib "$lib" \
  --center "smoke_nfp6_step0=$center" --scales 0.005 \
  --direction-count 1 --rk4-steps 256 --prepare-only
python scripts/qh_blackbox_gradient_reference.py \
  --output-dir "$output" --checkpoint "$checkpoint" --lib "$lib" \
  --score-only --rank 0 --world-size 1 --device-id 0
python scripts/qh_blackbox_gradient_reference.py \
  --output-dir "$output" --checkpoint "$checkpoint" --lib "$lib" --analyze-only

test -s "$output/summary.json"
