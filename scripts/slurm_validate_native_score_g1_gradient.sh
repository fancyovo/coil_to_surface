#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-grad-g1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator_worktrees/qh-blackbox-gradient}"
baseline_lib="$HOME/local_surface_evaluator_worktrees/qh-volume-qs-g-fix/gpu_backend/build_native_score/libstellarator_gpu.so"
build="$project/gpu_backend/build_gradient"
gradient_lib="$build/libstellarator_gpu.so"
case_path="$project/reports/assets/qh_small_condition_adam_nfp6_nc2_continue400_20260803/adam/trajectory/step_0200.json"
output="${OUTPUT_DIR:-$project/runs/qh_native_g1_validation_${SLURM_JOB_ID}}"

mkdir -p "$project/logs" "$output"
cd "$project"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project:$project/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -m pytest tests/test_qs_gradient_math.py tests/test_flow_vjp.py -q
cmake -S gpu_backend -B "$build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --parallel 16
test -f "$gradient_lib"

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader > "$output/gpu_preflight.csv"
python scripts/validate_native_score_g1_gradient.py \
  --case "$case_path" --baseline-lib "$baseline_lib" --gradient-lib "$gradient_lib" \
  --output "$output/validation.json" --directions 24 --steps 1,0.5,0.25 --repeats 3
python scripts/validate_qh_latent_gradient.py \
  --reference-dir "$project/runs/qh_blackbox_gradient_reference_31640" \
  --checkpoint "$HOME/local_surface_evaluator/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt" \
  --gradient-lib "$gradient_lib" --output-dir "$output/latent" \
  --center-id main_nfp6_step200 --scale 0.005 --directions 8 \
  --rk4-steps 256 --checkpoint-steps 8
sha256sum "$gradient_lib" > "$output/gradient_library_sha256.txt"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader > "$output/gpu_postflight.csv"
