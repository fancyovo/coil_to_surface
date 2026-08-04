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
run_native="${RUN_NATIVE_VALIDATION:-1}"
run_latent="${RUN_LATENT_VALIDATION:-1}"
build_jobs="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$project/logs" "$output"
cd "$project"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project:$project/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python - <<'PY'
import runpy

count = 0
for path in ("tests/test_qs_gradient_math.py", "tests/test_flow_vjp.py"):
    namespace = runpy.run_path(path)
    for name, value in sorted(namespace.items()):
        if name.startswith("test_") and callable(value):
            value()
            count += 1
print(f"standalone gradient checks passed: {count}")
PY
cmake -S gpu_backend -B "$build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --parallel "$build_jobs"
test -f "$gradient_lib"

idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS= read -r memory_used; do
    memory_used="${memory_used// /}"
    if (( memory_used > 16 )); then idle=0; fi
  done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    idle=0
  fi
  if (( idle )); then
    ((idle_streak += 1))
    if (( idle_streak >= 3 )); then break; fi
  else
    idle_streak=0
  fi
  sleep 2
done
if (( idle_streak < 3 )); then
  echo "allocated GPU retained memory or compute processes during idle probes" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader > "$output/gpu_preflight.csv"
if (( run_native )); then
  python scripts/validate_native_score_g1_gradient.py \
    --case "$case_path" --baseline-lib "$baseline_lib" --gradient-lib "$gradient_lib" \
    --output "$output/validation.json" --directions 24 --steps 1,0.5,0.25 --repeats 3
fi
if (( run_latent )); then
  python scripts/validate_qh_latent_gradient.py \
    --reference-dir "$project/runs/qh_blackbox_gradient_reference_31640" \
    --checkpoint "$HOME/local_surface_evaluator/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt" \
    --gradient-lib "$gradient_lib" --output-dir "$output/latent" \
    --center-id main_nfp6_step200 --scale 0.005 --directions 8 \
    --rk4-steps 256 --checkpoint-steps 8
fi
sha256sum "$gradient_lib" > "$output/gradient_library_sha256.txt"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader > "$output/gpu_postflight.csv"
