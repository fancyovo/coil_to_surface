#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-g2-close
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator_worktrees/qh-blackbox-gradient}"
run_dir="${RUN_DIR:-$project/runs/qh_physical_gradient_adam_start10_sweep_20260804/rk4_064_lr_0p003}"
iteration="${ITERATION:-120}"
output="${OUTPUT_DIR:-$project/runs/qh_g2_fixed_front_closure_${SLURM_JOB_ID}}"
checkpoint="${FLOW_CHECKPOINT:-$HOME/local_surface_evaluator/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
expected_checkpoint_sha="${EXPECTED_CHECKPOINT_SHA:-39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f}"
build="${BUILD_DIR:-$project/gpu_backend/build_gradient_g2_closure}"
gradient_lib="$build/libstellarator_gpu.so"
build_library="${BUILD_LIBRARY:-0}"
expected_gradient_sha="${EXPECTED_GRADIENT_SHA:-}"
random_directions="${RANDOM_DIRECTIONS:-32}"
scales="${SCALES:-0.0003125,0.000625,0.00125,0.0025}"
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
      --format=csv,noheader,nounits > "$output/gpu_postflight.csv" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$project/logs" "$output"
cd "$project"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project:$project/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_checkpoint_sha"
test -f "$run_dir/trajectory/step_$(printf '%04d' "$iteration").json"
test -f "$run_dir/trajectory/step_$(printf '%04d' "$((iteration + 1))").json"

if (( build_library )); then
  cmake -S gpu_backend -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DCMAKE_CUDA_ARCHITECTURES=120
  cmake --build "$build" --parallel "$SLURM_CPUS_PER_TASK"
fi
test -f "$gradient_lib"
actual_gradient_sha="$(sha256sum "$gradient_lib" | awk '{print $1}')"
if [[ -n "$expected_gradient_sha" ]]; then
  test "$actual_gradient_sha" = "$expected_gradient_sha"
fi
printf '%s  %s\n' "$actual_gradient_sha" "$gradient_lib" > "$output/gradient_library_sha256.txt"

idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS=',' read -r utilization memory_used; do
    utilization="${utilization// /}"
    memory_used="${memory_used// /}"
    if (( utilization != 0 || memory_used > 16 )); then idle=0; fi
  done < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
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
  echo "allocated GPU did not reach three consecutive idle probes" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$output/gpu_preflight.csv"

python scripts/diagnose_qh_g2_fixed_front_closure.py \
  --run-dir "$run_dir" \
  --iteration "$iteration" \
  --checkpoint "$checkpoint" \
  --gradient-lib "$gradient_lib" \
  --output-dir "$output" \
  --random-directions "$random_directions" \
  --scales "$scales" &
child="$!"
wait "$child"
child=""
test -s "$output/summary.json"
