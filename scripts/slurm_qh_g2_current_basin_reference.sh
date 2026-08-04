#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-g2-ref
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator_worktrees/qh-blackbox-gradient}"
checkpoint="${FLOW_CHECKPOINT:-$HOME/local_surface_evaluator/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
lib="${SCORE_LIB:-$HOME/local_surface_evaluator_worktrees/qh-volume-qs-g-fix/gpu_backend/build_native_score/libstellarator_gpu.so}"
expected_checkpoint_sha="${EXPECTED_CHECKPOINT_SHA:-39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f}"
expected_score_lib_sha="${EXPECTED_SCORE_LIB_SHA:-40dca7422995a91eab0a58285d9ced59a8e3be04a96b2b37686effbe6f1abff5}"
run_dir="${RUN_DIR:-$project/runs/qh_physical_gradient_adam_start10_sweep_20260804/rk4_064_lr_0p003}"
output="${OUTPUT_DIR:-$project/runs/qh_g2_current_basin_reference_${SLURM_JOB_ID}}"
scales="${SCALES:-0.005,0.0025}"
world_size=4
children=()

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if (( ${#children[@]} )); then
    kill "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
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
for step in 50 89 100 120; do
  test -f "$run_dir/trajectory/step_$(printf '%04d' "$step").json"
done
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_checkpoint_sha"
test "$(sha256sum "$lib" | awk '{print $1}')" = "$expected_score_lib_sha"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project:$project/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

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
  echo "allocated GPUs retained memory or compute processes during idle probes" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$output/gpu_preflight.csv"

python scripts/qh_blackbox_gradient_reference.py \
  --output-dir "$output" --checkpoint "$checkpoint" --lib "$lib" \
  --center "nc3_step50=$run_dir/trajectory/step_0050.json" \
  --center "nc3_step89=$run_dir/trajectory/step_0089.json" \
  --center "nc3_step100=$run_dir/trajectory/step_0100.json" \
  --center "nc3_step120=$run_dir/trajectory/step_0120.json" \
  --scales "$scales" --rk4-steps 64 --prepare-only

for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$rank" python scripts/qh_blackbox_gradient_reference.py \
    --output-dir "$output" --checkpoint "$checkpoint" --lib "$lib" \
    --score-only --rank "$rank" --world-size "$world_size" --device-id 0 \
    > "$output/rank_$(printf '%02d' "$rank").log" 2>&1 &
  children+=("$!")
done
for child in "${children[@]}"; do wait "$child"; done
children=()

python scripts/qh_blackbox_gradient_reference.py \
  --output-dir "$output" --checkpoint "$checkpoint" --lib "$lib" --analyze-only
test -s "$output/summary.json"
