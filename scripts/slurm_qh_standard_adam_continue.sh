#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-adam-cont
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
source_adam="${SOURCE_ADAM:?SOURCE_ADAM is required}"
run_root="${RUN_ROOT:?RUN_ROOT is required}"
asset_root="${ASSET_ROOT:-$HOME/local_surface_evaluator}"
checkpoint="${FLOW_CHECKPOINT:-$asset_root/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
lib="${SCORE_LIB:-$asset_root/gpu_backend/build_mixed/libstellarator_gpu.so}"
expected_flow_sha="${EXPECTED_FLOW_SHA:-39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f}"
expected_lib_sha="${EXPECTED_SCORE_LIB_SHA:-40dca7422995a91eab0a58285d9ced59a8e3be04a96b2b37686effbe6f1abff5}"
target_iterations="${TARGET_ITERATIONS:-400}"
source_iterations="${SOURCE_ITERATIONS:-200}"
nfp="${NFP:-6}"
n_base_coils="${N_BASE_COILS:-2}"
optimizer_seed="${OPTIMIZER_SEED:-2026180360}"
max_wall_s="${MAX_WALL_S:-9000}"
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

cd "$project"
test -f "$checkpoint"
test -f "$lib"
test -f "$source_adam/manifest.json"
test -f "$source_adam/history.jsonl"
test -f "$source_adam/progress.json"
test -f "$source_adam/best.json"
test -f "$source_adam/state_latest.npz"
test -f "$source_adam/summary.json"
test -d "$source_adam/trajectory"
test ! -e "$run_root"
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

idle_streak=0
for _ in {1..60}; do
  mapfile -t compute_processes < <(
    nvidia-smi --id="$gpu_selector" \
      --query-compute-apps=pid --format=csv,noheader,nounits |
      sed '/^[[:space:]]*$/d'
  )
  mapfile -t memory_values < <(
    nvidia-smi --id="$gpu_selector" \
      --query-gpu=memory.used --format=csv,noheader,nounits |
      tr -d ' '
  )
  idle=1
  if (( ${#compute_processes[@]} != 0 )); then idle=0; fi
  for memory_used in "${memory_values[@]}"; do
    if (( memory_used > 16 )); then idle=0; fi
  done
  if (( idle )); then
    ((idle_streak += 1))
    if (( idle_streak >= 3 )); then break; fi
  else
    idle_streak=0
  fi
  sleep 2
done
if (( idle_streak < 3 )); then
  printf 'allocated GPUs did not remain idle for three probes\n' >&2
  exit 42
fi

mkdir -p "$run_root"
cp -a "$source_adam" "$adam_root"
mv "$adam_root/summary.json" "$run_root/source_summary.json"
sha256sum "$source_adam/state_latest.npz" > "$run_root/source_state_sha256.txt"
sha256sum "$source_adam/best.json" > "$run_root/source_best_sha256.txt"
git rev-parse HEAD > "$run_root/code_commit.txt"
printf '%s\n' "$source_adam" > "$run_root/source_adam.txt"
nvidia-smi --id="$gpu_selector" \
  --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$run_root/gpu_preflight.csv"

python scripts/optimize_flow_prior_standard_adam.py \
  --checkpoint "$checkpoint" \
  --lib "$lib" \
  --out-dir "$adam_root" \
  --nfp "$nfp" \
  --n-base-coils "$n_base_coils" \
  --iterations "$target_iterations" \
  --directions 4 \
  --flow-steps 256 \
  --max-wall-s "$max_wall_s" \
  --learning-rate 0.01 \
  --perturbation 0.005 \
  --beta1 0.5 \
  --beta2 0.999 \
  --robust-direction-filter \
  --reject-invalid-center \
  --invalid-center-backtracking 0.5,0.25,0.125 \
  --gpus 0,1,2,3 \
  --seed "$optimizer_seed" \
  --resume &
children+=("$!")
wait "${children[0]}"
children=()

trajectory_count="$(find "$adam_root/trajectory" -maxdepth 1 -type f -name 'step_*.json' | wc -l)"
test "$trajectory_count" -eq "$((target_iterations + 1))"
python - "$adam_root/summary.json" "$source_iterations" "$target_iterations" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
source = int(sys.argv[2])
target = int(sys.argv[3])
assert summary["status"] == "ok"
assert summary["stop_reason"] == "completed_iterations"
assert summary["resumed_from_iteration"] == source
assert summary["completed_iterations"] == target
PY
touch "$run_root/completed.ok"
