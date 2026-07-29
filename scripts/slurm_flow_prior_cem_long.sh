#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=flow-prior-cem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --exclude=anode02
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator}"
checkpoint="${FLOW_CHECKPOINT:-$project/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
lib="${SCORE_LIB:-$project/gpu_backend/build_native_score/libstellarator_gpu.so}"
run_root="${RUN_ROOT:-$project/runs/qh_flow_prior_cem/${SLURM_JOB_ID}}"
target="${TARGET:-QH}"
nfp="${NFP:-4}"
n_base_coils="${N_BASE_COILS:-3}"
seed="${SEED:-2026073001}"
iterations="${ITERATIONS:-160}"
popsize="${POPSIZE:-160}"
elite="${ELITE:-40}"
flow_steps="${FLOW_STEPS:-32}"
max_wall_s="${MAX_WALL_S:-32400}"
gpu_ids="${GPU_IDS:-0,1,2,3}"
batch_timeout_s="${BATCH_TIMEOUT_S:-1800}"
children=()

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if (( ${#children[@]} )); then
    pkill -TERM -P "${children[0]}" 2>/dev/null || true
    kill "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
  fi
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader > "$run_root/gpu_postflight.csv" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$run_root"
cd "$project"
test -f "$checkpoint"
test -f "$lib"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS= read -r memory_used; do
    memory_used="${memory_used// /}"
    if (( memory_used > 16 )); then
      idle=0
    fi
  done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    idle=0
  fi
  if (( idle )); then
    ((idle_streak += 1))
    if (( idle_streak >= 3 )); then
      break
    fi
  else
    idle_streak=0
  fi
  sleep 2
done
if (( idle_streak < 3 )); then
  echo "allocated GPUs retained memory or compute processes during idle probes" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader > "$run_root/gpu_preflight.csv"

started="$(date +%s.%N)"
git_commit="$(git rev-parse HEAD)"
checkpoint_sha256="$(sha256sum "$checkpoint" | cut -d' ' -f1)"
library_sha256="$(sha256sum "$lib" | cut -d' ' -f1)"
printf '{"target":"%s","nfp":%d,"n_base_coils":%d,"seed":%d,"iterations":%d,"popsize":%d,"elite":%d,"flow_steps":%d,"max_wall_s":%s,"gpu_ids":"%s","git":"%s","checkpoint_sha256":"%s","library_sha256":"%s","started":%s}\n' \
  "$target" "$nfp" "$n_base_coils" "$seed" "$iterations" "$popsize" "$elite" \
  "$flow_steps" "$max_wall_s" "$gpu_ids" "$git_commit" "$checkpoint_sha256" \
  "$library_sha256" "$started" > "$run_root/job.json"

python "$project/scripts/optimize_flow_prior_cem.py" \
  --checkpoint "$checkpoint" \
  --lib "$lib" \
  --out-dir "$run_root/${target,,}" \
  --target "$target" \
  --nfp "$nfp" \
  --n-base-coils "$n_base_coils" \
  --iterations "$iterations" \
  --popsize "$popsize" \
  --elite "$elite" \
  --flow-steps "$flow_steps" \
  --gpus "$gpu_ids" \
  --batch-timeout-s "$batch_timeout_s" \
  --max-wall-s "$max_wall_s" \
  --seed "$seed" &
children=("$!")
wait "${children[0]}"
children=()

finished="$(date +%s.%N)"
python - "$run_root/job.json" "$finished" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["finished"] = float(sys.argv[2])
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
