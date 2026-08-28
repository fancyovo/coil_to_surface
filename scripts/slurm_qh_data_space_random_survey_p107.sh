#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-random-global
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=16G
#SBATCH --time=11:15:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
run_root="${RUN_ROOT:?RUN_ROOT is required}"
expected_commit="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
worker_index=$(( ${WORKER_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))
source "${VENV:-$HOME/coil/.venv}/bin/activate"
cd "$project"
test "$(git rev-parse HEAD)" = "$expected_commit"
test -z "$(git status --porcelain --untracked-files=no)"
mkdir -p "$run_root/logs"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

worker_dir="$run_root/workers/worker_$(printf '%02d' "$worker_index")"
mkdir -p "$worker_dir"
gpu_pre="$worker_dir/gpu_preflight.csv"
gpu_post="$worker_dir/gpu_postflight.csv"
child=""
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
    kill -TERM -- "-$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  nvidia-smi --query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$gpu_post" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "allocated GPU is not idle before survey timing" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$gpu_pre"

setsid python scripts/qh_data_space_random_survey.py worker \
  --run-root "$run_root" \
  --worker-index "$worker_index" \
  --device 0 \
  --chunk-size "${CHUNK_SIZE:-8}" \
  --max-wall-s "${MAX_WALL_S:-39600}" &
child=$!
wait "$child"
child=""
