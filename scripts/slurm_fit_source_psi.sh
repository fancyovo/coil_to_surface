#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=source-psi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${CASE_FILE:?CASE_FILE is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${A_VALUE:?A_VALUE is required}"

gpu_lib=${GPU_LIB:-$PROJECT/gpu_backend/build_mixed/libstellarator_gpu.so}
eval_env=${EVAL_ENV:-$HOME/local_surface_evaluator/.venv-desc016-py312}
levels=${LEVELS:-0.001,0.002,0.004,0.008,0.02,0.04,0.08,0.12,0.16,0.20,0.24,0.30,0.36,0.49,0.64,0.81}
gpu_selector=${CUDA_VISIBLE_DEVICES:-}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  mapfile -t children < <(jobs -pr)
  if (( ${#children[@]} )); then
    kill "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
  fi
  if [[ -n "$gpu_selector" && -d "$OUTPUT_DIR" ]]; then
    nvidia-smi --id="$gpu_selector" \
      --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits > "$OUTPUT_DIR/gpu_postflight.csv" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

for path in "$PROJECT" "$CASE_FILE" "$gpu_lib" "$eval_env" "$OUTPUT_DIR"; do
  resolved=$(realpath -m "$path")
  [[ $resolved == "$HOME"/* ]] || {
    printf 'path must stay under HOME: %s\n' "$resolved" >&2
    exit 2
  }
done
test -f "$CASE_FILE"
test -f "$gpu_lib"
test -d "$eval_env"
git -C "$PROJECT" diff --quiet
git -C "$PROJECT" diff --cached --quiet

mkdir -p "$OUTPUT_DIR"
: "${gpu_selector:?CUDA_VISIBLE_DEVICES is required}"
mapfile -t compute_processes < <(
  nvidia-smi --id="$gpu_selector" \
    --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
  printf 'the allocated GPU is not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi
nvidia-smi --id="$gpu_selector" \
  --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$OUTPUT_DIR/gpu_preflight.csv"
git -C "$PROJECT" rev-parse HEAD > "$OUTPUT_DIR/code_commit.txt"

source "$eval_env/bin/activate"
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PROJECT:$eval_env/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

cd /
python3 -m stellarator_eval.cli \
  --case-file "$CASE_FILE" \
  --key raw \
  --output-dir "$OUTPUT_DIR" \
  --current-unit A \
  --a "$A_VALUE" \
  --levels "$levels" \
  --max-boozer-candidates 0 \
  --psi-backend fullgpu \
  --psi-linear-solver qr \
  --psi-normal-eq-precision fp32 \
  --psi-gpu-lib "$gpu_lib" \
  --axis-gpu-lib "$gpu_lib" \
  --screen-gpu-lib "$gpu_lib" \
  --surface-gpu-lib "$gpu_lib"
