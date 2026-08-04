#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-g4-ref
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:-$HOME/local_surface_evaluator_worktrees/qh-blackbox-gradient}"
reference_dir="${REFERENCE_DIR:-$project/runs/qh_g2_current_basin_reference_20260804}"
closure="${CLOSURE:?CLOSURE must name the matching G2/G3 closure summary}"
center_index="${CENTER_INDEX:?CENTER_INDEX must be set}"
scale="${SCALE:-0.0025}"
gradient_lib="${GRADIENT_LIB:-$project/gpu_backend/build_g4_oracle/libstellarator_gpu.so}"
expected_gradient_sha="${EXPECTED_GRADIENT_SHA:-071f67925a8eca6cfc702ed6b8380f4677bb4c36711e6aae032b7dd227bdc88c}"
output="${OUTPUT_DIR:-$project/runs/qh_g4_reference_alignment_${center_index}_${SLURM_JOB_ID}}"
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
for path in "$reference_dir/manifest.json" "$reference_dir/summary.json" \
  "$reference_dir/latent_banks.npz" "$reference_dir/reference_gradients.npz" \
  "$reference_dir/raw_tokens.npy" "$reference_dir/cases.jsonl" "$closure" "$gradient_lib"; do
  test -f "$path"
done
test "$(sha256sum "$gradient_lib" | awk '{print $1}')" = "$expected_gradient_sha"

source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project:$project/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

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

python scripts/diagnose_qh_g4_reference_alignment.py \
  --reference-dir "$reference_dir" \
  --closure "$closure" \
  --gradient-lib "$gradient_lib" \
  --center-index "$center_index" \
  --scale "$scale" \
  --output-dir "$output" &
child="$!"
wait "$child"
child=""
test -s "$output/summary.json"
test -s "$output/directional_slopes.npz"
