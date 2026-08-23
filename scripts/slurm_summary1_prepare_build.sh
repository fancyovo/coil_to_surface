#!/usr/bin/env bash
#SBATCH --job-name=summary1-prepare
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

run_root="${RUN_ROOT:?RUN_ROOT is required}"
trajectory_root="${TRAJECTORY_ROOT:?TRAJECTORY_ROOT is required}"
expected_commit="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
bundle="$run_root/source.bundle"
project="$run_root/source"
build_dir="$run_root/build_native"
source "${VENV:-$HOME/coil/.venv}/bin/activate"
mkdir -p "$run_root/logs" "$run_root/flow_pairs"

git bundle verify "$bundle"
if [[ -e "$project" ]]; then
  actual_commit="$(git -C "$project" rev-parse HEAD)"
  if [[ "$actual_commit" != "$expected_commit" ]]; then
    echo "existing source commit $actual_commit != $expected_commit" >&2
    exit 43
  fi
else
  git clone --branch codex/summary1-project-report-qh "$bundle" "$project"
fi

actual_commit="$(git -C "$project" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "cloned source commit $actual_commit != $expected_commit" >&2
  exit 44
fi

module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cmake -S "$project/gpu_backend" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build "$build_dir" --target stellarator_gpu --parallel 4
python -m pytest "$project/tests/test_native_evaluator.py" -q
python "$project/scripts/prepare_summary1_flow_pairs.py" \
  --trajectory-root "$trajectory_root" \
  --output-dir "$run_root/flow_pairs" \
  --case-count 8

sha256sum "$bundle" > "$run_root/source_bundle.sha256"
sha256sum "$build_dir/libstellarator_gpu.so" > "$run_root/library.sha256"
git -C "$project" rev-parse HEAD > "$run_root/git_head.txt"
printf '{"status":"ok","job_id":"%s","commit":"%s"}\n' \
  "$SLURM_JOB_ID" "$actual_commit" > "$run_root/prepare_done.json"
