#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-survey-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
expected_commit="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
build_dir="${BUILD_DIR:-$project/gpu_backend/build_qh_random_survey}"
validation_case="${VALIDATION_CASE:?VALIDATION_CASE is required}"
validation_output="${VALIDATION_OUTPUT:?VALIDATION_OUTPUT is required}"
expected_lib_sha="${EXPECTED_LIB_SHA:-565c32073b145d97a1f2244705fb06e4b3458ce798cd74d0c97ee4e0129dc729}"
expected_case_sha="${EXPECTED_CASE_SHA:-6ee6f8e1f0290ec49093596a5f95b7f2aac98c61d51af3cad59410a771b7e8c1}"
allow_rebuilt_lib="${ALLOW_REBUILT_LIB:-0}"
expected_score="${EXPECTED_SCORE:-94.62541477362565}"

source "${VENV:-$HOME/coil/.venv}/bin/activate"
cd "$project"
test "$(git rev-parse HEAD)" = "$expected_commit"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$validation_case"
module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S gpu_backend -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build "$build_dir" --target stellarator_gpu --parallel "$SLURM_CPUS_PER_TASK"
lib="$build_dir/libstellarator_gpu.so"
actual_lib_sha="$(sha256sum "$lib" | awk '{print $1}')"
printf 'score_library_sha256=%s\n' "$actual_lib_sha"
lib_hash_args=(--expected-lib-sha "$expected_lib_sha")
if [[ "$allow_rebuilt_lib" == "1" ]]; then
  lib_hash_args=()
else
  test "$actual_lib_sha" = "$expected_lib_sha"
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
printf '%s' 'gpu_reference_environment='
nvidia-smi --query-gpu=index,uuid,name,driver_version \
  --format=csv,noheader,nounits
python scripts/validate_native_score_reference.py \
  --case "$validation_case" \
  --lib "$lib" \
  --output "$validation_output" \
  --expected-case-sha "$expected_case_sha" \
  "${lib_hash_args[@]}" \
  --expected-score "$expected_score" \
  --score-atol 1e-5 \
  --device 0
