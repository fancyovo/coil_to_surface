#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-random-adam-admin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
run_root="${RUN_ROOT:?RUN_ROOT is required}"
expected_commit="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
admin_mode="${ADMIN_MODE:?ADMIN_MODE is required}"
source "${VENV:-$HOME/coil/.venv}/bin/activate"
cd "$project"
test "$(git rev-parse HEAD)" = "$expected_commit"
test -z "$(git status --porcelain --untracked-files=no)"
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

case "$admin_mode" in
  prepare)
    python scripts/qh_data_space_random_adam_followup.py prepare \
      --survey-root "${SURVEY_ROOT:?SURVEY_ROOT is required}" \
      --run-root "$run_root" \
      --run-label "${RUN_LABEL:?RUN_LABEL is required}" \
      --checkpoint "${CHECKPOINT:?CHECKPOINT is required}" \
      --lib "${SCORE_LIB:?SCORE_LIB is required}" \
      --gradient-lib "${GRADIENT_LIB:?GRADIENT_LIB is required}" \
      --expected-gradient-lib-sha "${EXPECTED_GRADIENT_LIB_SHA:?EXPECTED_GRADIENT_LIB_SHA is required}" \
      --reference-case "${REFERENCE_CASE:?REFERENCE_CASE is required}" \
      --worker-count "${WORKER_COUNT:-6}" \
      --selection-seed "${SELECTION_SEED:-2026082901}" \
      --low-ok-quotas "${LOW_OK_QUOTAS:-1:4,2:8,3:7,4:7,5:12}" \
      --expected-selected-count "${EXPECTED_SELECTED_COUNT:-76}" \
      --expected-eligible-count "${EXPECTED_ELIGIBLE_COUNT:-72}"
    ;;
  summarize)
    summary_args=(--run-root "$run_root")
    if [[ -n "${SUMMARY_ALLOW_PARTIAL:-}" ]]; then
      summary_args+=(--allow-partial)
    fi
    python scripts/qh_data_space_random_adam_followup.py summarize "${summary_args[@]}"
    ;;
  *)
    echo "unsupported ADMIN_MODE=$admin_mode" >&2
    exit 64
    ;;
esac
