#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-survey-admin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
expected_commit="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
admin_mode="${ADMIN_MODE:?ADMIN_MODE is required}"
run_root="${RUN_ROOT:?RUN_ROOT is required}"

source "${VENV:-$HOME/coil/.venv}/bin/activate"
cd "$project"
test "$(git rev-parse HEAD)" = "$expected_commit"
test -z "$(git status --porcelain --untracked-files=no)"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

case "$admin_mode" in
  prepare)
    test ! -e "$run_root"
    python scripts/qh_data_space_random_survey.py prepare \
      --run-root "$run_root" \
      --run-label "${RUN_LABEL:?RUN_LABEL is required}" \
      --checkpoint "${CHECKPOINT:?CHECKPOINT is required}" \
      --lib "${SCORE_LIB:?SCORE_LIB is required}" \
      --reference-case "${REFERENCE_CASE:?REFERENCE_CASE is required}" \
      --data-dir "${DATA_DIR:?DATA_DIR is required}" \
      --worker-count "${WORKER_COUNT:-6}" \
      --worker-samples-per-condition "${WORKER_SAMPLES_PER_CONDITION:-2}" \
      --expected-target-sample-count "${EXPECTED_TARGET_SAMPLE_COUNT:?EXPECTED_TARGET_SAMPLE_COUNT is required}" \
      --seed-base "${SEED_BASE:-2026082801}" \
      --tail-retention-score "${TAIL_RETENTION_SCORE:-20.0}" \
      --expected-checkpoint-sha "${EXPECTED_CHECKPOINT_SHA:-39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f}" \
      --expected-lib-sha "${EXPECTED_LIB_SHA:-565c32073b145d97a1f2244705fb06e4b3458ce798cd74d0c97ee4e0129dc729}" \
      --expected-reference-case-sha "${EXPECTED_REFERENCE_CASE_SHA:-6ee6f8e1f0290ec49093596a5f95b7f2aac98c61d51af3cad59410a771b7e8c1}" \
      --expected-reference-score "${EXPECTED_REFERENCE_SCORE:-94.62541477362565}" \
      --reference-score-atol "${REFERENCE_SCORE_ATOL:-1e-5}"
    ;;
  summarize)
    python scripts/qh_data_space_random_survey.py summarize \
      --run-root "$run_root" \
      --followup-seed "${FOLLOWUP_SEED:-2026082802}" \
      --top-count "${TOP_COUNT:-64}"
    ;;
  calibrate)
    python scripts/qh_data_space_random_survey.py calibrate \
      --run-root "$run_root" \
      --target-wall-s "${TARGET_WALL_S:-36000}" \
      --safety-s "${SAFETY_S:-600}" \
      --output "${CALIBRATION_OUTPUT:?CALIBRATION_OUTPUT is required}"
    ;;
  *)
    echo "unsupported ADMIN_MODE: $admin_mode" >&2
    exit 2
    ;;
esac
