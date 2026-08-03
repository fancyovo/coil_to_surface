#!/usr/bin/env bash
set -euo pipefail

project="${PROJECT:?PROJECT is required}"
asset_root="${ASSET_ROOT:-$HOME/local_surface_evaluator}"
data_root="${DATA_ROOT:-$HOME/local_surface_evaluator_data}"
checkpoint="${FLOW_CHECKPOINT:-$asset_root/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt}"
training_manifest="${TRAINING_RUN_MANIFEST:-$(dirname "$checkpoint")/run_manifest.json}"
lib="${SCORE_LIB:-$project/gpu_backend/build_native_score/libstellarator_gpu.so}"
qh_data="${QH_DATA_DIR:-$data_root/quasr_qh_flow_v1}"
dataset_root="${SCORE_CORPUS_ROOT:-$data_root/qh_iid_score_corpus_v1}"
expected_flow_sha="${EXPECTED_FLOW_SHA:-39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f}"
expected_lib_sha="${EXPECTED_SCORE_LIB_SHA:-4bf7a12ea3dbdef9faf6de3ce4dc1840ecf48847ba795267500dd4179f730708}"
gpu_count="${GPU_COUNT:?GPU_COUNT is required}"
job_id="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
audit_dir="$dataset_root/job_audit/slurm_$job_id"
launcher_pid=""

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$launcher_pid" ]]; then
    pkill -TERM -P "$launcher_pid" 2>/dev/null || true
    kill "$launcher_pid" 2>/dev/null || true
    wait "$launcher_pid" 2>/dev/null || true
  fi
  if [[ -d "$audit_dir" ]]; then
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits > "$audit_dir/gpu_postflight.csv" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$audit_dir" "$project/logs"
cd "$project"
test -f "$checkpoint"
test -f "$training_manifest"
test -f "$lib"
test -f "$qh_data/manifest.json"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_flow_sha"
test "$(sha256sum "$lib" | awk '{print $1}')" = "$expected_lib_sha"
source "$HOME/coil/.venv/bin/activate"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mapfile -t compute_processes < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
  printf 'allocated GPUs are not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$audit_dir/gpu_preflight.csv"
git rev-parse HEAD > "$audit_dir/code_commit.txt"
printf '%s\n' "$checkpoint" > "$audit_dir/checkpoint_path.txt"
printf '%s\n' "$lib" > "$audit_dir/score_library_path.txt"

torchrun --standalone --nproc-per-node="$gpu_count" \
  scripts/collect_qh_iid_score_data.py \
  --checkpoint "$checkpoint" \
  --lib "$lib" \
  --data-dir "$qh_data" \
  --training-run-manifest "$training_manifest" \
  --dataset-root "$dataset_root" \
  --flow-steps "${FLOW_STEPS:-256}" \
  --shard-size "${SHARD_SIZE:-64}" \
  --max-shards "${MAX_SHARDS:-0}" \
  --max-wall-s "${MAX_WALL_S:-85800}" \
  --seed-base "${SEED_BASE:-$job_id}" \
  --job-id "$job_id" &
launcher_pid=$!
wait "$launcher_pid"
launcher_pid=""

nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$audit_dir/gpu_postflight.csv"
python scripts/summarize_qh_iid_score_corpus.py "$dataset_root" > "$audit_dir/corpus_summary_at_exit.json"
