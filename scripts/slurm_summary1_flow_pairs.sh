#!/usr/bin/env bash
#SBATCH --job-name=summary1-flow-pair
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/summary1-flow-pair-%A_%a.out
#SBATCH --error=logs/summary1-flow-pair-%A_%a.err
#SBATCH --array=0-3

set -euo pipefail

# DEPRECATED HISTORICAL 2-DIRECTION PROTOCOL (2026-08-24).
# Retained only as exact provenance for the old 32-case comparison. Current
# main forbids this protocol. Any future special comparison requires a new
# protocol ID, launcher, manifest, and explicit review.
echo "deprecated historical 2-direction launcher; no new runs are permitted" >&2
exit 64

project="${PROJECT:?PROJECT is required}"
run_root="${RUN_ROOT:?RUN_ROOT is required}"
native_lib="${NATIVE_LIB:?NATIVE_LIB is required}"
checkpoint="${CHECKPOINT:?CHECKPOINT is required}"
source "${VENV:-$HOME/coil/.venv}/bin/activate"
cd "$project"
mkdir -p logs "$run_root"

gpu_record="$run_root/gpu_preflight_${SLURM_ARRAY_TASK_ID}.csv"
gpu_post="$run_root/gpu_postflight_${SLURM_ARRAY_TASK_ID}.csv"
zombie_post="$run_root/zombies_postflight_${SLURM_ARRAY_TASK_ID}.txt"
assigned="${CUDA_VISIBLE_DEVICES%%,*}"
cleanup() {
  status=$?
  nvidia-smi -i "$assigned" --query-gpu=index,uuid,utilization.gpu,memory.used \
    --format=csv,noheader,nounits > "$gpu_post" 2>/dev/null || true
  ps -u "$USER" -o pid=,ppid=,stat=,comm= | awk '$3 ~ /^Z/ {print}' \
    > "$zombie_post" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

if nvidia-smi -i "$assigned" --query-compute-apps=pid \
    --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "allocated GPU is not idle before timing" >&2
  exit 42
fi
nvidia-smi -i "$assigned" --query-gpu=index,uuid,utilization.gpu,memory.used \
  --format=csv,noheader,nounits > "$gpu_record"

run_case() {
  local case_index="$1"
  local case_fields case_path case_id nfp ncoils seed
  readarray -t case_fields < <(python - "$run_root/pair_manifest.json" "$case_index" <<'PY'
import json
import sys
row = json.load(open(sys.argv[1], encoding="utf-8"))["cases"][int(sys.argv[2])]
for key in ("case_path", "case_id", "nfp", "n_base_coils"):
    print(row[key])
PY
  )
  case_path="${case_fields[0]}"
  case_id="${case_fields[1]}"
  nfp="${case_fields[2]}"
  ncoils="${case_fields[3]}"
  seed=$((2026082500 + case_index))

  python scripts/optimize_flow_prior_local_full_gradient_adam.py \
    --checkpoint "$checkpoint" \
    --initial-case "$case_path" \
    --lib "$native_lib" \
    --out-dir "$run_root/$case_id/latent" \
    --nfp "$nfp" \
    --n-base-coils "$ncoils" \
    --iterations 100 \
    --max-wall-s 4200 \
    --flow-steps 128 \
    --parameter-space latent \
    --perturbation 0.005 \
    --gradient-mode random-orthogonal \
    --random-directions 2 \
    --seed "$seed" \
    --optimizer adam \
    --learning-rate 0.01 \
    --beta1 0.7 \
    --beta2 0.999 \
    --flow-device 0 \
    --score-device 0 \
    --flow-pipeline \
    --plot-every 0 \
    --trajectory-every 0 \
    --progress-every 20 \
    --state-every 100

  python scripts/optimize_flow_prior_local_full_gradient_adam.py \
    --checkpoint "$checkpoint" \
    --initial-case "$case_path" \
    --lib "$native_lib" \
    --out-dir "$run_root/$case_id/data" \
    --nfp "$nfp" \
    --n-base-coils "$ncoils" \
    --iterations 100 \
    --max-wall-s 4200 \
    --flow-steps 128 \
    --parameter-space data \
    --perturbation 0.0025 \
    --gradient-mode random-orthogonal \
    --random-directions 2 \
    --seed "$seed" \
    --optimizer adam \
    --learning-rate 0.01 \
    --beta1 0.7 \
    --beta2 0.999 \
    --flow-device 0 \
    --score-device 0 \
    --plot-every 0 \
    --trajectory-every 0 \
    --progress-every 20 \
    --state-every 100
}

case_count="$(python - "$run_root/pair_manifest.json" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))["cases"]))
PY
)"
for ((case_index=SLURM_ARRAY_TASK_ID; case_index<case_count; case_index+=SLURM_ARRAY_TASK_COUNT)); do
  run_case "$case_index"
done
