#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=axisrl-radius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:RTX5090:2
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo="${AXIS_RL_REPO:?set AXIS_RL_REPO}"
dataset="${AXIS_RL_DATASET:?set AXIS_RL_DATASET}"
run_root="${AXIS_RL_RUN_ROOT:?set AXIS_RL_RUN_ROOT}"
commit="${AXIS_RL_COMMIT:?set AXIS_RL_COMMIT}"
score_lib="${AXIS_RL_SCORE_LIB:?set AXIS_RL_SCORE_LIB}"
score_lib_manifest="${AXIS_RL_SCORE_LIB_MANIFEST:?set AXIS_RL_SCORE_LIB_MANIFEST}"
score_lib_sha="${AXIS_RL_SCORE_LIB_SHA:?set AXIS_RL_SCORE_LIB_SHA}"
optimizer_checkpoint="${AXIS_RL_OPTIMIZER_CHECKPOINT:?set AXIS_RL_OPTIMIZER_CHECKPOINT}"
optimizer_checkpoint_sha="${AXIS_RL_OPTIMIZER_CHECKPOINT_SHA:?set AXIS_RL_OPTIMIZER_CHECKPOINT_SHA}"
minor_radius_m="${AXIS_RL_MINOR_RADIUS_M:?set AXIS_RL_MINOR_RADIUS_M}"
sample_seed="${AXIS_RL_SAMPLE_SEED:?set AXIS_RL_SAMPLE_SEED}"
protocol_manifest="${AXIS_RL_PROTOCOL_MANIFEST:?set AXIS_RL_PROTOCOL_MANIFEST}"
protocol_id="${AXIS_RL_PROTOCOL_ID:?set AXIS_RL_PROTOCOL_ID}"

case "$minor_radius_m" in
  0.15) generator_format="axis_surface_contour_prior_compact_flexible_axis_flip_r015_v1" ;;
  0.20) generator_format="axis_surface_contour_prior_compact_flexible_axis_flip_r020_v1" ;;
  *) echo "unsupported radius: $minor_radius_m" >&2; exit 2 ;;
esac
test -f "$protocol_manifest"
test "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["protocol_id"])' "$protocol_manifest")" = "$protocol_id"

cd "$repo"
mkdir -p "$repo/logs"
test ! -e "$run_root"
test -f "$score_lib"
test -f "$score_lib_manifest"
test -f "$optimizer_checkpoint"
export PYTHONPATH="$repo:$repo/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MPLBACKEND=Agg
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -e "$dataset/dataset_manifest.json" ]]; then
  echo "reusing completed teacher dataset: $dataset"
else
  test ! -e "$dataset" || { echo "dataset exists but is incomplete: $dataset" >&2; exit 1; }
  mkdir -p "$dataset"
  python scripts/generate_axisflip_prior_dataset.py generate \
    --output-dir "$dataset" \
    --shard-name p107_000000_200000 \
    --seed-start "$sample_seed" \
    --count 200000 \
    --workers 8 \
    --block-size 128 \
    --minor-radius-m "$minor_radius_m" \
    --expected-commit "$commit"
  python scripts/generate_axisflip_prior_dataset.py finalize \
    --output-dir "$dataset" \
    --shard-name p107_000000_200000 \
    --expected-total 200000 \
    --expected-commit "$commit"
fi
python - "$dataset/dataset_manifest.json" "$minor_radius_m" "$generator_format" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
expected = float(sys.argv[2])
expected_format = sys.argv[3]
if manifest.get("status") != "complete" or manifest.get("sample_count") != 200000:
    raise SystemExit("teacher dataset manifest is incomplete or has the wrong count")
generator = manifest["generator"]
if generator.get("format") != expected_format:
    raise SystemExit("teacher manifest generator format mismatch")
if generator.get("minor_radius_center_m") != expected:
    raise SystemExit("teacher manifest radius mismatch")
PY

export AXIS_RL_DISTILLATION_WORLD_SIZE=2
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/train_axisflip_prior_flow.py \
  --dataset-dir "$dataset" \
  --output-dir "$run_root/distillation" \
  --expected-commit "$commit" \
  --batch-per-gpu 256 \
  --validation-batch-per-gpu 256 \
  --learning-rate 1e-4 \
  --ema-decay 0.999 \
  --minimum-epochs 10 \
  --maximum-epochs 5000 \
  --patience 10 \
  --minimum-relative-improvement 0.003 \
  --monitor-count 512 \
  --monitor-flow-steps 32 \
  --width 256 \
  --layers 6 \
  --heads 8 \
  --hidden 704

export AXIS_RL_WORKER_COUNT=2
export AXIS_RL_SAMPLES_PER_WORKER=32
python scripts/axisflip_prior_online_rl.py prepare \
  --run-root "$run_root" \
  --distillation-dir "$run_root/distillation" \
  --dataset-dir "$dataset" \
  --score-lib "$score_lib" \
  --score-library-manifest "$score_lib_manifest" \
  --expected-score-lib-sha "$score_lib_sha" \
  --optimizer-checkpoint "$optimizer_checkpoint" \
  --expected-optimizer-checkpoint-sha "$optimizer_checkpoint_sha" \
  --expected-minor-radius-m "$minor_radius_m" \
  --expected-commit "$commit" \
  --sample-seed "$sample_seed" \
  --train-steps 250 \
  --train-batch-per-gpu 256 \
  --learning-rate 5e-5
cp "$protocol_manifest" "$run_root/protocol.json"

children=()
cleanup() {
  status=$?
  trap - EXIT INT TERM
  for child in "${children[@]:-}"; do
    if kill -0 "$child" 2>/dev/null; then kill "$child" 2>/dev/null || true; fi
  done
  wait 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
    > "$run_root/gpu_postflight.csv" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

for worker in 0 1; do
  python scripts/axisflip_prior_online_rl.py audit-worker \
    --run-root "$run_root" \
    --worker-index "$worker" \
    --device "$worker" \
    --samples-per-source 32 \
    > "$run_root/logs/q0_audit_worker_${worker}.log" 2>&1 &
  children+=("$!")
done
audit_status=0
for child in "${children[@]}"; do
  if ! wait "$child"; then audit_status=1; fi
done
children=()
if (( audit_status != 0 )); then
  echo "one or more q0 audit workers failed" >&2
  exit 1
fi
python scripts/axisflip_prior_online_rl.py summarize-audit --run-root "$run_root"

job_started=$SECONDS
round_index=0
while true; do
  if [[ -f "$run_root/STOP_AFTER_ROUND" ]]; then
    echo "graceful stop sentinel observed before round $round_index"
    break
  fi
  if (( SECONDS - job_started >= 342000 )); then
    echo "stopping before round $round_index with one-hour Slurm reserve"
    break
  fi
  round_label="$(printf '%03d' "$round_index")"
  mkdir -p "$run_root/rounds/round_${round_label}"
  children=()
  for worker in 0 1; do
    python scripts/axisflip_prior_online_rl.py collect-worker \
      --run-root "$run_root" \
      --round-index "$round_index" \
      --worker-index "$worker" \
      --device "$worker" \
      > "$run_root/logs/round_${round_label}_worker_${worker}.log" 2>&1 &
    children+=("$!")
  done
  collect_status=0
  for child in "${children[@]}"; do
    if ! wait "$child"; then collect_status=1; fi
  done
  children=()
  if (( collect_status != 0 )); then
    echo "one or more collection workers failed in round $round_index" >&2
    exit 1
  fi
  python scripts/axisflip_prior_online_rl.py summarize-round \
    --run-root "$run_root" \
    --round-index "$round_index"
  python -m torch.distributed.run --standalone --nproc-per-node=2 \
    scripts/axisflip_prior_online_rl.py train-round \
    --run-root "$run_root" \
    --round-index "$round_index"
  ((round_index += 1))
done

echo "axis-prior two-GPU online RL stopped cleanly; run_root=$run_root"
