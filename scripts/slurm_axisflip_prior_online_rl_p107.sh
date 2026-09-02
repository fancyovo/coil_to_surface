#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=axis-prior-rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo="${AXIS_RL_REPO:?set AXIS_RL_REPO}"
dataset="${AXIS_RL_DATASET:?set AXIS_RL_DATASET}"
run_root="${AXIS_RL_RUN_ROOT:?set AXIS_RL_RUN_ROOT}"
commit="${AXIS_RL_COMMIT:?set AXIS_RL_COMMIT}"
score_lib="${AXIS_RL_SCORE_LIB:?set AXIS_RL_SCORE_LIB}"
optimizer_checkpoint="${AXIS_RL_OPTIMIZER_CHECKPOINT:?set AXIS_RL_OPTIMIZER_CHECKPOINT}"
optimizer_checkpoint_sha="${AXIS_RL_OPTIMIZER_CHECKPOINT_SHA:?set AXIS_RL_OPTIMIZER_CHECKPOINT_SHA}"
max_rounds="${AXIS_RL_MAX_ROUNDS:-48}"

cd "$repo"
mkdir -p "$run_root" "$repo/logs"
test -f "$dataset/dataset_manifest.json"
test -f "$score_lib"
test -f "$optimizer_checkpoint"
export PYTHONPATH="$repo:$repo/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cuda_wheel_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib")')"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

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

idle_streak=0
for _ in {1..60}; do
  idle=1
  while IFS= read -r memory_used; do
    memory_used="${memory_used// /}"
    if (( memory_used > 32 )); then idle=0; fi
  done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
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
  echo "allocated GPUs retained memory or compute processes during idle probes" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
  > "$run_root/gpu_preflight.csv"

python -m torch.distributed.run --standalone --nproc-per-node=4 \
  scripts/train_axisflip_prior_flow.py \
  --dataset-dir "$dataset" \
  --output-dir "$run_root/distillation" \
  --expected-commit "$commit" \
  --batch-per-gpu 256 \
  --validation-batch-per-gpu 256 \
  --learning-rate 1e-4 \
  --ema-decay 0.999 \
  --minimum-epochs 10 \
  --maximum-epochs 200 \
  --patience 10 \
  --minimum-relative-improvement 0.003 \
  --monitor-count 512 \
  --monitor-flow-steps 32 \
  --width 256 \
  --layers 6 \
  --heads 8 \
  --hidden 704

python scripts/axisflip_prior_online_rl.py prepare \
  --run-root "$run_root" \
  --distillation-dir "$run_root/distillation" \
  --dataset-dir "$dataset" \
  --score-lib "$score_lib" \
  --optimizer-checkpoint "$optimizer_checkpoint" \
  --expected-optimizer-checkpoint-sha "$optimizer_checkpoint_sha" \
  --expected-commit "$commit" \
  --sample-seed 2026090302 \
  --train-steps 250 \
  --train-batch-per-gpu 256 \
  --learning-rate 5e-5

children=()
for worker in 0 1 2 3; do
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
for ((round_index=0; round_index<max_rounds; round_index+=1)); do
  if [[ -f "$run_root/STOP_AFTER_ROUND" ]]; then
    echo "graceful stop sentinel observed before round $round_index"
    break
  fi
  if (( SECONDS - job_started >= 82800 )); then
    echo "stopping before round $round_index with one-hour Slurm reserve"
    break
  fi
  round_label="$(printf '%03d' "$round_index")"
  mkdir -p "$run_root/rounds/round_${round_label}"
  children=()
  for worker in 0 1 2 3; do
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
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
    scripts/axisflip_prior_online_rl.py train-round \
    --run-root "$run_root" \
    --round-index "$round_index"
done

echo "axis-prior online RL stopped cleanly; run_root=$run_root"
