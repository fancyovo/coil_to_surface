#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-flow-rescore-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project=/home/scc/pb24511935/local_surface_evaluator
input_dir=${INPUT_DIR:-$project/runs/qh_flow_eval_28546}
output_dir=${OUTPUT_DIR:-$project/runs/qh_flow_rescore_v3_${SLURM_JOB_ID}}
lib=${SCORE_LIB:-$project/gpu_backend/build_native_score/libstellarator_gpu.so}
candidate_ids=${CANDIDATE_IDS:-}
world_size=4
children=()

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if (( ${#children[@]} )); then
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used \
        --format=csv,noheader,nounits > "$output_dir/gpu_postflight.csv" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

cd "$project"
module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
source /home/scc/pb24511935/coil/.venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

test -f "$lib"
for rank in 0 1 2 3; do
    test -f "$input_dir/rank_$(printf '%02d' "$rank").jsonl"
done
mkdir -p "$output_dir"

mapfile -t compute_processes < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
    printf 'an allocated GPU is not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
    exit 42
fi
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used \
    --format=csv,noheader,nounits > "$output_dir/gpu_preflight.csv"

started=$(date +%s.%N)
library_sha256=$(sha256sum "$lib" | cut -d' ' -f1)
printf '{"input_dir":"%s","candidate_ids":"%s","world_size":%d,"library_sha256":"%s","started":%s}\n' \
    "$input_dir" "$candidate_ids" "$world_size" "$library_sha256" "$started" \
    > "$output_dir/job.json"

for rank in 0 1 2 3; do
    python scripts/rescore_qh_flow_saved.py \
        --input-dir "$input_dir" \
        --output-dir "$output_dir" \
        --lib "$lib" \
        --rank "$rank" \
        --world-size "$world_size" \
        --candidate-ids "$candidate_ids" \
        > "$output_dir/rank_$(printf '%02d' "$rank").log" 2>&1 &
    children+=("$!")
done

for child in "${children[@]}"; do
    wait "$child"
done
children=()

python scripts/rescore_qh_flow_saved.py \
    --input-dir "$input_dir" \
    --output-dir "$output_dir" \
    --lib "$lib" \
    --analyze-only

finished=$(date +%s.%N)
python - "$output_dir/job.json" "$finished" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["finished"] = float(sys.argv[2])
payload["wall_s"] = payload["finished"] - payload["started"]
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
