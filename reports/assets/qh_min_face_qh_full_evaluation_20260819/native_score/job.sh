#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=min-face-qh-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${CASE_FILE:?CASE_FILE is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${SCORE_LIB:?SCORE_LIB is required}"

for path in "$PROJECT" "$CASE_FILE" "$OUTPUT_DIR" "$SCORE_LIB"; do
    resolved=$(realpath -m "$path")
    [[ $resolved == "$HOME"/* ]] || {
        printf 'path must stay under HOME: %s\n' "$resolved" >&2
        exit 2
    }
done

test -f "$CASE_FILE"
test -f "$SCORE_LIB"
mkdir -p "$OUTPUT_DIR/cases"
ln -s "$CASE_FILE" "$OUTPUT_DIR/cases/id_0000001.json"

module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mapfile -t compute_processes < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
    printf 'allocated GPU is not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
    exit 42
fi

source "$HOME/coil/.venv/bin/activate"
git -C "$PROJECT" rev-parse HEAD > "$OUTPUT_DIR/code_commit.txt"
sha256sum "$SCORE_LIB" > "$OUTPUT_DIR/score_library.sha256"
sha256sum "$CASE_FILE" > "$OUTPUT_DIR/input.sha256"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$OUTPUT_DIR/gpu_preflight.csv"

cd "$PROJECT"
python scripts/batch_native_score.py \
    --case-dir "$OUTPUT_DIR/cases" \
    --output "$OUTPUT_DIR/result.jsonl" \
    --lib "$SCORE_LIB" \
    --device 0 \
    --warmup \
    > "$OUTPUT_DIR/job_summary.json"

nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$OUTPUT_DIR/gpu_postflight.csv"
