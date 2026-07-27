#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=audit-axis-fallback
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --exclude=anode02
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project=/home/scc/pb24511935/local_surface_evaluator
case_dir=/home/scc/pb24511935/local_surface_evaluator_data/volume_score_2000/cases
source_run=${SOURCE_RUN:-runs/native_score/calibration_1000_27717}
fallback_grid=${FALLBACK_GRID:-64}
fallback_candidates=${FALLBACK_CANDIDATES:-48}
fallback_newton=${FALLBACK_NEWTON:-6}
nfp_min=${NFP_MIN:-0}
output_dir="$project/runs/native_score/axis_fallback${fallback_grid}_${SLURM_JOB_ID}"

cd "$project"
mapfile -t compute_processes < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
    printf 'allocated GPU is not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
    exit 42
fi

source /home/scc/pb24511935/coil/.venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$output_dir"
python - "$project/$source_run" "$output_dir/case_ids.txt" "$nfp_min" <<'PY'
import glob
import json
from pathlib import Path
import sys

rows = []
for path in glob.glob(str(Path(sys.argv[1]) / "worker_*.jsonl")):
    rows.extend(json.loads(line) for line in open(path, encoding="utf-8"))
ids = sorted(
    int(row["case_id"])
    for row in rows
    if row["native_score"]["diagnostics"]["axis_candidate_count"] == 16
    and int(row["nfp"]) >= int(sys.argv[3])
)
Path(sys.argv[2]).write_text("".join(f"{case_id}\n" for case_id in ids), encoding="utf-8")
PY

python scripts/batch_native_score.py \
    --case-dir "$case_dir" \
    --case-id-file "$output_dir/case_ids.txt" \
    --output "$output_dir/worker_0.jsonl" \
    --axis-fallback-grid "$fallback_grid" \
    --axis-fallback-max-candidates "$fallback_candidates" \
    --axis-fallback-newton-iters "$fallback_newton" \
    --warmup \
    > "$output_dir/worker_0.summary.json"
