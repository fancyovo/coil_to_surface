#!/usr/bin/env bash
#SBATCH --job-name=freeze-psi-qr
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/freeze-psi-qr-%j.out
#SBATCH --error=logs/freeze-psi-qr-%j.err

set -euo pipefail

project=${PROJECT:?PROJECT is required}
output_dir=${OUTPUT_DIR:?OUTPUT_DIR is required}
manifest=${MANIFEST:?MANIFEST is required}
case_id=${CASE_ID:-1739363}
cuda_root=${CUDA_ROOT:-/public/app/cuda/13.0}
python_bin=${PYTHON_BIN:-/home/scc/pb24511935/coil/.venv/bin/python}

mkdir -p "$project/logs" "$output_dir"
cd "$project"

export PATH="$cuda_root/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_root/lib64:${LD_LIBRARY_PATH:-}"

gpu_state_before=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)
if [[ "$gpu_state_before" -ne 0 ]]; then
    echo "allocated GPU is not idle before snapshot generation" >&2
    exit 1
fi

cmake -S gpu_backend -B build/qr_bench \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DSGPU_BUILD_QR_BENCHMARK=ON
cmake --build build/qr_bench -j 4

target_manifest="$output_dir/target_case.json"
"$python_bin" - "$manifest" "$target_manifest" "$case_id" <<'PY'
import json
import sys
from pathlib import Path

source = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
case_id = int(sys.argv[3])
matches = [case for case in source["cases"] if int(case["case_id"]) == case_id]
if len(matches) != 1:
    raise SystemExit(f"expected one case {case_id}, found {len(matches)}")
target = {key: value for key, value in source.items() if key != "cases"}
target["selected_count"] = 1
target["cases"] = matches
Path(sys.argv[2]).write_text(json.dumps(target, indent=2, allow_nan=True) + "\n", encoding="utf-8")
PY

snapshot="$output_dir/case_${case_id}_psi_qr_f32.bin"
export SGPU_PSI_QR_SNAPSHOT="$snapshot"
"$python_bin" scripts/benchmark_score_eval_hinted.py \
    --manifest "$target_manifest" \
    --lib build/qr_bench/libstellarator_gpu.so \
    --output "$output_dir/freeze_score.jsonl" \
    --device 0 --variants baseline --repeats 1 --warmups 0
unset SGPU_PSI_QR_SNAPSHOT

sha256sum "$snapshot" > "$snapshot.sha256"
stat --printf='%n %s bytes\n' "$snapshot" > "$output_dir/snapshot_size.txt"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader > "$output_dir/gpu.txt"
printf '{"status":"ok","job_id":"%s","case_id":%s,"snapshot":"%s"}\n' \
    "$SLURM_JOB_ID" "$case_id" "$snapshot" > "$output_dir/freeze_done.json"

if pgrep -u "$USER" -f 'benchmark_score_eval_hinted|psi_qr_benchmark' >/dev/null; then
    echo "snapshot worker remains after completion" >&2
    exit 1
fi
