#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=grid48-default-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project="${PROJECT:?PROJECT must point to the score-eval-compression checkout}"
output_dir="${OUTPUT_DIR:-$HOME/local_surface_evaluator_runs/grid48_default_smoke_${SLURM_JOB_ID}}"
build_dir="$project/gpu_backend/build_grid48_default_smoke_cuda13"
manifest="${MANIFEST:-$project/reports/assets/qh_psi_grid_reduction_20260810/cases.json}"

cleanup() {
    status=$?
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits > "$output_dir/gpu_postflight.csv" 2>/dev/null || true
    ps -u "$USER" -o pid=,ppid=,stat=,comm= | awk '$3 ~ /^Z/ {print}' \
        > "$output_dir/zombies_postflight.txt" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$output_dir" "$project/logs"
cd "$project"

assigned="${CUDA_VISIBLE_DEVICES%%,*}"
if nvidia-smi -i "$assigned" --query-compute-apps=pid --format=csv,noheader,nounits |
        grep -q '[0-9]'; then
    echo "allocated GPU is not idle" >&2
    exit 42
fi
nvidia-smi -i "$assigned" --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$output_dir/gpu_preflight.csv"

module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
source "$HOME/coil/.venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cmake -S gpu_backend -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build "$build_dir" -j4
lib="$build_dir/libstellarator_gpu.so"

python - "$project" "$lib" "$output_dir/default_config.json" <<'PY'
import ctypes
import json
from pathlib import Path
import sys

project = Path(sys.argv[1])
library_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
sys.path.insert(0, str(project / "gpu_backend" / "python"))
from stellarator_gpu import _SgpuScoreConfig, _bind_native_score

library = ctypes.CDLL(str(library_path))
_bind_native_score(library)
config = _SgpuScoreConfig()
code = library.sgpu_default_score_config(ctypes.byref(config))
if code != 0:
    raise RuntimeError(f"sgpu_default_score_config failed with code {code}")
values = {
    "psi_n_r": int(config.psi_n_r),
    "psi_n_z": int(config.psi_n_z),
    "psi_n_phi": int(config.psi_n_phi),
}
if tuple(values.values()) != (48, 48, 48):
    raise RuntimeError(f"unexpected default psi grid: {values}")
output_path.write_text(json.dumps(values, indent=2), encoding="utf-8")
PY

python scripts/benchmark_score_eval_hinted.py \
    --manifest "$manifest" --lib "$lib" --output "$output_dir/benchmark.jsonl" \
    --device 0 --variants baseline --case-limit 1 --repeats 1 --warmups 1

python - "$output_dir/benchmark.jsonl" "$output_dir/done.json" <<'PY'
import json
from pathlib import Path
import sys

benchmark_path = Path(sys.argv[1])
done_path = Path(sys.argv[2])
row = json.loads(benchmark_path.read_text(encoding="utf-8").strip())
if row["result"]["status"] != "ok":
    raise RuntimeError(f"default-grid score smoke failed: {row['result']['status']}")
if any(key.startswith("psi_n_") for key in row["config_overrides"]):
    raise RuntimeError("baseline smoke unexpectedly overrides the default psi grid")
done_path.write_text(
    json.dumps(
        {
            "status": "ok",
            "score": row["result"]["score"],
            "caller_wall_s": row["caller_wall_s"],
            "psi_fit_s": row["result"]["timing"]["psi_fit_s"],
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY

sha256sum "$lib" > "$output_dir/library.sha256"
git rev-parse HEAD > "$output_dir/git_head.txt"
git status --short > "$output_dir/git_status.txt"
