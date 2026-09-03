#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${EXPECTED_CHECKPOINT_SHA:?EXPECTED_CHECKPOINT_SHA is required}"

cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA"
source "$HOME/coil/.venv/bin/activate"
module load cuda/13.0 2>/dev/null || true
export CUDA_HOME=/public/app/cuda/13.0
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT:$PROJECT/gpu_backend/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

build_dir="$RUN_ROOT/build_curvature_r04"
cmake -S "$PROJECT/gpu_backend" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSGPU_COIL_CURVATURE_P95_SCALE=25.0
cmake --build "$build_dir" --parallel 4
score_lib="$build_dir/libstellarator_gpu.so"
library_sha=$(sha256sum "$score_lib" | awk '{print $1}')
printf '%s\n' "$library_sha" > "$RUN_ROOT/score_library.sha256"
python - "$RUN_ROOT/score_library_manifest.json" "$score_lib" "$library_sha" "$EXPECTED_COMMIT" <<'PY'
import json
from pathlib import Path
import sys

path, library, sha256, commit = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "format": "axisflip_r012_curvature_r04_score_library_v1",
            "interface_abi": 11,
            "path": library,
            "sha256": sha256,
            "code_commit": commit,
            "cmake_overrides": {"SGPU_COIL_CURVATURE_P95_SCALE": 25.0},
            "coil_curvature_p95_radius_m": 0.04,
            "coil_curvature_max_scale_m_inv": 35.0,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

smoke_root="$RUN_ROOT/smoke_validation"
python "$PROJECT/scripts/run_axis_surface_prior_axisflip_stream.py" \
  --run-root "$smoke_root" \
  --protocol-path "$RUN_ROOT/protocol.json" \
  --protocol-id qh-axis-surface-contour-compact-flexible-axisflip-r012-curvature-r04-adam200-64d-abi11-v1 \
  --checkpoint "$CHECKPOINT" \
  --score-lib "$score_lib" \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-lib-sha "$library_sha" \
  --expected-checkpoint-sha "$EXPECTED_CHECKPOINT_SHA" \
  --worker-index 0 \
  --worker-count 1 \
  --seed 20260906 \
  --device 0 \
  --discovery-wall-s 1800 \
  --hard-wall-s 2280 \
  --iterations 3 \
  --max-completed-cases 1 \
  --legality-audit-cases-per-worker 1 \
  --fixed-nfp 8 \
  --fixed-n-base-coils 3 \
  --minor-radius-m 0.12 \
  --smoke
python - "$smoke_root/workers/worker_00/done.json" "$smoke_root" "$library_sha" <<'PY'
import json
from pathlib import Path
import sys

done_path, smoke_root, expected_sha = sys.argv[1:]
done = json.loads(Path(done_path).read_text(encoding="utf-8"))
if done["completed_adam_count"] != 1 or done["failed_valid_count"] != 0:
    raise SystemExit("custom-score smoke did not complete exactly one trajectory")
manifests = list((Path(smoke_root) / "trajectories").glob("*/trajectory_manifest.json"))
if len(manifests) != 1:
    raise SystemExit("custom-score smoke trajectory count mismatch")
manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
if manifest["provenance"]["score_library_sha256"] != expected_sha:
    raise SystemExit("custom-score smoke library identity mismatch")
if manifest["initial_consistency_gate"]["absolute_delta"] > 0.1:
    raise SystemExit("custom-score smoke failed its initial consistency gate")
PY
