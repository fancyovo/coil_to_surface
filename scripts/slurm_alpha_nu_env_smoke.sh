#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=alpha-nu-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project=/home/scc/pb24511935/local_surface_evaluator
eval_env=${EVAL_ENV:-$project/.venv-desc016-py312}
cd "$project"

export CUDA_HOME=/public/app/cuda/13.0
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$project:$eval_env/lib/python3.12/site-packages:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cuda_wheel_lib="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
test -f "$cuda_wheel_lib/libcusolver.so.12"
export LD_LIBRARY_PATH="$cuda_wheel_lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 - <<'PY'
import json
import numpy as np
import scipy
import simsopt
import torch

A = torch.randn((8192, 256), dtype=torch.float64, device="cuda")
b = torch.randn((8192, 1), dtype=torch.float64, device="cuda")
x = torch.linalg.lstsq(A, b, driver="gels").solution
torch.cuda.synchronize()
assert torch.isfinite(x).all()
print(json.dumps({
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "simsopt": simsopt.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(),
    "qr_finite": True,
}))
PY
