#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=qh-flow-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo="${QH_FLOW_REPO:-$HOME/local_surface_evaluator}"
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
python3 - <<'PY'
import json
import matplotlib
import torch

from flow_matching.model import CoilFlowTransformer

model = CoilFlowTransformer().cuda().eval()
tokens = torch.randn(8, 5, 100, device="cuda")
time = torch.rand(8, device="cuda")
nfp = torch.full((8,), 4, dtype=torch.long, device="cuda")
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(tokens, time, nfp)
torch.cuda.synchronize()
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(),
    "bf16": torch.cuda.is_bf16_supported(),
    "matplotlib": matplotlib.__version__,
    "parameter_count": model.parameter_count,
    "output_shape": list(output.shape),
    "finite": bool(torch.isfinite(output).all()),
}))
PY
pytest -q tests/test_flow_data.py tests/test_flow_geometry.py tests/test_flow_model.py tests/test_qh_flow_export.py
