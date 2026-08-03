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

from flow_matching.flow import sample_heun
from flow_matching.model import CoilFlowTransformer

model = CoilFlowTransformer().cuda().eval()
tokens = torch.randn(8, 5, 100, device="cuda")
time = torch.rand(8, device="cuda")
nfp = torch.full((8,), 4, dtype=torch.long, device="cuda")
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(tokens, time, nfp)
torch.cuda.synchronize()
permutation = torch.tensor([2, 0, 4, 1, 3], device="cuda")
equivariant_reference = model(tokens.float(), time, nfp)[:, permutation]
equivariant_actual = model(tokens[:, permutation].float(), time, nfp)
torch.testing.assert_close(equivariant_actual, equivariant_reference, rtol=1.0e-5, atol=1.0e-5)
sample = sample_heun(model, tokens[:2], nfp[:2], steps=2)
assert torch.isfinite(sample).all()
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(),
    "bf16": torch.cuda.is_bf16_supported(),
    "matplotlib": matplotlib.__version__,
    "parameter_count": model.parameter_count,
    "output_shape": list(output.shape),
    "finite": bool(torch.isfinite(output).all()),
    "permutation_equivariance": True,
    "heun_finite": True,
}))
PY
