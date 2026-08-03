from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simsopt.geo import SurfaceXYZTensorFourier

from stellarator_eval.config import PsiFitConfig
from stellarator_eval.field import build_field
from stellarator_eval.psi import _make_gpu_field
from scripts.desc_psi_volume_initial_guess_experiment import load_field_input


def timed(function, repeats: int) -> tuple[list[float], object]:
    values = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        values.append(time.perf_counter() - started)
    return values, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile CPU and GPU field evaluation used by nu diagnostics.")
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--surface-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nphi", type=int, default=57)
    parser.add_argument("--ntheta", type=int, default=59)
    parser.add_argument("--cpu-repeats", type=int, default=2)
    parser.add_argument("--gpu-repeats", type=int, default=10)
    parser.add_argument(
        "--gpu-lib",
        type=Path,
        default=ROOT / "gpu_backend" / "build_mixed" / "libstellarator_gpu.so",
    )
    args = parser.parse_args()

    payload = np.load(args.surface_npz)
    nfp = int(payload["nfp"])
    order = int(payload["order"])
    stellsym = bool(payload["stellsym"])
    surface = SurfaceXYZTensorFourier(
        mpol=order,
        ntor=order,
        nfp=nfp,
        stellsym=stellsym,
        quadpoints_phi=(np.arange(args.nphi) + 0.371) / (nfp * args.nphi),
        quadpoints_theta=(np.arange(args.ntheta) + 0.413) / args.ntheta,
    )
    surface.set_dofs(np.asarray(payload["dofs"], dtype=float))
    xyz = np.asarray(surface.gamma(), dtype=float).reshape(-1, 3)
    field_input = load_field_input(args.case_file, "raw")

    cpu_field = build_field(field_input, current_unit="A").field

    def cpu_evaluate():
        cpu_field.set_points(xyz)
        return np.asarray(cpu_field.B(), dtype=float)

    cpu_times, cpu_B = timed(cpu_evaluate, args.cpu_repeats)

    gpu_config = PsiFitConfig(
        backend="gpu",
        gpu_lib_path=str(args.gpu_lib.resolve()),
        gpu_device=0,
        gpu_segments_per_coil=256,
    )
    gpu_field = _make_gpu_field(field_input, nfp, gpu_config, "A")
    gpu_field.eval_B(xyz, precision="fp64")
    gpu_fp64_times, gpu_fp64_B = timed(
        lambda: gpu_field.eval_B(xyz, precision="fp64"), args.gpu_repeats
    )
    gpu_field.eval_B(xyz, precision="fp32")
    gpu_fp32_times, gpu_fp32_B = timed(
        lambda: gpu_field.eval_B(xyz, precision="fp32"), args.gpu_repeats
    )
    cpu_B = np.asarray(cpu_B, dtype=float)
    point_norm = np.linalg.norm(cpu_B, axis=1)

    def accuracy(gpu_B: np.ndarray) -> dict[str, float]:
        difference = np.asarray(gpu_B, dtype=float) - cpu_B
        relative_point = np.linalg.norm(difference, axis=1) / np.maximum(
            point_norm, 1e-30
        )
        return {
            "field_relative_l2": float(
                np.linalg.norm(difference) / np.linalg.norm(cpu_B)
            ),
            "field_relative_point_p95": float(np.percentile(relative_point, 95)),
            "field_relative_point_max": float(np.max(relative_point)),
        }

    cpu_median = float(np.median(cpu_times))
    gpu_fp64_median = float(np.median(gpu_fp64_times))
    gpu_fp32_median = float(np.median(gpu_fp32_times))
    output = {
        "point_count": int(len(xyz)),
        "surface_grid": [args.nphi, args.ntheta],
        "cpu_times_s": cpu_times,
        "cpu_median_s": cpu_median,
        "gpu_fp64_times_s": gpu_fp64_times,
        "gpu_fp64_median_s": gpu_fp64_median,
        "gpu_fp64_speedup": float(cpu_median / gpu_fp64_median),
        "gpu_fp64_accuracy": accuracy(gpu_fp64_B),
        "gpu_fp32_times_s": gpu_fp32_times,
        "gpu_fp32_median_s": gpu_fp32_median,
        "gpu_fp32_speedup": float(cpu_median / gpu_fp32_median),
        "gpu_fp32_accuracy": accuracy(gpu_fp32_B),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
