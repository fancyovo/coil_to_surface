from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simsopt.geo import SurfaceXYZTensorFourier, boozer_surface_residual

from stellarator_eval.field import build_field, load_case_file


def evaluate_grid(field, dofs, *, nfp: int, order: int, iota: float, G: float, size: int, offset: float):
    surface = SurfaceXYZTensorFourier(
        mpol=order,
        ntor=order,
        nfp=nfp,
        stellsym=True,
        quadpoints_phi=(np.arange(size) + offset) / (nfp * size),
        quadpoints_theta=(np.arange(size) + offset) / size,
    )
    surface.set_dofs(dofs)
    xyz = surface.gamma()
    field.set_points(xyz.reshape(-1, 3))
    B = np.asarray(field.B(), dtype=float).reshape(xyz.shape)
    normal = np.asarray(surface.normal(), dtype=float)
    normal_sine = np.abs(np.sum(B * normal, axis=2)) / np.maximum(
        np.linalg.norm(B, axis=2) * np.linalg.norm(normal, axis=2), 1.0e-30
    )
    residual = boozer_surface_residual(surface, iota, G, field, derivatives=0)[0]
    return {
        "size": int(size),
        "offset": float(offset),
        "boozer_residual_norm": float(np.linalg.norm(residual)),
        "boozer_residual_rms_per_point": float(np.linalg.norm(residual) / np.sqrt(size * size)),
        "normal_B_sine_mean": float(np.mean(normal_sine)),
        "normal_B_sine_p95": float(np.percentile(normal_sine, 95)),
        "normal_B_sine_max": float(np.max(normal_sine)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a saved Boozer surface away from its solve grid.")
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--surface-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-key", default="raw")
    parser.add_argument("--current-unit", default="A")
    parser.add_argument("--surface-order", type=int, default=6)
    parser.add_argument("--grids", default="13,25,49,97")
    args = parser.parse_args()

    field_input = load_case_file(args.case_file, args.case_key)
    built = build_field(field_input, current_unit=args.current_unit)
    saved = np.load(args.surface_npz)
    dofs = np.asarray(saved["dofs"], dtype=float)
    iota = float(saved["iota"])
    G = float(saved["G"])
    sizes = [int(value) for value in args.grids.split(",") if value.strip()]

    rows = [
        evaluate_grid(
            built.field,
            dofs,
            nfp=field_input.nfp,
            order=args.surface_order,
            iota=iota,
            G=G,
            size=2 * args.surface_order + 1,
            offset=0.0,
        )
    ]
    rows.extend(
        evaluate_grid(
            built.field,
            dofs,
            nfp=field_input.nfp,
            order=args.surface_order,
            iota=iota,
            G=G,
            size=size,
            offset=0.371,
        )
        for size in sizes
    )
    result = {
        "case_file": str(args.case_file),
        "surface_npz": str(args.surface_npz),
        "nfp": field_input.nfp,
        "surface_order": args.surface_order,
        "iota": iota,
        "G": G,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
