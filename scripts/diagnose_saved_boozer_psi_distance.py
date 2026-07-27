from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from simsopt.geo import Volume

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stellarator_eval.psi import psi_and_gradient
from stellarator_eval.serialization import write_json
from scripts.desc_psi_volume_initial_guess_experiment import (
    load_json,
    load_psi_model,
    make_xyz_surface,
)

TWOPI = 2.0 * np.pi


def stats(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nphi", type=int, default=257)
    parser.add_argument("--ntheta", type=int, default=257)
    args = parser.parse_args()

    summary = load_json(args.run_dir / "summary.json")
    nfp = int(summary["nfp"])
    s_edge = float(summary["best_surface"]["psi_level"])
    level_dir = args.run_dir / f"level_{s_edge:.6g}".replace(".", "p")
    surface, metadata = make_xyz_surface(
        level_dir / "boozer_surface.npz",
        nfp=nfp,
        order=int(summary["config"]["boozer"]["surface_order"]),
        stellsym=bool(summary["config"]["boozer"]["stellsym"]),
        nphi=args.nphi,
        ntheta=args.ntheta,
    )
    xyz_grid = surface.gamma()
    xyz = xyz_grid.reshape((-1, 3))
    R = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    Z = xyz[:, 2]
    phi = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), TWOPI / nfp)
    model = load_psi_model(args.run_dir / "psi_model.npz")
    s, grad_R, grad_Z, grad_phi = psi_and_gradient(model, R, Z, phi)
    grad_norm = np.sqrt(grad_R**2 + grad_Z**2 + (grad_phi / R) ** 2)
    level_error = s - s_edge
    distance = np.abs(level_error) / np.maximum(grad_norm, 1e-30)
    geometric_phi = np.unwrap(np.arctan2(xyz_grid[:, :, 1], xyz_grid[:, :, 0]), axis=0)
    parameter_fraction = (args.nphi - 1) / args.nphi
    toroidal_winding = (
        (geometric_phi[-1] - geometric_phi[0])
        / parameter_fraction
        / (TWOPI / nfp)
    )
    output = {
        "s_edge": s_edge,
        "boozer_metadata": metadata,
        "point_count": int(len(s)),
        "s_on_boozer_surface": stats(s),
        "absolute_s_level_error": stats(np.abs(level_error)),
        "linearized_normal_distance_m": stats(distance),
        "signed_volume_m3": float(Volume(surface).J()),
        "geometric_toroidal_winding_per_field_period": stats(toroidal_winding),
    }
    write_json(args.output, output)


if __name__ == "__main__":
    main()
