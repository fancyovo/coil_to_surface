from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stellarator_eval.axis import rk4_period_samples
from stellarator_eval.field import build_field
from stellarator_eval.serialization import write_json
from scripts.desc_psi_volume_initial_guess_experiment import (
    load_field_input,
    load_json,
    load_psi_model,
    make_xyz_surface,
)

TWOPI = 2.0 * np.pi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--periods", type=int, default=32)
    parser.add_argument("--lines", type=int, default=8)
    parser.add_argument("--samples-per-period", type=int, default=32)
    parser.add_argument("--steps-per-period", type=int, default=480)
    args = parser.parse_args()

    summary = load_json(args.run_dir / "summary.json")
    nfp = int(summary["nfp"])
    current_unit = summary["config"].get("current_unit") or "A"
    s_edge = float(summary["best_surface"]["psi_level"])
    level_dir = args.run_dir / f"level_{s_edge:.6g}".replace(".", "p")
    surface, surface_meta = make_xyz_surface(
        level_dir / "boozer_surface.npz",
        nfp=nfp,
        order=int(summary["config"]["boozer"]["surface_order"]),
        stellsym=bool(summary["config"]["boozer"]["stellsym"]),
        nphi=3,
        ntheta=args.lines,
    )
    xyz = surface.gamma()[0]
    R_current = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    Z_current = xyz[:, 2]
    R_initial = R_current.copy()
    Z_initial = Z_current.copy()
    model = load_psi_model(args.run_dir / "psi_model.npz")
    field_input = load_field_input(args.case_file, "raw")
    field = build_field(field_input, current_unit=current_unit).field

    period_length = TWOPI / nfp
    phi_parts = []
    R_parts = []
    Z_parts = []
    for period_index in range(args.periods):
        phi, R_hist, Z_hist, R_current, Z_current = rk4_period_samples(
            field,
            R_current,
            Z_current,
            nfp,
            n_zeta=args.samples_per_period,
            steps=args.steps_per_period,
        )
        phi_parts.append(phi + period_index * period_length)
        R_parts.append(R_hist)
        Z_parts.append(Z_hist)
    phi = np.concatenate(phi_parts)
    R = np.concatenate(R_parts, axis=1)
    Z = np.concatenate(Z_parts, axis=1)
    design = np.column_stack([np.ones_like(phi), phi])
    rows = []
    for line in range(args.lines):
        local_phi = np.mod(phi, period_length)
        ra, za, _, _ = model.axis_at(local_phi)
        theta_clockwise = np.unwrap(-np.arctan2(Z[line] - za, R[line] - ra))
        coeffs, *_ = np.linalg.lstsq(design, theta_clockwise, rcond=None)
        residual = theta_clockwise - design @ coeffs
        rows.append(
            {
                "line": line,
                "iota_long_trace": float(coeffs[1]),
                "periodic_residual_rms_rad": float(np.sqrt(np.mean(residual**2))),
            }
        )
    iotas = np.asarray([row["iota_long_trace"] for row in rows])
    dense_surface, _ = make_xyz_surface(
        level_dir / "boozer_surface.npz",
        nfp=nfp,
        order=int(summary["config"]["boozer"]["surface_order"]),
        stellsym=bool(summary["config"]["boozer"]["stellsym"]),
        nphi=3,
        ntheta=512,
    )
    dense_xyz = dense_surface.gamma()[0]
    dense_contour = np.column_stack(
        [np.sqrt(dense_xyz[:, 0] ** 2 + dense_xyz[:, 1] ** 2), dense_xyz[:, 2]]
    )
    final_distance, _ = cKDTree(dense_contour).query(
        np.column_stack([R_current, Z_current])
    )
    output = {
        "periods": args.periods,
        "line_count": args.lines,
        "boozer_file_iota": surface_meta["iota"],
        "long_trace_iota_mean": float(np.mean(iotas)),
        "long_trace_iota_std": float(np.std(iotas)),
        "final_distance_to_boozer_contour_mean_m": float(np.mean(final_distance)),
        "final_distance_to_boozer_contour_max_m": float(np.max(final_distance)),
        "initial_to_final_direct_distance_mean_m": float(
            np.mean(np.sqrt((R_current - R_initial) ** 2 + (Z_current - Z_initial) ** 2))
        ),
        "rows": rows,
    }
    write_json(args.output, output)


if __name__ == "__main__":
    main()
