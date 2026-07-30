from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simsopt.geo import BoozerSurface, SurfaceXYZTensorFourier, Volume

from stellarator_eval.config import PsiFitConfig
from stellarator_eval.field import build_field, load_case_file
from stellarator_eval.psi import _make_gpu_field, psi_and_gradient
from stellarator_eval.serialization import write_json
from scripts.desc_psi_volume_initial_guess_experiment import load_psi_model
from scripts.diagnose_alpha_boozer_residual import (
    GpuBOnlyFieldAdapter,
    residual_for_iota_G,
)


TWOPI = 2.0 * np.pi


def surface_from_dofs(dofs, *, nfp: int, order: int, size: int, offset: float = 0.0):
    surface = SurfaceXYZTensorFourier(
        mpol=order,
        ntor=order,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=(np.arange(size) + offset) / (nfp * size),
        quadpoints_theta=(np.arange(size) + offset) / size,
    )
    surface.set_dofs(np.asarray(dofs, dtype=float))
    return surface


def stats(values) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def geometry_distance(reference_xyz, candidate_xyz) -> dict[str, dict[str, float]]:
    reference = np.asarray(reference_xyz, dtype=float).reshape((-1, 3))
    candidate = np.asarray(candidate_xyz, dtype=float).reshape((-1, 3))
    candidate_to_reference = cKDTree(reference).query(candidate, workers=-1)[0]
    reference_to_candidate = cKDTree(candidate).query(reference, workers=-1)[0]
    return {
        "candidate_to_reference_m": stats(candidate_to_reference),
        "reference_to_candidate_m": stats(reference_to_candidate),
    }


def psi_identity(model, xyz, target: float) -> dict[str, object]:
    points = np.asarray(xyz, dtype=float).reshape((-1, 3))
    R = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    Z = points[:, 2]
    phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), TWOPI / model.nfp)
    s, grad_R, grad_Z, grad_phi = psi_and_gradient(model, R, Z, phi)
    error = s - target
    grad_norm = np.sqrt(grad_R**2 + grad_Z**2 + (grad_phi / R) ** 2)
    distance = np.abs(error) / np.maximum(grad_norm, 1e-30)
    return {
        "target_s": float(target),
        "s": stats(s),
        "absolute_s_error": stats(np.abs(error)),
        "linearized_normal_distance_m": stats(distance),
    }


def geometric_metrics(surface) -> dict[str, object]:
    xyz = np.asarray(surface.gamma(), dtype=float)
    normal_norm = np.linalg.norm(np.asarray(surface.normal(), dtype=float), axis=2)
    geometric_phi = np.unwrap(np.arctan2(xyz[:, :, 1], xyz[:, :, 0]), axis=0)
    fraction = (surface.quadpoints_phi.size - 1) / surface.quadpoints_phi.size
    winding = (
        (geometric_phi[-1] - geometric_phi[0])
        / fraction
        / (TWOPI / surface.nfp)
    )
    return {
        "signed_volume_m3": float(Volume(surface).J()),
        "normal_norm": stats(normal_norm),
        "geometric_toroidal_winding": stats(winding),
    }


def evaluate_state(
    dofs,
    *,
    field,
    model,
    nfp: int,
    order: int,
    iota: float,
    G: float,
    target_s: float,
    reference_xyz,
    grid_sizes: list[int],
) -> dict[str, object]:
    grids = []
    dense_surface = None
    for size in grid_sizes:
        offset = 0.0 if size == 2 * order + 1 else 0.371
        surface = surface_from_dofs(
            dofs, nfp=nfp, order=order, size=size, offset=offset
        )
        grids.append(
            {
                "size": int(size),
                "offset": float(offset),
                **residual_for_iota_G(surface, field, iota=iota, G=G),
            }
        )
        dense_surface = surface
    assert dense_surface is not None
    xyz = dense_surface.gamma()
    return {
        "iota": float(iota),
        "G": float(G),
        "grids": grids,
        "psi_identity": psi_identity(model, xyz, target_s),
        "geometry": geometric_metrics(dense_surface),
        "distance_from_initial": geometry_distance(reference_xyz, xyz),
    }


def interpolate_state(start, trial, fraction: float):
    return tuple(a + fraction * (b - a) for a, b in zip(start, trial))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guard each BoozerExact Newton step with dense physical validation."
    )
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--surface-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-key", default="raw")
    parser.add_argument("--current-unit", default="A")
    parser.add_argument("--max-accepted-steps", type=int, default=8)
    parser.add_argument("--newton-tol", type=float, default=1e-12)
    parser.add_argument("--grids", default="25,49,97")
    parser.add_argument("--line-search", default="1,0.5,0.25,0.125,0.0625")
    parser.add_argument("--max-radius-fraction", type=float, default=0.05)
    parser.add_argument("--residual-growth-tolerance", type=float, default=1e-3)
    parser.add_argument("--normal-growth-limit", type=float, default=2.0)
    parser.add_argument("--psi-distance-growth-limit", type=float, default=2.0)
    parser.add_argument("--max-final-relative-l2", type=float, default=1e-4)
    parser.add_argument("--max-final-normal-p95", type=float, default=1e-4)
    parser.add_argument(
        "--gpu-lib",
        type=Path,
        default=ROOT / "gpu_backend" / "build_mixed" / "libstellarator_gpu.so",
    )
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--validation-field-precision", choices=("fp32", "fp64"), default="fp32")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    saved = np.load(args.surface_npz)
    dofs = np.asarray(saved["dofs"], dtype=float)
    iota = float(saved["iota"])
    G = float(saved["G"])
    nfp = int(saved["nfp"])
    order = int(saved["order"])
    target_s = float(saved["s_level"])
    radius_mean = float(saved["radius_mean_m"])
    projection_rms = float(saved["spectral_fit_rms_m"])
    grid_sizes = sorted(
        {2 * order + 1, *[int(value) for value in args.grids.split(",") if value]}
    )
    line_search = [float(value) for value in args.line_search.split(",") if value]

    field_input = load_case_file(args.case_file, args.case_key)
    if field_input.nfp != nfp:
        raise ValueError(f"surface nfp={nfp} but case nfp={field_input.nfp}")
    solver_field = build_field(field_input, current_unit=args.current_unit).field
    gpu_lib = args.gpu_lib if args.gpu_lib.is_absolute() else ROOT / args.gpu_lib
    gpu_config = PsiFitConfig(
        backend="gpu",
        gpu_lib_path=str(gpu_lib.resolve()),
        gpu_device=args.gpu_device,
        gpu_segments_per_coil=256,
    )
    gpu_field = _make_gpu_field(
        field_input, nfp, gpu_config, args.current_unit
    )
    atexit.register(gpu_field.close)
    validation_field = GpuBOnlyFieldAdapter(
        gpu_field, precision=args.validation_field_precision
    )
    model = load_psi_model(args.run_dir / "psi_model.npz")
    reference_surface = surface_from_dofs(
        dofs, nfp=nfp, order=order, size=max(grid_sizes), offset=0.371
    )
    reference_xyz = reference_surface.gamma()

    initial = evaluate_state(
        dofs,
        field=validation_field,
        model=model,
        nfp=nfp,
        order=order,
        iota=iota,
        G=G,
        target_s=target_s,
        reference_xyz=reference_xyz,
        grid_sizes=grid_sizes,
    )
    accepted = initial
    accepted_state = (dofs.copy(), iota, G)
    target_volume = float(initial["geometry"]["signed_volume_m3"])
    max_distance = max(3.0 * projection_rms, args.max_radius_fraction * radius_mean)
    attempts = []

    initial_dense = initial["grids"][-1]
    initial_normal = float(initial_dense["normal_B_sine_p95"])
    initial_psi_distance = float(
        initial["psi_identity"]["linearized_normal_distance_m"]["p95"]
    )

    for step in range(1, args.max_accepted_steps + 1):
        start_dofs, start_iota, start_G = accepted_state
        exact_surface = surface_from_dofs(
            start_dofs, nfp=nfp, order=order, size=2 * order + 1
        )
        boozer = BoozerSurface(
            solver_field, exact_surface, Volume(exact_surface), target_volume
        )
        result = boozer.solve_residual_equation_exactly_newton(
            tol=args.newton_tol,
            maxiter=1,
            iota=start_iota,
            G=start_G,
            verbose=False,
        )
        trial_state = (
            np.asarray(exact_surface.get_dofs(), dtype=float),
            float(result["iota"]),
            float(result["G"]),
        )
        proposal_record = {
            "step": step,
            "solver_success": bool(result["success"]),
            "solver_iter": int(result["iter"]),
            "line_search": [],
        }
        chosen = None
        previous_exact_residual = float(accepted["grids"][0]["relative_l2"])
        previous_dense_residual = float(accepted["grids"][-1]["relative_l2"])
        for fraction in line_search:
            candidate_state = interpolate_state(accepted_state, trial_state, fraction)
            candidate = evaluate_state(
                candidate_state[0],
                field=validation_field,
                model=model,
                nfp=nfp,
                order=order,
                iota=candidate_state[1],
                G=candidate_state[2],
                target_s=target_s,
                reference_xyz=reference_xyz,
                grid_sizes=grid_sizes,
            )
            dense = candidate["grids"][-1]
            distance_p95 = max(
                float(
                    candidate["distance_from_initial"][
                        "candidate_to_reference_m"
                    ]["p95"]
                ),
                float(
                    candidate["distance_from_initial"][
                        "reference_to_candidate_m"
                    ]["p95"]
                ),
            )
            psi_distance = float(
                candidate["psi_identity"]["linearized_normal_distance_m"]["p95"]
            )
            checks = {
                "exact_residual_nonincreasing": float(
                    candidate["grids"][0]["relative_l2"]
                )
                <= previous_exact_residual * (1.0 + args.residual_growth_tolerance),
                "dense_residual_nonincreasing": float(dense["relative_l2"])
                <= previous_dense_residual * (1.0 + args.residual_growth_tolerance),
                "geometry_distance": distance_p95 <= max_distance,
                "normal_field": float(dense["normal_B_sine_p95"])
                <= max(args.normal_growth_limit * initial_normal, 1e-8),
                "psi_identity": psi_distance
                <= max(args.psi_distance_growth_limit * initial_psi_distance, 1e-8),
                "toroidal_winding": abs(
                    float(candidate["geometry"]["geometric_toroidal_winding"]["mean"])
                    - 1.0
                )
                <= 0.02,
                "normal_nonzero": float(candidate["geometry"]["normal_norm"]["min"])
                > 1e-12,
            }
            proposal_record["line_search"].append(
                {
                    "fraction": fraction,
                    "checks": checks,
                    "state": candidate,
                }
            )
            if all(checks.values()):
                chosen = (candidate_state, candidate, fraction)
                break
        attempts.append(proposal_record)
        if chosen is None:
            break
        accepted_state, accepted, accepted_fraction = chosen
        attempts[-1]["accepted_fraction"] = accepted_fraction
        if bool(result["success"]) or float(accepted["grids"][0]["relative_l2"]) <= args.newton_tol:
            break

    final_dofs, final_iota, final_G = accepted_state
    final_dense = accepted["grids"][-1]
    final_distance_p95 = max(
        float(
            accepted["distance_from_initial"]["candidate_to_reference_m"]["p95"]
        ),
        float(
            accepted["distance_from_initial"]["reference_to_candidate_m"]["p95"]
        ),
    )
    absolute_checks = {
        "relative_l2": float(final_dense["relative_l2"])
        <= args.max_final_relative_l2,
        "normal_field_p95": float(final_dense["normal_B_sine_p95"])
        <= args.max_final_normal_p95,
        "geometry_distance": final_distance_p95 <= max_distance,
        "toroidal_winding": float(
            accepted["geometry"]["geometric_toroidal_winding"]["min"]
        )
        > 0.0,
        "normal_nonzero": float(accepted["geometry"]["normal_norm"]["min"])
        > 1e-12,
    }
    accepted_for_downstream = all(absolute_checks.values())
    surface_name = (
        "boozer_guarded.npz" if accepted_for_downstream else "boozer_rejected.npz"
    )
    output_surface = args.output_dir / surface_name
    np.savez(
        output_surface,
        dofs=final_dofs,
        iota=final_iota,
        G=final_G,
        nfp=nfp,
        order=order,
        stellsym=True,
        rho=float(saved["rho"]),
        s_edge=float(saved["s_edge"]),
        s_level=target_s,
        radius_mean_m=radius_mean,
        spectral_fit_rms_m=projection_rms,
        kind="alpha_nu_guarded_newton",
    )

    figure, axis = plt.subplots(figsize=(6.8, 6.2))
    for label, state, color in (
        ("alpha+nu initial", initial, "#0072b2"),
        ("guarded Newton", accepted, "#d55e00"),
    ):
        state_dofs = dofs if label == "alpha+nu initial" else final_dofs
        surface = surface_from_dofs(
            state_dofs, nfp=nfp, order=order, size=257, offset=0.0
        )
        xyz = surface.gamma()[0]
        axis.plot(
            np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2),
            xyz[:, 2],
            label=label,
            color=color,
            linewidth=1.5,
        )
    axis.set_aspect("equal")
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.legend(frameon=False)
    axis.set_title("Physical surface identity at phi=0")
    figure.tight_layout()
    figure.savefig(args.output_dir / "surface_identity_phi0.png", dpi=190)
    plt.close(figure)

    output = {
        "case_file": str(args.case_file),
        "source_surface": str(args.surface_npz),
        "nfp": nfp,
        "order": order,
        "rho": float(saved["rho"]),
        "target_s": target_s,
        "target_volume_m3": target_volume,
        "projection_rms_m": projection_rms,
        "radius_mean_m": radius_mean,
        "maximum_allowed_geometry_p95_m": max_distance,
        "initial": initial,
        "attempts": attempts,
        "accepted_step_count": sum("accepted_fraction" in row for row in attempts),
        "final": accepted,
        "acceptance_thresholds": {
            "max_final_relative_l2": args.max_final_relative_l2,
            "max_final_normal_p95": args.max_final_normal_p95,
            "max_geometry_distance_p95_m": max_distance,
        },
        "absolute_checks": absolute_checks,
        "backends": {
            "boozer_newton_with_spatial_derivatives": "Simsopt CPU",
            "dense_line_search_field_validation": (
                f"C++/CUDA eval_B {args.validation_field_precision}"
            ),
            "psi_and_geometry_validation": "NumPy/SciPy CPU",
        },
        "accepted_for_downstream": accepted_for_downstream,
        "output_surface": str(output_surface),
    }
    write_json(args.output_dir / "summary.json", output)
    print(json.dumps(output, indent=2), flush=True)
    gpu_field.close()
    atexit.unregister(gpu_field.close)
    if not accepted_for_downstream:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
