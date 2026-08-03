from __future__ import annotations

import argparse
import atexit
import inspect
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from simsopt.geo import BoozerSurface, Volume, boozer_surface_residual

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.desc_psi_volume_initial_guess_experiment import load_psi_model
from scripts.diagnose_alpha_boozer_residual import GpuBOnlyFieldAdapter
from scripts.guarded_boozer_from_alpha_nu import evaluate_state, surface_from_dofs
from stellarator_eval.config import PsiFitConfig
from stellarator_eval.field import build_field, load_case_file
from stellarator_eval.psi import _make_gpu_field
from stellarator_eval.serialization import write_json
from stellarator_eval.surface import helical_qs_metric


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run standard Boozer LS and full Newton from an alpha+nu surface."
    )
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--surface-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-key", default="raw")
    parser.add_argument("--current-unit", default="A")
    parser.add_argument("--ls-tol", type=float, default=1e-10)
    parser.add_argument("--ls-maxiter", type=int, default=100)
    parser.add_argument("--newton-tol", type=float, default=1e-12)
    parser.add_argument("--newton-maxiter", type=int, default=30)
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--grids", default="25,49,97")
    parser.add_argument("--max-final-relative-l2", type=float, default=1e-4)
    parser.add_argument("--max-final-normal-p95", type=float, default=1e-4)
    parser.add_argument("--max-newton-residual-norm", type=float, default=1e-8)
    parser.add_argument("--qs-sdim", type=int, default=16)
    parser.add_argument(
        "--gpu-lib",
        type=Path,
        default=ROOT / "gpu_backend" / "build_mixed" / "libstellarator_gpu.so",
    )
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument(
        "--validation-field-precision", choices=("fp32", "fp64"), default="fp32"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = np.load(args.surface_npz)
    initial_dofs = np.asarray(saved["dofs"], dtype=float)
    initial_iota = float(saved["iota"])
    initial_G = float(saved["G"])
    nfp = int(saved["nfp"])
    order = int(saved["order"])
    target_s = float(saved["s_level"])
    radius_mean = float(saved["radius_mean_m"])
    projection_rms = float(saved["spectral_fit_rms_m"])
    grid_sizes = sorted(
        {2 * order + 1, *[int(value) for value in args.grids.split(",") if value]}
    )

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
    gpu_field = _make_gpu_field(field_input, nfp, gpu_config, args.current_unit)
    atexit.register(gpu_field.close)
    validation_field = GpuBOnlyFieldAdapter(
        gpu_field, precision=args.validation_field_precision
    )
    model = load_psi_model(args.run_dir / "psi_model.npz")
    reference_surface = surface_from_dofs(
        initial_dofs, nfp=nfp, order=order, size=max(grid_sizes), offset=0.371
    )
    reference_xyz = reference_surface.gamma()

    initial = evaluate_state(
        initial_dofs,
        field=validation_field,
        model=model,
        nfp=nfp,
        order=order,
        iota=initial_iota,
        G=initial_G,
        target_s=target_s,
        reference_xyz=reference_xyz,
        grid_sizes=grid_sizes,
    )
    target_volume = float(initial["geometry"]["signed_volume_m3"])
    exact_size = 2 * order + 1
    ls_surface = surface_from_dofs(
        initial_dofs, nfp=nfp, order=order, size=exact_size
    )
    ls_problem = BoozerSurface(
        solver_field, ls_surface, Volume(ls_surface), target_volume
    )
    ls_kwargs = {
        "tol": args.ls_tol,
        "maxiter": args.ls_maxiter,
        "iota": initial_iota,
        "G": initial_G,
        "constraint_weight": args.constraint_weight,
    }
    if "weight_inv_modB" in inspect.signature(
        ls_problem.minimize_boozer_penalty_constraints_ls
    ).parameters:
        ls_kwargs["weight_inv_modB"] = True
    started = time.perf_counter()
    ls_result = ls_problem.minimize_boozer_penalty_constraints_ls(**ls_kwargs)
    ls_time_s = time.perf_counter() - started
    ls_iota = float(ls_result["iota"])
    ls_G = float(ls_result["G"])
    ls_dofs = np.asarray(ls_surface.get_dofs(), dtype=float)
    ls_residual_norm = float(
        np.linalg.norm(
            boozer_surface_residual(
                ls_surface, ls_iota, ls_G, solver_field, derivatives=0
            )[0]
        )
    )
    ls_state = evaluate_state(
        ls_dofs,
        field=validation_field,
        model=model,
        nfp=nfp,
        order=order,
        iota=ls_iota,
        G=ls_G,
        target_s=target_s,
        reference_xyz=reference_xyz,
        grid_sizes=grid_sizes,
    )

    newton_surface = surface_from_dofs(
        ls_dofs, nfp=nfp, order=order, size=exact_size
    )
    newton_problem = BoozerSurface(
        solver_field, newton_surface, Volume(newton_surface), target_volume
    )
    started = time.perf_counter()
    newton_result = newton_problem.solve_residual_equation_exactly_newton(
        tol=args.newton_tol,
        maxiter=args.newton_maxiter,
        iota=ls_iota,
        G=ls_G,
        verbose=False,
    )
    newton_time_s = time.perf_counter() - started
    final_iota = float(newton_result["iota"])
    final_G = float(newton_result["G"])
    final_dofs = np.asarray(newton_surface.get_dofs(), dtype=float)
    newton_residual_norm = float(
        np.linalg.norm(
            boozer_surface_residual(
                newton_surface, final_iota, final_G, solver_field, derivatives=0
            )[0]
        )
    )
    final = evaluate_state(
        final_dofs,
        field=validation_field,
        model=model,
        nfp=nfp,
        order=order,
        iota=final_iota,
        G=final_G,
        target_s=target_s,
        reference_xyz=reference_xyz,
        grid_sizes=grid_sizes,
    )
    dense = final["grids"][-1]
    solver_converged = bool(newton_result.get("success", False)) or (
        newton_residual_norm <= args.max_newton_residual_norm
    )
    acceptance_checks = {
        "newton_converged": solver_converged,
        "dense_relative_l2": float(dense["relative_l2"])
        <= args.max_final_relative_l2,
        "dense_normal_field_p95": float(dense["normal_B_sine_p95"])
        <= args.max_final_normal_p95,
        "toroidal_winding": float(
            final["geometry"]["geometric_toroidal_winding"]["min"]
        )
        > 0.0,
        "normal_nonzero": float(final["geometry"]["normal_norm"]["min"])
        > 1e-12,
    }
    accepted_for_downstream = all(acceptance_checks.values())
    output_surface = args.output_dir / (
        "boozer_standard.npz" if accepted_for_downstream else "boozer_rejected.npz"
    )
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
        kind="alpha_nu_standard_ls_newton",
    )
    qs_errors = {}
    if accepted_for_downstream:
        qs_errors = {
            "QA_1_0": helical_qs_metric(
                newton_problem, solver_field, 1, 0, args.qs_sdim
            ),
            "QH_1_1": helical_qs_metric(
                newton_problem, solver_field, 1, 1, args.qs_sdim
            ),
            "QP_0_1": helical_qs_metric(
                newton_problem, solver_field, 0, 1, args.qs_sdim
            ),
        }

    output = {
        "case_file": str(args.case_file),
        "source_surface": str(args.surface_npz),
        "nfp": nfp,
        "order": order,
        "target_s": target_s,
        "target_volume_m3": target_volume,
        "initial": initial,
        "least_squares": {
            "success": bool(ls_result.get("success", False)),
            "iter": int(ls_result.get("iter", -1)),
            "iota": ls_iota,
            "G": ls_G,
            "residual_norm": ls_residual_norm,
            "time_s": ls_time_s,
            "state": ls_state,
        },
        "newton": {
            "success": bool(newton_result.get("success", False)),
            "iter": int(newton_result.get("iter", -1)),
            "iota": final_iota,
            "G": final_G,
            "residual_norm": newton_residual_norm,
            "time_s": newton_time_s,
            "state": final,
        },
        "acceptance_thresholds": {
            "max_newton_residual_norm": args.max_newton_residual_norm,
            "max_final_relative_l2": args.max_final_relative_l2,
            "max_final_normal_p95": args.max_final_normal_p95,
        },
        "acceptance_checks": acceptance_checks,
        "accepted_for_downstream": accepted_for_downstream,
        "branch_diagnostics": {
            "distance_from_initial": final["distance_from_initial"],
            "psi_identity": final["psi_identity"],
            "distance_p95_over_radius": max(
                float(
                    final["distance_from_initial"]["candidate_to_reference_m"][
                        "p95"
                    ]
                ),
                float(
                    final["distance_from_initial"]["reference_to_candidate_m"][
                        "p95"
                    ]
                ),
            )
            / max(radius_mean, 1e-30),
        },
        "surface_qs_error": qs_errors,
        "backends": {
            "ls_newton": "Simsopt CPU",
            "dense_validation": (
                f"C++/CUDA eval_B {args.validation_field_precision}"
            ),
        },
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
