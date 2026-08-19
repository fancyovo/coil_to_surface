from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from simsopt.geo import BoozerSurface, Volume, boozer_surface_residual

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.guarded_boozer_from_alpha_nu import surface_from_dofs  # noqa: E402
from stellarator_eval.field import build_field, load_case_file  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU-only standard Simsopt LS/Newton for one prepared face-QS surface.")
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--surface-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ls-tol", type=float, default=1e-10)
    parser.add_argument("--ls-maxiter", type=int, default=100)
    parser.add_argument("--newton-tol", type=float, default=1e-12)
    parser.add_argument("--newton-maxiter", type=int, default=30)
    parser.add_argument("--max-newton-residual-norm", type=float, default=1e-8)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "status": "failed",
        "case_file": str(args.case_file.resolve()),
        "source_surface": str(args.surface_npz.resolve()),
        "backend": "Simsopt CPU, one process with one BLAS/OpenMP thread",
    }
    try:
        with np.load(args.surface_npz) as saved:
            initial_dofs = np.asarray(saved["dofs"], dtype=float)
            initial_iota = float(saved["iota"])
            initial_G = float(saved["G"])
            nfp = int(saved["nfp"])
            order = int(saved["order"])
            surface_metadata = {
                key: np.asarray(saved[key]).item()
                for key in ("rho", "s_edge", "s_level", "radius_mean_m", "spectral_fit_rms_m")
                if key in saved
            }
        field_input = load_case_file(args.case_file, "raw")
        if field_input.nfp != nfp:
            raise ValueError("surface and case nfp do not match")
        field = build_field(field_input, current_unit="A").field
        exact_size = 2 * order + 1
        initial_surface = surface_from_dofs(initial_dofs, nfp=nfp, order=order, size=exact_size)
        target_volume = float(Volume(initial_surface).J())
        problem = BoozerSurface(field, initial_surface, Volume(initial_surface), target_volume)
        ls_kwargs = {
            "tol": args.ls_tol,
            "maxiter": args.ls_maxiter,
            "iota": initial_iota,
            "G": initial_G,
            "constraint_weight": 1.0,
        }
        if "weight_inv_modB" in inspect.signature(problem.minimize_boozer_penalty_constraints_ls).parameters:
            ls_kwargs["weight_inv_modB"] = True
        stage_started = time.perf_counter()
        ls_result = problem.minimize_boozer_penalty_constraints_ls(**ls_kwargs)
        ls_time = time.perf_counter() - stage_started
        ls_iota = float(ls_result["iota"])
        ls_G = float(ls_result["G"])
        ls_dofs = np.asarray(initial_surface.get_dofs(), dtype=float)
        ls_residual = float(np.linalg.norm(boozer_surface_residual(initial_surface, ls_iota, ls_G, field, derivatives=0)[0]))

        final_surface = surface_from_dofs(ls_dofs, nfp=nfp, order=order, size=exact_size)
        final_problem = BoozerSurface(field, final_surface, Volume(final_surface), target_volume)
        stage_started = time.perf_counter()
        newton_result = final_problem.solve_residual_equation_exactly_newton(
            tol=args.newton_tol,
            maxiter=args.newton_maxiter,
            iota=ls_iota,
            G=ls_G,
            verbose=False,
        )
        newton_time = time.perf_counter() - stage_started
        final_iota = float(newton_result["iota"])
        final_G = float(newton_result["G"])
        final_dofs = np.asarray(final_surface.get_dofs(), dtype=float)
        final_residual = float(np.linalg.norm(boozer_surface_residual(final_surface, final_iota, final_G, field, derivatives=0)[0]))
        solver_converged = bool(newton_result.get("success", False)) or final_residual <= args.max_newton_residual_norm
        output_surface = args.output_dir / "boozer_cpu_candidate.npz"
        np.savez(
            output_surface,
            dofs=final_dofs,
            iota=final_iota,
            G=final_G,
            nfp=nfp,
            order=order,
            stellsym=True,
            target_volume_m3=target_volume,
            initial_iota=initial_iota,
            **surface_metadata,
        )
        summary.update(
            {
                "status": "ok" if solver_converged else "solver_rejected",
                "nfp": nfp,
                "order": order,
                "target_volume_m3": target_volume,
                "initial_iota": initial_iota,
                "initial_G": initial_G,
                "least_squares": {
                    "success": bool(ls_result.get("success", False)),
                    "iterations": int(ls_result.get("iter", -1)),
                    "iota": ls_iota,
                    "G": ls_G,
                    "residual_norm": ls_residual,
                    "time_s": ls_time,
                },
                "newton": {
                    "success": bool(newton_result.get("success", False)),
                    "iterations": int(newton_result.get("iter", -1)),
                    "iota": final_iota,
                    "G": final_G,
                    "residual_norm": final_residual,
                    "time_s": newton_time,
                },
                "solver_converged": solver_converged,
                "output_surface": str(output_surface.resolve()),
            }
        )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
    summary["total_wall_s"] = time.perf_counter() - started
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "wall_s": summary["total_wall_s"]}), flush=True)
    if summary["status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
