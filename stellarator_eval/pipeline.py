from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .axis import find_axis
from .config import EvalConfig
from .field import FieldInput, build_field, input_from_flat_vector, load_case_file
from .psi import fit_psi, model_to_npz_dict
from .serialization import write_json
from .surface import evaluate_boozer_surface, screen_level


def _set_thread_env(threads: int) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)


def _axis_summary(axis) -> dict:
    return {
        "has_axis": axis.has_axis,
        "best_R": axis.best_R,
        "best_Z": axis.best_Z,
        "best_residual": axis.best_residual,
        "generation": axis.generation,
        "time_s": axis.time_s,
        "history": axis.history,
    }


def _psi_summary(model) -> dict:
    return {
        "a": model.a,
        "nfp": model.nfp,
        "fit_info": model.fit_info,
        "modes": [{"a": m.a, "b": m.b, "m": m.m, "kind": m.kind} for m in model.modes],
        "coeffs": model.coeffs,
    }


def evaluate_field_input(field_input: FieldInput, config: EvalConfig | None = None, output_dir: str | Path = "runs/eval") -> dict:
    config = config or EvalConfig()
    _set_thread_env(config.omp_threads)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    result: dict[str, Any] = {
        "input_name": field_input.name,
        "nfp": int(field_input.nfp),
        "config": config.to_dict(),
        "warnings": [],
    }

    t0 = time.perf_counter()
    built = build_field(field_input, config.current_unit)
    result["field"] = {
        "n_base_coils": built.n_base_coils,
        "n_total_coils": built.n_total_coils,
        "coil_r0": built.coil_r0,
        "build_time_s": time.perf_counter() - t0,
    }

    axis = find_axis(built.field, built.nfp, built.coil_r0, config.axis)
    result["axis"] = _axis_summary(axis)
    np.savez(
        out / "axis_data.npz",
        phi=axis.phi,
        R=axis.R,
        Z=axis.Z,
        R_phi=axis.R_phi,
        Z_phi=axis.Z_phi,
        best_R=axis.best_R,
        best_Z=axis.best_Z,
        best_residual=axis.best_residual,
        nfp=built.nfp,
    )
    if not axis.has_axis:
        result["warnings"].append("axis residual did not reach configured tolerance; downstream steps skipped")
        result["total_time_s"] = time.perf_counter() - t_all
        write_json(out / "summary.json", result)
        return result

    model = fit_psi(built.field, axis, built.nfp, config.psi)
    result["psi"] = _psi_summary(model)
    np.savez(out / "psi_model.npz", **model_to_npz_dict(model))

    screen_results = []
    t_screen = time.perf_counter()
    for level in config.scan.levels:
        try:
            screen = screen_level(built.field, model, float(level), config.scan)
            screen_results.append(screen.__dict__)
        except Exception as exc:
            screen_results.append({"psi_level": float(level), "ok": False, "reason": repr(exc)})
    result["surface_screen"] = {
        "time_s": time.perf_counter() - t_screen,
        "levels": screen_results,
    }
    ok_levels = [r for r in screen_results if r.get("ok")]
    ok_levels.sort(key=lambda r: r["psi_level"], reverse=True)
    if not ok_levels:
        result["warnings"].append("no psi level passed the cheap fieldline drift screen")
        result["best_surface"] = None
        result["total_time_s"] = time.perf_counter() - t_all
        write_json(out / "summary.json", result)
        return result

    surface_results = []
    for item in ok_levels[: config.scan.max_boozer_candidates]:
        level = float(item["psi_level"])
        level_dir = out / f"level_{level:.6g}".replace(".", "p")
        level_dir.mkdir(parents=True, exist_ok=True)
        try:
            surf_result = evaluate_boozer_surface(
                built.field,
                model,
                level,
                config.scan,
                config.boozer,
                out_npz=level_dir / "boozer_surface.npz",
            )
        except Exception as exc:
            surf_result = {"psi_level": level, "error": repr(exc)}
        surface_results.append(surf_result)
        if surf_result.get("newton_success"):
            break

    result["surface_candidates"] = surface_results
    successes = [r for r in surface_results if r.get("newton_success")]
    result["best_surface"] = max(successes, key=lambda r: r.get("volume", -np.inf)) if successes else None
    if result["best_surface"] is None:
        result["warnings"].append("no screened psi level converged in Boozer LS/Newton")

    result["timing"] = {
        "field_build_s": result["field"]["build_time_s"],
        "axis_s": axis.time_s,
        "psi_fit_s": model.fit_info["time_s"],
        "surface_screen_s": result["surface_screen"]["time_s"],
        "boozer_candidates_s": sum(float(r.get("total_time_s", 0.0)) for r in surface_results),
    }
    result["total_time_s"] = time.perf_counter() - t_all
    write_json(out / "summary.json", result)
    return result


def evaluate_case_file(case_file: str | Path, key: str = "raw", config: EvalConfig | None = None, output_dir: str | Path = "runs/eval") -> dict:
    return evaluate_field_input(load_case_file(case_file, key), config=config, output_dir=output_dir)


def evaluate_coils(
    coil_coefficients: Any,
    currents: Any | None = None,
    nfp: int | None = None,
    config: EvalConfig | None = None,
    output_dir: str | Path = "runs/eval",
) -> dict:
    if isinstance(coil_coefficients, FieldInput):
        field_input = coil_coefficients
    elif currents is None:
        if nfp is None:
            raise ValueError("nfp is required when coil_coefficients is a flat vector")
        field_input = input_from_flat_vector(coil_coefficients, nfp=nfp)
    else:
        if nfp is None:
            raise ValueError("nfp is required")
        arr = np.asarray(coil_coefficients, dtype=float)
        if arr.ndim == 3 and arr.shape[1] == 3:
            x, y, z = arr[:, 0, :], arr[:, 1, :], arr[:, 2, :]
        elif isinstance(coil_coefficients, dict):
            x, y, z = coil_coefficients["x"], coil_coefficients["y"], coil_coefficients["z"]
        else:
            raise ValueError("coil_coefficients must be FieldInput, flat vector, dict{x,y,z}, or array (ncoil,3,ncoef)")
        field_input = FieldInput(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float), np.asarray(currents, float), int(nfp))
    return evaluate_field_input(field_input, config=config, output_dir=output_dir)
