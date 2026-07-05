from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .axis import find_axis, find_axis_gpu
from .config import EvalConfig
from .field import FieldInput, build_field, input_from_flat_vector, load_case_file
from .psi import fit_psi, model_to_npz_dict
from .serialization import write_json
from .surface import evaluate_boozer_surface, screen_level, screen_levels_gpu
from .timing import timing_phase, timing_session


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
        "search_time_s": axis.search_time_s,
        "trace_time_s": axis.trace_time_s,
        "backend": axis.backend,
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

    with timing_session() as timings:
        t0 = time.perf_counter()
        built = build_field(field_input, config.current_unit)
        result["field"] = {
            "n_base_coils": built.n_base_coils,
            "n_total_coils": built.n_total_coils,
            "coil_r0": built.coil_r0,
            "build_time_s": time.perf_counter() - t0,
        }

        with timing_phase("axis_search"):
            if config.axis.backend.lower() == "gpu":
                axis = find_axis_gpu(field_input, built.field, built.nfp, built.coil_r0, config.axis, config.current_unit)
            else:
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
            result["timing_b_calls"] = timings.as_dict()
            result["total_time_s"] = time.perf_counter() - t_all
            write_json(out / "summary.json", result)
            return result

        with timing_phase("psi_fit"):
            model = fit_psi(built.field, axis, built.nfp, config.psi, field_input=field_input, current_unit=config.current_unit)
        result["psi"] = _psi_summary(model)
        np.savez(out / "psi_model.npz", **model_to_npz_dict(model))

        screen_results = []
        t_screen = time.perf_counter()
        with timing_phase("psi0_screen_fieldline"):
            if config.scan.trace_backend.lower() == "gpu":
                try:
                    screen_results = screen_levels_gpu(field_input, model, config.scan.levels, config.scan, config.current_unit)
                except Exception as exc:
                    result["warnings"].append(f"GPU psi0 screen failed; falling back to CPU: {exc!r}")
                    screen_results = []
            if not screen_results:
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
            result["timing_b_calls"] = timings.as_dict()
            result["total_time_s"] = time.perf_counter() - t_all
            write_json(out / "summary.json", result)
            return result

        surface_results = []
        for item in ok_levels[: config.scan.max_boozer_candidates]:
            level = float(item["psi_level"])
            level_dir = out / f"level_{level:.6g}".replace(".", "p")
            level_dir.mkdir(parents=True, exist_ok=True)
            try:
                with timing_phase("boozer_candidate_pre_ls"):
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

        result["timing_b_calls"] = timings.as_dict()
        result["timing"] = {
            "field_build_s": result["field"]["build_time_s"],
            "axis_s": axis.time_s,
            "axis_search_s": axis.search_time_s,
            "axis_trace_s": axis.trace_time_s,
            "psi_fit_s": model.fit_info["time_s"],
            "psi_gpu_create_s": model.fit_info.get("gpu_create_s", 0.0),
            "psi_training_point_s": model.fit_info.get("training_point_s", 0.0),
            "psi_assemble_s": model.fit_info.get("assemble_s", 0.0),
            "psi_assemble_interp_s": model.fit_info.get("assemble_interp_s", 0.0),
            "psi_assemble_b_sample_s": model.fit_info.get("assemble_b_sample_s", 0.0),
            "psi_assemble_basis_s": model.fit_info.get("assemble_basis_s", 0.0),
            "psi_assemble_normal_eq_s": model.fit_info.get("assemble_normal_eq_s", 0.0),
            "psi_assemble_normal_eq_cpu_s": model.fit_info.get("assemble_normal_eq_cpu_s", 0.0),
            "psi_assemble_normal_eq_gpu_s": model.fit_info.get("assemble_normal_eq_gpu_s", 0.0),
            "psi_solve_s": model.fit_info.get("solve_s", 0.0),
            "psi_validation_s": model.fit_info.get("validation_s", 0.0),
            "psi_validation_b_sample_s": model.fit_info.get("validation_b_sample_s", 0.0),
            "surface_screen_s": result["surface_screen"]["time_s"],
            "surface_screen_curve_newton_s": sum(float(r.get("curve_newton_time_s", 0.0)) for r in screen_results),
            "surface_screen_fieldline_trace_s": sum(float(r.get("trace_time_s", 0.0)) for r in screen_results),
            "surface_extract_1d_newton_s": sum(float(r.get("level_surface_1d_newton_time_s", 0.0)) for r in surface_results),
            "boozer_ls_s": sum(float(r.get("ls_time_s", 0.0)) for r in surface_results),
            "boozer_newton_s": sum(float(r.get("newton_time_s", 0.0)) for r in surface_results),
            "boozer_qs_s": sum(float(r.get("qs_time_s", 0.0)) for r in surface_results),
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
