from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .axis import find_axis, find_axis_gpu
from .config import EvalConfig
from .config import BoozerConfig
from .field import FieldInput, build_field, input_from_flat_vector, input_from_packed_vector, load_case_file
from .psi import fit_psi, model_to_npz_dict
from .score import ScoreConfig, evaluate_quality_score
from .serialization import write_json
from .surface import evaluate_boozer_surface, screen_level, screen_levels_gpu
from .timing import timing_phase, timing_session
from .visualization import export_axis_residual_heatmap, export_psi_slices


def _set_thread_env(threads: int) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)


def _axis_summary(axis) -> dict:
    return {
        "has_axis": axis.has_axis,
        "best_R": axis.best_R,
        "best_Z": axis.best_Z,
        "best_residual": axis.best_residual,
        "search_best_residual": axis.search_best_residual,
        "best_residual_definition": "closure distance after tracing one field period from the best candidate",
        "generation": axis.generation,
        "time_s": axis.time_s,
        "search_time_s": axis.search_time_s,
        "trace_time_s": axis.trace_time_s,
        "backend": axis.backend,
        "trace_error": axis.trace_error,
        "failure_reason": axis.failure_reason,
        "topology_class": axis.topology_class,
        "topology_trace": axis.topology_trace,
        "topology_det": axis.topology_det,
        "topology_ellipse_aspect": axis.topology_ellipse_aspect,
        "topology_time_s": axis.topology_time_s,
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


def _timing_summary(*, field_build_s: float = 0.0, axis=None, model=None, screen_results=None, surface_results=None) -> dict:
    screen_results = screen_results or []
    surface_results = surface_results or []
    fit_info = {} if model is None else model.fit_info
    return {
        "field_build_s": float(field_build_s),
        "axis_s": 0.0 if axis is None else float(axis.time_s),
        "axis_search_s": 0.0 if axis is None else float(axis.search_time_s),
        "axis_trace_s": 0.0 if axis is None else float(axis.trace_time_s),
        "axis_topology_s": 0.0 if axis is None else float(axis.topology_time_s),
        "psi_fit_s": float(fit_info.get("time_s", 0.0)),
        "psi_gpu_create_s": float(fit_info.get("gpu_create_s", 0.0)),
        "psi_training_point_s": float(fit_info.get("training_point_s", 0.0)),
        "psi_assemble_s": float(fit_info.get("assemble_s", 0.0)),
        "psi_assemble_interp_s": float(fit_info.get("assemble_interp_s", 0.0)),
        "psi_assemble_b_sample_s": float(fit_info.get("assemble_b_sample_s", 0.0)),
        "psi_assemble_basis_s": float(fit_info.get("assemble_basis_s", 0.0)),
        "psi_assemble_normal_eq_s": float(fit_info.get("assemble_normal_eq_s", 0.0)),
        "psi_assemble_normal_eq_cpu_s": float(fit_info.get("assemble_normal_eq_cpu_s", 0.0)),
        "psi_assemble_normal_eq_gpu_s": float(fit_info.get("assemble_normal_eq_gpu_s", 0.0)),
        "psi_qr_prep_s": float(fit_info.get("qr_prep_s", 0.0)),
        "psi_qr_transpose_s": float(fit_info.get("qr_transpose_s", 0.0)),
        "psi_qr_scale_s": float(fit_info.get("qr_scale_s", 0.0)),
        "psi_qr_factor_s": float(fit_info.get("qr_factor_s", 0.0)),
        "psi_qr_apply_qtb_s": float(fit_info.get("qr_apply_qtb_s", 0.0)),
        "psi_qr_tri_s": float(fit_info.get("qr_tri_s", 0.0)),
        "psi_fullgpu_copy_in_s": float(fit_info.get("fullgpu_copy_in_s", 0.0)),
        "psi_fullgpu_residual_s": float(fit_info.get("fullgpu_residual_s", 0.0)),
        "psi_fullgpu_copy_out_s": float(fit_info.get("fullgpu_copy_out_s", 0.0)),
        "psi_fullgpu_total_kernel_s": float(fit_info.get("fullgpu_total_kernel_s", 0.0)),
        "psi_solve_s": float(fit_info.get("solve_s", 0.0)),
        "psi_validation_s": float(fit_info.get("validation_s", 0.0)),
        "psi_validation_b_sample_s": float(fit_info.get("validation_b_sample_s", 0.0)),
        "surface_screen_s": 0.0,
        "surface_screen_curve_newton_s": sum(float(r.get("curve_newton_time_s", 0.0)) for r in screen_results),
        "surface_screen_fieldline_trace_s": sum(float(r.get("trace_time_s", 0.0)) for r in screen_results),
        "surface_screen_verify_trace_s": sum(float(r.get("verify_trace_time_s", 0.0)) for r in screen_results),
        "surface_extract_1d_newton_s": sum(float(r.get("level_surface_1d_newton_time_s", 0.0)) for r in surface_results),
        "surface_extract_coeff_build_s": sum(float(r.get("level_surface_coeff_build_time_s", 0.0)) for r in surface_results),
        "surface_extract_copy_in_s": sum(float(r.get("level_surface_copy_in_time_s", 0.0)) for r in surface_results),
        "surface_extract_copy_out_s": sum(float(r.get("level_surface_copy_out_time_s", 0.0)) for r in surface_results),
        "boozer_ls_s": sum(float(r.get("ls_time_s", 0.0)) for r in surface_results),
        "boozer_newton_s": sum(float(r.get("newton_time_s", 0.0)) for r in surface_results),
        "boozer_qs_s": sum(float(r.get("qs_time_s", 0.0)) for r in surface_results),
        "boozer_candidates_s": sum(float(r.get("total_time_s", 0.0)) for r in surface_results),
    }


def _field_input_from_any(
    coil_coefficients: Any,
    currents: Any | None = None,
    nfp: int | None = None,
    *,
    coeff_count: int = 33,
) -> FieldInput:
    if isinstance(coil_coefficients, FieldInput):
        return coil_coefficients
    if currents is None:
        arr = np.asarray(coil_coefficients, dtype=float).ravel()
        if nfp is None:
            return input_from_packed_vector(arr, coeff_count=coeff_count)
        return input_from_flat_vector(arr, nfp=nfp, coeff_count=coeff_count)
    if nfp is None:
        raise ValueError("nfp is required when currents are passed separately")
    if isinstance(coil_coefficients, dict):
        x, y, z = coil_coefficients["x"], coil_coefficients["y"], coil_coefficients["z"]
    else:
        arr = np.asarray(coil_coefficients, dtype=float)
        if arr.ndim == 3 and arr.shape[1] == 3:
            x, y, z = arr[:, 0, :], arr[:, 1, :], arr[:, 2, :]
        else:
            raise ValueError("coil_coefficients must be FieldInput, flat vector, dict{x,y,z}, or array (ncoil,3,ncoef)")
    return FieldInput(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float), np.asarray(currents, float), int(nfp))


def _attach_quality_score(
    result: dict,
    field_input: FieldInput,
    config: EvalConfig,
    *,
    metadata: dict | None = None,
    target: str | None = None,
    score_config: ScoreConfig | None = None,
) -> None:
    result["quality_score"] = evaluate_quality_score(
        result,
        field_input=field_input,
        current_unit=config.current_unit,
        metadata=metadata,
        target=target,
        config=score_config,
    )


def _auto_boozer_config(base: BoozerConfig, ok_levels: list[dict]) -> tuple[BoozerConfig, dict]:
    info = {
        "initial_iota_config": float(base.initial_iota),
        "initial_iota_used": float(base.initial_iota),
        "initial_iota_source": "config",
    }
    if not base.auto_initial_iota:
        return base, info
    if base.auto_initial_iota_default_only and abs(float(base.initial_iota) - float(base.auto_initial_iota_default_value)) > 1e-12:
        return base, info
    candidates = []
    for item in ok_levels:
        value = item.get("iota_estimate")
        spread = item.get("iota_estimate_std")
        try:
            value_f = float(value)
            spread_f = float(spread)
        except Exception:
            continue
        if not (np.isfinite(value_f) and np.isfinite(spread_f)):
            continue
        candidates.append((float(item.get("psi_level", 0.0)), value_f, spread_f))
    if not candidates:
        info["initial_iota_source"] = "config_no_screen_estimate"
        return base, info
    candidates.sort(key=lambda x: (x[0], -x[2]), reverse=True)
    level, estimate, spread = candidates[0]
    info.update(
        {
            "initial_iota_used": estimate,
            "initial_iota_source": "screen_fieldline_estimate",
            "initial_iota_estimate_level": level,
            "initial_iota_estimate_std": spread,
        }
    )
    return replace(base, initial_iota=estimate), info


def evaluate_field_input(
    field_input: FieldInput,
    config: EvalConfig | None = None,
    output_dir: str | Path = "runs/eval",
    *,
    metadata: dict | None = None,
    target: str | None = None,
    score_config: ScoreConfig | None = None,
) -> dict:
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
        "diagnostics": {},
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
        if config.diagnostics.export_axis_heatmap:
            t_plot = time.perf_counter()
            try:
                info = export_axis_residual_heatmap(
                    field_input,
                    built,
                    axis,
                    config.axis,
                    config.current_unit,
                    out / config.diagnostics.axis_heatmap_filename,
                    grid=config.diagnostics.axis_heatmap_grid,
                    dpi=config.diagnostics.plot_dpi,
                )
                info["time_s"] = float(time.perf_counter() - t_plot)
                result["diagnostics"]["axis_residual_heatmap"] = info
            except Exception as exc:
                result["warnings"].append(f"axis residual heatmap export failed: {exc!r}")
        if not axis.has_axis:
            result["warnings"].append("axis residual did not reach configured tolerance; downstream steps skipped")
            result["timing_b_calls"] = timings.as_dict()
            result["timing"] = _timing_summary(field_build_s=result["field"]["build_time_s"], axis=axis)
            result["total_time_s"] = time.perf_counter() - t_all
            _attach_quality_score(result, field_input, config, metadata=metadata, target=target, score_config=score_config)
            write_json(out / "summary.json", result)
            return result

        with timing_phase("psi_fit"):
            model = fit_psi(built.field, axis, built.nfp, config.psi, field_input=field_input, current_unit=config.current_unit)
        result["psi"] = _psi_summary(model)
        np.savez(out / "psi_model.npz", **model_to_npz_dict(model))
        if config.diagnostics.export_psi_slices:
            t_plot = time.perf_counter()
            try:
                info = export_psi_slices(
                    model,
                    out / config.diagnostics.psi_slice_filename,
                    levels=config.scan.levels,
                    grid=config.diagnostics.psi_slice_grid,
                    phi_count=config.diagnostics.psi_slice_phi_count,
                    dpi=config.diagnostics.plot_dpi,
                )
                info["time_s"] = float(time.perf_counter() - t_plot)
                result["diagnostics"]["psi_slices"] = info
            except Exception as exc:
                result["warnings"].append(f"psi slice export failed: {exc!r}")

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
            result["timing"] = _timing_summary(
                field_build_s=result["field"]["build_time_s"],
                axis=axis,
                model=model,
                screen_results=screen_results,
            )
            result["timing"]["surface_screen_s"] = result["surface_screen"]["time_s"]
            result["total_time_s"] = time.perf_counter() - t_all
            _attach_quality_score(result, field_input, config, metadata=metadata, target=target, score_config=score_config)
            write_json(out / "summary.json", result)
            return result

        boozer_cfg, iota_info = _auto_boozer_config(config.boozer, ok_levels)
        result["boozer_initial_iota"] = iota_info

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
                        boozer_cfg,
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
        result["timing"] = _timing_summary(
            field_build_s=result["field"]["build_time_s"],
            axis=axis,
            model=model,
            screen_results=screen_results,
            surface_results=surface_results,
        )
        result["timing"]["surface_screen_s"] = result["surface_screen"]["time_s"]
        result["total_time_s"] = time.perf_counter() - t_all
        _attach_quality_score(result, field_input, config, metadata=metadata, target=target, score_config=score_config)
        write_json(out / "summary.json", result)
        return result


def evaluate_case_file(
    case_file: str | Path,
    key: str = "raw",
    config: EvalConfig | None = None,
    output_dir: str | Path = "runs/eval",
    *,
    metadata: dict | None = None,
    target: str | None = None,
    score_config: ScoreConfig | None = None,
) -> dict:
    return evaluate_field_input(
        load_case_file(case_file, key),
        config=config,
        output_dir=output_dir,
        metadata=metadata,
        target=target,
        score_config=score_config,
    )


def evaluate_coils(
    coil_coefficients: Any,
    currents: Any | None = None,
    nfp: int | None = None,
    config: EvalConfig | None = None,
    output_dir: str | Path = "runs/eval",
    *,
    coeff_count: int = 33,
    metadata: dict | None = None,
    target: str | None = None,
    score_config: ScoreConfig | None = None,
) -> dict:
    field_input = _field_input_from_any(coil_coefficients, currents, nfp, coeff_count=coeff_count)
    return evaluate_field_input(
        field_input,
        config=config,
        output_dir=output_dir,
        metadata=metadata,
        target=target,
        score_config=score_config,
    )


def evaluate_coil_quality(
    coil_coefficients: Any,
    currents: Any | None = None,
    nfp: int | None = None,
    config: EvalConfig | None = None,
    output_dir: str | Path = "runs/eval",
    *,
    coeff_count: int = 33,
    metadata: dict | None = None,
    target: str | None = None,
    score_config: ScoreConfig | None = None,
    include_result: bool = True,
) -> dict:
    result = evaluate_coils(
        coil_coefficients,
        currents,
        nfp,
        config=config,
        output_dir=output_dir,
        coeff_count=coeff_count,
        metadata=metadata,
        target=target,
        score_config=score_config,
    )
    quality = result["quality_score"]
    out = {
        "score": quality["score"],
        "status": quality["status"],
        "quality_score": quality,
    }
    if include_result:
        out["result"] = result
    return out
