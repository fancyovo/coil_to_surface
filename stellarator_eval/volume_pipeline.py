from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np

from .axis import find_axis, find_axis_gpu
from .config import EvalConfig
from .field import FieldInput, build_field
from .psi import fit_psi, model_to_npz_dict
from .serialization import write_json
from .surface import screen_level, screen_levels_gpu
from .volume_qs import evaluate_volume_qs_model


def _set_thread_env(threads: int) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)


def _write_run_artifacts(output_dir, result, *, axis=None, model=None, nfp=None) -> None:
    if output_dir is None:
        return
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if axis is not None:
        np.savez(
            output / "axis_data.npz",
            phi=axis.phi,
            R=axis.R,
            Z=axis.Z,
            R_phi=axis.R_phi,
            Z_phi=axis.Z_phi,
            nfp=nfp,
        )
    if model is not None:
        np.savez(output / "psi_model.npz", **model_to_npz_dict(model))
    write_json(output / "summary.json", result)


def evaluate_coils_to_volume_qs(
    field_input: FieldInput,
    config: EvalConfig,
    *,
    target_helicity: tuple[int, int],
    output_dir: str | Path | None = None,
) -> dict:
    """Run the stable coil-to-psi route followed by the linear volume-QS backend."""
    _set_thread_env(min(int(config.omp_threads), 16))
    started = time.perf_counter()
    timing = {}

    field_start = time.perf_counter()
    built = build_field(field_input, config.current_unit)
    timing["field_build_s"] = float(time.perf_counter() - field_start)

    axis_start = time.perf_counter()
    if config.axis.backend.lower() == "gpu":
        axis = find_axis_gpu(
            field_input,
            built.field,
            built.nfp,
            built.coil_r0,
            config.axis,
            config.current_unit,
        )
    else:
        axis = find_axis(built.field, built.nfp, built.coil_r0, config.axis)
    timing["axis_s"] = float(time.perf_counter() - axis_start)
    result = {
        "input_name": field_input.name,
        "nfp": int(field_input.nfp),
        "target_helicity": list(target_helicity),
        "axis": {
            "has_axis": bool(axis.has_axis),
            "best_R": float(axis.best_R),
            "best_Z": float(axis.best_Z),
            "best_residual": float(axis.best_residual),
            "search_best_residual": float(axis.search_best_residual),
            "generation": int(axis.generation),
            "backend": axis.backend,
            "trace_error": axis.trace_error,
            "failure_reason": axis.failure_reason,
            "topology_class": axis.topology_class,
            "topology_trace": axis.topology_trace,
            "topology_det": axis.topology_det,
            "topology_ellipse_aspect": axis.topology_ellipse_aspect,
        },
    }
    if not axis.has_axis:
        result.update(status="failed", reason="stable magnetic-axis search failed")
        result["timing"] = {**timing, "total_s": float(time.perf_counter() - started)}
        _write_run_artifacts(output_dir, result)
        return result

    psi_start = time.perf_counter()
    model = fit_psi(
        built.field,
        axis,
        built.nfp,
        config.psi,
        field_input=field_input,
        current_unit=config.current_unit,
    )
    timing["psi_fit_s"] = float(time.perf_counter() - psi_start)
    result["psi"] = {"a": float(model.a), "fit_info": model.fit_info}

    screen_start = time.perf_counter()
    screen_results = []
    if config.scan.trace_backend.lower() == "gpu":
        try:
            screen_results = screen_levels_gpu(
                field_input, model, config.scan.levels, config.scan, config.current_unit
            )
        except Exception:
            screen_results = []
    if not screen_results:
        for level in config.scan.levels:
            try:
                screen_results.append(
                    screen_level(built.field, model, float(level), config.scan).__dict__
                )
            except Exception as exc:
                screen_results.append(
                    {"psi_level": float(level), "ok": False, "reason": repr(exc)}
                )
    timing["surface_screen_s"] = float(time.perf_counter() - screen_start)
    stable_ok_levels = [float(item["psi_level"]) for item in screen_results if item.get("ok")]
    ok_levels = [
        float(item["psi_level"])
        for item in screen_results
        if item.get("ok")
        and float(item.get("rel_end_distance_p95", np.inf))
        <= config.volume_qs.screen_relative_drift_tolerance
    ]
    result["surface_screen"] = {
        "levels": screen_results,
        "stable_accepted_levels": stable_ok_levels,
        "volume_qs_accepted_levels": ok_levels,
    }
    if not stable_ok_levels:
        result.update(
            status="failed",
            reason="no psi level passed the stable surface screen",
        )
        result["timing"] = {**timing, "total_s": float(time.perf_counter() - started)}
        _write_run_artifacts(output_dir, result, axis=axis, model=model, nfp=built.nfp)
        return result
    if not ok_levels:
        result.update(
            status="failed",
            reason="no psi level passed the volume-QS fieldline drift gate",
        )
        result["timing"] = {**timing, "total_s": float(time.perf_counter() - started)}
        _write_run_artifacts(output_dir, result, axis=axis, model=model, nfp=built.nfp)
        return result

    downstream = evaluate_volume_qs_model(
        field_input,
        model,
        ok_levels,
        config.volume_qs,
        current_unit=config.current_unit,
        target_helicity=target_helicity,
    )
    result["volume_qs"] = downstream
    result["status"] = downstream["status"]
    if downstream["status"] != "ok":
        result["reason"] = downstream["reason"]
    result["timing"] = {
        **timing,
        "volume_qs_s": float(downstream["timing"]["downstream_total_s"]),
        "total_s": float(time.perf_counter() - started),
    }

    _write_run_artifacts(output_dir, result, axis=axis, model=model, nfp=built.nfp)
    return result
