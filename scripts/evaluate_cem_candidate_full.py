from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_float_list(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("expected a non-empty comma-separated list of positive values")
    return values


def select_largest_surface(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        row
        for row in rows
        if row.get("best_surface") is not None
        and np.isfinite(float(row["best_surface"].get("volume", float("nan"))))
    ]
    return max(valid, key=lambda row: float(row["best_surface"]["volume"])) if valid else None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def surface_path(run_dir: Path, summary: dict[str, Any]) -> Path:
    level = float(summary["best_surface"]["psi_level"])
    return run_dir / f"level_{level:.6g}".replace(".", "p") / "boozer_surface.npz"


def rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    out = points.copy()
    out[..., 0] = cosine * points[..., 0] - sine * points[..., 1]
    out[..., 1] = sine * points[..., 0] + cosine * points[..., 1]
    return out


def render_boozer_and_geometry(
    *,
    case_file: Path,
    surface_npz: Path,
    output_dir: Path,
    current_unit: str,
    surface_order: int,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    from scripts.desc_psi_volume_initial_guess_experiment import make_xyz_surface
    from scripts.plot_poincare_validation import plot_poincare_from_dofs
    from stellarator_eval.field import build_field, load_case_file

    output_dir.mkdir(parents=True, exist_ok=True)
    field_input = load_case_file(case_file, "raw")
    built = build_field(field_input, current_unit=current_unit)
    surface, surface_meta = make_xyz_surface(
        surface_npz,
        nfp=field_input.nfp,
        order=surface_order,
        stellsym=True,
        nphi=96,
        ntheta=192,
    )
    xyz = np.asarray(surface.gamma(), dtype=float)
    built.field.set_points(xyz.reshape(-1, 3))
    b_abs = np.linalg.norm(np.asarray(built.field.B(), dtype=float), axis=1).reshape(xyz.shape[:2])
    zeta = np.linspace(0.0, 2.0 * np.pi, xyz.shape[0], endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, xyz.shape[1], endpoint=False)

    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    image = axis.pcolormesh(zeta, theta, b_abs.T, shading="auto", cmap="turbo")
    axis.set_xlabel(r"field-period angle $N_{\rm FP}\phi$")
    axis.set_ylabel(r"Boozer $\theta$")
    axis.set_title(r"$|B|$ on largest Boozer-solvable surface")
    fig.colorbar(image, ax=axis, label=r"$|B|$ [T]")
    fig.tight_layout()
    fig.savefig(output_dir / "boozer_b.png", dpi=190)
    plt.close(fig)

    heatmap = go.Figure(
        data=go.Heatmap(
            x=zeta,
            y=theta,
            z=b_abs.T,
            colorscale="Turbo",
            colorbar={"title": "|B| [T]"},
        )
    )
    heatmap.update_layout(
        title="|B| on largest Boozer-solvable surface",
        xaxis_title="NFP phi",
        yaxis_title="Boozer theta",
    )
    heatmap.write_html(output_dir / "boozer_b.html", include_plotlyjs="cdn")

    scene = go.Figure()
    for period in range(field_input.nfp):
        angle = 2.0 * np.pi * period / field_input.nfp
        gamma = rotate_z(xyz, angle)
        scene.add_trace(
            go.Surface(
                x=gamma[..., 0],
                y=gamma[..., 1],
                z=gamma[..., 2],
                surfacecolor=b_abs,
                colorscale="Turbo",
                cmin=float(np.min(b_abs)),
                cmax=float(np.max(b_abs)),
                showscale=period == 0,
                colorbar={"title": "|B| [T]"},
                opacity=0.86,
                name="Boozer surface",
            )
        )
    for index, coil in enumerate(built.field.coils):
        gamma = np.asarray(coil.curve.gamma(), dtype=float)
        scene.add_trace(
            go.Scatter3d(
                x=gamma[:, 0],
                y=gamma[:, 1],
                z=gamma[:, 2],
                mode="lines",
                line={"color": "#222222", "width": 5},
                name="coils" if index == 0 else None,
                showlegend=index == 0,
            )
        )
    scene.update_layout(
        title="Optimized coils and largest Boozer-solvable surface",
        scene={"aspectmode": "data", "xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    scene.write_html(output_dir / "coils_surface.html", include_plotlyjs="cdn")

    surface_data = np.load(surface_npz)
    coil_dofs = np.concatenate(
        [field_input.coeffs_x, field_input.coeffs_y, field_input.coeffs_z], axis=1
    )
    poincare = plot_poincare_from_dofs(
        surface_data["dofs"],
        coil_dofs,
        field_input.currents,
        field_input.nfp,
        mpol=surface_order,
        ntor=surface_order,
        coil_order=field_input.order,
        nfieldlines=16,
        tmax_fl=1.0e8,
        save_path=output_dir / "poincare.png",
        marker_size=5,
        dpi=180,
    )
    return {
        "surface_meta": surface_meta,
        "b_abs_min": float(np.min(b_abs)),
        "b_abs_mean": float(np.mean(b_abs)),
        "b_abs_max": float(np.max(b_abs)),
        "poincare": poincare,
        "boozer_b_png": str(output_dir / "boozer_b.png"),
        "boozer_b_html": str(output_dir / "boozer_b.html"),
        "coils_surface_html": str(output_dir / "coils_surface.html"),
    }


def save_desc_plot(name: str, output_dir: Path, plotter, *args, **kwargs) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    try:
        figure, _ = plotter(*args, **kwargs)
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(figure)
        return {"success": True, "path": str(path), "time_s": time.perf_counter() - started}
    except Exception as exc:
        plt.close("all")
        return {
            "success": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "time_s": time.perf_counter() - started,
        }


def run_desc_boundary_solve(
    *,
    case_file: Path,
    surface_npz: Path,
    output_dir: Path,
    current_unit: str,
    surface_order: int,
    resolution: int,
    maxiter: int,
    ftol: float,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)

    from desc.geometry import FourierRZToroidalSurface
    from desc.plotting import (
        plot_1d,
        plot_boozer_modes,
        plot_boozer_surface,
        plot_boundary,
        plot_qs_error,
    )
    from simsopt.geo import ToroidalFlux

    from scripts.desc_psi_volume_initial_guess_experiment import (
        build_equilibrium,
        force_stats,
        make_xyz_surface,
        write_vmec_input_from_surface,
    )
    from stellarator_eval.field import build_field, load_case_file

    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    try:
        field_input = load_case_file(case_file, "raw")
        built = build_field(field_input, current_unit=current_unit)
        surface, _ = make_xyz_surface(
            surface_npz,
            nfp=field_input.nfp,
            order=surface_order,
            stellsym=True,
            nphi=96,
            ntheta=144,
        )
        toroidal_flux = float(ToroidalFlux(surface, built.field).J())
        result["toroidal_flux"] = toroidal_flux
        input_path = output_dir / "input.check"
        result["input"] = write_vmec_input_from_surface(surface, toroidal_flux, input_path)
        desc_surface = FourierRZToroidalSurface.from_input_file(str(input_path))
        eq = build_equilibrium(
            desc_surface,
            toroidal_flux,
            resolution,
            resolution,
            resolution,
            constructor_ensure_nested=True,
        )
        result["nested_initial"] = bool(eq.is_nested())
        result.update(force_stats(eq, "initial"))
        result["plots_initial"] = {
            "boundary": save_desc_plot("boundary_initial", output_dir, plot_boundary, eq)
        }
        started = time.perf_counter()
        try:
            solve_result = eq.solve(maxiter=maxiter, ftol=ftol, verbose=1)
            result["solve_call_success"] = True
            result["solve_return"] = str(solve_result)[:4000]
            optimizer = solve_result[1] if isinstance(solve_result, tuple) and len(solve_result) > 1 else None
            if optimizer is not None:
                result["optimizer_success"] = bool(getattr(optimizer, "success", False))
                result["optimizer_message"] = str(getattr(optimizer, "message", ""))
                for name in ("cost", "nit", "nfev", "njev", "optimality"):
                    if hasattr(optimizer, name):
                        value = getattr(optimizer, name)
                        result[f"optimizer_{name}"] = float(value)
        except Exception as exc:
            result["solve_call_success"] = False
            result["solve_error"] = repr(exc)
            result["solve_traceback"] = traceback.format_exc()
        result["solve_time_s"] = time.perf_counter() - started
        result["nested_final"] = bool(eq.is_nested())
        result.update(force_stats(eq, "final"))
        try:
            eq.save(str(output_dir / "equilibrium.h5"))
            result["equilibrium"] = str(output_dir / "equilibrium.h5")
        except Exception as exc:
            result["equilibrium_save_error"] = repr(exc)

        if result.get("solve_call_success"):
            result["plots_final"] = {
                "boundary": save_desc_plot("boundary", output_dir, plot_boundary, eq),
                "boozer_modes": save_desc_plot(
                    "boozer_modes", output_dir, plot_boozer_modes, eq, rho=10, num_modes=12, log=True
                ),
                "boozer_B": save_desc_plot(
                    "boozer_B", output_dir, plot_boozer_surface, eq, rho=1.0, fill=True, ncontours=32
                ),
                "qs_QA": save_desc_plot(
                    "qs_QA", output_dir, plot_qs_error, eq, helicity=(1, 0), log=True
                ),
                "qs_QH": save_desc_plot(
                    "qs_QH", output_dir, plot_qs_error, eq, helicity=(1, field_input.nfp), log=True
                ),
                "qs_QP": save_desc_plot(
                    "qs_QP", output_dir, plot_qs_error, eq, helicity=(0, 1), log=True
                ),
                "iota": save_desc_plot("iota", output_dir, plot_1d, eq, "iota"),
            }
    except Exception as exc:
        result["setup_error"] = repr(exc)
        result["setup_traceback"] = traceback.format_exc()
    write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stable Boozer/DESC evaluation for a CEM candidate.")
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("QA", "QH"), required=True)
    parser.add_argument("--a-values", default="0.05,0.08,0.12,0.16,0.20")
    parser.add_argument(
        "--levels",
        default="0.001,0.002,0.004,0.008,0.02,0.04,0.08,0.12,0.16,0.20,0.24,0.30,0.36,0.49,0.64,0.81",
    )
    parser.add_argument("--gpu-lib", default="gpu_backend/build_mixed/libstellarator_gpu.so")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--desc-resolution", type=int, default=8)
    parser.add_argument("--desc-maxiter", type=int, default=50)
    parser.add_argument("--desc-ftol", type=float, default=1.0e-8)
    args = parser.parse_args()

    from stellarator_eval.config import EvalConfig
    from stellarator_eval.pipeline import evaluate_case_file

    args.output_dir.mkdir(parents=True, exist_ok=True)
    a_values = parse_float_list(args.a_values)
    levels = parse_float_list(args.levels)
    rows = []
    total_started = time.perf_counter()
    for a_value in a_values:
        run_dir = args.output_dir / f"a_{a_value:.4g}".replace(".", "p")
        base = EvalConfig(current_unit="A")
        config = replace(
            base,
            axis=replace(base.axis, gpu_lib_path=args.gpu_lib, gpu_device=args.gpu_device),
            psi=replace(base.psi, a=a_value, gpu_lib_path=args.gpu_lib, gpu_device=args.gpu_device),
            scan=replace(
                base.scan,
                levels=levels,
                max_boozer_candidates=6,
                gpu_lib_path=args.gpu_lib,
                gpu_device=args.gpu_device,
            ),
            boozer=replace(base.boozer, gpu_lib_path=args.gpu_lib, gpu_device=args.gpu_device),
        )
        started = time.perf_counter()
        try:
            summary = evaluate_case_file(
                args.case_file,
                key="raw",
                config=config,
                output_dir=run_dir,
                target=args.target,
            )
            row = {
                "a": a_value,
                "run_dir": str(run_dir),
                "best_surface": summary.get("best_surface"),
                "quality_score": summary.get("quality_score"),
                "warnings": summary.get("warnings"),
                "total_time_s": time.perf_counter() - started,
            }
        except Exception as exc:
            row = {
                "a": a_value,
                "run_dir": str(run_dir),
                "best_surface": None,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "total_time_s": time.perf_counter() - started,
            }
        rows.append(row)
        write_json(args.output_dir / "sweep_progress.json", {"rows": rows})
        print(json.dumps(row, separators=(",", ":"), allow_nan=True), flush=True)

    best = select_largest_surface(rows)
    result: dict[str, Any] = {
        "case_file": str(args.case_file),
        "target": args.target,
        "rows": rows,
        "best": best,
        "total_sweep_time_s": time.perf_counter() - total_started,
    }
    if best is None:
        result["status"] = "no_boozer_surface"
        write_json(args.output_dir / "full_summary.json", result)
        raise SystemExit("no Boozer LS/Newton surface succeeded")

    best_run_dir = Path(best["run_dir"])
    best_summary = json.loads((best_run_dir / "summary.json").read_text(encoding="utf-8"))
    source_surface = surface_path(best_run_dir, best_summary)
    saved_surface = args.output_dir / "best_boozer_surface.npz"
    shutil.copy2(source_surface, saved_surface)
    result["best_surface_npz"] = str(saved_surface)
    result["visualization"] = render_boozer_and_geometry(
        case_file=args.case_file,
        surface_npz=saved_surface,
        output_dir=args.output_dir / "assets",
        current_unit="A",
        surface_order=int(best_summary["config"]["boozer"]["surface_order"]),
    )
    result["desc"] = run_desc_boundary_solve(
        case_file=args.case_file,
        surface_npz=saved_surface,
        output_dir=args.output_dir / "desc",
        current_unit="A",
        surface_order=int(best_summary["config"]["boozer"]["surface_order"]),
        resolution=args.desc_resolution,
        maxiter=args.desc_maxiter,
        ftol=args.desc_ftol,
    )
    result["status"] = "completed"
    result["total_time_s"] = time.perf_counter() - total_started
    write_json(args.output_dir / "full_summary.json", result)
    print(json.dumps({"status": result["status"], "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
