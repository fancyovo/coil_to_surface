from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stellarator_eval.config import AxisGAConfig
from stellarator_eval.psi import PolyMode, PsiModel, psi_and_gradient
from stellarator_eval.quasr import load_quasr_field_input
from stellarator_eval.serialization import jsonable, write_json


QUASR_ROOT_ENV = "QUASR_ROOT"
TWOPI = 2.0 * np.pi


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser()


def parse_ids(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]


def configure_plot_fonts() -> None:
    try:
        import matplotlib
        from matplotlib import font_manager
    except Exception:
        return
    candidates = (
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "PingFang SC",
        "Arial Unicode MS",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans", "sans-serif"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
    matplotlib.rcParams["axes.unicode_minus"] = False


def load_rows(run_dir: Path) -> dict[int, dict]:
    path = run_dir / "batch_summary.csv"
    with path.open(encoding="utf-8-sig") as f:
        return {int(row["ID"]): row for row in csv.DictReader(f)}


def load_summary(run_dir: Path, device_id: int) -> dict:
    path = run_dir / f"id_{device_id:07d}" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_psi_model(path: Path) -> PsiModel | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    modes = [
        PolyMode(int(a), int(b), int(m), str(kind))
        for a, b, m, kind in zip(data["mode_a"], data["mode_b"], data["mode_m"], data["mode_kind"])
    ]
    fit_info = {}
    for key in data.files:
        if key.startswith("info_"):
            value = data[key]
            if isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            fit_info[key[5:]] = value
    return PsiModel(
        coeffs=np.asarray(data["coeffs"], dtype=float),
        modes=modes,
        nfp=int(data["nfp"]),
        a=float(data["a"]),
        phi_axis=np.asarray(data["phi_axis"], dtype=float),
        R_axis=np.asarray(data["R_axis"], dtype=float),
        Z_axis=np.asarray(data["Z_axis"], dtype=float),
        R_axis_phi=np.asarray(data["R_axis_phi"], dtype=float),
        Z_axis_phi=np.asarray(data["Z_axis_phi"], dtype=float),
        fit_info=fit_info,
    )


def axis_cfg_from_summary(summary: dict, *, gpu_device: int) -> AxisGAConfig:
    src = summary["config"]["axis"]
    values = {key: src[key] for key in AxisGAConfig.__dataclass_fields__ if key in src}
    values["gpu_device"] = int(gpu_device)
    return AxisGAConfig(**values)


def make_gpu_field(field_input, cfg: AxisGAConfig, current_unit: str):
    gpu_python = REPO_ROOT / "gpu_backend" / "python"
    if str(gpu_python) not in sys.path:
        sys.path.insert(0, str(gpu_python))
    from stellarator_gpu import CoilFieldGpu

    unit = current_unit.lower()
    if unit in {"ma", "megaamp", "megaamps"}:
        currents = np.asarray(field_input.currents, dtype=float) * 1e6
    elif unit in {"a", "amp", "amps"}:
        currents = np.asarray(field_input.currents, dtype=float)
    else:
        raise ValueError(f"unknown current unit {current_unit!r}")
    lib_path = Path(cfg.gpu_lib_path)
    if not lib_path.is_absolute():
        lib_path = REPO_ROOT / lib_path
    return CoilFieldGpu(
        lib_path,
        field_input.coeffs_x,
        field_input.coeffs_y,
        field_input.coeffs_z,
        currents,
        nfp=field_input.nfp,
        segments_per_coil=cfg.gpu_segments_per_coil,
        device_id=cfg.gpu_device,
    )


def trace_closure_heatmap(field_input, summary: dict, *, grid: int, gpu_device: int) -> dict:
    axis = summary["axis"]
    a = float(summary["config"]["psi"]["a"])
    cfg = axis_cfg_from_summary(summary, gpu_device=gpu_device)
    r0 = float(axis["best_R"])
    z0 = float(axis["best_Z"])
    r_min = max(float(cfg.fixed_point_r_floor), r0 - a)
    r_max = max(r_min + 1e-6, r0 + a)
    z_min = z0 - a
    z_max = z0 + a
    rs = np.linspace(r_min, r_max, grid)
    zs = np.linspace(z_min, z_max, grid)
    rg, zg = np.meshgrid(rs, zs, indexing="xy")
    flat_r = np.ascontiguousarray(rg.ravel(), dtype=float)
    flat_z = np.ascontiguousarray(zg.ravel(), dtype=float)
    t0 = time.perf_counter()
    gpu_field = make_gpu_field(field_input, cfg, summary["config"].get("current_unit", "A"))
    try:
        re, ze = gpu_field.trace_period_blockline_precision(
            flat_r,
            flat_z,
            steps=cfg.rk4_steps,
            precision=cfg.gpu_trace_precision,
            threads_per_line=cfg.gpu_threads_per_line,
            nfp=field_input.nfp,
        )
    finally:
        gpu_field.close()
    trace_s = time.perf_counter() - t0
    residual = np.hypot(re - flat_r, ze - flat_z).reshape(grid, grid)
    radius = np.hypot(rg - r0, zg - z0)
    inside = radius <= a
    inside_vals = residual[inside & np.isfinite(residual)]
    if inside_vals.size:
        stats = {
            "closure_inside_min": float(np.min(inside_vals)),
            "closure_inside_p50": float(np.percentile(inside_vals, 50)),
            "closure_inside_p95": float(np.percentile(inside_vals, 95)),
            "closure_inside_p99": float(np.percentile(inside_vals, 99)),
            "closure_inside_max": float(np.max(inside_vals)),
            "closure_inside_frac_lt_1e-4": float(np.mean(inside_vals < 1e-4)),
            "closure_inside_frac_lt_1e-3": float(np.mean(inside_vals < 1e-3)),
        }
    else:
        stats = {}
    return {
        "R": rs,
        "Z": zs,
        "residual": residual,
        "axis_R": r0,
        "axis_Z": z0,
        "a": a,
        "trace_s": trace_s,
        **stats,
    }


def draw_closure_heatmap(data: dict, output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    configure_plot_fonts()
    rs = data["R"]
    zs = data["Z"]
    residual = np.asarray(data["residual"], dtype=float)
    values = np.log10(np.maximum(residual, 1e-12))
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    im = ax.imshow(
        values,
        origin="lower",
        extent=[float(rs[0]), float(rs[-1]), float(zs[0]), float(zs[-1])],
        aspect="equal",
        cmap="viridis",
    )
    circle = plt.Circle((data["axis_R"], data["axis_Z"]), data["a"], color="white", fill=False, lw=1.2)
    ax.add_patch(circle)
    ax.plot([data["axis_R"]], [data["axis_Z"]], "r+", ms=9, mew=1.5)
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10 one-period closure residual")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def signed_sqrt(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.sqrt(np.abs(values))


def draw_psi_slices(model: PsiModel, summary: dict, output: Path, title: str, *, grid: int, phi_count: int) -> dict:
    import matplotlib.pyplot as plt

    configure_plot_fonts()
    a = float(model.a)
    period = TWOPI / model.nfp
    phis = np.linspace(0.0, period, phi_count)
    levels = [float(x) for x in summary["config"]["scan"]["levels"] if float(x) > 0.0]
    psi_max = max(levels) if levels else 0.16
    v = np.sqrt(psi_max)
    cols = min(5, phi_count)
    rows = int(np.ceil(phi_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.1 * rows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    stats = []
    for ax, phi in zip(axes_arr, phis):
        ra, za, _, _ = model.axis_at(np.array([phi]))
        rc = float(ra[0])
        zc = float(za[0])
        rs = np.linspace(max(1e-5, rc - 1.05 * a), rc + 1.05 * a, grid)
        zs = np.linspace(zc - 1.05 * a, zc + 1.05 * a, grid)
        rg, zg = np.meshgrid(rs, zs, indexing="xy")
        pp = np.full(rg.size, float(phi))
        psi, *_ = psi_and_gradient(model, rg.ravel(), zg.ravel(), pp)
        psi = psi.reshape(grid, grid)
        transformed = signed_sqrt(psi)
        im = ax.imshow(
            transformed,
            origin="lower",
            extent=[float(rs[0]), float(rs[-1]), float(zs[0]), float(zs[-1])],
            aspect="equal",
            cmap="coolwarm",
            vmin=-v,
            vmax=v,
        )
        contour_levels = [x for x in levels if np.nanmin(psi) <= x <= np.nanmax(psi)]
        if contour_levels:
            ax.contour(rg, zg, psi, levels=contour_levels, colors="black", linewidths=0.45, alpha=0.65)
        ax.plot([rc], [zc], "k+", ms=7, mew=1.2)
        ax.set_title(f"Phi={phi / period:.2f} period")
        ax.set_xlabel("R")
        ax.set_ylabel("Z")
        stats.append(
            {
                "phi": float(phi),
                "psi_min": float(np.nanmin(psi)),
                "psi_max": float(np.nanmax(psi)),
                "psi_abs_p95": float(np.nanpercentile(np.abs(psi), 95)),
            }
        )
    for ax in axes_arr[len(phis) :]:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes_arr.tolist())
    cbar.set_label("signed sqrt(psi), saturated at scan max")
    fig.suptitle(title)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return {"psi_slice_stats": stats}


def selected_screen_rows(summary: dict, limit: int = 10) -> list[dict]:
    levels = (summary.get("surface_screen") or {}).get("levels") or []
    rows = []
    for item in levels[:limit]:
        rows.append(
            {
                "psi_level": item.get("psi_level"),
                "ok": item.get("ok"),
                "radius_mean": item.get("radius_mean"),
                "radius_max": item.get("radius_max"),
                "end_distance_p95": item.get("end_distance_p95"),
                "rel_end_distance_p95": item.get("rel_end_distance_p95"),
                "reason": item.get("reason"),
            }
        )
    return rows


def summarize_candidates(summary: dict) -> list[dict]:
    out = []
    for item in summary.get("surface_candidates") or []:
        out.append(
            {
                "psi_level": item.get("psi_level"),
                "error": item.get("error"),
                "initial_boozer_residual_norm": item.get("initial_boozer_residual_norm"),
                "ls_success": item.get("ls_success"),
                "ls_residual_norm": item.get("ls_residual_norm"),
                "newton_success": item.get("newton_success"),
                "newton_residual_norm": item.get("newton_residual_norm"),
                "volume": item.get("volume"),
            }
        )
    return out


def diagnose_one(run_dir: Path, row: dict | None, device_id: int, output_dir: Path, args) -> dict:
    summary = load_summary(run_dir, device_id)
    field_input, info = load_quasr_field_input(args.quasr_root, device_id)
    device_dir = output_dir / f"id_{device_id:07d}"
    device_dir.mkdir(parents=True, exist_ok=True)
    closure = trace_closure_heatmap(field_input, summary, grid=args.closure_grid, gpu_device=args.gpu_device)
    closure_png = device_dir / "closure_residual_heatmap.png"
    draw_closure_heatmap(closure, closure_png, f"ID {device_id} closure residual")
    model = load_psi_model(run_dir / f"id_{device_id:07d}" / "psi_model.npz")
    psi_png = None
    psi_stats = {}
    if model is not None:
        psi_png = device_dir / "psi_sqrt_slices.png"
        psi_stats = draw_psi_slices(
            model,
            summary,
            psi_png,
            f"ID {device_id} signed sqrt psi",
            grid=args.psi_grid,
            phi_count=args.phi_count,
        )
    result = {
        "device_id": int(device_id),
        "run_dir": str(run_dir),
        "status": None if row is None else row.get("status"),
        "metadata_nfp": None if row is None else row.get("meta_nfp"),
        "metadata_helicity": None if row is None else row.get("meta_helicity"),
        "quasr_info": info,
        "axis": summary.get("axis"),
        "psi_fit_info": (summary.get("psi") or {}).get("fit_info"),
        "surface_screen_rows": selected_screen_rows(summary),
        "surface_candidates": summarize_candidates(summary),
        "best_surface": summary.get("best_surface"),
        "warnings": summary.get("warnings"),
        "closure": {k: v for k, v in closure.items() if k not in {"R", "Z", "residual"}},
        "closure_heatmap": str(closure_png.relative_to(output_dir)),
        "psi_slices": None if psi_png is None else str(psi_png.relative_to(output_dir)),
        **psi_stats,
    }
    write_json(device_dir / "diagnostic_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose no-surface cases after fixed-point axis search.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--ids", required=True, help="Comma/space-separated QUASR IDs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quasr-root", type=Path, default=_env_path(QUASR_ROOT_ENV))
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--closure-grid", type=int, default=96)
    parser.add_argument("--psi-grid", type=int, default=160)
    parser.add_argument("--phi-count", type=int, default=9)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.quasr_root is None:
        raise ValueError(f"QUASR root is required; pass --quasr-root or set {QUASR_ROOT_ENV}")
    row_by_id: dict[int, tuple[Path, dict]] = {}
    for run_dir in args.run_dir:
        for device_id, row in load_rows(run_dir).items():
            row_by_id[int(device_id)] = (run_dir, row)

    results = []
    for device_id in parse_ids(args.ids):
        if device_id not in row_by_id:
            raise KeyError(f"ID {device_id} not found in provided run dirs")
        run_dir, row = row_by_id[device_id]
        print(f"diagnose ID={device_id} status={row.get('status')} run={run_dir}", flush=True)
        results.append(diagnose_one(run_dir, row, device_id, args.output_dir, args))
    write_json(args.output_dir / "diagnostic_batch_summary.json", jsonable(results))


if __name__ == "__main__":
    main()
