from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .axis import _fixed_point_domain, rk4_one_period
from .config import AxisGAConfig
from .psi import PsiModel, psi_and_gradient

TWOPI = 2.0 * np.pi


def _configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

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
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


def _currents_in_amp(currents, current_unit: str) -> np.ndarray:
    unit = current_unit.lower()
    if unit in {"ma", "megaamp", "megaamps"}:
        return np.asarray(currents, dtype=float) * 1e6
    if unit in {"a", "amp", "amps"}:
        return np.asarray(currents, dtype=float)
    raise ValueError(f"unknown current_unit={current_unit!r}; use 'MA' or 'A'")


def _make_gpu_field(field_input, nfp: int, cfg: AxisGAConfig, current_unit: str):
    import sys

    gpu_python = Path(__file__).resolve().parents[1] / "gpu_backend" / "python"
    if str(gpu_python) not in sys.path:
        sys.path.insert(0, str(gpu_python))
    from stellarator_gpu import CoilFieldGpu

    lib_path = Path(cfg.gpu_lib_path)
    if not lib_path.is_absolute():
        lib_path = Path.cwd() / lib_path
    return CoilFieldGpu(
        lib_path,
        field_input.coeffs_x,
        field_input.coeffs_y,
        field_input.coeffs_z,
        _currents_in_amp(field_input.currents, current_unit),
        nfp=nfp,
        segments_per_coil=cfg.gpu_segments_per_coil,
        device_id=cfg.gpu_device,
    )


def compute_axis_residual_grid(field_input, built, axis, cfg: AxisGAConfig, current_unit: str, *, grid: int) -> dict[str, Any]:
    domain = _fixed_point_domain(field_input, float(built.coil_r0), cfg)
    rs = np.linspace(domain["r_min"], domain["r_max"], int(grid))
    zs = np.linspace(domain["z_min"], domain["z_max"], int(grid))
    rg, zg = np.meshgrid(rs, zs, indexing="xy")
    r0 = np.ascontiguousarray(rg.ravel(), dtype=float)
    z0 = np.ascontiguousarray(zg.ravel(), dtype=float)
    t0 = time.perf_counter()
    backend = cfg.backend.lower()
    if backend == "gpu":
        gpu_field = _make_gpu_field(field_input, built.nfp, cfg, current_unit)
        try:
            r1, z1 = gpu_field.trace_period_blockline_precision(
                r0,
                z0,
                steps=cfg.rk4_steps,
                precision=cfg.gpu_trace_precision,
                threads_per_line=cfg.gpu_threads_per_line,
                nfp=built.nfp,
            )
        finally:
            gpu_field.close()
    else:
        r1, z1 = rk4_one_period(built.field, r0, z0, built.nfp, cfg.rk4_steps)
    residual = np.hypot(r1 - r0, z1 - z0).reshape(int(grid), int(grid))
    elapsed = time.perf_counter() - t0
    finite = residual[np.isfinite(residual)]
    best_idx = int(np.nanargmin(residual))
    best_j, best_i = np.unravel_index(best_idx, residual.shape)
    return {
        "rs": rs,
        "zs": zs,
        "residual": residual,
        "domain": domain,
        "grid": int(grid),
        "trace_time_s": float(elapsed),
        "residual_min": float(np.min(finite)) if finite.size else float("nan"),
        "residual_p50": float(np.percentile(finite, 50)) if finite.size else float("nan"),
        "residual_p95": float(np.percentile(finite, 95)) if finite.size else float("nan"),
        "grid_best_R": float(rs[best_i]),
        "grid_best_Z": float(zs[best_j]),
        "axis_R": float(axis.best_R),
        "axis_Z": float(axis.best_Z),
        "axis_residual": float(axis.best_residual),
        "axis_has_axis": bool(axis.has_axis),
        "axis_topology_class": str(axis.topology_class),
    }


def export_axis_residual_heatmap(
    field_input,
    built,
    axis,
    cfg: AxisGAConfig,
    current_unit: str,
    output_path: Path,
    *,
    grid: int,
    dpi: int,
) -> dict[str, Any]:
    data = compute_axis_residual_grid(field_input, built, axis, cfg, current_unit, grid=grid)
    plt = _configure_matplotlib()
    rs = data["rs"]
    zs = data["zs"]
    residual = np.asarray(data["residual"], dtype=float)
    values = np.log10(np.maximum(residual, 1e-12))
    vmax = float(np.nanpercentile(values, 99.5))
    vmin = float(np.nanmin(values))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(values))
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    im = ax.imshow(
        values,
        origin="lower",
        extent=[float(rs[0]), float(rs[-1]), float(zs[0]), float(zs[-1])],
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    levels = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    logs = [math.log10(x) for x in levels if np.nanmin(residual) <= x <= np.nanmax(residual)]
    if logs:
        ax.contour(rs, zs, values, levels=logs, colors="white", linewidths=0.55, alpha=0.75)
    ax.plot([data["grid_best_R"]], [data["grid_best_Z"]], marker="x", color="#00e5ff", ms=7, mew=1.5, label="grid best")
    if np.isfinite(data["axis_R"]) and np.isfinite(data["axis_Z"]):
        ax.plot([data["axis_R"]], [data["axis_Z"]], marker="+", color="#ff3b30", ms=10, mew=1.8, label="selected axis")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("One-period magnetic-axis closure residual")
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}$ closure residual [m]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi))
    plt.close(fig)
    return {
        "path": str(output_path),
        "grid": int(grid),
        "trace_time_s": data["trace_time_s"],
        "residual_min": data["residual_min"],
        "residual_p50": data["residual_p50"],
        "residual_p95": data["residual_p95"],
        "grid_best_R": data["grid_best_R"],
        "grid_best_Z": data["grid_best_Z"],
    }


def _signed_sqrt(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.sqrt(np.abs(values))


def export_psi_slices(
    model: PsiModel,
    output_path: Path,
    *,
    levels,
    grid: int,
    phi_count: int,
    dpi: int,
) -> dict[str, Any]:
    plt = _configure_matplotlib()
    grid = int(grid)
    phi_count = int(phi_count)
    period = TWOPI / model.nfp
    phis = np.linspace(0.0, period, phi_count, endpoint=False)
    offsets = np.linspace(-model.a, model.a, grid)
    xg, zg = np.meshgrid(offsets, offsets, indexing="xy")
    positive_levels = [float(x) for x in levels if float(x) > 0.0]
    clip_level = max(positive_levels) if positive_levels else 1.0
    vmax = math.sqrt(max(clip_level, 1e-14))
    ncols = min(5, phi_count)
    nrows = int(math.ceil(phi_count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.15 * ncols, 3.0 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    stats = []
    im = None
    for ax, phi in zip(axes, phis):
        ra, za, _, _ = model.axis_at(np.array([phi]))
        R = float(ra[0]) + xg
        Z = float(za[0]) + zg
        psi, *_ = psi_and_gradient(model, R.ravel(), Z.ravel(), np.full(R.size, phi))
        psi = psi.reshape(R.shape)
        shown = np.clip(_signed_sqrt(psi), -vmax, vmax)
        im = ax.imshow(
            shown,
            origin="lower",
            extent=[-model.a, model.a, -model.a, model.a],
            aspect="equal",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        contour_levels = [lev for lev in positive_levels if np.nanmin(psi) <= lev <= np.nanmax(psi)]
        if contour_levels:
            ax.contour(xg, zg, psi, levels=contour_levels, colors="#1f1f1f", linewidths=0.45, alpha=0.75)
        ax.plot([0.0], [0.0], "k+", ms=6, mew=1.0)
        ax.set_title(rf"$\Phi={phi / period:.2f}$ period")
        ax.set_xlabel("R - R_axis [m]")
        ax.set_ylabel("Z - Z_axis [m]")
        stats.append(
            {
                "phi": float(phi),
                "psi_min": float(np.nanmin(psi)),
                "psi_max": float(np.nanmax(psi)),
                "psi_abs_p95": float(np.nanpercentile(np.abs(psi), 95)),
            }
        )
    for ax in axes[len(phis) :]:
        ax.axis("off")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.92, pad=0.012)
        cbar.set_label(r"$\mathrm{sgn}(\psi)\sqrt{|\psi|}$")
    fig.suptitle(r"Local $\psi$ slices around the magnetic axis", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi))
    plt.close(fig)
    return {
        "path": str(output_path),
        "grid": int(grid),
        "phi_count": int(phi_count),
        "clip_level": float(clip_level),
        "scale": "signed_sqrt",
        "slice_stats": stats,
    }
