from __future__ import annotations

"""Exploratory Poincare plot using raw psi level curves instead of Boozer surfaces."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

TWOPI = 2.0 * np.pi


def _parse_levels(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _load_packed_case(path: Path):
    from stellarator_eval.field import input_from_packed_vector

    payload = json.loads(path.read_text(encoding="utf-8"))
    inp = payload["input"]
    coeff_count = int(inp.get("coeff_count", 33))
    field_input = input_from_packed_vector(inp["packed_values"], coeff_count=coeff_count)
    field_input.name = str(payload.get("sample_key") or path.stem)
    return field_input


def _rebuild_model(summary_path: Path, field, field_input, current_unit: str):
    from stellarator_eval.axis import trace_axis
    from stellarator_eval.psi import PolyMode, PsiModel

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    nfp = int(summary["nfp"])
    axis = summary["axis"]
    axis_steps = int(summary.get("config", {}).get("axis", {}).get("axis_trace_steps", 240))
    phi, R, Z, R_phi, Z_phi = trace_axis(field, float(axis["best_R"]), float(axis["best_Z"]), nfp, axis_steps)
    psi = summary["psi"]
    modes = [PolyMode(int(m["a"]), int(m["b"]), int(m["m"]), str(m["kind"])) for m in psi["modes"]]
    return PsiModel(
        coeffs=np.asarray(psi["coeffs"], dtype=float),
        modes=modes,
        nfp=nfp,
        a=float(psi["a"]),
        phi_axis=phi,
        R_axis=R,
        Z_axis=Z,
        R_axis_phi=R_phi,
        Z_axis_phi=Z_phi,
        fit_info=dict(psi.get("fit_info", {})),
    )


def _psi_level_curve(model, psi_level: float, phi: float, ntheta: int, max_radius_scale: float):
    from stellarator_eval.surface import _quadratic_radius
    from stellarator_eval.psi import psi_ray_value_and_derivative

    theta = np.linspace(0.0, TWOPI, int(ntheta), endpoint=False)
    max_radius = float(max_radius_scale) * model.a
    rho = _quadratic_radius(model, float(psi_level), theta, float(phi), max_radius)
    for _ in range(40):
        psi, dpsi = psi_ray_value_and_derivative(model, rho, theta, float(phi))
        f = psi - float(psi_level)
        if np.nanmax(np.abs(f)) <= 1e-12:
            break
        denom = np.where(np.abs(dpsi) > 1e-14, dpsi, np.where(dpsi >= 0.0, 1e-14, -1e-14))
        step = np.clip(f / denom, -0.45 * np.maximum(np.abs(rho), 1e-8 * model.a), 0.45 * np.maximum(np.abs(rho), 1e-8 * model.a))
        rho = np.clip(rho - step, 1e-12 * model.a, max_radius)
    ra, za, _, _ = model.axis_at(np.full_like(theta, float(phi)))
    R = ra + rho * np.cos(theta)
    Z = za + rho * np.sin(theta)
    return theta, R, Z, rho


def _psi_level_curve_scan(model, psi_level: float, phi: float, ntheta: int, max_radius_scale: float, nrho: int = 360):
    from stellarator_eval.psi import psi_ray_value_and_derivative

    theta = np.linspace(0.0, TWOPI, int(ntheta), endpoint=False)
    max_radius = float(max_radius_scale) * model.a
    rho_grid = np.linspace(0.0, max_radius, int(nrho))
    rr = np.repeat(rho_grid[:, None], theta.size, axis=1)
    tt = np.repeat(theta[None, :], rho_grid.size, axis=0)
    vals, _ = psi_ray_value_and_derivative(model, rr.ravel(), tt.ravel(), float(phi))
    f = vals.reshape(rho_grid.size, theta.size) - float(psi_level)
    crossing = (f[:-1] <= 0.0) & (f[1:] >= 0.0)
    any_crossing = np.any(crossing, axis=0)
    first = np.argmax(crossing, axis=0)
    rho = np.full_like(theta, np.nan, dtype=float)
    if np.any(any_crossing):
        idx = np.where(any_crossing)[0]
        lo = rho_grid[first[idx]].astype(float)
        hi = rho_grid[first[idx] + 1].astype(float)
        th = theta[idx]
        flo = psi_ray_value_and_derivative(model, lo, th, float(phi))[0] - float(psi_level)
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            fmid = psi_ray_value_and_derivative(model, mid, th, float(phi))[0] - float(psi_level)
            left = ((flo <= 0.0) & (fmid >= 0.0)) | ((flo >= 0.0) & (fmid <= 0.0))
            hi = np.where(left, mid, hi)
            lo_new = np.where(left, lo, mid)
            flo = np.where(left, flo, fmid)
            lo = lo_new
        rho[idx] = 0.5 * (lo + hi)
    ra, za, _, _ = model.axis_at(np.full_like(theta, float(phi)))
    R = ra + rho * np.cos(theta)
    Z = za + rho * np.sin(theta)
    return theta, R, Z, rho


def _make_normalized_field(field_input):
    from dataclasses import replace
    from stellarator_eval.field import build_field

    currents = np.asarray(field_input.currents, dtype=float)
    mean_abs = float(np.mean(np.abs(currents)))
    normalized = replace(field_input, currents=currents / mean_abs)
    return build_field(normalized, current_unit="A"), mean_abs


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _median_point_to_segment_distance(points: np.ndarray, segment: np.ndarray) -> float:
    if points.size == 0 or segment.size == 0:
        return float("inf")
    # This is small for the diagnostic grids used here, and avoids adding a
    # scipy dependency to this exploratory script.
    d2 = (points[:, None, 0] - segment[None, :, 0]) ** 2 + (points[:, None, 1] - segment[None, :, 1]) ** 2
    return float(np.median(np.sqrt(np.min(d2, axis=1))))


def _selected_psi_contours(plt, model, phi: float, levels: list[float], grid: int, radius_scale: float, ref_points_by_level=None):
    from matplotlib.path import Path as MplPath
    from stellarator_eval.psi import psi_and_gradient

    ra, za, _, _ = model.axis_at(np.array([phi]))
    axis_point = (float(ra[0]), float(za[0]))
    span = float(radius_scale) * model.a
    Rg = np.linspace(axis_point[0] - span, axis_point[0] + span, int(grid))
    Zg = np.linspace(axis_point[1] - span, axis_point[1] + span, int(grid))
    RR, ZZ = np.meshgrid(Rg, Zg, indexing="xy")
    psi_grid, *_ = psi_and_gradient(model, RR.ravel(), ZZ.ravel(), np.full(RR.size, phi))
    psi_grid = psi_grid.reshape(RR.shape)

    tmp_fig, tmp_ax = plt.subplots()
    try:
        cs = tmp_ax.contour(RR, ZZ, psi_grid, levels=levels)
        selected = []
        for level, segs in zip(cs.levels, cs.allsegs):
            usable = [np.asarray(seg, dtype=float) for seg in segs if len(seg) >= 8]
            if not usable:
                continue
            ref_points = None if ref_points_by_level is None else ref_points_by_level.get(float(level))
            if ref_points is not None and len(ref_points) >= 3:
                chosen = min(usable, key=lambda seg: _median_point_to_segment_distance(ref_points, seg))
            else:
                enclosing = [seg for seg in usable if MplPath(seg).contains_point(axis_point)]
                if enclosing:
                    # Use the smallest branch enclosing the axis; outer false branches
                    # can also enclose it on coarse grids.
                    chosen = min(enclosing, key=_polygon_area)
                else:
                    # Fallback for a slightly open contour: choose the branch closest to the axis.
                    chosen = min(usable, key=lambda seg: float(np.min(np.hypot(seg[:, 0] - axis_point[0], seg[:, 1] - axis_point[1]))))
            selected.append((float(level), chosen))
        return selected
    finally:
        plt.close(tmp_fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poincare diagnostic with raw psi level curves.")
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", default="0.04,0.08,0.12,0.16,0.18")
    parser.add_argument("--n-theta", type=int, default=192)
    parser.add_argument("--seed-levels", default=None)
    parser.add_argument("--seed-theta-stride", type=int, default=8)
    parser.add_argument("--max-radius-scale", type=float, default=1.0)
    parser.add_argument("--curve-method", choices=["scan", "contour", "ray"], default="scan")
    parser.add_argument("--contour-grid", type=int, default=260)
    parser.add_argument("--contour-radius-scale", type=float, default=1.65)
    parser.add_argument("--stop-fraction", type=float, default=1.5)
    parser.add_argument("--tmax-fl", type=float, default=2.0e8)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--dpi", type=int, default=170)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from simsopt.field import (
        MaxRStoppingCriterion,
        MaxZStoppingCriterion,
        MinRStoppingCriterion,
        MinZStoppingCriterion,
        compute_fieldlines,
    )

    field_input = _load_packed_case(args.case_file)
    built, current_mean = _make_normalized_field(field_input)
    model = _rebuild_model(args.summary, built.field, field_input, "A")
    levels = _parse_levels(args.levels)
    seed_levels = _parse_levels(args.seed_levels) if args.seed_levels else levels
    phis = [(i / 4.0) * (TWOPI / model.nfp) for i in range(4)]

    all_boundary_r = []
    all_boundary_z = []
    curves_by_phi: list[list[tuple[float, np.ndarray, np.ndarray]]] = []
    for phi in phis:
        curves = []
        for level in levels:
            if args.curve_method == "scan":
                _, R, Z, _ = _psi_level_curve_scan(model, level, phi, args.n_theta, args.max_radius_scale)
            else:
                _, R, Z, _ = _psi_level_curve(model, level, phi, args.n_theta, args.max_radius_scale)
            curves.append((level, R, Z))
            all_boundary_r.append(R[np.isfinite(R)])
            all_boundary_z.append(Z[np.isfinite(Z)])
        curves_by_phi.append(curves)

    all_r = np.concatenate(all_boundary_r)
    all_z = np.concatenate(all_boundary_z)
    r_center = 0.5 * (float(np.min(all_r)) + float(np.max(all_r)))
    z_center = 0.5 * (float(np.min(all_z)) + float(np.max(all_z)))
    r_half_width = 0.5 * (float(np.max(all_r)) - float(np.min(all_r))) * float(args.stop_fraction)
    z_half_width = 0.5 * (float(np.max(all_z)) - float(np.min(all_z))) * float(args.stop_fraction)
    stopping_criteria = [
        MinRStoppingCriterion(max(r_center - r_half_width, 0.0)),
        MaxRStoppingCriterion(r_center + r_half_width),
        MinZStoppingCriterion(z_center - z_half_width),
        MaxZStoppingCriterion(z_center + z_half_width),
    ]

    seed_R = []
    seed_Z = []
    seed_tags = []
    for level in seed_levels:
        theta, R, Z, _ = _psi_level_curve(model, level, 0.0, args.n_theta, args.max_radius_scale)
        sl = slice(0, len(theta), max(1, int(args.seed_theta_stride)))
        seed_R.extend(R[sl])
        seed_Z.extend(Z[sl])
        seed_tags.extend([float(level)] * len(R[sl]))
    seed_R = np.asarray(seed_R, dtype=float)
    seed_Z = np.asarray(seed_Z, dtype=float)

    print(f"Current normalization mean|I|={current_mean:.6g}")
    print(f"Tracing {len(seed_R)} field lines from psi levels {seed_levels}")
    fieldlines_phi_hits = []
    for i, (r0, z0) in enumerate(zip(seed_R, seed_Z), start=1):
        _, hits_one = compute_fieldlines(
            built.field,
            [float(r0)],
            [float(z0)],
            tmax=float(args.tmax_fl),
            tol=float(args.tol),
            phis=phis,
            stopping_criteria=stopping_criteria,
        )
        fieldlines_phi_hits.append(np.asarray(hits_one[0]))
        if i % 20 == 0 or i == len(seed_R):
            print(f"  traced {i}/{len(seed_R)}")
    hit_counts = [
        0 if hits.ndim != 2 else int(np.sum(hits[:, 1] >= 0))
        for hits in fieldlines_phi_hits
    ]
    print("Poincare hits per line =", hit_counts)

    cmap = plt.get_cmap("viridis")
    level_to_color = {level: cmap(i / max(1, len(levels) - 1)) for i, level in enumerate(levels)}
    nrowcol = int(np.ceil(np.sqrt(len(phis))))
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(9, 6))
    axs = np.asarray(axs).reshape((nrowcol, nrowcol))
    for i, phi in enumerate(phis):
        ax = axs[i // nrowcol, i % nrowcol]
        ax.set_aspect("equal")
        ax.set_title(f"$\\phi = {phi / np.pi:.2f}\\pi$", loc="left")
        if i // nrowcol == nrowcol - 1:
            ax.set_xlabel("$R$")
        if i % nrowcol == 0:
            ax.set_ylabel("$Z$")
        ax.grid(True, linewidth=0.5, alpha=0.8)
        ref_points_by_level = {float(level): [] for level in levels}
        for hits, tag in zip(fieldlines_phi_hits, seed_tags):
            if hits.ndim != 2 or hits.shape[0] == 0:
                continue
            data = hits[np.where(hits[:, 1] == i)[0], :]
            if data.size == 0:
                continue
            r = np.sqrt(data[:, 2] ** 2 + data[:, 3] ** 2)
            ref_points_by_level.setdefault(float(tag), []).append(np.column_stack([r, data[:, 4]]))
            ax.scatter(r, data[:, 4], s=5, linewidths=0, color=level_to_color.get(tag, "0.3"), alpha=0.75)
        ref_points_by_level = {
            level: np.vstack(parts) for level, parts in ref_points_by_level.items() if parts
        }
        if args.curve_method == "contour":
            selected_contours = _selected_psi_contours(
                plt,
                model,
                phi,
                levels,
                grid=args.contour_grid,
                radius_scale=args.contour_radius_scale,
                ref_points_by_level=ref_points_by_level,
            )
            for _, seg in selected_contours:
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    linewidth=0.85,
                    color="black",
                    alpha=0.94,
                    path_effects=[pe.Stroke(linewidth=1.55, foreground="white", alpha=0.85), pe.Normal()],
                )
        else:
            for level, R, Z in curves_by_phi[i]:
                ax.plot(
                    R,
                    Z,
                    linewidth=0.85,
                    color="black",
                    linestyle="-",
                    alpha=0.92,
                    path_effects=[pe.Stroke(linewidth=1.55, foreground="white", alpha=0.85), pe.Normal()],
                )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=level_to_color[level], markersize=5, label=f"{level:g}")
        for level in levels
    ]
    handles.append(Line2D([0], [0], color="black", linewidth=0.9, label="$\\psi$ contour"))
    fig.legend(handles=handles, title="$\\psi$ level", loc="center right", frameon=False)
    for i in range(len(phis), nrowcol * nrowcol):
        axs[i // nrowcol, i % nrowcol].axis("off")
    fig.suptitle("cem_3 Poincare from raw $\\psi$ level curves (no Boozer optimization)", y=0.98)
    plt.tight_layout(rect=[0.0, 0.0, 0.90, 0.95])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=int(args.dpi))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
