from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stellarator_eval import EvalConfig
from stellarator_eval.axis import find_axis_gpu
from stellarator_eval.field import build_field
from stellarator_eval.psi import PolyMode, PsiModel, psi_and_gradient
from stellarator_eval.quasr import build_quasr_metadata_index, load_quasr_field_input, load_quasr_metadata
from stellarator_eval.serialization import write_json


REMOTE_QUASR_ROOT = Path("/data/zhouyebi/QUASR_08072024")
REMOTE_PRIVATE_META = Path("/home/cyfan/stellarator_gpu_eval/quasr_private/QUASR_08072024_meta.csv")


def configure_plot_fonts() -> bool:
    try:
        import matplotlib
        from matplotlib import font_manager
    except Exception:
        return False
    candidates = (
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "PingFang SC",
        "Sarasa Gothic SC",
        "Arial Unicode MS",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans", "sans-serif"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return False


def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]


def load_batch_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_psi_model_from_npz(path: Path) -> PsiModel:
    data = np.load(path, allow_pickle=True)
    modes = [
        PolyMode(int(a), int(b), int(m), str(kind))
        for a, b, m, kind in zip(data["mode_a"], data["mode_b"], data["mode_m"], data["mode_kind"])
    ]
    fit_info = {}
    for key in data.files:
        if not key.startswith("info_"):
            continue
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


def batch_label(path: Path) -> str:
    return path.resolve().parent.name


def sample_phi_grid(nfp: int, count: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi / nfp, count, endpoint=False)


def axis_sweep_for_id(device_id: int, quasr_root: Path, generations: list[int]) -> list[dict]:
    field_input, info = load_quasr_field_input(quasr_root, device_id)
    cfg = EvalConfig()
    cfg.current_unit = "A"
    built = build_field(field_input, cfg.current_unit)
    rows = []
    for max_gen in generations:
        cfg_local = EvalConfig(
            axis=cfg.axis,
            psi=cfg.psi,
            scan=cfg.scan,
            boozer=cfg.boozer,
            current_unit=cfg.current_unit,
            omp_threads=cfg.omp_threads,
        )
        cfg_local.axis.backend = "gpu"
        cfg_local.axis.max_generations = int(max_gen)
        t0 = time.perf_counter()
        axis = find_axis_gpu(field_input, built.field, built.nfp, built.coil_r0, cfg_local.axis, cfg_local.current_unit)
        rows.append(
            {
                "device_id": int(device_id),
                "max_generations": int(max_gen),
                "has_axis": bool(axis.has_axis),
                "best_residual": float(axis.best_residual),
                "search_best_residual": float(axis.search_best_residual),
                "generation": int(axis.generation),
                "failure_reason": axis.failure_reason,
                "trace_error": axis.trace_error,
                "time_s": float(time.perf_counter() - t0),
                "history_last": axis.history[-1] if axis.history else None,
                "nfp": info["nfp"],
                "nc_per_hp": info["nc_per_hp"],
            }
        )
    return rows


def _signed_sqrt(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.sqrt(np.abs(values))


def _format_psi_value(value: float) -> str:
    if abs(value) < 5e-7:
        return "0"
    if abs(value) >= 0.1:
        return f"{value:.2f}"
    if abs(value) >= 0.01:
        return f"{value:.3f}"
    return f"{value:.4g}"


def _select_colorbar_ticks(vmin: float, vmax: float, screen_levels: list[float]) -> list[float]:
    positive = [lev for lev in screen_levels if 0.0 < lev <= vmax]
    if len(positive) > 6:
        step = int(np.ceil(len(positive) / 6))
        positive = positive[::step]
        if positive[-1] != max(screen_levels):
            positive.append(max(screen_levels))
    negative_refs = [-0.001, -0.004, -0.016, -0.064, -0.16, -0.64, -2.56, -10.24]
    negative = [val for val in negative_refs if vmin <= val < 0.0]
    if vmin < 0.0 and (not negative or negative[0] != vmin):
        negative = [vmin] + negative
    ticks = sorted(set(negative + [0.0] + positive))
    return ticks


def classify_axis_quadratic(model: PsiModel, *, phi_count: int, eps_fraction: float = 2e-3) -> dict:
    phis = sample_phi_grid(model.nfp, phi_count)
    eps = max(model.a * eps_fraction, 1e-6)
    dets = []
    eigmins = []
    eigmaxs = []
    dxxs = []
    dzzs = []
    dxzs = []
    for phi in phis:
        phi_arr = np.full(1, float(phi))
        ra, za, _, _ = model.axis_at(phi_arr)
        r0 = float(ra[0])
        z0 = float(za[0])

        def psi_at(dr: float, dz: float) -> float:
            psi, *_ = psi_and_gradient(model, np.array([r0 + dr]), np.array([z0 + dz]), phi_arr)
            return float(psi[0])

        p00 = psi_at(0.0, 0.0)
        ppx = psi_at(+eps, 0.0)
        pmx = psi_at(-eps, 0.0)
        ppz = psi_at(0.0, +eps)
        pmz = psi_at(0.0, -eps)
        ppp = psi_at(+eps, +eps)
        ppm = psi_at(+eps, -eps)
        pmp = psi_at(-eps, +eps)
        pmm = psi_at(-eps, -eps)

        dxx = (ppx - 2.0 * p00 + pmx) / (eps * eps)
        dzz = (ppz - 2.0 * p00 + pmz) / (eps * eps)
        dxz = (ppp - ppm - pmp + pmm) / (4.0 * eps * eps)
        tr = dxx + dzz
        det = dxx * dzz - dxz * dxz
        disc = max(tr * tr - 4.0 * det, 0.0)
        disc = float(np.sqrt(disc))
        eigmin = 0.5 * (tr - disc)
        eigmax = 0.5 * (tr + disc)

        dxxs.append(dxx)
        dzzs.append(dzz)
        dxzs.append(dxz)
        dets.append(det)
        eigmins.append(eigmin)
        eigmaxs.append(eigmax)

    dets = np.asarray(dets)
    eigmins = np.asarray(eigmins)
    eigmaxs = np.asarray(eigmaxs)
    tol = 1e-8 / max(model.a * model.a, 1e-12)
    if np.min(dets) < -tol:
        quad_class = "saddle"
    elif np.min(eigmins) > tol:
        quad_class = "elliptic"
    else:
        quad_class = "degenerate"
    return {
        "class": quad_class,
        "eps": float(eps),
        "det_min": float(np.min(dets)),
        "det_max": float(np.max(dets)),
        "eigmin_min": float(np.min(eigmins)),
        "eigmax_max": float(np.max(eigmaxs)),
        "dxx_mean": float(np.mean(dxxs)),
        "dzz_mean": float(np.mean(dzzs)),
        "dxz_abs_max": float(np.max(np.abs(dxzs))),
    }


def draw_heatmap_for_run(
    run_dir: Path,
    output_path: Path,
    *,
    phi_count: int,
    grid: int,
    heatmap_clip: float,
    title_prefix: str,
    scale: str,
) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    use_zh = configure_plot_fonts()
    model = load_psi_model_from_npz(run_dir / "psi_model.npz")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    screen_levels = [float(x["psi_level"]) for x in summary.get("surface_screen", {}).get("levels", [])]
    if not screen_levels:
        screen_levels = [0.001, 0.002, 0.004, 0.008, 0.012, 0.02, 0.04, 0.08, 0.12, 0.16]
    vmax = max(screen_levels) if heatmap_clip <= 0 else float(heatmap_clip)
    phis = sample_phi_grid(model.nfp, phi_count)
    xs = np.linspace(-model.a, model.a, grid)
    zs = np.linspace(-model.a, model.a, grid)
    X, Z = np.meshgrid(xs, zs, indexing="xy")

    psi_slices = []
    slice_mins = []
    slice_maxs = []
    for phi in phis:
        ra, za, _, _ = model.axis_at(np.array([phi]))
        rr = ra[0] + X
        zz = za[0] + Z
        psi, *_ = psi_and_gradient(model, rr.ravel(), zz.ravel(), np.full(rr.size, phi))
        psi = psi.reshape(rr.shape)
        psi_slices.append(psi)
        slice_mins.append(float(np.min(psi)))
        slice_maxs.append(float(np.max(psi)))
    vmin = min(slice_mins)

    ncols = 6 if len(phis) >= 15 else 4
    nrows = int(np.ceil(len(phis) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    if scale == "sqrt":
        shown_vmin = float(_signed_sqrt(vmin))
        shown_vmax = float(_signed_sqrt(vmax))
        transform = _signed_sqrt
        color_label = r"$\mathrm{sgn}(\psi)\sqrt{|\psi|}$"
    else:
        shown_vmin = float(vmin)
        shown_vmax = float(vmax)
        transform = lambda x: np.asarray(x, dtype=float)
        color_label = r"$\psi$"
    if shown_vmin < 0.0 < shown_vmax:
        norm = TwoSlopeNorm(vmin=shown_vmin, vcenter=0.0, vmax=shown_vmax)
    else:
        norm = Normalize(vmin=shown_vmin, vmax=shown_vmax)

    im = None
    for ax, phi, psi in zip(axes, phis, psi_slices):
        shown = transform(np.clip(psi, vmin, vmax))
        im = ax.imshow(
            shown,
            origin="lower",
            extent=[xs[0], xs[-1], zs[0], zs[-1]],
            aspect="equal",
            cmap="RdBu_r",
            norm=norm,
        )
        contour_levels = [lev for lev in screen_levels if lev <= vmax]
        if contour_levels:
            ax.contour(X, Z, psi, levels=contour_levels, colors="#222222", linewidths=0.65, alpha=0.9)
        deg = float(phi * 180.0 / np.pi)
        ax.set_title(f"$\\Phi={deg:.1f}^\\circ$", fontsize=11)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
    for ax in axes[len(phis) :]:
        ax.axis("off")
    if im is not None:
        tick_values = _select_colorbar_ticks(vmin, vmax, screen_levels)
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.92, pad=0.012)
        cbar.set_ticks(transform(tick_values))
        cbar.set_ticklabels([_format_psi_value(v) for v in tick_values])
        cbar.set_label(color_label)
    if use_zh:
        if scale == "sqrt":
            title = f"{title_prefix} 局部 x-z 截面 $\\psi$ 热力图（颜色按 $\\mathrm{{sgn}}(\\psi)\\sqrt{{|\\psi|}}$ 缩放）"
        else:
            title = f"{title_prefix} 局部 x-z 截面 $\\psi$ 热力图"
    else:
        if scale == "sqrt":
            title = f"{title_prefix} local x-z $\\psi$ slices (signed-sqrt color scale)"
        else:
            title = f"{title_prefix} local x-z $\\psi$ slices"
    fig.suptitle(title, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "phi_count": int(phi_count),
        "grid": int(grid),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "screen_levels": screen_levels,
        "slice_min": float(min(slice_mins)),
        "slice_max": float(max(slice_maxs)),
        "scale": scale,
    }


def collect_runs(batch_paths: list[Path]) -> tuple[list[dict], list[dict]]:
    batches = []
    runs = []
    for path in batch_paths:
        batch = load_batch_summary(path)
        label = batch_label(path)
        batches.append(
            {
                "label": label,
                "path": str(path),
                "status_counts": batch.get("status_counts", {}),
                "success_rate": batch.get("success_rate"),
                "sample_size": len(batch.get("rows", [])),
            }
        )
        runs_root = path.resolve().parent
        for run in batch["runs"]:
            item = dict(run)
            item["_batch_label"] = label
            item["_runs_root"] = str(runs_root)
            runs.append(item)
    return batches, runs


def build_surface_row(run: dict, meta_index: dict[int, dict], quad_phi_count: int) -> dict:
    device_id = int(run["device_id"])
    run_dir = Path(run["_runs_root"]) / f"id_{device_id:07d}"
    model = load_psi_model_from_npz(run_dir / "psi_model.npz")
    quad = classify_axis_quadratic(model, phi_count=quad_phi_count)
    fit = run["result"]["psi"]["fit_info"]
    levels = run["result"].get("surface_screen", {}).get("levels", [])
    ok_levels = [float(item["psi_level"]) for item in levels if item.get("ok")]
    stage = "boozer_failed" if ok_levels else "screen_failed"
    return {
        "device_id": device_id,
        "batch_label": run["_batch_label"],
        "status": run["status"],
        "train_rms": float(fit["train_rms"]),
        "validation_rms": float(fit["validation_rms"]),
        "validation_angle_mean": float(fit["validation_angle_mean"]),
        "validation_angle_p95": float(fit["validation_angle_p95"]),
        "validation_angle_l2": float(fit["validation_angle_l2"]),
        "warnings": run["result"]["warnings"],
        "ok_level_count": len(ok_levels),
        "ok_levels": ok_levels,
        "surface_stage": stage,
        "quadratic": quad,
        "metadata": meta_index.get(device_id),
        "run_dir": str(run_dir),
        "boozer_time_s": float(run["result"].get("timing", {}).get("boozer_candidates_s", 0.0)),
    }


def write_report(path: Path, batches: list[dict], axis_rows: list[dict], surface_rows: list[dict], heatmaps: list[dict]) -> None:
    lines = ["# QUASR 失败样本诊断", ""]
    lines.append("## 1. 数据来源")
    lines.append("")
    for batch in batches:
        lines.append(
            f"- `{batch['label']}`: 样本数 `{batch['sample_size']}`，"
            f"成功率 `{batch['success_rate']:.6g}`，状态统计 `{batch['status_counts']}`"
        )
    lines.append("")

    lines.append("## 2. no_axis：GA 代数扫描")
    lines.append("")
    if axis_rows:
        lines.append("| ID | batch | max_generations | best_residual | search_best_residual | failure_reason | time [s] |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- | ---: |")
        for row in axis_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["device_id"]),
                        str(row.get("batch_label", "")),
                        str(row["max_generations"]),
                        f"{row['best_residual']:.6g}",
                        f"{row['search_best_residual']:.6g}",
                        str(row["failure_reason"]),
                        f"{row['time_s']:.3f}",
                    ]
                )
                + " |"
            )
    else:
        lines.append("- 无 `no_axis` 样本。")
    lines.append("")

    lines.append("## 3. no_surface：拟合与局部二次型")
    lines.append("")
    if surface_rows:
        lines.append("| ID | batch | stage | quad_class | angle_mean | angle_p95 | ok_level_count | boozer_time [s] |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
        for row in surface_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["device_id"]),
                        str(row["batch_label"]),
                        str(row["surface_stage"]),
                        str(row["quadratic"]["class"]),
                        f"{row['validation_angle_mean']:.6g}",
                        f"{row['validation_angle_p95']:.6g}",
                        str(row["ok_level_count"]),
                        f"{row['boozer_time_s']:.3f}",
                    ]
                )
                + " |"
            )
        lines.append("")
        stage_counts = {}
        quad_counts = {}
        for row in surface_rows:
            stage_counts[row["surface_stage"]] = stage_counts.get(row["surface_stage"], 0) + 1
            quad_class = row["quadratic"]["class"]
            quad_counts[quad_class] = quad_counts.get(quad_class, 0) + 1
        lines.append(f"- `surface_stage` 统计：`{stage_counts}`")
        lines.append(f"- `quadratic.class` 统计：`{quad_counts}`")
    else:
        lines.append("- 无 `no_surface` 样本。")
    lines.append("")

    lines.append("## 4. 热力图")
    lines.append("")
    if heatmaps:
        for item in heatmaps:
            lines.append(f"### ID {item['device_id']} ({item['batch_label']})")
            lines.append("")
            lines.append(f"![ID {item['device_id']}]({item['filename']})")
            lines.append("")
            lines.append(
                f"- `stage={item['surface_stage']}`，`quad_class={item['quadratic_class']}`，"
                f"`validation_angle_mean={item['validation_angle_mean']:.6g}`，"
                f"`validation_angle_p95={item['validation_angle_p95']:.6g}`"
            )
            lines.append(
                f"- `ok_level_count={item['ok_level_count']}`，"
                f"`vmin={item['plot']['vmin']:.6g}`，`vmax={item['plot']['vmax']:.6g}`，"
                f"`scale={item['plot']['scale']}`"
            )
            lines.append("")
    else:
        lines.append("- 本次未生成热力图。")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose QUASR failure cases from one or more batch runs.")
    parser.add_argument("--batch-summary", type=Path, action="append", required=True, help="Path to batch_summary.json. Repeatable.")
    parser.add_argument("--quasr-root", type=Path, default=REMOTE_QUASR_ROOT)
    parser.add_argument("--metadata", type=Path, default=REMOTE_PRIVATE_META)
    parser.add_argument("--axis-generations", default="32,64,96,128,192")
    parser.add_argument("--quad-phi-count", type=int, default=36)
    parser.add_argument("--heatmap-phi-count", type=int, default=18)
    parser.add_argument("--heatmap-grid", type=int, default=241)
    parser.add_argument("--heatmap-clip", type=float, default=0.16)
    parser.add_argument("--heatmap-scale", choices=("sqrt", "linear"), default="sqrt")
    parser.add_argument("--heatmap-limit", type=int, default=0, help="0 means all no_surface samples.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/quasr_failure_diagnose"))
    args = parser.parse_args()

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")

    meta_rows = load_quasr_metadata(args.metadata)
    meta_index = build_quasr_metadata_index(meta_rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    batches, runs = collect_runs(args.batch_summary)
    generations = parse_int_list(args.axis_generations)

    no_axis_runs = [run for run in runs if run["status"] == "no_axis"]
    no_axis_map: dict[int, list[str]] = {}
    for run in no_axis_runs:
        no_axis_map.setdefault(int(run["device_id"]), []).append(run["_batch_label"])
    axis_rows: list[dict] = []
    for device_id in sorted(no_axis_map):
        print(f"[axis sweep] ID={device_id}")
        rows = axis_sweep_for_id(device_id, args.quasr_root, generations)
        for row in rows:
            row["batch_label"] = ",".join(no_axis_map[device_id])
        axis_rows.extend(rows)

    no_surface_runs = [run for run in runs if run["status"] == "no_surface"]
    surface_rows = [build_surface_row(run, meta_index, args.quad_phi_count) for run in no_surface_runs]
    surface_rows.sort(key=lambda row: (row["validation_angle_mean"], row["device_id"]))

    heatmap_targets = surface_rows
    if args.heatmap_limit > 0:
        heatmap_targets = surface_rows[: args.heatmap_limit]
    heatmaps: list[dict] = []
    for row in heatmap_targets:
        device_id = int(row["device_id"])
        plot_path = output_dir / f"{row['batch_label']}__id_{device_id:07d}_phi_slices.png"
        print(f"[heatmap] ID={device_id}")
        plot_info = draw_heatmap_for_run(
            Path(row["run_dir"]),
            plot_path,
            phi_count=args.heatmap_phi_count,
            grid=args.heatmap_grid,
            heatmap_clip=args.heatmap_clip,
            title_prefix=f"ID {device_id}",
            scale=args.heatmap_scale,
        )
        heatmaps.append(
            {
                "device_id": device_id,
                "batch_label": row["batch_label"],
                "filename": plot_path.name,
                "plot": plot_info,
                "surface_stage": row["surface_stage"],
                "quadratic_class": row["quadratic"]["class"],
                "validation_angle_mean": row["validation_angle_mean"],
                "validation_angle_p95": row["validation_angle_p95"],
                "ok_level_count": row["ok_level_count"],
            }
        )

    payload = {
        "batch_summaries": [str(path) for path in args.batch_summary],
        "quasr_root": str(args.quasr_root),
        "metadata": str(args.metadata),
        "axis_generations": generations,
        "batches": batches,
        "axis_rows": axis_rows,
        "surface_rows": surface_rows,
        "heatmaps": heatmaps,
    }
    write_json(output_dir / "diagnose_summary.json", payload)
    write_report(output_dir / "report.md", batches, axis_rows, surface_rows, heatmaps)


if __name__ == "__main__":
    main()
