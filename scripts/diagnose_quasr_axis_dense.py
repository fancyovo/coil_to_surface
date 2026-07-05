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

from stellarator_eval.config import AxisGAConfig
from stellarator_eval.field import build_field
from stellarator_eval.quasr import load_quasr_field_input


REMOTE_QUASR_ROOT = Path("/data/zhouyebi/QUASR_08072024")


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


def load_batch_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def batch_label(path: Path) -> str:
    return path.resolve().parent.name


def _axis_cfg_from_run(run: dict) -> AxisGAConfig:
    src = run["result"]["config"]["axis"]
    allowed = AxisGAConfig.__dataclass_fields__.keys()
    values = {key: src[key] for key in allowed if key in src}
    return AxisGAConfig(**values)


def collect_no_axis_runs(batch_paths: list[Path]) -> dict[int, dict]:
    collected: dict[int, dict] = {}
    for path in batch_paths:
        batch = load_batch_summary(path)
        label = batch_label(path)
        for run in batch["runs"]:
            if run["status"] != "no_axis":
                continue
            device_id = int(run["device_id"])
            item = dict(run)
            item["_batch_label"] = label
            item["_batch_summary"] = str(path)
            if device_id not in collected:
                collected[device_id] = item
            else:
                collected[device_id]["_batch_label"] += f",{label}"
    return collected


def _make_gpu_field(field_input, nfp: int, cfg: AxisGAConfig, current_unit: str):
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
        raise ValueError(f"unknown current_unit={current_unit!r}")

    lib_path = Path(cfg.gpu_lib_path)
    if not lib_path.is_absolute():
        lib_path = Path.cwd() / lib_path
    return CoilFieldGpu(
        lib_path,
        field_input.coeffs_x,
        field_input.coeffs_y,
        field_input.coeffs_z,
        currents,
        nfp=nfp,
        segments_per_coil=cfg.gpu_segments_per_coil,
        device_id=cfg.gpu_device,
    )


def dense_axis_scan(
    run: dict,
    quasr_root: Path,
    *,
    grid: int,
    precision: str,
    verify_precision: str,
    verify_candidates: int,
) -> dict:
    device_id = int(run["device_id"])
    field_input, info = load_quasr_field_input(quasr_root, device_id)
    current_unit = run["result"]["config"].get("current_unit", "A")
    axis_cfg = _axis_cfg_from_run(run)
    built = build_field(field_input, current_unit)
    r_center = float(built.coil_r0)
    z_center = float(axis_cfg.z_center)
    span = float(axis_cfg.span)
    ga_axis = run["result"]["axis"]
    history = ga_axis.get("history", [])
    path_r = np.array([float(item["best_R"]) for item in history], dtype=float)
    path_z = np.array([float(item["best_Z"]) for item in history], dtype=float)
    ga_best_r = float(ga_axis["best_R"])
    ga_best_z = float(ga_axis["best_Z"])
    domain_pad = max(0.04 * span, 0.01)
    r_min = min(r_center - span, ga_best_r - domain_pad, float(np.min(path_r)) - domain_pad if path_r.size else ga_best_r - domain_pad)
    r_max = max(r_center + span, ga_best_r + domain_pad, float(np.max(path_r)) + domain_pad if path_r.size else ga_best_r + domain_pad)
    z_min = min(z_center - span, ga_best_z - domain_pad, float(np.min(path_z)) - domain_pad if path_z.size else ga_best_z - domain_pad)
    z_max = max(z_center + span, ga_best_z + domain_pad, float(np.max(path_z)) + domain_pad if path_z.size else ga_best_z + domain_pad)

    rs = np.linspace(r_min, r_max, grid)
    zs = np.linspace(z_min, z_max, grid)
    rg, zg = np.meshgrid(rs, zs, indexing="xy")
    r0 = np.ascontiguousarray(rg.ravel(), dtype=float)
    z0 = np.ascontiguousarray(zg.ravel(), dtype=float)

    t_create = time.perf_counter()
    gpu_field = _make_gpu_field(field_input, int(info["nfp"]), axis_cfg, current_unit)
    create_s = time.perf_counter() - t_create
    try:
        t_scan = time.perf_counter()
        r1, z1 = gpu_field.trace_period_blockline_precision(
            r0,
            z0,
            steps=int(axis_cfg.rk4_steps),
            precision=precision,
            threads_per_line=int(axis_cfg.gpu_threads_per_line),
            nfp=int(info["nfp"]),
        )
        scan_s = time.perf_counter() - t_scan
        residual = np.hypot(r1 - r0, z1 - z0)
        order = np.argsort(residual)
        top = order[: min(int(verify_candidates), residual.size)]
        t_verify = time.perf_counter()
        rv, zv = gpu_field.trace_period_blockline_precision(
            r0[top],
            z0[top],
            steps=int(axis_cfg.rk4_steps),
            precision=verify_precision,
            threads_per_line=int(axis_cfg.gpu_threads_per_line),
            nfp=int(info["nfp"]),
        )
        verify_s = time.perf_counter() - t_verify
    finally:
        gpu_field.close()

    verify_residual = np.hypot(rv - r0[top], zv - z0[top])
    best_local = int(np.argmin(verify_residual))
    best_idx = int(top[best_local])

    residual_grid = residual.reshape(grid, grid)
    ga_best_resid = float(ga_axis["best_residual"])
    dense_best_r = float(r0[best_idx])
    dense_best_z = float(z0[best_idx])
    dense_best_mixed = float(residual[best_idx])
    dense_best_verify = float(verify_residual[best_local])
    grid_step_r = float(rs[1] - rs[0]) if grid > 1 else 0.0
    grid_step_z = float(zs[1] - zs[0]) if grid > 1 else 0.0
    dense_vs_ga_distance = float(np.hypot(dense_best_r - ga_best_r, dense_best_z - ga_best_z))

    tol = float(axis_cfg.tol)
    same_scale = max(grid_step_r, grid_step_z, 1e-15)
    if dense_best_verify <= tol:
        interpretation = "dense_grid_contains_subtol_point"
    elif ga_best_resid <= max(10.0 * tol, 1e-6) and dense_vs_ga_distance <= 4.0 * same_scale:
        interpretation = "ga_found_subgrid_narrow_minimum"
    elif dense_vs_ga_distance <= 2.5 * same_scale and 0.5 * ga_best_resid <= dense_best_verify <= 1.25 * ga_best_resid:
        interpretation = "consistent_with_ga_minimum"
    elif dense_best_verify < 0.5 * ga_best_resid and dense_vs_ga_distance > 4.0 * same_scale:
        interpretation = "ga_missed_better_basin"
    else:
        interpretation = "better_region_exists_but_not_axis"

    return {
        "device_id": device_id,
        "batch_label": run["_batch_label"],
        "batch_summary": run["_batch_summary"],
        "nfp": int(info["nfp"]),
        "nc_per_hp": int(info["nc_per_hp"]),
        "current_unit": current_unit,
        "center_R": r_center,
        "center_Z": z_center,
        "span": span,
        "r_min": float(r_min),
        "r_max": float(r_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "grid": int(grid),
        "precision": precision,
        "verify_precision": verify_precision,
        "verify_candidates": int(len(top)),
        "rk4_steps": int(axis_cfg.rk4_steps),
        "ga_best_R": ga_best_r,
        "ga_best_Z": ga_best_z,
        "ga_best_residual": ga_best_resid,
        "ga_search_best_residual": float(ga_axis.get("search_best_residual", ga_best_resid)),
        "dense_best_R": dense_best_r,
        "dense_best_Z": dense_best_z,
        "dense_best_residual_mixed": dense_best_mixed,
        "dense_best_residual_verify": dense_best_verify,
        "dense_vs_ga_distance": dense_vs_ga_distance,
        "grid_step_R": grid_step_r,
        "grid_step_Z": grid_step_z,
        "interpretation": interpretation,
        "timing": {
            "gpu_create_s": float(create_s),
            "scan_s": float(scan_s),
            "verify_s": float(verify_s),
            "total_s": float(create_s + scan_s + verify_s),
        },
        "history_path_R": path_r.tolist(),
        "history_path_Z": path_z.tolist(),
        "residual_grid": residual_grid.tolist(),
        "rs": rs.tolist(),
        "zs": zs.tolist(),
    }


def draw_heatmap(scan: dict, output_path: Path, *, floor: float) -> None:
    import matplotlib.pyplot as plt

    configure_plot_fonts()
    rs = np.asarray(scan["rs"], dtype=float)
    zs = np.asarray(scan["zs"], dtype=float)
    residual = np.asarray(scan["residual_grid"], dtype=float)
    log_resid = np.log10(np.maximum(residual, floor))
    vmin = float(np.min(log_resid))
    vmax = float(np.percentile(log_resid, 99.5))
    if vmax <= vmin:
        vmax = float(np.max(log_resid))
    ga_r = float(scan["ga_best_R"])
    ga_z = float(scan["ga_best_Z"])
    dense_r = float(scan["dense_best_R"])
    dense_z = float(scan["dense_best_Z"])
    path_r = np.asarray(scan["history_path_R"], dtype=float)
    path_z = np.asarray(scan["history_path_Z"], dtype=float)
    span = float(scan["span"])
    zoom_span = max(8.0 * max(float(scan["grid_step_R"]), float(scan["grid_step_Z"])), 0.08 * span)
    levels = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    level_logs = [np.log10(x) for x in levels if floor <= x <= np.max(residual)]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), constrained_layout=True)
    extent = [rs[0], rs[-1], zs[0], zs[-1]]
    for ax in axes:
        im = ax.imshow(
            log_resid,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        if level_logs:
            ax.contour(
                rs,
                zs,
                log_resid,
                levels=level_logs,
                colors="white",
                linewidths=0.55,
                alpha=0.75,
            )
        if len(path_r) >= 2:
            ax.plot(path_r, path_z, color="#ff6b6b", linewidth=1.2, alpha=0.9, label="GA best path")
        if len(path_r) == 1:
            ax.plot(path_r, path_z, "o", color="#ff6b6b", markersize=3.0, alpha=0.9, label="GA best path")
        ax.scatter([ga_r], [ga_z], s=54, facecolors="none", edgecolors="#ff3b30", linewidths=1.6, label="GA final")
        ax.scatter([dense_r], [dense_z], s=60, marker="x", color="#00e5ff", linewidths=1.8, label="Dense best")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
    axes[0].set_title("Global 128x128 residual heatmap")
    axes[1].set_title("Local zoom near dense minimum")
    axes[1].set_xlim(dense_r - zoom_span, dense_r + zoom_span)
    axes[1].set_ylim(dense_z - zoom_span, dense_z + zoom_span)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper right", fontsize=8)
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.93, pad=0.02)
    tick_values = [x for x in levels if floor <= x <= np.max(residual)]
    if tick_values:
        cbar.set_ticks([np.log10(x) for x in tick_values])
        cbar.set_ticklabels([f"{x:.0e}" if x < 0.1 else f"{x:g}" for x in tick_values])
    cbar.set_label("closure residual after one field period [m]")
    title = (
        f"ID {scan['device_id']} | GA={scan['ga_best_residual']:.3e}, "
        f"dense(best, {scan['verify_precision']})={scan['dense_best_residual_verify']:.3e}, "
        f"interpretation={scan['interpretation']}"
    )
    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(path: Path, scans: list[dict]) -> None:
    lines = ["# QUASR no-axis 密集残差热力图诊断", ""]
    lines.append("| ID | batch | GA residual | dense residual | distance | interpretation | total [s] |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | ---: |")
    for row in scans:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["device_id"]),
                    str(row["batch_label"]),
                    f"{row['ga_best_residual']:.6g}",
                    f"{row['dense_best_residual_verify']:.6g}",
                    f"{row['dense_vs_ga_distance']:.6g}",
                    str(row["interpretation"]),
                    f"{row['timing']['total_s']:.3f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("解释规则：")
    lines.append("")
    lines.append("- `dense_grid_contains_subtol_point`：128x128 网格里已经直接出现低于阈值的点，GA 很可能没抓住。")
    lines.append("- `ga_found_subgrid_narrow_minimum`：GA 最终点已经接近阈值，而且与 dense 图上的最优格点非常接近；更像是存在一个比网格更窄的尖锐低谷。")
    lines.append("- `consistent_with_ga_minimum`：dense-grid 最优点和 GA 最优点基本一致，支持“在当前搜索框里确实没有磁轴”。")
    lines.append("- `ga_missed_better_basin`：dense-grid 上存在明显更好的区域，且位置与 GA 最终点分离，怀疑 GA 收敛到了错误盆地。")
    lines.append("- `better_region_exists_but_not_axis`：dense-grid 的最优点比 GA 稍好，但仍远高于阈值，说明有更好的区域但仍不像真磁轴。")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense residual heatmaps for QUASR no-axis samples.")
    parser.add_argument("--batch-summary", type=Path, action="append", required=True)
    parser.add_argument("--quasr-root", type=Path, default=REMOTE_QUASR_ROOT)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--precision", default="mixed64")
    parser.add_argument("--verify-precision", default="fp64")
    parser.add_argument("--verify-candidates", type=int, default=16)
    parser.add_argument("--floor", type=float, default=1e-8)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/quasr_axis_dense"))
    args = parser.parse_args()

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_no_axis_runs(args.batch_summary)

    scans: list[dict] = []
    for device_id in sorted(runs):
        print(f"[dense axis scan] ID={device_id}")
        scan = dense_axis_scan(
            runs[device_id],
            args.quasr_root,
            grid=int(args.grid),
            precision=str(args.precision),
            verify_precision=str(args.verify_precision),
            verify_candidates=int(args.verify_candidates),
        )
        plot_path = output_dir / f"id_{device_id:07d}_axis_residual_heatmap.png"
        draw_heatmap(scan, plot_path, floor=float(args.floor))
        scan["heatmap"] = plot_path.name
        scans.append(scan)

    payload = {
        "batch_summaries": [str(p) for p in args.batch_summary],
        "grid": int(args.grid),
        "precision": str(args.precision),
        "verify_precision": str(args.verify_precision),
        "verify_candidates": int(args.verify_candidates),
        "scans": scans,
    }
    write_json(output_dir / "dense_axis_summary.json", payload)
    write_report(output_dir / "report.md", scans)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
