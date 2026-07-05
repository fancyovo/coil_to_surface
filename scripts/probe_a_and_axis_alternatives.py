from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_quasr_failures import classify_axis_quadratic
from scripts.explore_axis_fixed_point import (
    AxisFixedPointConfig,
    choose_domain,
    dedupe_candidates,
    evaluate_residual,
    local_min_candidates,
    make_gpu_field,
    refine_candidates_newton,
    verify_refined,
    winding_candidates,
)
from stellarator_eval.axis import AxisResult, trace_axis
from stellarator_eval.config import EvalConfig
from stellarator_eval.field import build_field
from stellarator_eval.pipeline import evaluate_field_input
from stellarator_eval.psi import fit_psi
from stellarator_eval.quasr import load_quasr_field_input
from stellarator_eval.serialization import jsonable, write_json
from stellarator_eval.surface import evaluate_boozer_surface, screen_levels_gpu


REMOTE_QUASR_ROOT = Path("/data/zhouyebi/QUASR_08072024")


def parse_ids(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(",", " ").split() if x.strip()]


def configure_plot_fonts() -> None:
    try:
        import matplotlib
        from matplotlib import font_manager
    except Exception:
        return
    candidates = ("Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei")
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans", "sans-serif"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
    matplotlib.rcParams["axes.unicode_minus"] = False


def make_eval_config(*, a: float, gpu_device: int, psi_n: int, qs_sdim: int) -> EvalConfig:
    cfg = EvalConfig()
    cfg.current_unit = "A"
    cfg.omp_threads = 1
    cfg.psi.a = float(a)
    cfg.psi.n_r = int(psi_n)
    cfg.psi.n_z = int(psi_n)
    cfg.psi.n_phi = int(psi_n)
    cfg.psi.gpu_device = int(gpu_device)
    cfg.axis.gpu_device = int(gpu_device)
    cfg.scan.gpu_device = int(gpu_device)
    cfg.boozer.gpu_device = int(gpu_device)
    cfg.boozer.qs_sdim = int(qs_sdim)
    return cfg


def result_status(result: dict) -> str:
    if result.get("best_surface") is not None:
        return "surface"
    if not result["axis"]["has_axis"]:
        return "no_axis"
    return "no_surface"


def summarize_result(device_id: int, a: float, result: dict) -> dict:
    psi_fit = ((result.get("psi") or {}).get("fit_info") or {})
    screen_levels = ((result.get("surface_screen") or {}).get("levels") or [])
    ok_screen = [x for x in screen_levels if x.get("ok")]
    dists = [float(x["end_distance_p95"]) for x in screen_levels if x.get("end_distance_p95") is not None]
    rels = [float(x["rel_end_distance_p95"]) for x in screen_levels if x.get("rel_end_distance_p95") is not None]
    candidates = result.get("surface_candidates") or []
    best = result.get("best_surface") or {}
    return {
        "device_id": int(device_id),
        "a": float(a),
        "status": result_status(result),
        "axis_has_axis": bool(result["axis"]["has_axis"]),
        "axis_residual": float(result["axis"]["best_residual"]),
        "psi_angle_p95": psi_fit.get("validation_angle_p95"),
        "psi_angle_l2": psi_fit.get("validation_angle_l2"),
        "psi_train_rms": psi_fit.get("train_rms"),
        "screen_ok_count": len(ok_screen),
        "screen_best_level": max((float(x["psi_level"]) for x in ok_screen), default=None),
        "screen_min_distance_p95": min(dists) if dists else None,
        "screen_min_rel_distance_p95": min(rels) if rels else None,
        "surface_candidate_count": len(candidates),
        "best_surface_psi_level": best.get("psi_level"),
        "best_surface_iota": best.get("iota"),
        "best_surface_volume": best.get("volume"),
        "total_time_s": float(result.get("total_time_s", 0.0)),
        "warnings": result.get("warnings", []),
    }


def run_a_sweep(args) -> list[dict]:
    rows = []
    for device_id in parse_ids(args.a_sweep_ids):
        field_input, _ = load_quasr_field_input(args.quasr_root, device_id)
        for a in parse_floats(args.a_values):
            out_dir = args.output_dir / "a_sweep" / f"id_{device_id:07d}" / f"a_{a:.5g}".replace(".", "p")
            print(f"a-sweep ID={device_id} a={a}", flush=True)
            cfg = make_eval_config(a=a, gpu_device=args.gpu_device, psi_n=args.psi_n, qs_sdim=args.qs_sdim)
            try:
                result = evaluate_field_input(field_input, config=cfg, output_dir=out_dir)
                row = summarize_result(device_id, a, result)
            except Exception as exc:
                row = {"device_id": int(device_id), "a": float(a), "status": "error", "error": repr(exc)}
            rows.append(row)
    return rows


def draw_axis_scan(scan: dict, output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    configure_plot_fonts()
    rs = np.asarray(scan["rs"], dtype=float)
    zs = np.asarray(scan["zs"], dtype=float)
    residual = np.asarray(scan["residual"], dtype=float)
    values = np.log10(np.maximum(residual, 1e-12))
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    im = ax.imshow(
        values,
        origin="lower",
        extent=[float(rs[0]), float(rs[-1]), float(zs[0]), float(zs[-1])],
        aspect="auto",
        cmap="viridis",
    )
    for cand in scan["candidates"]:
        color = "white" if cand.get("residual_verify", 1.0) <= scan["tol"] else "red"
        ax.plot([cand["R"]], [cand["Z"]], marker="x", color=color, ms=7, mew=1.3)
        ax.text(cand["R"], cand["Z"], f"{cand.get('rank','')}", color=color, fontsize=7)
    if scan.get("original_axis_R") is not None:
        ax.plot([scan["original_axis_R"]], [scan["original_axis_Z"]], "r+", ms=10, mew=1.6)
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10 one-period closure residual")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def scan_axis_candidates(field_input, current_unit: str, args, *, original_axis: dict | None = None) -> dict:
    cfg = AxisFixedPointConfig(
        grid=args.axis_grid,
        local_min_candidates=args.axis_local_min_candidates,
        max_candidates=args.axis_max_candidates,
        newton_iters=args.axis_newton_iters,
        domain_span=args.axis_domain_span,
        domain_mode=args.axis_domain_mode,
        verify_top=args.axis_verify_top,
        gpu_device=args.gpu_device,
    )
    domain = choose_domain(field_input, current_unit, cfg)
    rs = np.linspace(domain["r_min"], domain["r_max"], cfg.grid)
    zs = np.linspace(domain["z_min"], domain["z_max"], cfg.grid)
    rg, zg = np.meshgrid(rs, zs, indexing="xy")
    r0 = np.ascontiguousarray(rg.ravel(), dtype=float)
    z0 = np.ascontiguousarray(zg.ravel(), dtype=float)
    gpu_field = make_gpu_field(field_input, cfg, current_unit)
    try:
        dr, dz, residual = evaluate_residual(gpu_field, r0, z0, cfg, precision=cfg.precision)
        residual_grid = residual.reshape(cfg.grid, cfg.grid)
        wind = winding_candidates(rs, zs, dr.reshape(cfg.grid, cfg.grid), dz.reshape(cfg.grid, cfg.grid))
        mins = local_min_candidates(rs, zs, residual_grid, cfg.local_min_candidates)
        min_distance = 0.75 * max(float(rs[1] - rs[0]), float(zs[1] - zs[0]))
        candidates = dedupe_candidates(
            wind + mins,
            min_distance=min_distance,
            max_candidates=cfg.max_candidates,
            r_floor=cfg.r_floor,
        )
        refined, stats = refine_candidates_newton(gpu_field, candidates, cfg, domain)
        verified, verify_s = verify_refined(gpu_field, refined, cfg)
    finally:
        gpu_field.close()
    for i, cand in enumerate(verified):
        cand["rank"] = i
    return {
        "config": asdict(cfg),
        "domain": domain,
        "rs": rs.tolist(),
        "zs": zs.tolist(),
        "residual": residual_grid.tolist(),
        "grid_best_residual": float(np.min(residual)),
        "winding_count": len(wind),
        "candidate_count": len(candidates),
        "newton_stats": stats,
        "verify_s": verify_s,
        "tol": cfg.tol,
        "candidates": verified,
        "original_axis_R": None if original_axis is None else original_axis.get("best_R"),
        "original_axis_Z": None if original_axis is None else original_axis.get("best_Z"),
    }


def make_forced_axis(built, nfp: int, R0: float, Z0: float, residual: float, cfg: EvalConfig) -> AxisResult:
    t0 = time.perf_counter()
    phi, R, Z, R_phi, Z_phi = trace_axis(built.field, float(R0), float(Z0), nfp, cfg.axis.axis_trace_steps)
    trace_s = time.perf_counter() - t0
    return AxisResult(
        has_axis=True,
        best_R=float(R0),
        best_Z=float(Z0),
        best_residual=float(residual),
        generation=0,
        history=[{"method": "forced_alternative_axis", "best_R": float(R0), "best_Z": float(Z0), "best_residual": float(residual)}],
        phi=phi,
        R=R,
        Z=Z,
        R_phi=R_phi,
        Z_phi=Z_phi,
        time_s=trace_s,
        search_time_s=0.0,
        trace_time_s=trace_s,
        backend="forced",
        search_best_residual=float(residual),
    )


def evaluate_forced_axis(field_input, built, candidate: dict, args, *, candidate_rank: int) -> dict:
    cfg = make_eval_config(a=args.alt_a, gpu_device=args.gpu_device, psi_n=args.psi_n, qs_sdim=args.qs_sdim)
    axis = make_forced_axis(built, field_input.nfp, candidate["R"], candidate["Z"], candidate["residual_verify"], cfg)
    t0 = time.perf_counter()
    model = fit_psi(built.field, axis, built.nfp, cfg.psi, field_input=field_input, current_unit=cfg.current_unit)
    fit_s = time.perf_counter() - t0
    quad = classify_axis_quadratic(model, phi_count=args.hessian_phi_count)
    try:
        screens = screen_levels_gpu(field_input, model, cfg.scan.levels, cfg.scan, cfg.current_unit)
    except Exception as exc:
        screens = [{"ok": False, "reason": repr(exc)}]
    ok = [x for x in screens if x.get("ok")]
    surface_results = []
    best_surface = None
    for item in sorted(ok, key=lambda x: x["psi_level"], reverse=True)[: args.max_boozer_candidates]:
        try:
            res = evaluate_boozer_surface(built.field, model, float(item["psi_level"]), cfg.scan, cfg.boozer)
        except Exception as exc:
            res = {"psi_level": float(item["psi_level"]), "error": repr(exc)}
        surface_results.append(res)
        if res.get("newton_success"):
            best_surface = res
            break
    dists = [float(x["end_distance_p95"]) for x in screens if x.get("end_distance_p95") is not None]
    rels = [float(x["rel_end_distance_p95"]) for x in screens if x.get("rel_end_distance_p95") is not None]
    return {
        "candidate_rank": int(candidate_rank),
        "R": float(candidate["R"]),
        "Z": float(candidate["Z"]),
        "residual_verify": float(candidate["residual_verify"]),
        "candidate_kind": candidate.get("kind"),
        "fit_time_s": float(fit_s),
        "psi_angle_p95": model.fit_info.get("validation_angle_p95"),
        "psi_angle_l2": model.fit_info.get("validation_angle_l2"),
        "psi_train_rms": model.fit_info.get("train_rms"),
        "hessian": quad,
        "screen_ok_count": len(ok),
        "screen_best_level": max((float(x["psi_level"]) for x in ok), default=None),
        "screen_min_distance_p95": min(dists) if dists else None,
        "screen_min_rel_distance_p95": min(rels) if rels else None,
        "surface_candidate_count": len(surface_results),
        "best_surface_found": best_surface is not None,
        "best_surface": best_surface,
        "surface_results": surface_results,
    }


def run_axis_alternatives(args) -> list[dict]:
    rows = []
    for device_id in parse_ids(args.axis_alt_ids):
        print(f"axis-alternatives ID={device_id}", flush=True)
        field_input, _ = load_quasr_field_input(args.quasr_root, device_id)
        built = build_field(field_input, "A")
        original_axis = None
        for run_dir in args.reference_run_dir:
            summary_path = run_dir / f"id_{device_id:07d}" / "summary.json"
            if summary_path.exists():
                original_axis = json.loads(summary_path.read_text(encoding="utf-8")).get("axis")
                break
        scan = scan_axis_candidates(field_input, "A", args, original_axis=original_axis)
        out_dir = args.output_dir / "axis_alternatives" / f"id_{device_id:07d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        draw_axis_scan(scan, out_dir / "large_axis_residual_scan.png", f"ID {device_id} large fixed-point residual")
        forced = []
        for cand in scan["candidates"][: args.force_top_candidates]:
            if cand.get("residual_verify", 1.0) > args.force_residual_max:
                continue
            try:
                forced.append(evaluate_forced_axis(field_input, built, cand, args, candidate_rank=int(cand["rank"])))
            except Exception as exc:
                forced.append(
                    {
                        "candidate_rank": int(cand["rank"]),
                        "R": cand.get("R"),
                        "Z": cand.get("Z"),
                        "residual_verify": cand.get("residual_verify"),
                        "error": repr(exc),
                    }
                )
        result = {
            "device_id": int(device_id),
            "scan": {k: v for k, v in scan.items() if k not in {"residual"}},
            "scan_heatmap": str((out_dir / "large_axis_residual_scan.png").relative_to(args.output_dir)),
            "forced_axis_results": forced,
        }
        write_json(out_dir / "axis_alternative_summary.json", result)
        rows.append(result)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe smaller a and alternative magnetic axes for QUASR failures.")
    parser.add_argument("--quasr-root", type=Path, default=REMOTE_QUASR_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--psi-n", type=int, default=64)
    parser.add_argument("--qs-sdim", type=int, default=8)
    parser.add_argument("--a-sweep-ids", default="")
    parser.add_argument("--a-values", default="0.05,0.0375,0.03,0.02,0.0125")
    parser.add_argument("--axis-alt-ids", default="")
    parser.add_argument("--reference-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--axis-grid", type=int, default=144)
    parser.add_argument("--axis-domain-mode", choices=["hybrid", "quantile", "ga"], default="quantile")
    parser.add_argument("--axis-domain-span", type=float, default=0.8)
    parser.add_argument("--axis-max-candidates", type=int, default=32)
    parser.add_argument("--axis-local-min-candidates", type=int, default=128)
    parser.add_argument("--axis-newton-iters", type=int, default=8)
    parser.add_argument("--axis-verify-top", type=int, default=16)
    parser.add_argument("--force-top-candidates", type=int, default=6)
    parser.add_argument("--force-residual-max", type=float, default=1e-6)
    parser.add_argument("--alt-a", type=float, default=0.05)
    parser.add_argument("--hessian-phi-count", type=int, default=17)
    parser.add_argument("--max-boozer-candidates", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": vars(args), "a_sweep": [], "axis_alternatives": []}
    if args.a_sweep_ids:
        a_rows = run_a_sweep(args)
        payload["a_sweep"] = a_rows
        write_csv(args.output_dir / "a_sweep_summary.csv", a_rows)
    if args.axis_alt_ids:
        payload["axis_alternatives"] = run_axis_alternatives(args)
    write_json(args.output_dir / "probe_summary.json", jsonable(payload))


if __name__ == "__main__":
    main()
