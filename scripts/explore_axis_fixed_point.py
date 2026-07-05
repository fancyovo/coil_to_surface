from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stellarator_eval.axis import search_axis_ga_gpu
from stellarator_eval.config import AxisGAConfig
from stellarator_eval.field import FieldInput, build_field, load_case_file
from stellarator_eval.quasr import load_quasr_field_input, load_quasr_metadata
from stellarator_eval.serialization import jsonable


REMOTE_QUASR_ROOT = Path("/data/zhouyebi/QUASR_08072024")
REMOTE_PRIVATE_META = Path("/home/cyfan/stellarator_gpu_eval/quasr_private/QUASR_08072024_meta.csv")


@dataclass
class AxisFixedPointConfig:
    grid: int = 96
    local_min_candidates: int = 64
    max_candidates: int = 96
    newton_iters: int = 8
    newton_fd_rel: float = 2e-4
    newton_fd_abs: float = 2e-6
    r_floor: float = 1e-4
    domain_span: float = 0.5
    domain_mode: str = "hybrid"
    domain_margin: float = 0.10
    precision: str = "mixed64"
    verify_precision: str = "fp64"
    verify_top: int = 8
    tol: float = 1e-7
    rk4_steps: int = 800
    gpu_threads_per_line: int = 256
    gpu_lib_path: str = "gpu_backend/build_mixed/libstellarator_gpu.so"
    gpu_segments_per_coil: int = 256
    gpu_device: int = 0


def parse_ids(items: Iterable[str]) -> list[int]:
    out: list[int] = []
    for item in items:
        out.extend(int(x) for x in item.replace(",", " ").split() if x.strip())
    return out


def normalize_currents(currents: np.ndarray, unit: str) -> np.ndarray:
    unit_l = unit.lower()
    if unit_l in {"ma", "megaamp", "megaamps"}:
        return np.asarray(currents, dtype=float) * 1e6
    if unit_l in {"a", "amp", "amps"}:
        return np.asarray(currents, dtype=float)
    raise ValueError(f"unknown current unit {unit!r}")


def make_gpu_field(field_input: FieldInput, cfg: AxisFixedPointConfig, current_unit: str):
    gpu_python = REPO_ROOT / "gpu_backend" / "python"
    if str(gpu_python) not in sys.path:
        sys.path.insert(0, str(gpu_python))
    from stellarator_gpu import CoilFieldGpu

    lib_path = Path(cfg.gpu_lib_path)
    if not lib_path.is_absolute():
        lib_path = REPO_ROOT / lib_path
    return CoilFieldGpu(
        lib_path,
        field_input.coeffs_x,
        field_input.coeffs_y,
        field_input.coeffs_z,
        normalize_currents(field_input.currents, current_unit),
        nfp=field_input.nfp,
        segments_per_coil=cfg.gpu_segments_per_coil,
        device_id=cfg.gpu_device,
    )


def coil_rz_points(field_input: FieldInput, current_unit: str) -> tuple[np.ndarray, np.ndarray, float]:
    built = build_field(field_input, current_unit)
    rs = []
    zs = []
    for curve in built.base_curves:
        gamma = curve.gamma()
        rs.append(np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2))
        zs.append(gamma[:, 2])
    return np.concatenate(rs), np.concatenate(zs), float(built.coil_r0)


def choose_domain(field_input: FieldInput, current_unit: str, cfg: AxisFixedPointConfig) -> dict:
    coil_r, coil_z, coil_r0 = coil_rz_points(field_input, current_unit)
    mode = cfg.domain_mode.lower()
    if mode == "ga":
        r_min = coil_r0 - cfg.domain_span
        r_max = coil_r0 + cfg.domain_span
        z_min = -cfg.domain_span
        z_max = cfg.domain_span
    elif mode == "quantile":
        rq0, rq1 = np.quantile(coil_r, [0.02, 0.98])
        zq0, zq1 = np.quantile(coil_z, [0.02, 0.98])
        r_pad = max(cfg.domain_margin * (rq1 - rq0), 0.05)
        z_pad = max(cfg.domain_margin * (zq1 - zq0), 0.05)
        r_min, r_max = float(rq0 - r_pad), float(rq1 + r_pad)
        z_min, z_max = float(zq0 - z_pad), float(zq1 + z_pad)
    elif mode == "hybrid":
        rq0, rq1 = np.quantile(coil_r, [0.05, 0.95])
        z_abs = float(np.quantile(np.abs(coil_z), 0.95))
        r_span = max(cfg.domain_span, 0.45 * float(rq1 - rq0))
        z_span = max(cfg.domain_span, 1.15 * z_abs)
        r_min = min(coil_r0 - cfg.domain_span, coil_r0 - r_span)
        r_max = max(coil_r0 + cfg.domain_span, coil_r0 + r_span)
        z_min = -z_span
        z_max = z_span
    else:
        raise ValueError(f"unknown domain mode {cfg.domain_mode!r}")
    r_min = max(float(cfg.r_floor), float(r_min))
    if r_max <= r_min:
        r_max = r_min + max(cfg.domain_span, 0.1)
    return {
        "mode": cfg.domain_mode,
        "coil_r0": coil_r0,
        "r_min": float(r_min),
        "r_max": float(r_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "coil_r_q02": float(np.quantile(coil_r, 0.02)),
        "coil_r_q98": float(np.quantile(coil_r, 0.98)),
        "coil_z_q02": float(np.quantile(coil_z, 0.02)),
        "coil_z_q98": float(np.quantile(coil_z, 0.98)),
    }


def trace_map(gpu_field, r: np.ndarray, z: np.ndarray, cfg: AxisFixedPointConfig, *, precision: str) -> tuple[np.ndarray, np.ndarray]:
    r = np.ascontiguousarray(np.asarray(r, dtype=float).ravel())
    z = np.ascontiguousarray(np.asarray(z, dtype=float).ravel())
    return gpu_field.trace_period_blockline_precision(
        r,
        z,
        steps=cfg.rk4_steps,
        precision=precision,
        threads_per_line=cfg.gpu_threads_per_line,
        nfp=gpu_field.nfp,
    )


def vector_field(gpu_field, r: np.ndarray, z: np.ndarray, cfg: AxisFixedPointConfig, *, precision: str):
    r1, z1 = trace_map(gpu_field, r, z, cfg, precision=precision)
    return r1 - r, z1 - z


def angle_diff(a: float, b: float) -> float:
    d = b - a
    while d <= -math.pi:
        d += 2.0 * math.pi
    while d > math.pi:
        d -= 2.0 * math.pi
    return d


def winding_candidates(rs: np.ndarray, zs: np.ndarray, d_r: np.ndarray, d_z: np.ndarray) -> list[dict]:
    grid = len(rs)
    angles = np.arctan2(d_z, d_r)
    residual = np.hypot(d_r, d_z)
    out: list[dict] = []
    for j in range(grid - 1):
        for i in range(grid - 1):
            vals = [
                angles[j, i],
                angles[j, i + 1],
                angles[j + 1, i + 1],
                angles[j + 1, i],
            ]
            if not np.all(np.isfinite(vals)):
                continue
            total = (
                angle_diff(vals[0], vals[1])
                + angle_diff(vals[1], vals[2])
                + angle_diff(vals[2], vals[3])
                + angle_diff(vals[3], vals[0])
            )
            if abs(total) > math.pi:
                out.append(
                    {
                        "R": 0.5 * float(rs[i] + rs[i + 1]),
                        "Z": 0.5 * float(zs[j] + zs[j + 1]),
                        "kind": "winding",
                        "index": int(round(total / (2.0 * math.pi))),
                        "cell_residual_min": float(np.min(residual[j : j + 2, i : i + 2])),
                    }
                )
    return out


def local_min_candidates(
    rs: np.ndarray,
    zs: np.ndarray,
    residual: np.ndarray,
    limit: int,
) -> list[dict]:
    candidates: list[dict] = []
    ny, nx = residual.shape
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            val = residual[j, i]
            if not np.isfinite(val):
                continue
            nb = residual[j - 1 : j + 2, i - 1 : i + 2]
            if val <= np.min(nb):
                candidates.append({"R": float(rs[i]), "Z": float(zs[j]), "kind": "local_min", "cell_residual_min": float(val)})
    candidates.sort(key=lambda x: x["cell_residual_min"])
    return candidates[:limit]


def dedupe_candidates(candidates: list[dict], *, min_distance: float, max_candidates: int, r_floor: float) -> list[dict]:
    candidates = [c for c in candidates if c["R"] >= r_floor and np.isfinite(c["R"]) and np.isfinite(c["Z"])]
    candidates.sort(key=lambda x: (0 if x["kind"] == "winding" else 1, x["cell_residual_min"]))
    out: list[dict] = []
    for cand in candidates:
        if all(math.hypot(cand["R"] - prev["R"], cand["Z"] - prev["Z"]) >= min_distance for prev in out):
            out.append(dict(cand))
        if len(out) >= max_candidates:
            break
    return out


def evaluate_residual(gpu_field, r: np.ndarray, z: np.ndarray, cfg: AxisFixedPointConfig, *, precision: str):
    d_r, d_z = vector_field(gpu_field, r, z, cfg, precision=precision)
    return d_r, d_z, np.hypot(d_r, d_z)


def refine_candidates_newton(
    gpu_field,
    candidates: list[dict],
    cfg: AxisFixedPointConfig,
    domain: dict,
) -> tuple[list[dict], dict]:
    if not candidates:
        return [], {"newton_s": 0.0, "iterations": 0, "evaluated_points": 0}
    r = np.array([c["R"] for c in candidates], dtype=float)
    z = np.array([c["Z"] for c in candidates], dtype=float)
    kind = [c["kind"] for c in candidates]
    best_dr, best_dz, best_res = evaluate_residual(gpu_field, r, z, cfg, precision=cfg.precision)
    evaluated = len(r)
    span = max(domain["r_max"] - domain["r_min"], domain["z_max"] - domain["z_min"])
    h = max(cfg.newton_fd_abs, cfg.newton_fd_rel * span)
    alpha_values = (1.0, 0.5, 0.25, 0.125)
    t0 = time.perf_counter()
    actual_iters = 0
    for it in range(cfg.newton_iters):
        active = np.where(best_res > cfg.tol)[0]
        if active.size == 0:
            break
        actual_iters = it + 1
        ra = r[active]
        za = z[active]
        eval_r = np.concatenate(
            [
                np.maximum(cfg.r_floor, ra + h),
                np.maximum(cfg.r_floor, ra - h),
                ra,
                ra,
            ]
        )
        eval_z = np.concatenate([za, za, za + h, za - h])
        fdr, fdz, _ = evaluate_residual(gpu_field, eval_r, eval_z, cfg, precision=cfg.precision)
        evaluated += eval_r.size
        n = active.size
        j11 = (fdr[:n] - fdr[n : 2 * n]) / (np.maximum(cfg.r_floor, ra + h) - np.maximum(cfg.r_floor, ra - h))
        j21 = (fdz[:n] - fdz[n : 2 * n]) / (np.maximum(cfg.r_floor, ra + h) - np.maximum(cfg.r_floor, ra - h))
        j12 = (fdr[2 * n : 3 * n] - fdr[3 * n :]) / (2.0 * h)
        j22 = (fdz[2 * n : 3 * n] - fdz[3 * n :]) / (2.0 * h)
        det = j11 * j22 - j12 * j21
        good = np.abs(det) > 1e-14
        step_r = np.zeros(n, dtype=float)
        step_z = np.zeros(n, dtype=float)
        step_r[good] = (-best_dr[active][good] * j22[good] + j12[good] * best_dz[active][good]) / det[good]
        step_z[good] = (j21[good] * best_dr[active][good] - j11[good] * best_dz[active][good]) / det[good]
        max_step = 0.25 * span
        step_norm = np.hypot(step_r, step_z)
        scale = np.minimum(1.0, max_step / np.maximum(step_norm, 1e-300))
        step_r *= scale
        step_z *= scale
        accepted = np.zeros(n, dtype=bool)
        for alpha in alpha_values:
            trial_r = np.clip(ra + alpha * step_r, cfg.r_floor, domain["r_max"])
            trial_z = np.clip(za + alpha * step_z, domain["z_min"], domain["z_max"])
            tdr, tdz, tres = evaluate_residual(gpu_field, trial_r, trial_z, cfg, precision=cfg.precision)
            evaluated += trial_r.size
            improve = (~accepted) & np.isfinite(tres) & (tres < best_res[active])
            if not np.any(improve):
                continue
            take_idx = active[improve]
            r[take_idx] = trial_r[improve]
            z[take_idx] = trial_z[improve]
            best_dr[take_idx] = tdr[improve]
            best_dz[take_idx] = tdz[improve]
            best_res[take_idx] = tres[improve]
            accepted[improve] = True
    newton_s = time.perf_counter() - t0
    order = np.argsort(best_res)
    refined: list[dict] = []
    for idx in order:
        refined.append(
            {
                "R": float(r[idx]),
                "Z": float(z[idx]),
                "residual_search": float(best_res[idx]),
                "dR_search": float(best_dr[idx]),
                "dZ_search": float(best_dz[idx]),
                "kind": kind[idx],
            }
        )
    return refined, {"newton_s": float(newton_s), "iterations": int(actual_iters), "evaluated_points": int(evaluated)}


def verify_refined(gpu_field, refined: list[dict], cfg: AxisFixedPointConfig) -> tuple[list[dict], float]:
    if not refined:
        return [], 0.0
    top = refined[: max(1, cfg.verify_top)]
    r = np.array([c["R"] for c in top], dtype=float)
    z = np.array([c["Z"] for c in top], dtype=float)
    t0 = time.perf_counter()
    d_r, d_z, res = evaluate_residual(gpu_field, r, z, cfg, precision=cfg.verify_precision)
    verify_s = time.perf_counter() - t0
    for i, cand in enumerate(top):
        cand["residual_verify"] = float(res[i])
        cand["dR_verify"] = float(d_r[i])
        cand["dZ_verify"] = float(d_z[i])
    top.sort(key=lambda c: c["residual_verify"])
    return top, float(verify_s)


def run_fixed_point_search(field_input: FieldInput, current_unit: str, cfg: AxisFixedPointConfig, *, include_ga: bool) -> dict:
    timings: dict[str, float] = {}
    t_domain = time.perf_counter()
    domain = choose_domain(field_input, current_unit, cfg)
    timings["domain_s"] = time.perf_counter() - t_domain
    rs = np.linspace(domain["r_min"], domain["r_max"], cfg.grid)
    zs = np.linspace(domain["z_min"], domain["z_max"], cfg.grid)
    rg, zg = np.meshgrid(rs, zs, indexing="xy")
    r0 = np.ascontiguousarray(rg.ravel(), dtype=float)
    z0 = np.ascontiguousarray(zg.ravel(), dtype=float)
    t_create = time.perf_counter()
    gpu_field = make_gpu_field(field_input, cfg, current_unit)
    timings["gpu_create_s"] = time.perf_counter() - t_create
    try:
        ga_result = None
        if include_ga:
            ga_cfg = AxisGAConfig(
                span=cfg.domain_span,
                rk4_steps=cfg.rk4_steps,
                tol=cfg.tol,
                gpu_lib_path=cfg.gpu_lib_path,
                gpu_segments_per_coil=cfg.gpu_segments_per_coil,
                gpu_device=cfg.gpu_device,
                gpu_trace_precision=cfg.precision,
                gpu_verify_precision=cfg.verify_precision,
                gpu_threads_per_line=cfg.gpu_threads_per_line,
            )
            t_ga = time.perf_counter()
            ga_best, ga_history = search_axis_ga_gpu(gpu_field, field_input.nfp, domain["coil_r0"], ga_cfg)
            timings["ga_baseline_s"] = time.perf_counter() - t_ga
            ga_result = {"best": ga_best, "history": ga_history}
        t_grid = time.perf_counter()
        d_r_flat, d_z_flat, residual_flat = evaluate_residual(gpu_field, r0, z0, cfg, precision=cfg.precision)
        timings["grid_trace_s"] = time.perf_counter() - t_grid
        d_r = d_r_flat.reshape(cfg.grid, cfg.grid)
        d_z = d_z_flat.reshape(cfg.grid, cfg.grid)
        residual = residual_flat.reshape(cfg.grid, cfg.grid)
        t_candidates = time.perf_counter()
        wind = winding_candidates(rs, zs, d_r, d_z)
        mins = local_min_candidates(rs, zs, residual, cfg.local_min_candidates)
        min_distance = 0.75 * max((rs[1] - rs[0]) if cfg.grid > 1 else 0.0, (zs[1] - zs[0]) if cfg.grid > 1 else 0.0)
        candidates = dedupe_candidates(wind + mins, min_distance=min_distance, max_candidates=cfg.max_candidates, r_floor=cfg.r_floor)
        timings["candidate_s"] = time.perf_counter() - t_candidates
        refined, refine_stats = refine_candidates_newton(gpu_field, candidates, cfg, domain)
        timings.update(refine_stats)
        verified, verify_s = verify_refined(gpu_field, refined, cfg)
        timings["verify_s"] = verify_s
    finally:
        gpu_field.close()
    best = verified[0] if verified else None
    timings["total_search_s"] = sum(
        float(v)
        for k, v in timings.items()
        if k.endswith("_s") and isinstance(v, (int, float))
    )
    return {
        "name": field_input.name,
        "nfp": int(field_input.nfp),
        "current_unit": current_unit,
        "config": asdict(cfg),
        "domain": domain,
        "has_axis": bool(best is not None and best["residual_verify"] <= cfg.tol),
        "best": best,
        "candidate_counts": {
            "winding": len(wind),
            "local_min": len(mins),
            "deduped": len(candidates),
            "refined": len(refined),
            "verified": len(verified),
        },
        "grid_best": {
            "R": float(r0[int(np.argmin(residual_flat))]),
            "Z": float(z0[int(np.argmin(residual_flat))]),
            "residual": float(np.min(residual_flat)),
        },
        "timing": timings,
        "ga_baseline": ga_result,
    }


def write_csv_summary(path: Path, rows: list[dict]) -> None:
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flatten_result(label: str, result: dict) -> dict:
    best = result.get("best") or {}
    ga_best = ((result.get("ga_baseline") or {}).get("best") or {})
    timing = result.get("timing") or {}
    return {
        "label": label,
        "name": result["name"],
        "nfp": result["nfp"],
        "has_axis": result["has_axis"],
        "best_R": best.get("R"),
        "best_Z": best.get("Z"),
        "best_residual_verify": best.get("residual_verify"),
        "best_residual_search": best.get("residual_search"),
        "best_kind": best.get("kind"),
        "grid_best_residual": result["grid_best"]["residual"],
        "winding_candidates": result["candidate_counts"]["winding"],
        "deduped_candidates": result["candidate_counts"]["deduped"],
        "ga_best_residual": ga_best.get("best_residual"),
        "ga_best_R": ga_best.get("best_R"),
        "ga_best_Z": ga_best.get("best_Z"),
        "domain_r_min": result["domain"]["r_min"],
        "domain_r_max": result["domain"]["r_max"],
        "domain_z_min": result["domain"]["z_min"],
        "domain_z_max": result["domain"]["z_max"],
        **{f"timing_{k}": v for k, v in timing.items()},
    }


def load_work_items(args) -> list[tuple[str, FieldInput, str]]:
    items: list[tuple[str, FieldInput, str]] = []
    for case_file in args.case_file:
        field_input = load_case_file(case_file, args.key)
        unit = args.current_unit or field_input.__dict__.get("current_unit") or "MA"
        items.append((Path(case_file).stem, field_input, unit))
    ids = parse_ids(args.id)
    if args.sample_size:
        rows = load_quasr_metadata(args.metadata_path)
        if args.helicity is not None:
            rows = [row for row in rows if int(row.get("helicity", -999)) == int(args.helicity)]
        if args.nfp is not None:
            rows = [row for row in rows if int(row.get("nfp", -999)) == int(args.nfp)]
        if args.sample_size > len(rows):
            raise ValueError(f"sample_size={args.sample_size} exceeds filtered metadata count {len(rows)}")
        rng = np.random.default_rng(args.sample_seed)
        take = rng.choice(len(rows), size=args.sample_size, replace=False)
        ids.extend(int(rows[i]["ID"]) for i in take)
    seen: set[int] = set()
    for device_id in ids:
        if device_id in seen:
            continue
        seen.add(device_id)
        field_input, _ = load_quasr_field_input(args.quasr_root, device_id)
        items.append((f"quasr_{device_id:07d}", field_input, "A"))
    return items


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Explore fixed-point magnetic-axis search on GPU.")
    p.add_argument("--case-file", action="append", default=[], help="JSON case file such as examples/01.json.")
    p.add_argument("--key", default="raw")
    p.add_argument("--current-unit", default=None, choices=[None, "MA", "A"])
    p.add_argument("--quasr-root", type=Path, default=REMOTE_QUASR_ROOT)
    p.add_argument("--metadata-path", type=Path, default=REMOTE_PRIVATE_META)
    p.add_argument("--id", action="append", default=[], help="QUASR ID, comma list allowed.")
    p.add_argument("--sample-size", type=int, default=0, help="Randomly sample this many QUASR IDs from metadata.")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--helicity", type=int, choices=[0, 1], default=None)
    p.add_argument("--nfp", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("runs/axis_fixed_point_explore"))
    p.add_argument("--grid", type=int, default=96)
    p.add_argument("--local-min-candidates", type=int, default=64)
    p.add_argument("--max-candidates", type=int, default=96)
    p.add_argument("--newton-iters", type=int, default=8)
    p.add_argument("--precision", default="mixed64", choices=["mixed64", "fp32", "fp64"])
    p.add_argument("--verify-precision", default="fp64", choices=["mixed64", "fp32", "fp64"])
    p.add_argument("--rk4-steps", type=int, default=800)
    p.add_argument("--tol", type=float, default=1e-7)
    p.add_argument("--r-floor", type=float, default=1e-4)
    p.add_argument("--domain-span", type=float, default=0.5)
    p.add_argument("--domain-mode", default="hybrid", choices=["ga", "quantile", "hybrid"])
    p.add_argument("--threads-per-line", type=int, default=256)
    p.add_argument("--gpu-lib-path", default="gpu_backend/build_mixed/libstellarator_gpu.so")
    p.add_argument("--gpu-segments-per-coil", type=int, default=256)
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--include-ga", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.case_file and not args.id and not args.sample_size:
        raise SystemExit("provide at least one --case-file, --id, or --sample-size")
    cfg = AxisFixedPointConfig(
        grid=args.grid,
        local_min_candidates=args.local_min_candidates,
        max_candidates=args.max_candidates,
        newton_iters=args.newton_iters,
        r_floor=args.r_floor,
        domain_span=args.domain_span,
        domain_mode=args.domain_mode,
        precision=args.precision,
        verify_precision=args.verify_precision,
        tol=args.tol,
        rk4_steps=args.rk4_steps,
        gpu_threads_per_line=args.threads_per_line,
        gpu_lib_path=args.gpu_lib_path,
        gpu_segments_per_coil=args.gpu_segments_per_coil,
        gpu_device=args.gpu_device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    details: dict[str, dict] = {}
    for label, field_input, current_unit in load_work_items(args):
        print(f"[axis-fixed-point] {label} nfp={field_input.nfp} unit={current_unit}", flush=True)
        result = run_fixed_point_search(field_input, current_unit, cfg, include_ga=args.include_ga)
        rows.append(flatten_result(label, result))
        details[label] = result
        best_text = "None" if result["best"] is None else f"{result['best']['residual_verify']:.6g}"
        print(
            f"  has_axis={result['has_axis']} best={best_text} "
            f"grid={result['grid_best']['residual']:.6g} total={result['timing']['total_search_s']:.3f}s",
            flush=True,
        )
    (args.output_dir / "axis_fixed_point_details.json").write_text(
        json.dumps(jsonable(details), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv_summary(args.output_dir / "axis_fixed_point_summary.csv", rows)


if __name__ == "__main__":
    main()
