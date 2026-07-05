from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stellarator_eval import EvalConfig
from stellarator_eval.pipeline import evaluate_field_input
from stellarator_eval.quasr import (
    build_quasr_metadata_index,
    choose_quasr_eval_params,
    load_quasr_field_input,
    load_quasr_metadata,
    quasr_failure_case_payload,
)
from stellarator_eval.serialization import jsonable, write_json


WINDOWS_QUASR_ROOT = Path(r"D:\FPC\2.4.0\bin\i386-win32\new\ML\lhls\stellarator\quasr\data")
REMOTE_QUASR_ROOT = Path("/data/zhouyebi/QUASR_08072024")
REMOTE_PRIVATE_META = Path("/home/cyfan/stellarator_gpu_eval/quasr_private/QUASR_08072024_meta.csv")


def default_quasr_root() -> Path:
    if REMOTE_QUASR_ROOT.exists():
        return REMOTE_QUASR_ROOT
    return WINDOWS_QUASR_ROOT


def default_metadata_path() -> Path | None:
    candidates = [
        REMOTE_PRIVATE_META,
        default_quasr_root() / "QUASR_08072024.csv",
        default_quasr_root() / "data.csv",
        default_quasr_root() / "QUASR_08072024.pkl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_ids(args) -> list[int]:
    ids: list[int] = []
    for item in args.id:
        ids.append(int(item))
    if args.ids:
        ids.extend(int(x) for x in args.ids.replace(",", " ").split() if x.strip())
    return ids


def parse_levels(text: str | None):
    if text is None:
        return None
    return tuple(float(x) for x in text.replace(",", " ").split() if x.strip())


def choose_sample_ids(rows: list[dict], *, sample_size: int, seed: int, helicity: int | None, nfp: int | None) -> list[int]:
    filtered = rows
    if helicity is not None:
        filtered = [row for row in filtered if int(row.get("helicity", -999)) == helicity]
    if nfp is not None:
        filtered = [row for row in filtered if int(row.get("nfp", -999)) == nfp]
    if not filtered:
        raise ValueError("no metadata rows satisfy the requested filters")
    if sample_size > len(filtered):
        raise ValueError(f"sample_size={sample_size} exceeds filtered metadata count {len(filtered)}")
    rng = np.random.default_rng(seed)
    order = rng.choice(len(filtered), size=sample_size, replace=False)
    return [int(filtered[i]["ID"]) for i in order]


def make_config(args, eval_params: dict[str, float]) -> EvalConfig:
    cfg = EvalConfig()
    cfg.current_unit = "A"
    cfg.omp_threads = args.omp_threads
    cfg.psi.a = float(eval_params["a"])
    cfg.psi.linear_solver = args.psi_linear_solver
    cfg.psi.normal_eq_precision = args.psi_normal_eq_precision
    cfg.psi.n_r = args.psi_n_r
    cfg.psi.n_z = args.psi_n_z
    cfg.psi.n_phi = args.psi_n_phi
    cfg.psi.gpu_device = args.gpu_device
    cfg.axis.gpu_device = args.gpu_device
    cfg.scan.gpu_device = args.gpu_device
    cfg.boozer.gpu_device = args.gpu_device
    cfg.boozer.initial_iota = float(eval_params["initial_iota"])
    cfg.boozer.qs_sdim = args.qs_sdim
    cfg.boozer.surface_order = args.surface_order
    cfg.scan.max_boozer_candidates = args.max_boozer_candidates
    if args.levels is not None:
        cfg.scan.levels = args.levels
    return cfg


def result_status(result: dict) -> str:
    if result.get("best_surface") is not None:
        return "surface"
    if not result["axis"]["has_axis"]:
        return "no_axis"
    return "no_surface"


def flatten_record(device_id: int, meta: dict | None, adapt: dict, result: dict) -> dict:
    psi_fit = ((result.get("psi") or {}).get("fit_info") or {})
    screen_levels = ((result.get("surface_screen") or {}).get("levels") or [])
    ok_screen = [x for x in screen_levels if x.get("ok")]
    finite_dist = [float(x["end_distance_p95"]) for x in screen_levels if x.get("end_distance_p95") is not None]
    finite_rel = [float(x["rel_end_distance_p95"]) for x in screen_levels if x.get("rel_end_distance_p95") is not None]
    candidates = result.get("surface_candidates") or []
    row = {
        "ID": int(device_id),
        "status": result_status(result),
        "a_used": float(adapt["a"]),
        "initial_iota_used": float(adapt["initial_iota"]),
        "initial_iota_source": adapt["initial_iota_source"],
        "minor_radius_used_fraction": adapt.get("minor_radius_used_fraction"),
        "axis_has_axis": bool(result["axis"]["has_axis"]),
        "axis_best_residual": float(result["axis"]["best_residual"]),
        "axis_failure_reason": result["axis"].get("failure_reason", ""),
        "total_time_s": float(result["total_time_s"]),
        "psi_train_rms": psi_fit.get("train_rms"),
        "psi_validation_rms": psi_fit.get("validation_rms"),
        "psi_validation_angle_mean": psi_fit.get("validation_angle_mean"),
        "psi_validation_angle_p95": psi_fit.get("validation_angle_p95"),
        "psi_validation_angle_l2": psi_fit.get("validation_angle_l2"),
        "screen_level_count": len(screen_levels),
        "screen_ok_count": len(ok_screen),
        "screen_best_psi_level": max((float(x["psi_level"]) for x in ok_screen), default=None),
        "screen_min_distance_p95": min(finite_dist) if finite_dist else None,
        "screen_min_rel_distance_p95": min(finite_rel) if finite_rel else None,
        "surface_candidate_count": len(candidates),
    }
    if meta:
        for key in ("helicity", "nfp", "nc_per_hp", "mean_iota", "minor_radius", "aspect_ratio", "volume", "qs_error", "Nsurfaces"):
            if key in meta:
                row[f"meta_{key}"] = meta[key]
    best = result.get("best_surface")
    if best is not None:
        row.update(
            {
                "best_surface_psi_level": best.get("psi_level"),
                "best_surface_iota": best.get("iota"),
                "best_surface_volume": best.get("volume"),
                "best_surface_G": best.get("G"),
            }
        )
    if candidates:
        first = candidates[0]
        row.update(
            {
                "first_candidate_psi_level": first.get("psi_level"),
                "first_candidate_error": first.get("error"),
                "first_candidate_initial_residual": first.get("initial_boozer_residual_norm"),
                "first_candidate_ls_success": first.get("ls_success"),
                "first_candidate_ls_residual": first.get("ls_residual_norm"),
                "first_candidate_newton_success": first.get("newton_success"),
                "first_candidate_newton_residual": first.get("newton_residual_norm"),
            }
        )
    timing = result.get("timing", {})
    for key in ("axis_s", "psi_fit_s", "surface_screen_s", "boozer_candidates_s"):
        if key in timing:
            row[f"timing_{key}"] = timing[key]
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def export_failure_cases(
    out_dir: Path,
    *,
    failures: list[tuple[int, dict, Any]],
    quasr_root: Path,
    export_dir: Path,
    start_index: int,
    limit: int,
) -> list[str]:
    export_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    _ = out_dir
    _ = start_index
    for device_id, metadata_row, field_input in failures[:limit]:
        payload = quasr_failure_case_payload(
            field_input,
            device_id=device_id,
            metadata_row=metadata_row,
            source_root=quasr_root,
        )
        stem = f"quasr_id_{int(device_id):07d}"
        path = export_dir / f"{stem}.json"
        suffix = 1
        while path.exists():
            path = export_dir / f"{stem}__{suffix}.json"
            suffix += 1
        path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(path.name)
    return written


def write_report(path: Path, rows: list[dict], written_failures: list[str]) -> None:
    status_counts = Counter(row["status"] for row in rows)
    total = len(rows)
    lines = [
        "# QUASR 小规模评测报告",
        "",
        f"- 样本数: `{total}`",
        f"- `surface`: `{status_counts.get('surface', 0)}`",
        f"- `no_surface`: `{status_counts.get('no_surface', 0)}`",
        f"- `no_axis`: `{status_counts.get('no_axis', 0)}`",
        "",
        "## 逐样本结果",
        "",
        "| ID | status | nfp | helicity | a | initial_iota | axis_residual | total_time_s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["ID"]),
                    str(row["status"]),
                    str(row.get("meta_nfp", "")),
                    str(row.get("meta_helicity", "")),
                    f"{float(row['a_used']):.6g}",
                    f"{float(row['initial_iota_used']):.6g}",
                    f"{float(row['axis_best_residual']):.6g}",
                    f"{float(row['total_time_s']):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## 导出的失败样本", ""])
    if written_failures:
        for name in written_failures:
            lines.append(f"- `{name}`")
    else:
        lines.append("- 无")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QUASR devices with the local surface evaluator.")
    parser.add_argument("--quasr-root", type=Path, default=default_quasr_root())
    parser.add_argument("--metadata", type=Path, default=default_metadata_path())
    parser.add_argument("--id", action="append", default=[], help="Single device ID. Can be repeated.")
    parser.add_argument("--ids", help="Comma- or space-separated device IDs.")
    parser.add_argument("--sample-size", type=int, default=0, help="Random sample size. Requires metadata.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--helicity", type=int, choices=[0, 1], default=None)
    parser.add_argument("--nfp", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/quasr_eval"))
    parser.add_argument("--a", type=float, default=None, help="Explicit evaluation radius. If omitted, use min(default-a, frac*minor_radius).")
    parser.add_argument("--default-a", type=float, default=0.05)
    parser.add_argument("--a-minor-fraction", type=float, default=0.9)
    parser.add_argument("--initial-iota", type=float, default=None)
    parser.add_argument("--levels", type=parse_levels, default=None)
    parser.add_argument("--surface-order", type=int, default=6)
    parser.add_argument("--qs-sdim", type=int, default=16)
    parser.add_argument("--max-boozer-candidates", type=int, default=3)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--psi-linear-solver", choices=["qr", "normal_eq"], default="qr")
    parser.add_argument("--psi-normal-eq-precision", choices=["fp32", "fp64"], default="fp32")
    parser.add_argument("--psi-n-r", type=int, default=80)
    parser.add_argument("--psi-n-z", type=int, default=80)
    parser.add_argument("--psi-n-phi", type=int, default=80)
    parser.add_argument("--export-failures-dir", type=Path, default=Path("examples"))
    parser.add_argument("--export-failure-start", type=int, default=2)
    parser.add_argument("--export-failure-limit", type=int, default=8)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = None
    metadata_index = {}
    if args.metadata is not None:
        metadata_rows = load_quasr_metadata(args.metadata)
        metadata_index = build_quasr_metadata_index(metadata_rows)

    ids = parse_ids(args)
    if args.sample_size:
        if metadata_rows is None:
            raise ValueError("--sample-size requires --metadata")
        ids.extend(
            choose_sample_ids(
                metadata_rows,
                sample_size=args.sample_size,
                seed=args.sample_seed,
                helicity=args.helicity,
                nfp=args.nfp,
            )
        )
    if not ids:
        raise ValueError("please provide --id/--ids or --sample-size")

    ids = list(dict.fromkeys(int(x) for x in ids))
    rows = []
    failures: list[tuple[int, dict | None, Any]] = []
    run_summaries = []
    for device_id in ids:
        field_input, info = load_quasr_field_input(args.quasr_root, device_id)
        meta = metadata_index.get(int(device_id))
        adapt = choose_quasr_eval_params(
            meta,
            default_a=args.default_a,
            a_minor_fraction=args.a_minor_fraction,
            explicit_a=args.a,
            explicit_initial_iota=args.initial_iota,
        )
        cfg = make_config(args, adapt)
        run_dir = output_dir / f"id_{int(device_id):07d}"
        result = evaluate_field_input(field_input, config=cfg, output_dir=run_dir)
        status = result_status(result)
        record = flatten_record(device_id, meta, adapt, result)
        record.update(
            {
                "serial_path": info["serial_path"],
                "nc_per_hp_loaded": info["nc_per_hp"],
                "n_total_coils_loaded": info["n_total_coils"],
                "curve_order": info["curve_order"],
            }
        )
        rows.append(record)
        run_summaries.append(
            {
                "device_id": int(device_id),
                "status": status,
                "adapt": adapt,
                "metadata": meta,
                "result": result,
                "quasr_info": info,
            }
        )
        print(
            f"ID={int(device_id):7d} status={status:>10s} "
            f"a={adapt['a']:.6g} iota0={adapt['initial_iota']:.6g} "
            f"axis_res={result['axis']['best_residual']:.3e} total={result['total_time_s']:.3f}s"
        )
        if status != "surface":
            failures.append((int(device_id), meta, field_input))

    rows.sort(key=lambda row: int(row["ID"]))
    written_failures = export_failure_cases(
        output_dir,
        failures=failures,
        quasr_root=args.quasr_root.resolve(),
        export_dir=args.export_failures_dir.resolve(),
        start_index=args.export_failure_start,
        limit=args.export_failure_limit,
    )

    batch = {
        "quasr_root": str(args.quasr_root),
        "metadata": None if args.metadata is None else str(args.metadata),
        "ids": ids,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "success_rate": 0.0 if not rows else sum(row["status"] == "surface" for row in rows) / len(rows),
        "rows": rows,
        "written_failure_files": written_failures,
        "runs": run_summaries,
    }
    write_json(output_dir / "batch_summary.json", batch)
    write_csv(output_dir / "batch_summary.csv", rows)
    write_report(output_dir / "report.md", rows, written_failures)


if __name__ == "__main__":
    main()
