from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stellarator_eval.config import EvalConfig
from stellarator_eval.field import load_case_file
from stellarator_eval.quasr import (
    choose_quasr_eval_params,
    load_quasr_field_input,
    load_quasr_metadata,
)
from stellarator_eval.serialization import write_json
from stellarator_eval.volume_pipeline import evaluate_coils_to_volume_qs


def stratified_rows(rows, helicity: int, count: int):
    eligible = []
    for row in rows:
        try:
            qs_error = float(row["qs_error"])
            if int(row["helicity"]) == helicity and math.isfinite(qs_error) and qs_error > 0.0:
                eligible.append(row)
        except (KeyError, TypeError, ValueError):
            continue
    eligible.sort(key=lambda row: float(row["qs_error"]))
    indices = np.floor(np.linspace(0, len(eligible), count, endpoint=False) + 0.5).astype(int)
    return [eligible[min(index, len(eligible) - 1)] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--quasr-root", type=Path)
    source.add_argument("--case-dir", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-helicity", type=int, default=20)
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--alpha-fit-points", type=int, default=30000)
    parser.add_argument("--alpha-order", type=int, default=12)
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_quasr_metadata(args.metadata)
    selected = stratified_rows(metadata, 0, args.per_helicity) + stratified_rows(
        metadata, 1, args.per_helicity
    )
    warmup_started = time.perf_counter()
    import torch

    warmup_a = torch.eye(32, device="cuda", dtype=torch.float32)
    warmup_b = torch.ones((32, 1), device="cuda", dtype=torch.float32)
    torch.linalg.lstsq(warmup_a, warmup_b)
    torch.cuda.synchronize()
    warmup_s = float(time.perf_counter() - warmup_started)

    rows = []
    batch_started = time.perf_counter()
    for index, metadata_row in enumerate(selected, start=1):
        device_id = int(metadata_row["ID"])
        case_dir = args.output_dir / f"id_{device_id:07d}"
        started = time.perf_counter()
        resource_exception = None
        try:
            if args.case_dir is not None:
                field_input = load_case_file(
                    args.case_dir / f"id_{device_id:07d}.json", "raw"
                )
            else:
                field_input, _ = load_quasr_field_input(args.quasr_root, device_id)
            params = choose_quasr_eval_params(metadata_row)
            base = EvalConfig(current_unit="A", omp_threads=args.threads)
            volume = replace(
                base.volume_qs,
                point_count=args.points,
                alpha_fit_point_count=args.alpha_fit_points,
                alpha_radial_order=args.alpha_order,
                alpha_poloidal_order=args.alpha_order,
                alpha_toroidal_order=args.alpha_order,
                precision=args.precision,
            )
            config = replace(
                base,
                psi=replace(base.psi, a=float(params["a"])),
                volume_qs=volume,
            )
            helicity = int(metadata_row["helicity"])
            target = (1, 0 if helicity == 0 else field_input.nfp)
            result = evaluate_coils_to_volume_qs(
                field_input,
                config,
                target_helicity=target,
                output_dir=case_dir,
            )
            volume_result = result.get("volume_qs") or {}
            target_metric = ((volume_result.get("metrics") or {}).get("target") or {}).get(
                "f_C_over_B3_rms"
            )
            iota = ((volume_result.get("alpha") or {}).get("iota_coefficients") or [None])[0]
            row = {
                "id": device_id,
                "status": result["status"],
                "helicity": helicity,
                "nfp": int(field_input.nfp),
                "metadata_qs_error": float(metadata_row["qs_error"]),
                "metadata_mean_iota": float(metadata_row["mean_iota"]),
                "s_edge": volume_result.get("s_edge"),
                "iota": iota,
                "target_f_C_over_B3_rms": target_metric,
                "total_s": float(result["timing"]["total_s"]),
                "reason": result.get("reason"),
            }
        except Exception as exc:
            resource_exception = exc
            reason = repr(exc)
            row = {
                "id": device_id,
                "status": "error",
                "helicity": int(metadata_row.get("helicity", -1)),
                "metadata_qs_error": float(metadata_row.get("qs_error", "nan")),
                "total_s": float(time.perf_counter() - started),
                "reason": reason,
            }
        rows.append(row)
        print(
            f"[{index:02d}/{len(selected):02d}] id={device_id} status={row['status']} "
            f"time={row['total_s']:.3f}s",
            flush=True,
        )
        write_json(args.output_dir / "batch_summary.json", {"rows": rows})
        resource_failure = reason.lower() if row["status"] == "error" else ""
        if any(
            marker in resource_failure
            for marker in (
                "out of memory",
                "cudaerrormemoryallocation",
                "cuda-capable device is busy",
                "all cuda-capable devices are busy",
            )
        ):
            raise RuntimeError(
                "aborting batch after a CUDA resource failure; timing run requires an idle GPU"
            ) from resource_exception

    successful = [row for row in rows if row["status"] == "ok"]
    times = np.asarray([row["total_s"] for row in successful], dtype=float)
    summary = {
        "selection": {
            "kind": "qs_error_rank_stratified_per_helicity",
            "per_helicity": int(args.per_helicity),
            "requested": len(selected),
        },
        "config": {
            "points": int(args.points),
            "alpha_fit_points": int(args.alpha_fit_points),
            "alpha_order": int(args.alpha_order),
            "precision": args.precision,
            "threads": int(args.threads),
            "one_time_cuda_warmup_s": warmup_s,
        },
        "success_count": len(successful),
        "failure_count": len(rows) - len(successful),
        "timing": {
            "batch_wall_s": float(time.perf_counter() - batch_started),
            "mean_s": float(np.mean(times)) if len(times) else None,
            "median_s": float(np.median(times)) if len(times) else None,
            "p95_s": float(np.percentile(times, 95)) if len(times) else None,
            "max_s": float(np.max(times)) if len(times) else None,
            "under_10s_fraction": float(np.mean(times < 10.0)) if len(times) else None,
        },
        "rows": rows,
    }
    write_json(args.output_dir / "batch_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("success_count", "failure_count", "timing")}, indent=2))


if __name__ == "__main__":
    main()
