from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_cem_candidate_full import (
    render_boozer_and_geometry,
    run_desc_boundary_solve,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render and run DESC for an independently validated Boozer surface."
    )
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--surface-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-unit", default="A")
    parser.add_argument("--desc-resolution", type=int, default=8)
    parser.add_argument("--desc-maxiter", type=int, default=100)
    parser.add_argument("--desc-ftol", type=float, default=1.0e-8)
    args = parser.parse_args()

    args.case_file = args.case_file.resolve()
    args.surface_npz = args.surface_npz.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "full_summary.json"
    progress_path = args.output_dir / "progress.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            raise FileExistsError(f"output already complete: {args.output_dir}")

    with np.load(args.surface_npz) as saved:
        surface_order = int(saved["order"])
        surface_meta = {
            key: np.asarray(saved[key]).item()
            for key in ("nfp", "order", "iota", "G", "rho", "s_edge", "s_level")
            if key in saved
        }

    started = time.perf_counter()
    result = {
        "case_file": str(args.case_file),
        "surface_npz": str(args.surface_npz),
        "surface": surface_meta,
    }
    try:
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("surface_npz") != str(args.surface_npz):
                raise ValueError("saved progress belongs to a different surface")
            result["visualization"] = progress["visualization"]
            result["visualization_time_s"] = progress["visualization_time_s"]
            result["visualization_reused"] = True
        else:
            stage_started = time.perf_counter()
            result["visualization"] = render_boozer_and_geometry(
                case_file=args.case_file,
                surface_npz=args.surface_npz,
                output_dir=args.output_dir / "assets",
                current_unit=args.current_unit,
                surface_order=surface_order,
            )
            result["visualization_time_s"] = time.perf_counter() - stage_started
            write_json(progress_path, result)

        stage_started = time.perf_counter()
        result["desc"] = run_desc_boundary_solve(
            case_file=args.case_file,
            surface_npz=args.surface_npz,
            output_dir=args.output_dir / "desc",
            current_unit=args.current_unit,
            surface_order=surface_order,
            resolution=args.desc_resolution,
            maxiter=args.desc_maxiter,
            ftol=args.desc_ftol,
        )
        result["desc_time_s"] = time.perf_counter() - stage_started
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        raise
    finally:
        result["total_time_s"] = time.perf_counter() - started
        write_json(summary_path, result)
        print(json.dumps({"status": result["status"], "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
