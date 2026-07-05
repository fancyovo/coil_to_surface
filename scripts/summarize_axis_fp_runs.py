from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "axis_fixed_point_summary.csv"
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fmean(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize axis fixed-point experiment CSV files.")
    parser.add_argument("run_dir", nargs="+", type=Path)
    args = parser.parse_args()
    print("| run | ok | max residual | mean total s | mean grid s | mean newton s | mean verify s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run_dir in args.run_dir:
        rows = load_rows(run_dir)
        ok = sum(row.get("has_axis") == "True" for row in rows)
        vals = [float(row["best_residual_verify"]) for row in rows if row.get("best_residual_verify")]
        max_res = max(vals) if vals else float("nan")
        print(
            f"| {run_dir.name} | {ok}/{len(rows)} | {max_res:.3g} | "
            f"{fmean(rows, 'timing_total_search_s'):.3f} | "
            f"{fmean(rows, 'timing_grid_trace_s'):.3f} | "
            f"{fmean(rows, 'timing_newton_s'):.3f} | "
            f"{fmean(rows, 'timing_verify_s'):.3f} |"
        )


if __name__ == "__main__":
    main()
