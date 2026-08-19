from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GRIDS = (64, 96, 128)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def distribution(values) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    return {
        "count": int(len(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check strict equal-s QH convergence against the 128x128 grid.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case_dir in sorted(path for path in (args.experiment_root / "cases").iterdir() if path.is_dir()):
        payloads = {}
        for grid in GRIDS:
            path = case_dir / "face_qs" / f"equal_s_qs_grid{grid}.json"
            if path.is_file():
                payload = read_json(path)
                if payload.get("status") == "ok":
                    payloads[grid] = payload
        if len(payloads) != len(GRIDS):
            continue
        by_grid = {grid: {row["name"]: row for row in payloads[grid]["surfaces"]} for grid in GRIDS}
        for surface_name in ("fixed_probe", "adaptive_edge"):
            reference = by_grid[128][surface_name]["metrics"]["per_helicity_area_rms"]
            row = {
                "case_id": case_dir.name,
                "surface_name": surface_name,
                "qh_128": float(reference),
            }
            for grid in (64, 96):
                value = float(by_grid[grid][surface_name]["metrics"]["per_helicity_area_rms"])
                row[f"qh_{grid}"] = value
                row[f"relative_error_{grid}_vs_128"] = abs(value - reference) / max(abs(reference), 1e-300)
            rows.append(row)
    if not rows:
        raise RuntimeError("no cases have all requested grid outputs")
    summary = {
        "format": "qh_equal_s_grid_convergence_v1",
        "case_count": len({row["case_id"] for row in rows}),
        "surface_count": len(rows),
        "relative_error": {
            str(grid): distribution(row[f"relative_error_{grid}_vs_128"] for row in rows)
            for grid in (64, 96)
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "equal_s_grid_convergence.json", summary)
    figure, axis = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    for grid, color in ((64, "#176b87"), (96, "#d18b2c")):
        x = np.asarray([row["qh_128"] for row in rows])
        y = np.asarray([row[f"relative_error_{grid}_vs_128"] for row in rows])
        axis.scatter(x, y, s=28, alpha=0.72, color=color, label=f"{grid}x{grid} vs 128x128")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("128x128 strict equal-s differential QH")
    axis.set_ylabel("relative discretization difference")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(args.output_dir / "equal_s_grid_convergence.png", dpi=190)
    plt.close(figure)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
