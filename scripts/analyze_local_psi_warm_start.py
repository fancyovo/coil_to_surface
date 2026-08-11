from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import struct

import matplotlib.pyplot as plt
import numpy as np


HEADER = struct.Struct("<8sIIIIQQQdQQQ")


def rhs_norm(snapshot: Path) -> tuple[int, float]:
    with snapshot.open("rb") as handle:
        values = HEADER.unpack(handle.read(HEADER.size))
        if values[0] != b"SGPUQR1\0":
            raise ValueError(f"invalid snapshot magic: {snapshot}")
        data_rows = int(values[7])
        matrix_bytes = int(values[9])
        handle.seek(HEADER.size + matrix_bytes)
        rhs = np.fromfile(handle, dtype="<f4", count=data_rows)
    if rhs.size != data_rows:
        raise ValueError(f"truncated rhs in {snapshot}")
    return data_rows, float(np.linalg.norm(rhs.astype(np.float64)))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (args.run_root / "snapshots" / "manifest.json").read_text(encoding="utf-8")
    )
    endpoint_metadata = {
        row["label"]: row for row in manifest["rows"] if row["label"] != "center"
    }
    baselines: dict[str, float] = {}
    for label, row in endpoint_metadata.items():
        data_rows, norm = rhs_norm(Path(row["snapshot"]))
        baselines[label] = float(row["psi_train_rms"]) * math.sqrt(data_rows) / norm

    raw_rows = []
    benchmark_path = args.benchmark or (args.run_root / "benchmark.jsonl")
    with benchmark_path.open(encoding="utf-8") as handle:
        for line in handle:
            raw_rows.append(json.loads(line))

    endpoint_order = [
        path.stem
        for pattern in ("direction_*_minus.bin", "direction_*_plus.bin")
        for path in sorted((args.run_root / "snapshots").glob(pattern))
    ]
    if len(raw_rows) % len(endpoint_order) != 0:
        raise ValueError(
            f"{len(raw_rows)} benchmark rows cannot be assigned to "
            f"{len(endpoint_order)} endpoints"
        )
    rows_per_endpoint = len(raw_rows) // len(endpoint_order)
    reference_methods = [row["method"] for row in raw_rows[:rows_per_endpoint]]

    rows = []
    for endpoint_index, endpoint in enumerate(endpoint_order):
        start = endpoint_index * rows_per_endpoint
        block = raw_rows[start:start + rows_per_endpoint]
        if [row["method"] for row in block] != reference_methods:
            raise ValueError(f"inconsistent iteration order for {endpoint}")
        if endpoint not in endpoint_metadata:
            raise ValueError(f"benchmark endpoint missing from manifest: {endpoint}")
        for row in block:
            match = re.search(r"(\d+)$", row["method"])
            if not match:
                raise ValueError(f"method does not end in an iteration count: {row['method']}")
            iterations = int(match.group(1))
            physical = float(row["physical_residual_relative"])
            rows.append({
                **row,
                "endpoint": endpoint,
                "iterations": iterations,
                "qr_physical_residual_relative": baselines[endpoint],
                "physical_residual_ratio": physical / baselines[endpoint],
            })

    iterations = sorted({row["iterations"] for row in rows})
    aggregate = []
    for count in iterations:
        group = [row for row in rows if row["iterations"] == count]
        aggregate.append({
            "iterations": count,
            "endpoint_count": len(group),
            "solve_ms_p50": percentile([row["solve_ms_p50"] for row in group], 50),
            "solve_ms_p95": percentile([row["solve_ms_p50"] for row in group], 95),
            "physical_residual_ratio_p50": percentile(
                [row["physical_residual_ratio"] for row in group], 50
            ),
            "physical_residual_ratio_max": max(
                row["physical_residual_ratio"] for row in group
            ),
            "coefficient_relative_error_p50": percentile(
                [row["coefficient_relative_error"] for row in group], 50
            ),
            "coefficient_relative_error_max": max(
                row["coefficient_relative_error"] for row in group
            ),
            "normal_residual_relative_p50": percentile(
                [row["normal_residual_relative"] for row in group], 50
            ),
            "normal_residual_relative_max": max(
                row["normal_residual_relative"] for row in group
            ),
        })

    accepted = [
        row for row in aggregate
        if row["physical_residual_ratio_max"] <= 1.05
        and row["coefficient_relative_error_max"] <= 0.05
    ]
    summary = {
        "format": "local_psi_warm_start_analysis_v1",
        "method_prefix": re.sub(r"\d+$", "", raw_rows[0]["method"]),
        "scale": manifest["scale"],
        "direction_count": manifest["direction_count"],
        "endpoint_count": len(endpoint_metadata),
        "qr_physical_residual_relative": baselines,
        "aggregate": aggregate,
        "first_accepted_iterations": accepted[0]["iterations"] if accepted else None,
        "acceptance": {
            "maximum_physical_residual_ratio": 1.05,
            "maximum_scaled_coefficient_relative_error": 0.05,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    x = np.asarray(iterations, dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    for endpoint in sorted(endpoint_metadata):
        group = sorted(
            (row for row in rows if row["endpoint"] == endpoint),
            key=lambda row: row["iterations"],
        )
        axes[0].plot(x, [row["physical_residual_ratio"] for row in group], alpha=0.55)
        axes[1].plot(x, [row["coefficient_relative_error"] for row in group], alpha=0.55)
    axes[0].axhline(1.05, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("physical residual / endpoint QR")
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("scaled coefficient error vs endpoint QR")
    axes[2].plot(x, [row["solve_ms_p50"] for row in aggregate], marker="o")
    axes[2].set_ylabel("warm solve time (ms)")
    for axis in axes[:2]:
        axis.set_yscale("log")
    for axis in axes:
        axis.set_xlabel("CGLS iterations")
        axis.grid(True, alpha=0.25)
    fig.suptitle(
        f"Full-grid48 {summary['method_prefix']} endpoint fits at fixed h=0.005"
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "warm_start_convergence.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
