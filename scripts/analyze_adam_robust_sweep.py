from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def invalid_recovery(rows: list[dict[str, Any]]) -> tuple[int | None, int]:
    longest = 0
    unrecovered = 0
    for index, row in enumerate(rows):
        if float(row["valid_endpoint_fraction"]) >= 1.0:
            continue
        prior_best = max(
            [float(item["current_score"]) for item in rows[:index]],
            default=float(row["current_score"]),
        )
        recovery = next(
            (
                int(item["iteration"])
                for item in rows[index + 1 :]
                if float(item["current_score"]) >= prior_best
            ),
            None,
        )
        if recovery is None:
            unrecovered += 1
        else:
            longest = max(longest, recovery - int(row["iteration"]))
    return (None if unrecovered else longest), unrecovered


def summarize(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(run_dir / "history.jsonl")
    longest_recovery, unrecovered = invalid_recovery(rows)
    raw_gradient = np.asarray(
        [row.get("raw_gradient_rms", row["gradient_rms"]) for row in rows], dtype=float
    )
    used_gradient = np.asarray([row["gradient_rms"] for row in rows], dtype=float)
    updates = np.asarray([row["update_rms"] for row in rows], dtype=float)
    return (
        {
            "name": run_dir.name,
            "beta1": float(manifest["betas"][0]),
            "beta2": float(manifest["betas"][1]),
            "perturbation": float(manifest["perturbation"]),
            "robust_direction_filter": bool(manifest["robust_direction_filter"]),
            "initial_score": float(summary["initial_score"]),
            "best_score": float(summary["best_score"]),
            "best_iteration": int(summary["best_iteration"]),
            "final_score": float(summary["final_score"]),
            "final_drawdown": float(summary["best_score"] - summary["final_score"]),
            "maximum_running_best_drawdown": max(
                float(row["best_score"] - row["current_score"]) for row in rows
            ),
            "invalid_endpoint_steps": sum(
                float(row["valid_endpoint_fraction"]) < 1.0 for row in rows
            ),
            "filtered_invalid_directions": sum(
                sum(bool(value) for value in row.get("filtered_invalid_directions", []))
                for row in rows
            ),
            "filtered_valid_outliers": sum(
                sum(bool(value) for value in row.get("filtered_outlier_directions", []))
                for row in rows
            ),
            "maximum_raw_gradient_rms": float(np.max(raw_gradient)),
            "maximum_used_gradient_rms": float(np.max(used_gradient)),
            "median_update_rms": float(np.median(updates)),
            "longest_invalid_recovery_steps": longest_recovery,
            "unrecovered_invalid_events": unrecovered,
            "total_wall_s": float(summary["total_wall_s"]),
        },
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare robust-gradient Adam runs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    results = []
    histories = {}
    for run_dir in sorted(path for path in args.run_root.iterdir() if path.is_dir()):
        if not (run_dir / "summary.json").is_file():
            continue
        result, rows = summarize(run_dir)
        results.append(result)
        histories[run_dir.name] = rows
    if not results:
        raise ValueError(f"no completed runs under {args.run_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps({"runs": results}, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for name, rows in histories.items():
        iterations = [row["iteration"] for row in rows]
        axes[0, 0].plot(iterations, [row["current_score"] for row in rows], label=name)
        axes[0, 1].plot(
            iterations,
            [row.get("raw_gradient_rms", row["gradient_rms"]) for row in rows],
            label=f"{name} raw",
            alpha=0.45,
        )
        axes[0, 1].plot(
            iterations,
            [row["gradient_rms"] for row in rows],
            label=f"{name} used",
        )
        axes[1, 0].plot(iterations, [row["update_rms"] for row in rows], label=name)
        axes[1, 1].plot(
            iterations,
            [row["best_score"] - row["current_score"] for row in rows],
            label=name,
        )
    axes[0, 0].set(title="Current score", ylabel="native score")
    axes[0, 1].set(title="Raw and used gradient RMS", yscale="log")
    axes[1, 0].set(title="Adam update RMS", xlabel="iteration", yscale="log")
    axes[1, 1].set(title="Running-best drawdown", xlabel="iteration", ylabel="score")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.savefig(args.out_dir / "sweep_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"runs": results}, indent=2))


if __name__ == "__main__":
    main()
