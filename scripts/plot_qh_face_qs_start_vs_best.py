from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def usable(row: dict[str, str]) -> bool:
    try:
        value = float(row["face_qh"])
    except (KeyError, TypeError, ValueError):
        return False
    return parse_bool(row.get("accepted", False)) and math.isfinite(value) and value > 0.0


def build_pairs(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_trajectory: dict[str, dict[int, dict[str, str]]] = {}
    for row in rows:
        if row.get("surface_name") != "fixed_probe":
            continue
        by_trajectory.setdefault(row["trajectory_id"], {})[int(row["iteration"])] = row

    pairs: list[dict[str, Any]] = []
    for trajectory_id, stages in sorted(by_trajectory.items()):
        start = stages.get(0)
        if start is None or not usable(start):
            continue
        later = [row for iteration, row in stages.items() if iteration > 0 and usable(row)]
        endpoint = stages.get(200)
        pairs.append(
            {
                "trajectory_id": trajectory_id,
                "nfp": int(start["nfp"]),
                "n_base_coils": int(start["n_base_coils"]),
                "start_face_qh": float(start["face_qh"]),
                "endpoint_face_qh": (
                    float(endpoint["face_qh"])
                    if endpoint is not None and usable(endpoint)
                    else None
                ),
                "best_observed_later_face_qh": (
                    min(float(row["face_qh"]) for row in later) if later else None
                ),
                "best_observed_later_iteration": (
                    min(later, key=lambda row: float(row["face_qh"]))["iteration"]
                    if later
                    else None
                ),
            }
        )
    return pairs


def pair_summary(pairs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    usable_pairs = [row for row in pairs if row[key] is not None]
    ratios = np.asarray(
        [row[key] / row["start_face_qh"] for row in usable_pairs], dtype=float
    )
    return {
        "count": len(usable_pairs),
        "improved_count": int(np.sum(ratios < 1.0)),
        "improved_fraction": float(np.mean(ratios < 1.0)) if ratios.size else None,
        "ratio_p10": float(np.percentile(ratios, 10)) if ratios.size else None,
        "ratio_p50": float(np.percentile(ratios, 50)) if ratios.size else None,
        "ratio_p90": float(np.percentile(ratios, 90)) if ratios.size else None,
    }


def plot_panel(axis: Any, pairs: list[dict[str, Any]], key: str, title: str) -> None:
    usable_pairs = [row for row in pairs if row[key] is not None]
    x = np.asarray([row["start_face_qh"] for row in usable_pairs], dtype=float)
    y = np.asarray([row[key] for row in usable_pairs], dtype=float)
    colors = np.asarray([row["nfp"] for row in usable_pairs], dtype=float)
    axis.scatter(x, y, c=colors, cmap="viridis", s=29, alpha=0.72, linewidths=0)
    lower = min(np.min(x), np.min(y)) / 1.5
    upper = max(np.max(x), np.max(y)) * 1.5
    axis.plot([lower, upper], [lower, upper], "--", color="#222222", linewidth=1.2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.grid(True, which="both", alpha=0.18)
    axis.set(
        xlabel="Initial Simsopt face QH",
        ylabel="Later Simsopt face QH",
        title=title,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot initial versus later fixed-probe Simsopt face QH."
    )
    parser.add_argument("--surface-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.surface_records.open(encoding="utf-8", newline="") as stream:
        pairs = build_pairs(list(csv.DictReader(stream)))
    if not pairs:
        raise RuntimeError("no accepted initial fixed-probe surfaces were found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "face_qh_start_vs_later.csv", pairs)
    summary = {
        "format": "qh_face_qs_start_vs_later_v1",
        "source": str(args.surface_records.resolve()),
        "trajectory_count_with_accepted_start": len(pairs),
        "endpoint_iteration_200": pair_summary(pairs, "endpoint_face_qh"),
        "best_observed_later": {
            **pair_summary(pairs, "best_observed_later_face_qh"),
            "definition": (
                "minimum accepted fixed-probe face QH among the seven measured "
                "post-start stages 10,25,50,75,100,150,200"
            ),
        },
        "limitation": (
            "The historical face-QS experiment did not evaluate each optimizer's "
            "exact score-best iteration."
        ),
    }
    (args.output_dir / "face_qh_start_vs_later_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), constrained_layout=True)
    plot_panel(
        axes[0],
        pairs,
        "endpoint_face_qh",
        f"Iteration 200 (n={summary['endpoint_iteration_200']['count']})",
    )
    plot_panel(
        axes[1],
        pairs,
        "best_observed_later_face_qh",
        f"Best of 7 measured later stages (n={summary['best_observed_later']['count']})",
    )
    colorbar = figure.colorbar(axes[1].collections[0], ax=axes, shrink=0.82)
    colorbar.set_label("NFP")
    figure.savefig(args.output_dir / "face_qh_start_vs_later.png", dpi=200)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
