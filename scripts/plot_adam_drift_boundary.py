from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def endpoint_label(scan: dict) -> str:
    sign = "+" if scan["sign"] > 0 else "-"
    return f"step {scan['iteration']}, dir {scan['direction']}{sign}"


def first_accepted_long_drift(scan: dict) -> float:
    for item in scan["tolerance_scan"]:
        if item["status"] == "ok":
            return float(item["diagnostics"]["surface_drift_relative_p95"])
    return np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the Adam long-horizon drift audit.")
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.audit.read_text(encoding="utf-8"))
    scans = payload["drift_boundary_scan"]
    if not scans:
        raise ValueError("audit contains no drift-boundary scans")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors

    for index, scan in enumerate(scans):
        color = colors[index % len(colors)]
        label = endpoint_label(scan)
        alphas = np.asarray([item["alpha"] for item in scan["center_to_probe"]])
        scores = np.asarray([item["score"] for item in scan["center_to_probe"]])
        accepted = np.asarray(
            [item["status"] == "ok" for item in scan["center_to_probe"]]
        )
        axes[0].plot(alphas, scores, color=color, marker="o", label=label)
        axes[0].scatter(
            alphas[~accepted],
            scores[~accepted],
            color=color,
            marker="x",
            s=55,
            linewidths=2,
        )

        period_items = scan["period_scan"]
        periods = [int(item["periods"]) for item in period_items if item["status"] == "ok"]
        drifts = [
            100.0 * float(item["diagnostics"]["surface_drift_relative_p95"])
            for item in period_items
            if item["status"] == "ok"
        ]
        periods.append(16)
        drifts.append(100.0 * first_accepted_long_drift(scan))
        axes[1].plot(periods, drifts, color=color, marker="o", label=label)

    axes[0].set(
        xlabel="fraction from center to rejected probe",
        ylabel="native score",
        title="The score cliff occurs only at the acceptance boundary",
        xlim=(-0.02, 1.02),
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].axhline(5.0, color="black", linestyle="--", linewidth=1.2, label="5% limit")
    axes[1].set(
        xlabel="field periods traced",
        ylabel="relative drift P95 (%)",
        title="Drift accumulates over the long horizon",
        xticks=[1, 2, 4, 8, 16],
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
