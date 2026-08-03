from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PATH_STYLES = {
    "latent": ("-", "o"),
    "tangent_direct": ("--", "x"),
    "random_direct": (":", "+"),
}


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for row in rows:
        if row["kind"] != "landscape":
            continue
        output.append(
            {
                **row,
                "source_id": int(row["source_id"]),
                "direction": int(row["direction"]),
                "alpha": float(row["alpha"]),
                "position_delta_rms_m": float(row["position_delta_rms_m"]),
                "score": float(row["score"]),
            }
        )
    return output


def plot_alpha(
    rows: list[dict],
    source_ids: list[int],
    directions: int,
    output: Path,
    *,
    zoom: bool,
) -> None:
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, directions))
    figure, axes = plt.subplots(
        len(source_ids), 1, figsize=(11.0, 3.8 * len(source_ids)), squeeze=False
    )
    for axis, source_id in zip(axes[:, 0], source_ids, strict=True):
        selected = [row for row in rows if row["source_id"] == source_id]
        for direction, color in enumerate(colors):
            for path, (linestyle, _) in PATH_STYLES.items():
                curve = sorted(
                    [
                        row
                        for row in selected
                        if row["direction"] == direction and row["path"] == path
                    ],
                    key=lambda row: row["alpha"],
                )
                axis.plot(
                    [row["alpha"] for row in curve],
                    [row["score"] for row in curve],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    alpha=0.9,
                    label=f"d{direction} {path}" if source_id == source_ids[0] else None,
                )
        axis.axvline(0.0, color="#444444", linewidth=0.8)
        axis.set(title=f"QUASR {source_id}", xlabel="direction coordinate alpha", ylabel="score v3")
        if zoom:
            axis.set_xlim(-0.035, 0.035)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(ncol=3, fontsize=7)
    figure.suptitle(
        "QH score landscapes near the reference coils"
        if zoom
        else "QH score landscapes: latent, transported tangent, and random data directions"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_displacement(
    rows: list[dict],
    source_ids: list[int],
    directions: int,
    output: Path,
) -> None:
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, directions))
    figure, axes = plt.subplots(
        len(source_ids), 1, figsize=(11.0, 3.8 * len(source_ids)), squeeze=False
    )
    for axis, source_id in zip(axes[:, 0], source_ids, strict=True):
        selected = [row for row in rows if row["source_id"] == source_id]
        for direction, color in enumerate(colors):
            for path, (_, marker) in PATH_STYLES.items():
                curve = [
                    row
                    for row in selected
                    if row["direction"] == direction and row["path"] == path
                ]
                axis.scatter(
                    [row["position_delta_rms_m"] for row in curve],
                    [row["score"] for row in curve],
                    color=color,
                    marker=marker,
                    s=14,
                    alpha=0.65,
                )
        axis.set(
            title=f"QUASR {source_id}",
            xlabel="coil position RMS displacement [m]",
            ylabel="score v3",
        )
        axis.set_xscale("symlog", linthresh=5.0e-4)
        axis.grid(alpha=0.2)
    path_handles = [
        Line2D([], [], color="black", marker=marker, linestyle="none", label=path)
        for path, (_, marker) in PATH_STYLES.items()
    ]
    direction_handles = [
        Line2D([], [], color=color, linewidth=2.0, label=f"direction {direction}")
        for direction, color in enumerate(colors)
    ]
    axes[0, 0].legend(
        handles=path_handles + direction_handles,
        ncol=4,
        fontsize=7,
        loc="lower left",
    )
    figure.suptitle("Score retention at matched physical displacement")
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a saved QH flow landscape CSV.")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = load_rows(args.rows)
    source_ids = [int(value) for value in manifest["source_ids"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_alpha(
        rows,
        source_ids,
        int(manifest["directions"]),
        args.output_dir / "landscape_score_vs_alpha.png",
        zoom=False,
    )
    plot_alpha(
        rows,
        source_ids,
        int(manifest["directions"]),
        args.output_dir / "landscape_score_vs_alpha_zoom.png",
        zoom=True,
    )
    plot_displacement(
        rows,
        source_ids,
        int(manifest["directions"]),
        args.output_dir / "landscape_score_vs_displacement.png",
    )


if __name__ == "__main__":
    main()
