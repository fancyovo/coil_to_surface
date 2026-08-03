from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_convergence(qh: dict, qa: dict, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors = {"QH": "#c43c2f", "QA": "#177e89"}
    for label, summary in (("QH", qh), ("QA", qa)):
        generations = summary["generations"]
        x = np.asarray([row["generation"] for row in generations])
        best = np.asarray([row["best_score"] for row in generations])
        median = np.asarray([row["score_median"] for row in generations])
        ok = np.asarray([row["statuses"].get("ok", 0) for row in generations])
        axes[0].plot(x, best, "o-", color=colors[label], linewidth=2.0, label=f"{label} best")
        axes[0].plot(
            x,
            median,
            "--",
            color=colors[label],
            linewidth=1.25,
            alpha=0.65,
            label=f"{label} median",
        )
        axes[1].plot(x, ok, "o-", color=colors[label], linewidth=2.0, label=label)

    axes[0].axhline(70.89, color="#555555", linestyle=":", linewidth=1.4, label="QUASR P90")
    axes[0].axhline(78.02, color="#999999", linestyle=":", linewidth=1.2, label="QUASR max")
    axes[0].set_xlabel("CEM generation")
    axes[0].set_ylabel("native score")
    axes[0].set_xticks(range(1, 9))
    axes[0].set_ylim(0, 82)
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_xlabel("CEM generation")
    axes[1].set_ylabel("valid candidates / 32")
    axes[1].set_xticks(range(1, 9))
    axes[1].set_ylim(0, 32)
    axes[1].grid(alpha=0.22)
    axes[1].legend()
    figure.suptitle("Random-start diagonal CEM optimization")
    figure.tight_layout()
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_surface_sweep(full: dict, output_path: Path) -> None:
    rows = [row for row in full["rows"] if row.get("best_surface")]
    a_values = np.asarray([row["a"] for row in rows])
    volumes = np.asarray([row["best_surface"]["volume"] for row in rows])
    qh_percent = 100.0 * np.asarray(
        [row["best_surface"]["qs_error_QH_1_1"] for row in rows]
    )
    psi_levels = np.asarray([row["best_surface"]["psi_level"] for row in rows])

    figure, axis_volume = plt.subplots(figsize=(7.4, 4.6))
    axis_qs = axis_volume.twinx()
    axis_volume.plot(a_values, volumes, "o-", color="#177e89", linewidth=2.2, label="volume")
    axis_qs.plot(a_values, qh_percent, "s-", color="#c43c2f", linewidth=2.2, label="QH error")
    for a_value, volume, psi_level in zip(a_values, volumes, psi_levels, strict=True):
        axis_volume.annotate(
            f"psi={psi_level:g}",
            (a_value, volume),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#145d64",
        )
    axis_volume.set_xlabel("psi fit radius parameter a [m]")
    axis_volume.set_ylabel("largest solved surface volume [m$^3$]", color="#177e89")
    axis_qs.set_ylabel("Simsopt QH error [%]", color="#c43c2f")
    axis_volume.tick_params(axis="y", labelcolor="#177e89")
    axis_qs.tick_params(axis="y", labelcolor="#c43c2f")
    axis_volume.grid(alpha=0.22)
    lines = axis_volume.lines + axis_qs.lines
    axis_volume.legend(lines, [line.get_label() for line in lines], loc="upper left")
    figure.suptitle("Stable Boozer LS/Newton cross-check")
    figure.tight_layout()
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot summary figures for native-score CEM validation.")
    parser.add_argument("--qh-summary", type=Path, required=True)
    parser.add_argument("--qa-summary", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qh = load_json(args.qh_summary)
    qa = load_json(args.qa_summary)
    full = load_json(args.full_summary)
    plot_convergence(qh, qa, args.output_dir / "cem_convergence.png")
    plot_surface_sweep(full, args.output_dir / "surface_sweep.png")


if __name__ == "__main__":
    main()
