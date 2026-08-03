from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_metrics(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n")


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_metrics(jsonl_path: str | Path, output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_metrics(jsonl_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)

    def line(axis, key: str, label: str | None = None, **kwargs) -> None:
        selected = [(row["step"], row[key]) for row in rows if key in row]
        if selected:
            axis.plot(
                [item[0] for item in selected],
                [item[1] for item in selected],
                label=label or key,
                **kwargs,
            )

    line(axes[0, 0], "train_loss", "train")
    line(axes[0, 0], "validation_loss", "validation")
    line(
        axes[0, 0],
        "validation_geometry_physical_loss",
        "validation, physical geometry",
        linestyle="--",
    )
    line(
        axes[0, 0],
        "validation_geometry_relative_loss",
        "validation, relative geometry",
        linestyle=":",
    )
    line(
        axes[0, 0],
        "validation_current_loss",
        "validation, current",
        linestyle="-.",
    )
    axes[0, 0].set(yscale="log", title="Flow matching loss", xlabel="step")
    axes[0, 0].legend()

    line(axes[0, 1], "samples_per_s", "samples/s", color="#1976a3")
    axes[0, 1].set(title="Training throughput", xlabel="step")

    line(axes[1, 0], "grad_norm", "gradient norm", color="#b34d2e")
    learning_axis = axes[1, 0].twinx()
    line(learning_axis, "learning_rate", "learning rate", color="#2b7a3d")
    axes[1, 0].set(title="Optimization", xlabel="step")

    line(axes[1, 1], "geometry_eligible_rate", "geometry eligible")
    line(axes[1, 1], "finite_rate", "finite")
    line(axes[1, 1], "score_ok_rate", "native score ok")
    axes[1, 1].set(title="Generated validity", xlabel="step", ylim=(-0.02, 1.02))
    axes[1, 1].legend()

    line(axes[2, 0], "length_mean_median", "length")
    line(axes[2, 0], "curvature_p95_median", "curvature P95")
    line(axes[2, 0], "high_mode_fraction_median", "high-mode fraction")
    axes[2, 0].set(title="Generated geometry", xlabel="step", yscale="log")
    axes[2, 0].legend()

    line(axes[2, 1], "score_mean_all", "score, all candidates")
    line(axes[2, 1], "score_mean_ok", "score, status=ok")
    line(axes[2, 1], "score_p90", "score P90")
    axes[2, 1].set(title="Sparse native QH score", xlabel="step", ylim=(0, 100))
    if axes[2, 1].has_data():
        axes[2, 1].legend()

    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    figure.savefig(temporary, dpi=150)
    plt.close(figure)
    temporary.replace(output)
