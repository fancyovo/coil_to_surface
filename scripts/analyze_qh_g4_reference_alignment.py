from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


COMPONENTS = ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else float("nan")


def reconstruct(slopes: np.ndarray, directions: np.ndarray) -> np.ndarray:
    finite = np.isfinite(slopes)
    if not np.any(finite):
        return np.full(directions.shape[1], np.nan, dtype=np.float64)
    return np.sum(slopes[finite, None] * directions[finite], axis=0) / directions.shape[1]


def orthogonal_random_basis(
    rng: np.random.Generator,
    dimension: int,
    count: int,
    fixed: np.ndarray | None = None,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    if fixed is not None:
        rows.append(np.asarray(fixed, dtype=np.float64) / rms(fixed))
    while len(rows) < count + (fixed is not None):
        candidate = rng.standard_normal(dimension)
        for row in rows:
            candidate -= np.dot(candidate, row) / dimension * row
        scale = rms(candidate)
        if scale > 1.0e-12:
            rows.append(candidate / scale)
    start = 1 if fixed is not None else 0
    selected = rows[start:]
    if not selected:
        return np.empty((0, dimension), dtype=np.float64)
    return np.asarray(selected, dtype=np.float64)


def exact_subspace_projection(full: np.ndarray, basis: np.ndarray) -> np.ndarray:
    dimension = full.size
    slopes = basis @ full
    return np.sum(slopes[:, None] * basis, axis=0) / dimension


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p05": float(np.percentile(array, 5.0)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze G4 alignment and G3-informed subspaces.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026080503)
    parser.add_argument("--random-directions", default="0,1,2,4,8,16")
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("trials must be positive")
    random_counts = tuple(int(value) for value in args.random_directions.split(","))
    if not random_counts or any(value < 0 for value in random_counts):
        raise ValueError("random direction counts must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    alignment_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for step in (50, 89, 100, 120):
        directory = args.input_root / f"step_{step:04d}"
        metadata = read_json(directory / "summary.json")
        data = np.load(directory / "directional_slopes.npz")
        directions = np.asarray(data["directions"], dtype=np.float64)
        full = reconstruct(np.asarray(data["full"], dtype=np.float64), directions)
        g2 = reconstruct(np.asarray(data["g2"], dtype=np.float64), directions)
        g3 = reconstruct(np.asarray(data["g3"], dtype=np.float64), directions)
        g4 = reconstruct(np.asarray(data["g4"], dtype=np.float64), directions)
        state = read_json(args.trajectory / f"step_{step:04d}.json")
        next_state = read_json(args.trajectory / f"step_{step + 1:04d}.json")
        adam = np.asarray(next_state["noise"], dtype=np.float64).ravel() - np.asarray(
            state["noise"], dtype=np.float64
        ).ravel()
        adam /= rms(adam)
        alignment = {
            "step": step,
            "score": float(metadata["center_score"]),
            "valid_g4_directions": int(metadata["valid_direction_count"]),
            "full_g2_cosine": cosine(full, g2),
            "full_g3_cosine": cosine(full, g3),
            "full_g4_cosine": cosine(full, g4),
            "full_on_adam": float(np.dot(full, adam)),
            "g2_on_adam": float(np.dot(g2, adam)),
            "g3_on_adam": float(np.dot(g3, adam)),
            "g4_on_adam": float(np.dot(g4, adam)),
            "full_rms": rms(full),
            "g2_rms": rms(g2),
            "g3_rms": rms(g3),
            "g4_projected_rms": rms(g4),
            "g4_batch_wall_s": float(metadata["batch_wall_s"]),
            "g4_query_mean_wall_s": float(metadata["mean_query_wall_s"]),
        }
        alignment_rows.append(alignment)

        names = [str(value) for value in data["component_names"]]
        g4_components = np.asarray(data["g4_components"], dtype=np.float64)
        for component in COMPONENTS:
            index = names.index(component)
            full_component = metadata["component_alignment"][component]
            g4_component = reconstruct(g4_components[:, index], directions)
            component_rows.append(
                {
                    "step": step,
                    "component": component,
                    "full_g4_cosine": float(full_component["full_g4_cosine"]),
                    "full_on_adam": float(full_component["full_on_adam"]),
                    "g4_on_adam": float(np.dot(g4_component, adam)),
                    "full_gradient_rms": float(full_component["full_gradient_rms"]),
                    "g4_projected_rms": rms(g4_component),
                }
            )

        e0 = g3 / rms(g3)
        for random_count in random_counts:
            informed_values: list[float] = []
            pure_values: list[float] = []
            for _ in range(args.trials):
                random_informed = orthogonal_random_basis(
                    rng, full.size, random_count, fixed=e0
                )
                informed_basis = np.concatenate((e0[None], random_informed), axis=0)
                informed = exact_subspace_projection(full, informed_basis)
                pure_basis = orthogonal_random_basis(
                    rng, full.size, random_count + 1
                )
                pure = exact_subspace_projection(full, pure_basis)
                informed_values.append(cosine(full, informed))
                pure_values.append(cosine(full, pure))
            informed_summary = summarize(informed_values)
            pure_summary = summarize(pure_values)
            subspace_rows.append(
                {
                    "step": step,
                    "random_direction_count": random_count,
                    "total_directions": random_count + 1,
                    **{f"informed_{key}": value for key, value in informed_summary.items()},
                    **{f"pure_{key}": value for key, value in pure_summary.items()},
                }
            )
        summaries.append({"step": step, "alignment": alignment})

    write_csv(args.output_dir / "gradient_alignment.csv", alignment_rows)
    write_csv(args.output_dir / "component_alignment.csv", component_rows)
    write_csv(args.output_dir / "subspace_simulation.csv", subspace_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([row["step"] for row in alignment_rows])
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for key, label in (
        ("full_g2_cosine", "G2"),
        ("full_g3_cosine", "G3"),
        ("full_g4_cosine", "fixed-axis G4"),
    ):
        axes[0, 0].plot(steps, [row[key] for row in alignment_rows], marker="o", label=label)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set(title="Alignment with full ABI-9 gradient", xlabel="saved step", ylabel="cosine")
    axes[0, 0].legend()

    for key, label in (
        ("full_on_adam", "full"),
        ("g3_on_adam", "G3"),
        ("g4_on_adam", "fixed-axis G4"),
    ):
        axes[0, 1].plot(steps, [row[key] for row in alignment_rows], marker="o", label=label)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set(title="Slope along recorded Adam update", xlabel="saved step", ylabel="score / latent RMS")
    axes[0, 1].legend()

    image = np.asarray(
        [
            [
                next(
                    row["full_g4_cosine"]
                    for row in component_rows
                    if row["step"] == step and row["component"] == component
                )
                for component in COMPONENTS
            ]
            for step in steps
        ],
        dtype=np.float64,
    )
    masked = np.ma.masked_invalid(image)
    rendered = axes[1, 0].imshow(masked, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    axes[1, 0].set_xticks(range(len(COMPONENTS)), COMPONENTS, rotation=35, ha="right")
    axes[1, 0].set_yticks(range(len(steps)), [str(value) for value in steps])
    axes[1, 0].set(title="Fixed-axis G4 component cosine", ylabel="saved step")
    figure.colorbar(rendered, ax=axes[1, 0], label="cosine")

    for step in steps:
        selected = [row for row in subspace_rows if row["step"] == step]
        axes[1, 1].plot(
            [row["total_directions"] for row in selected],
            [row["informed_mean"] for row in selected],
            marker="o",
            label=f"step {step}",
        )
    axes[1, 1].set(
        title="Exact G3-informed subspace",
        xlabel="full-score secant directions",
        ylabel="mean cosine",
        xscale="log",
    )
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.savefig(args.output_dir / "g4_and_subspace_alignment.png", dpi=180)
    plt.close(figure)

    output = {
        "format": "qh_g4_reference_alignment_analysis_v1",
        "trial_count": int(args.trials),
        "seed": int(args.seed),
        "random_direction_counts": list(random_counts),
        "alignments": alignment_rows,
        "subspace_simulation": subspace_rows,
    }
    write_json(args.output_dir / "analysis.json", output)


if __name__ == "__main__":
    main()
