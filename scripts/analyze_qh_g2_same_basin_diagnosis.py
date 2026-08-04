from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")
COMPONENT_WEIGHTS = np.asarray((10.0, 10.0, 10.0, 10.0, 42.0, 10.0, 8.0)) / 100.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def normalized(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array / rms(array)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else float("nan")


def projection(gradient: np.ndarray, direction: np.ndarray) -> float:
    return float(np.sum(np.asarray(gradient, dtype=np.float64) * direction))


def trajectory_noise(trajectory: Path, step: int) -> np.ndarray:
    payload = read_json(trajectory / f"step_{step:04d}.json")
    return np.asarray(payload["noise"], dtype=np.float64)


def pair_at(summary: dict[str, Any], name: str, scale: float) -> dict[str, Any]:
    return next(
        row
        for row in summary["pairs"]
        if row["name"] == name and math.isclose(float(row["scale"]), scale)
    )


def size_factor(inverse_aspect_ratio: float) -> float:
    x = float(np.clip(inverse_aspect_ratio / 0.03, 0.0, 1.0))
    size_score = x * x * (3.0 - 2.0 * x)
    return 0.65 + 0.35 * size_score


def direct_adam_rows(probe_root: Path, step: int, nfp: int) -> list[dict[str, float]]:
    path = probe_root / f"step_{step:04d}" / "summary.json"
    if not path.is_file():
        return []
    payload = read_json(path)
    grouped: dict[float, dict[int, dict[str, Any]]] = {}
    for row in payload["rows"]:
        if row["method"] != "adam":
            continue
        grouped.setdefault(float(row["step"]), {})[int(row["sign"])] = row["result"]
    output = []
    for scale, endpoints in sorted(grouped.items()):
        minus = endpoints[-1]
        plus = endpoints[1]

        def slope(getter) -> float:
            return (float(getter(plus)) - float(getter(minus))) / (2.0 * scale)

        component_slopes = {
            name: slope(lambda result, key=name: result["components"][key])
            for name in COMPONENTS
        }
        minus_size = size_factor(float(minus["diagnostics"]["surface_inverse_aspect_ratio"]))
        plus_size = size_factor(float(plus["diagnostics"]["surface_inverse_aspect_ratio"]))
        output.append(
            {
                "step": step,
                "scale": scale,
                "score_slope": slope(lambda result: result["score"]),
                **{f"{name}_slope": value for name, value in component_slopes.items()},
                "target_error_slope": slope(
                    lambda result: result["diagnostics"]["qs_global_error"]
                    / math.hypot(1.0, float(nfp))
                ),
                "qa_error_slope": slope(lambda result: result["diagnostics"]["qs_qa_global_error"]),
                "qp_error_slope": slope(lambda result: result["diagnostics"]["qs_qp_global_error"]),
                "iota_slope": slope(lambda result: result["diagnostics"]["iota_min"]),
                "size_factor_slope": (plus_size - minus_size) / (2.0 * scale),
                "residual_component_slope": (
                    float(plus["components"]["volume_qs"]) / plus_size
                    - float(minus["components"]["volume_qs"]) / minus_size
                )
                / (2.0 * scale),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze same-basin G2/G3 closure and full gradients.")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--closure", type=Path, action="append", required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    closures = {int(payload["iteration"]): payload for payload in map(read_json, args.closure)}
    reference_summary = read_json(args.reference_dir / "summary.json")
    reference = np.load(args.reference_dir / "reference_gradients.npz")
    full_gradients = np.asarray(reference["gradients"], dtype=np.float64)
    component_gradients = np.asarray(reference["component_gradients"], dtype=np.float64)
    component_names = [str(value) for value in reference["component_names"]]
    scales = np.asarray(reference["scales"], dtype=np.float64)
    component_indices = [component_names.index(name) for name in COMPONENTS]

    gradient_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, float]] = []
    for center_index, center in enumerate(reference_summary["centers"]):
        step = int(center["center_id"].rsplit("step", 1)[1])
        closure = closures[step]
        nfp = int(closure["nfp"])
        g2 = np.asarray(closure["latent_gradients"]["g2"], dtype=np.float64)
        g3 = np.asarray(closure["latent_gradients"]["g3"], dtype=np.float64)
        adam = normalized(trajectory_noise(args.trajectory, step + 1) - trajectory_noise(args.trajectory, step))
        for scale_index, scale in enumerate(scales):
            full = full_gradients[center_index, scale_index]
            row = {
                "step": step,
                "scale": float(scale),
                "smooth_fraction": float(center["scales"][scale_index]["smooth_fraction"]),
                "full_gradient_rms": rms(full),
                "full_g2_cosine": cosine(full, g2),
                "full_g3_cosine": cosine(full, g3),
                "g2_g3_cosine": cosine(g2, g3),
                "full_on_adam": projection(full, adam),
                "g2_on_adam": projection(g2, adam),
                "g3_on_adam": projection(g3, adam),
                "full_on_g2": projection(full, normalized(g2)),
                "full_on_g3": projection(full, normalized(g3)),
            }
            gradient_rows.append(row)
            component_projection = []
            for component_index, name in zip(component_indices, COMPONENTS, strict=True):
                value = projection(component_gradients[center_index, scale_index, component_index], adam)
                component_projection.append(value)
                component_rows.append(
                    {
                        "step": step,
                        "scale": float(scale),
                        "component": name,
                        "component_slope_on_adam": value,
                        "weighted_score_contribution": value * COMPONENT_WEIGHTS[len(component_projection) - 1],
                    }
                )
            row["weighted_component_sum_on_adam"] = float(
                np.dot(COMPONENT_WEIGHTS, component_projection)
            )

        for scale_row in closure["scale_summaries"]:
            scale = float(scale_row["scale"])
            adam_pair = pair_at(closure, "adam", scale)
            closure_rows.append(
                {
                    "step": step,
                    "scale": scale,
                    "g2_frozen_vs_native_cosine": float(scale_row["frozen_vs_physical_cosine"]),
                    "g3_frozen_vs_native_cosine": float(scale_row["g3_frozen_vs_physical_cosine"]),
                    "flow_vjp_g2_cosine": float(scale_row["physical_vs_latent_cosine"]),
                    "flow_vjp_g3_cosine": float(scale_row["g3_physical_vs_latent_cosine"]),
                    "g2_adam_prediction": float(adam_pair["g2_prediction"]),
                    "g2_adam_frozen_slope": float(adam_pair["frozen_score_slope"]),
                    "g3_adam_prediction": float(adam_pair["g3_prediction"]),
                    "g3_adam_frozen_slope": float(adam_pair["g3_frozen_score_slope"]),
                    "g2_volume_slope": float(adam_pair["volume_qs_component_slope"]),
                    "g3_volume_slope": float(adam_pair["g3_volume_qs_component_slope"]),
                    "g2_target_error_slope": float(adam_pair["target_error_slope"]),
                    "g3_target_error_slope": float(adam_pair["g3_target_error_slope"]),
                }
            )
        direct_rows.extend(direct_adam_rows(args.probe_root, step, nfp))

    gradient_rows.sort(key=lambda row: (row["step"], row["scale"]))
    component_rows.sort(key=lambda row: (row["step"], row["scale"], row["component"]))
    closure_rows.sort(key=lambda row: (row["step"], row["scale"]))
    direct_rows.sort(key=lambda row: (row["step"], row["scale"]))
    write_csv(args.output_dir / "full_gradient_comparison.csv", gradient_rows)
    write_csv(args.output_dir / "component_directional_slopes.csv", component_rows)
    write_csv(args.output_dir / "closure_comparison.csv", closure_rows)
    write_csv(args.output_dir / "direct_adam_probes.csv", direct_rows)

    steps = sorted(closures)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    for scale in scales:
        selected = [row for row in gradient_rows if math.isclose(row["scale"], float(scale))]
        axes[0].plot(steps, [row["full_g2_cosine"] for row in selected], "o-", label=f"G2, h={scale:g}")
        axes[0].plot(steps, [row["full_g3_cosine"] for row in selected], "s--", label=f"G3, h={scale:g}")
        axes[1].plot(steps, [row["full_on_adam"] for row in selected], "o-", label=f"full, h={scale:g}")
    smallest = min(closure_rows, key=lambda row: row["scale"])["scale"]
    selected_closure = [row for row in closure_rows if math.isclose(row["scale"], smallest)]
    axes[1].plot(steps, [row["g2_adam_frozen_slope"] for row in selected_closure], "^-", label="G2 frozen")
    axes[1].plot(steps, [row["g3_adam_frozen_slope"] for row in selected_closure], "v--", label="G3 refit alpha/iota")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set(xlabel="Adam step", ylabel="cosine with full gradient", title="Same-basin gradient direction")
    axes[1].set(xlabel="Adam step", ylabel="directional score slope", title="Actual Adam proposal direction")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(args.output_dir / "same_basin_gradient_comparison.png", dpi=180)
    plt.close(fig)

    selected_scale = float(np.min(scales))
    width = 0.10
    x = np.arange(len(steps), dtype=np.float64)
    fig, axis = plt.subplots(figsize=(10.8, 4.6), constrained_layout=True)
    for component_index, name in enumerate(COMPONENTS):
        values = [
            next(
                row["weighted_score_contribution"]
                for row in component_rows
                if row["step"] == step
                and math.isclose(row["scale"], selected_scale)
                and row["component"] == name
            )
            for step in steps
        ]
        axis.bar(x + (component_index - 3) * width, values, width=width, label=name)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, [str(step) for step in steps])
    axis.set(xlabel="Adam step", ylabel="weighted contribution to full score slope", title=f"Full-score component response along Adam direction, h={selected_scale:g}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, fontsize=8)
    fig.savefig(args.output_dir / "adam_component_contributions.png", dpi=180)
    plt.close(fig)

    step = max(steps)
    comparison_scale = 0.000625
    closure = next(
        row for row in closure_rows
        if row["step"] == step and math.isclose(row["scale"], comparison_scale)
    )
    direct = next(
        row for row in direct_rows
        if row["step"] == step and math.isclose(row["scale"], comparison_scale)
    )
    labels = ("G2: fixed alpha/iota", "G3: refit alpha/iota", "full geometry")
    values = (
        (closure["g2_adam_frozen_slope"], closure["g3_adam_frozen_slope"], direct["score_slope"]),
        (closure["g2_volume_slope"], closure["g3_volume_slope"], direct["volume_qs_slope"]),
        (closure["g2_target_error_slope"], closure["g3_target_error_slope"], direct["target_error_slope"]),
    )
    titles = ("score slope", "volume-QS component slope", "target QH error slope")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    colors = ("#3b82f6", "#14b8a6", "#dc2626")
    for axis, panel, title in zip(axes, values, titles, strict=True):
        axis.bar(np.arange(3), panel, color=colors)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(3), labels, rotation=18, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Step {step}: sign reversal appears only after geometry/psi recomputation")
    fig.savefig(args.output_dir / "step120_qs_sign_reversal.png", dpi=180)
    plt.close(fig)

    write_json(
        args.output_dir / "summary.json",
        {
            "steps": steps,
            "reference_scales": scales.tolist(),
            "closure_scales": sorted({row["scale"] for row in closure_rows}),
            "full_gradient_comparison": gradient_rows,
            "closure_comparison": closure_rows,
            "direct_adam_probes": direct_rows,
        },
    )


if __name__ == "__main__":
    main()
