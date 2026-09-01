from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.axis_surface_prior import evaluate_fourier


STATUS_ORDER = (
    "ok",
    "flux_rejected",
    "drift_rejected",
    "no_surface",
    "no_axis",
    "alpha_failed",
    "internal_error",
    "error",
)


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def statistics(values: Iterable[float]) -> dict[str, float | int | None]:
    array = finite(values)
    if not len(array):
        return {"count": 0, "mean": None, "p10": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("shard_*.jsonl")):
        with path.open("r", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    if not rows:
        raise ValueError(f"no shard JSONL files found under {input_dir}")
    case_ids = [int(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case IDs found across shards")
    return sorted(rows, key=lambda row: int(row["case_id"]))


def score(row: dict[str, Any]) -> float:
    return float(row["native"]["score"]) if row.get("native") else 0.0


def component(row: dict[str, Any], name: str) -> float:
    return float(row["native"]["components"][name]) if row.get("native") else float("nan")


def diagnostic(row: dict[str, Any], name: str) -> float:
    return float(row["native"]["diagnostics"].get(name, float("nan"))) if row.get("native") else float("nan")


def status(row: dict[str, Any]) -> str:
    return str(row["native"]["status"]) if row.get("native") else "error"


def grouped_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "status_counts": dict(Counter(status(row) for row in rows)),
        "status_ok_rate": float(np.mean([status(row) == "ok" for row in rows])),
        "score": statistics(score(row) for row in rows),
        "coil_component": statistics(component(row, "coil") for row in rows),
        "score_exceedance_rate": {
            str(threshold): float(np.mean([score(row) >= threshold for row in rows]))
            for threshold in (10, 20, 30, 40, 50)
        },
        "coil_exceedance_rate": {
            str(threshold): float(np.mean([component(row, "coil") >= threshold for row in rows]))
            for threshold in (50, 60, 70, 80, 90)
        },
        "engineering_diagnostics": {
            name: statistics(diagnostic(row, name) for row in rows)
            for name in (
                "coil_length_mean",
                "coil_curvature_p95",
                "coil_curvature_max",
                "coil_min_intercoil_distance",
                "coil_min_axis_distance",
                "coil_high_mode_energy_fraction",
                "coil_current_abs_max_a",
            )
        },
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {}
    for family in sorted({str(row["family"]) for row in rows}):
        by_family[family] = grouped_summary([row for row in rows if row["family"] == family])
    by_nc = {}
    for n_base_coils in sorted({int(row["n_base_coils"]) for row in rows}):
        by_nc[str(n_base_coils)] = grouped_summary([row for row in rows if int(row["n_base_coils"]) == n_base_coils])
    by_condition = {}
    for nfp, n_base_coils in sorted({(int(row["nfp"]), int(row["n_base_coils"])) for row in rows}):
        key = f"nfp{nfp}_nc{n_base_coils}"
        by_condition[key] = grouped_summary(
            [row for row in rows if int(row["nfp"]) == nfp and int(row["n_base_coils"]) == n_base_coils]
        )
    return {
        "format": "axis_surface_prior_analysis_v1",
        "overall": grouped_summary(rows),
        "by_family": by_family,
        "by_n_base_coils": by_nc,
        "by_condition": by_condition,
    }


def plot_distributions(rows: list[dict[str, Any]], output_path: Path) -> None:
    family_colors = {"near_circular": "#2d6a4f", "balanced": "#3d5a80", "helical": "#9c6644"}
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), constrained_layout=True)
    for family, color in family_colors.items():
        group = [row for row in rows if row["family"] == family]
        scores = np.asarray([score(row) for row in group])
        coils = np.asarray([component(row, "coil") for row in group])
        axes[0, 0].hist(scores, bins=np.linspace(0, 100, 81), histtype="step", linewidth=1.6, color=color, label=family)
        axes[0, 1].hist(coils, bins=np.linspace(0, 100, 81), histtype="step", linewidth=1.6, color=color, label=family)
        ordered = np.sort(scores)
        survival = (len(ordered) - np.arange(len(ordered))) / len(ordered)
        axes[1, 0].step(ordered, survival, where="post", color=color, linewidth=1.5, label=family)
        axes[1, 1].scatter(coils, scores, s=7, alpha=0.18, color=color, edgecolors="none", label=family)
    axes[0, 0].set(xlabel="ABI-11 total score", ylabel="samples", title="QH score distribution")
    axes[0, 1].set(xlabel="ABI-11 coil component", ylabel="samples", title="Coil-engineering distribution")
    axes[1, 0].set(xlabel="ABI-11 total score", ylabel="empirical survival", yscale="log", title="Upper-tail survival")
    axes[1, 1].set(xlabel="ABI-11 coil component", ylabel="ABI-11 total score", title="Engineering versus total score")
    for axis in axes.ravel():
        axis.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def plot_engineering(rows: list[dict[str, Any]], output_path: Path) -> None:
    families = ("near_circular", "balanced", "helical")
    metrics = (
        ("coil", "Coil component", lambda row: component(row, "coil")),
        ("curvature", "Curvature p95 [1/m]", lambda row: diagnostic(row, "coil_curvature_p95")),
        ("spacing", "Minimum intercoil distance [m]", lambda row: diagnostic(row, "coil_min_intercoil_distance")),
        ("high_mode", "High-mode energy fraction", lambda row: diagnostic(row, "coil_high_mode_energy_fraction")),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), constrained_layout=True)
    for axis, (_, label, getter) in zip(axes.ravel(), metrics, strict=True):
        values = [finite(getter(row) for row in rows if row["family"] == family) for family in families]
        axis.boxplot(values, tick_labels=["near circular", "balanced", "helical"], showfliers=False)
        axis.set(ylabel=label)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("ABI-11 engineering diagnostics by analytic-prior family")
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def physical_curves(row: dict[str, Any], samples: int = 192) -> list[tuple[np.ndarray, int]]:
    base = evaluate_fourier(np.asarray(row["tokens"], dtype=float), samples=samples)
    nfp = int(row["nfp"])
    full: list[tuple[np.ndarray, int]] = []
    for base_index, curve in enumerate(base):
        for reflected in range(2):
            transformed = curve.copy()
            if reflected:
                transformed[:, 1] *= -1.0
                transformed[:, 2] *= -1.0
            for period in range(nfp):
                angle = 2.0 * math.pi * period / nfp
                cosine = math.cos(angle)
                sine = math.sin(angle)
                rotated = transformed.copy()
                rotated[:, 0] = cosine * transformed[:, 0] - sine * transformed[:, 1]
                rotated[:, 1] = sine * transformed[:, 0] + cosine * transformed[:, 1]
                full.append((rotated, base_index))
    return full


def write_coil_html(row: dict[str, Any], output_path: Path, label: str) -> None:
    import plotly.graph_objects as go

    palette = ("#1f4e79", "#b04a3a", "#3f7d20", "#7b4f9d")
    traces = []
    for curve, base_index in physical_curves(row):
        closed = np.vstack((curve, curve[:1]))
        traces.append(
            go.Scatter3d(
                x=closed[:, 0],
                y=closed[:, 1],
                z=closed[:, 2],
                mode="lines",
                line={"color": palette[base_index % len(palette)], "width": 5},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure = go.Figure(traces)
    figure.update_layout(
        title=(
            f"{label}: case {row['case_id']} | {row['family']} | nfp={row['nfp']}, nc={row['n_base_coils']} | "
            f"score={score(row):.3f}, coil={component(row, 'coil'):.3f}"
        ),
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "camera": {"eye": {"x": 1.45, "y": -1.55, "z": 1.05}},
        },
    )
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)


def representative_rows(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    for family in ("near_circular", "balanced", "helical"):
        group = [row for row in rows if row["family"] == family]
        selected.append((f"{family}_top_score", max(group, key=score)))
        median = float(np.median([component(row, "coil") for row in group]))
        selected.append((f"{family}_median_engineering", min(group, key=lambda row: abs(component(row, "coil") - median))))
    return selected


def write_tables(rows: list[dict[str, Any]], output_dir: Path) -> None:
    with (output_dir / "condition_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("nfp", "n_base_coils", "count", "ok_rate", "score_median", "score_max", "coil_median", "coil_p10", "coil_p90"),
        )
        writer.writeheader()
        for nfp, n_base_coils in sorted({(int(row["nfp"]), int(row["n_base_coils"])) for row in rows}):
            group = [row for row in rows if int(row["nfp"]) == nfp and int(row["n_base_coils"]) == n_base_coils]
            score_stats = statistics(score(row) for row in group)
            coil_stats = statistics(component(row, "coil") for row in group)
            writer.writerow(
                {
                    "nfp": nfp,
                    "n_base_coils": n_base_coils,
                    "count": len(group),
                    "ok_rate": np.mean([status(row) == "ok" for row in group]),
                    "score_median": score_stats["median"],
                    "score_max": score_stats["max"],
                    "coil_median": coil_stats["median"],
                    "coil_p10": coil_stats["p10"],
                    "coil_p90": coil_stats["p90"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze scored axis/surface/contour-prior samples.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input_dir)
    summary = build_summary(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_distributions(rows, args.output_dir / "score_and_coil_distributions.png")
    plot_engineering(rows, args.output_dir / "engineering_diagnostics.png")
    write_tables(rows, args.output_dir)
    representatives = representative_rows(rows)
    representative_payload = []
    for label, row in representatives:
        filename = f"coils_{label}_case_{int(row['case_id']):05d}.html"
        write_coil_html(row, args.output_dir / filename, label.replace("_", " "))
        representative_payload.append(
            {
                "label": label,
                "case_id": int(row["case_id"]),
                "family": row["family"],
                "nfp": int(row["nfp"]),
                "n_base_coils": int(row["n_base_coils"]),
                "score": score(row),
                "coil_component": component(row, "coil"),
                "html": filename,
            }
        )
    (args.output_dir / "representative_samples.json").write_text(
        json.dumps(representative_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "analysis_complete", "count": len(rows), "output_dir": str(args.output_dir)}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
