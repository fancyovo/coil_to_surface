from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = ("axis_qr16k", "axis_ne16k", "axis_ne8k", "fixed_ne8k")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    output = np.empty(len(values), dtype=np.float64)
    output[order] = np.arange(len(values), dtype=np.float64)
    return output


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def load_rows(result_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(result_dir.glob("rank_*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    return sorted(rows, key=lambda row: int(row["candidate_index"]))


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(
        (args.candidate_dir / "candidates.json").read_text(encoding="utf-8")
    )
    rows = load_rows(args.result_dir)
    if len(rows) != len(manifest["candidates"]):
        raise ValueError("result rows do not cover every candidate")
    lookup = {
        (
            int(row["center_index"]), int(row["direction_index"]),
            float(row["scale"]), int(row["sign"]),
        ): row
        for row in rows
    }

    summaries = []
    slope_bank: dict[tuple[str, float, str], np.ndarray] = {}
    for center in manifest["centers"]:
        center_index = int(center["center_index"])
        label = str(center["label"])
        for scale in manifest["scales"]:
            exact_slopes = []
            variant_slopes = {name: [] for name in VARIANTS}
            valid = {name: [] for name in VARIANTS}
            for direction_index in range(int(manifest["direction_count"])):
                minus = lookup[(center_index, direction_index, float(scale), -1)]
                plus = lookup[(center_index, direction_index, float(scale), 1)]
                exact_ok = all(
                    endpoint["variants"]["exact"]["status"] == "ok"
                    for endpoint in (minus, plus)
                )
                exact_slopes.append(
                    (plus["variants"]["exact"]["score"] - minus["variants"]["exact"]["score"])
                    / (2.0 * float(scale)) if exact_ok else float("nan")
                )
                for name in VARIANTS:
                    ok = exact_ok and all(
                        endpoint["variants"][name]["status"] == "ok"
                        for endpoint in (minus, plus)
                    )
                    valid[name].append(ok)
                    variant_slopes[name].append(
                        (plus["variants"][name]["score"] - minus["variants"][name]["score"])
                        / (2.0 * float(scale)) if ok else float("nan")
                    )
            exact = np.asarray(exact_slopes, dtype=np.float64)
            slope_bank[(label, float(scale), "exact")] = exact
            for name in VARIANTS:
                proxy = np.asarray(variant_slopes[name], dtype=np.float64)
                mask = np.asarray(valid[name], dtype=bool) & np.isfinite(exact) & np.isfinite(proxy)
                slope_bank[(label, float(scale), name)] = proxy
                summaries.append(
                    {
                        "center": label,
                        "scale": float(scale),
                        "variant": name,
                        "valid_count": int(mask.sum()),
                        "direction_count": len(mask),
                        "cosine": cosine(exact[mask], proxy[mask]),
                        "pearson": correlation(exact[mask], proxy[mask]),
                        "spearman": correlation(ranks(exact[mask]), ranks(proxy[mask])),
                        "sign_agreement": float(np.mean(np.sign(exact[mask]) == np.sign(proxy[mask]))) if mask.any() else float("nan"),
                        "exact_rms": float(np.sqrt(np.mean(exact[mask] ** 2))) if mask.any() else float("nan"),
                        "proxy_rms": float(np.sqrt(np.mean(proxy[mask] ** 2))) if mask.any() else float("nan"),
                    }
                )

    timings = {}
    for name in ("exact", *VARIANTS):
        timings[name] = quantiles([
            float(row["variants"][name]["call_wall_s"])
            for row in rows if row["variants"][name]["status"] == "ok"
        ])

    scale_consistency = []
    scales = [float(value) for value in manifest["scales"]]
    for center in manifest["centers"]:
        label = str(center["label"])
        for left_index, left in enumerate(scales):
            for right in scales[left_index + 1 :]:
                a = slope_bank[(label, left, "exact")]
                b = slope_bank[(label, right, "exact")]
                mask = np.isfinite(a) & np.isfinite(b)
                scale_consistency.append(
                    {"center": label, "left": left, "right": right, "cosine": cosine(a[mask], b[mask])}
                )

    finite_summaries = [
        row for row in summaries
        if math.isfinite(row["cosine"]) and row["valid_count"] == row["direction_count"]
    ]
    aggregate = []
    for scale in scales:
        for name in VARIANTS:
            selected = [
                row for row in finite_summaries
                if row["scale"] == scale and row["variant"] == name
            ]
            if selected:
                aggregate.append(
                    {
                        "scale": scale,
                        "variant": name,
                        "minimum_cosine": float(min(row["cosine"] for row in selected)),
                        "mean_cosine": float(np.mean([row["cosine"] for row in selected])),
                        "mean_sign_agreement": float(np.mean([row["sign_agreement"] for row in selected])),
                        "timing_p50_s": timings[name]["p50"],
                    }
                )
    recommended = max(aggregate, key=lambda row: (row["minimum_cosine"], -row["timing_p50_s"])) if aggregate else None

    output = {
        "format": "local_score_gradient_calibration_v1",
        "candidate_manifest": manifest,
        "timings": timings,
        "scale_consistency": scale_consistency,
        "summaries": summaries,
        "aggregate": aggregate,
        "recommended": recommended,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(output), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_scale = float(recommended["scale"]) if recommended else scales[0]
    center_labels = [str(center["label"]) for center in manifest["centers"]]
    figure, axes = plt.subplots(
        len(center_labels), 2, figsize=(12, 4 * len(center_labels)), squeeze=False,
        constrained_layout=True,
    )
    colors = {"axis_qr16k": "#2667a8", "axis_ne16k": "#d1495b", "axis_ne8k": "#edae49", "fixed_ne8k": "#3a7d44"}
    for row_index, label in enumerate(center_labels):
        exact = slope_bank[(label, selected_scale, "exact")]
        for name in VARIANTS:
            proxy = slope_bank[(label, selected_scale, name)]
            mask = np.isfinite(exact) & np.isfinite(proxy)
            axes[row_index, 0].scatter(exact[mask], proxy[mask], s=20, alpha=0.75, color=colors[name], label=name)
        axes[row_index, 0].axhline(0.0, color="#222222", linewidth=0.6)
        axes[row_index, 0].axvline(0.0, color="#222222", linewidth=0.6)
        axes[row_index, 0].set(title=f"{label}: directional slopes, h={selected_scale:g}", xlabel="formal score slope", ylabel="proxy slope")
        axes[row_index, 0].legend(fontsize=8)

        selected_rows = [row for row in summaries if row["center"] == label and row["scale"] == selected_scale]
        for summary in selected_rows:
            name = summary["variant"]
            axes[row_index, 1].scatter(timings[name]["p50"], summary["cosine"], s=70, color=colors[name], label=name)
        axes[row_index, 1].set(title=f"{label}: accuracy versus query latency", xlabel="P50 query wall time (s)", ylabel="gradient cosine", ylim=(-1.05, 1.05))
        axes[row_index, 1].legend(fontsize=8)
        for axis in axes[row_index]:
            axis.grid(alpha=0.22)
    figure.savefig(args.output_dir / "gradient_accuracy_timing.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"timings": timings, "aggregate": aggregate, "recommended": recommended}, indent=2))


if __name__ == "__main__":
    main()
