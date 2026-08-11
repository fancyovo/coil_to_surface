from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


COMPONENTS = ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")


def load_rows(result_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(result_dir.glob("rank_*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    return rows


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


def finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="axis_qr16k")
    args = parser.parse_args()

    manifest = json.loads(
        (args.candidate_dir / "candidates.json").read_text(encoding="utf-8")
    )
    rows = load_rows(args.result_dir)
    lookup = {
        (
            int(row["center_index"]), int(row["direction_index"]),
            float(row["scale"]), int(row["sign"]),
        ): row
        for row in rows if row["kind"] == "endpoint"
    }
    metrics = ("score", *COMPONENTS)
    summaries = []
    gradients: dict[tuple[str, float, str, str], np.ndarray] = {}
    for center in manifest["centers"]:
        center_index = int(center["center_index"])
        label = str(center["label"])
        for scale_value in manifest["scales"]:
            scale = float(scale_value)
            banks = {
                (source, metric): []
                for source in ("exact", args.variant)
                for metric in metrics
            }
            valid = []
            for direction in range(int(manifest["direction_count"])):
                minus = lookup[(center_index, direction, scale, -1)]
                plus = lookup[(center_index, direction, scale, 1)]
                pair_ok = all(
                    endpoint["variants"][source]["status"] == "ok"
                    for endpoint in (minus, plus)
                    for source in ("exact", args.variant)
                )
                valid.append(pair_ok)
                for source in ("exact", args.variant):
                    for metric in metrics:
                        if not pair_ok:
                            banks[(source, metric)].append(float("nan"))
                            continue
                        if metric == "score":
                            lo = minus["variants"][source]["score"]
                            hi = plus["variants"][source]["score"]
                        else:
                            lo = minus["variants"][source]["components"][metric]
                            hi = plus["variants"][source]["components"][metric]
                        banks[(source, metric)].append((hi - lo) / (2.0 * scale))
            mask = np.asarray(valid, dtype=bool)
            for metric in metrics:
                exact = np.asarray(banks[("exact", metric)], dtype=np.float64)
                proxy = np.asarray(banks[(args.variant, metric)], dtype=np.float64)
                metric_mask = mask & np.isfinite(exact) & np.isfinite(proxy)
                exact_valid = exact[metric_mask]
                proxy_valid = proxy[metric_mask]
                exact_norm = np.linalg.norm(exact_valid)
                proxy_norm = np.linalg.norm(proxy_valid)
                gradients[(label, scale, "exact", metric)] = exact
                gradients[(label, scale, args.variant, metric)] = proxy
                summaries.append({
                    "center": label,
                    "scale": scale,
                    "metric": metric,
                    "valid_count": int(metric_mask.sum()),
                    "direction_count": len(mask),
                    "cosine": finite(cosine(exact_valid, proxy_valid)),
                    "norm_ratio": finite(proxy_norm / exact_norm) if exact_norm > 0.0 else None,
                    "sign_agreement": finite(float(np.mean(
                        np.sign(exact_valid) == np.sign(proxy_valid)
                    ))) if len(exact_valid) else None,
                })

    exact_scale_consistency = []
    scales = [float(value) for value in manifest["scales"]]
    for center in manifest["centers"]:
        label = str(center["label"])
        for metric in metrics:
            for left_index, left in enumerate(scales):
                for right in scales[left_index + 1:]:
                    a = gradients[(label, left, "exact", metric)]
                    b = gradients[(label, right, "exact", metric)]
                    mask = np.isfinite(a) & np.isfinite(b)
                    exact_scale_consistency.append({
                        "center": label,
                        "metric": metric,
                        "left": left,
                        "right": right,
                        "cosine": finite(cosine(a[mask], b[mask])),
                    })

    weights = {
        "axis": 0.10,
        "psi": 0.10,
        "surface": 0.10,
        "coordinate": 0.10,
        "volume_qs": 0.42,
        "iota": 0.10,
        "coil": 0.08,
    }
    score_alignment = []
    for center in manifest["centers"]:
        label = str(center["label"])
        for scale in scales:
            exact_score = gradients[(label, scale, "exact", "score")]
            proxy_volume = gradients[(label, scale, args.variant, "volume_qs")]
            proxy_coil = gradients[(label, scale, args.variant, "coil")]
            exact_volume = gradients[(label, scale, "exact", "volume_qs")]
            exact_coil = gradients[(label, scale, "exact", "coil")]
            proxy_all = sum(
                weights[metric] * gradients[(label, scale, args.variant, metric)]
                for metric in COMPONENTS
            )
            candidates = {
                "proxy_volume_qs": proxy_volume,
                "proxy_volume_qs_plus_coil": 0.42 * proxy_volume + 0.08 * proxy_coil,
                "proxy_linear_all_components": proxy_all,
                "exact_volume_qs_plus_coil_ceiling": 0.42 * exact_volume + 0.08 * exact_coil,
            }
            for name, candidate in candidates.items():
                mask = np.isfinite(exact_score) & np.isfinite(candidate)
                target = exact_score[mask]
                estimate = candidate[mask]
                target_norm = np.linalg.norm(target)
                score_alignment.append({
                    "center": label,
                    "scale": scale,
                    "estimate": name,
                    "valid_count": int(mask.sum()),
                    "cosine": finite(cosine(target, estimate)),
                    "norm_ratio": finite(np.linalg.norm(estimate) / target_norm)
                    if target_norm > 0.0 else None,
                    "sign_agreement": finite(float(np.mean(
                        np.sign(target) == np.sign(estimate)
                    ))) if len(target) else None,
                })

    output = {
        "format": "local_score_gradient_components_v1",
        "variant": args.variant,
        "summaries": summaries,
        "exact_scale_consistency": exact_scale_consistency,
        "score_alignment": score_alignment,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "component_summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(center["label"]) for center in manifest["centers"]]
    figure, axes = plt.subplots(
        len(labels), 1, figsize=(8.2, 3.2 * len(labels)), squeeze=False,
        constrained_layout=True,
    )
    for row_index, label in enumerate(labels):
        values = np.full((len(metrics), len(scales)), np.nan)
        for metric_index, metric in enumerate(metrics):
            for scale_index, scale in enumerate(scales):
                row = next(
                    item for item in summaries
                    if item["center"] == label and item["scale"] == scale
                    and item["metric"] == metric
                )
                values[metric_index, scale_index] = (
                    row["cosine"] if row["cosine"] is not None else np.nan
                )
        axis = axes[row_index, 0]
        image = axis.imshow(values, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
        axis.set_xticks(range(len(scales)), [f"h={scale:g}" for scale in scales])
        axis.set_yticks(range(len(metrics)), metrics)
        axis.set_title(f"{label}: formal versus {args.variant} gradient cosine")
        for metric_index in range(len(metrics)):
            for scale_index in range(len(scales)):
                value = values[metric_index, scale_index]
                if np.isfinite(value):
                    axis.text(
                        scale_index, metric_index, f"{value:.2f}",
                        ha="center", va="center",
                        color="white" if abs(value) > 0.55 else "black",
                        fontsize=9,
                    )
        figure.colorbar(image, ax=axis, label="cosine")
    figure.savefig(args.output_dir / "component_gradient_cosine.png", dpi=180)
    plt.close(figure)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
