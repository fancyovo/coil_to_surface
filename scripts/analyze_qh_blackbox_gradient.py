from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


K_VALUES = (1, 2, 4, 8, 16, 32, 64)
METHODS = ("g1", "g2", "g3")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_validation(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("validation must have CENTER_ID=OUTPUT_DIR form")
    return label, Path(raw_path)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float:
    if rank:
        left = rankdata(left)
        right = rankdata(right)
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def load_reference(reference_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, Any], np.ndarray, np.ndarray]:
    manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))
    cases = load_jsonl(reference_dir / "cases.jsonl")
    scored: dict[int, Any] = {}
    for path in sorted(reference_dir.glob("scores_rank_*.jsonl")):
        for row in load_jsonl(path):
            scored[int(row["case_id"])] = row["result"]
    banks = np.load(reference_dir / "latent_banks.npz")
    return manifest, cases, scored, np.asarray(banks["directions"], dtype=np.float64), np.asarray(banks["scales"], dtype=np.float64)


def reference_tables(
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    scored: dict[int, Any],
    direction_count: int,
    scales: np.ndarray,
) -> dict[str, Any]:
    case_index = {
        (
            int(row["center_index"]),
            row["kind"],
            -1 if row.get("direction_index") is None else int(row["direction_index"]),
            int(row.get("sign", 0)),
            -1 if row.get("scale_index") is None else int(row["scale_index"]),
        ): int(row["case_id"])
        for row in cases
    }
    output = {}
    for center_index, center in enumerate(manifest["centers"]):
        center_result = scored[case_index[(center_index, "center", -1, 0, -1)]]
        fingerprint = tuple(center_result["branch_fingerprint"])
        scale_rows = []
        slope_table = np.full((scales.size, direction_count), np.nan, dtype=np.float64)
        safe_table = np.zeros((scales.size, direction_count), dtype=bool)
        for scale_index, scale in enumerate(scales):
            endpoint_status = Counter()
            for direction_index in range(direction_count):
                results = [
                    scored.get(case_index[(center_index, "endpoint", direction_index, sign, scale_index)])
                    for sign in (-1, 1)
                ]
                endpoint_status.update(
                    "worker_error" if result is None else result["status"] for result in results
                )
                if all(
                    result is not None
                    and result["status"] == "ok"
                    and tuple(result["branch_fingerprint"]) == fingerprint
                    for result in results
                ):
                    safe_table[scale_index, direction_index] = True
                    slope_table[scale_index, direction_index] = (
                        float(results[1]["score"]) - float(results[0]["score"])
                    ) / (2.0 * scale)
            scale_rows.append(
                {
                    "scale": float(scale),
                    "safe_count": int(np.sum(safe_table[scale_index])),
                    "safe_fraction": float(np.mean(safe_table[scale_index])),
                    "endpoint_status_counts": dict(endpoint_status),
                    "slope_rms": float(np.sqrt(np.nanmean(slope_table[scale_index] ** 2))),
                }
            )
        convergence = []
        for scale_index in range(scales.size - 1):
            common = safe_table[scale_index] & safe_table[scale_index + 1]
            large = slope_table[scale_index, common]
            small = slope_table[scale_index + 1, common]
            denominator = np.maximum.reduce(
                [np.abs(large), np.abs(small), np.full(large.size, 1.0e-8)]
            )
            inconsistency = np.abs(large - small) / denominator
            convergence.append(
                {
                    "large_scale": float(scales[scale_index]),
                    "small_scale": float(scales[scale_index + 1]),
                    "common_safe_count": int(np.sum(common)),
                    "converged_count": int(np.sum(inconsistency <= 0.5)),
                    "converged_fraction_of_common": float(np.mean(inconsistency <= 0.5)) if large.size else float("nan"),
                    "inconsistency_median": float(np.median(inconsistency)) if large.size else float("nan"),
                    "inconsistency_p95": float(np.percentile(inconsistency, 95.0)) if large.size else float("nan"),
                }
            )
        output[center["center_id"]] = {
            "center_index": center_index,
            "recorded_score": center["recorded_score"],
            "reevaluated_score": float(center_result["score"]),
            "center_status": center_result["status"],
            "scale_rows": scale_rows,
            "scale_convergence": convergence,
            "safe_table": safe_table,
            "slope_table": slope_table,
        }
    return output


def random_k_statistics(
    slopes: np.ndarray,
    directions: np.ndarray,
    safe_mask: np.ndarray,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    safe_indices = np.flatnonzero(safe_mask)
    reference = np.mean(slopes[safe_indices, None] * directions[safe_indices], axis=0)
    rng = np.random.default_rng(seed)
    rows = []
    for k in K_VALUES:
        if k > safe_indices.size:
            continue
        values = []
        for _ in range(repeats):
            selected = rng.choice(safe_indices, size=k, replace=False)
            estimate = np.mean(slopes[selected, None] * directions[selected], axis=0)
            values.append(cosine(estimate, reference))
        values = np.asarray(values)
        rows.append(
            {
                "k": k,
                "repeats": repeats,
                "cosine_mean": float(np.mean(values)),
                "cosine_std": float(np.std(values)),
                "cosine_p10": float(np.percentile(values, 10.0)),
                "cosine_p50": float(np.percentile(values, 50.0)),
                "cosine_p90": float(np.percentile(values, 90.0)),
                "blackbox_calls": 2 * k,
            }
        )
    return rows


def equivalent_k(cosine_value: float, random_rows: list[dict[str, Any]]) -> str | int:
    if not np.isfinite(cosine_value):
        return "undefined"
    for row in random_rows:
        if row["cosine_mean"] >= cosine_value:
            return int(row["k"])
    return f">{random_rows[-1]['k']}" if random_rows else "undefined"


def analyze_validation(
    center_id: str,
    validation_dir: Path,
    center_reference: dict[str, Any],
    center_directions: np.ndarray,
    scales: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    summary = json.loads((validation_dir / "summary.json").read_text(encoding="utf-8"))
    arrays = np.load(validation_dir / "gradients.npz")
    scale_index = int(np.flatnonzero(np.isclose(scales, float(summary["scale"]), rtol=0.0, atol=1.0e-12))[0])
    safe_mask = center_reference["safe_table"][scale_index]
    slopes = center_reference["slope_table"][scale_index]
    random_rows = random_k_statistics(slopes, center_directions, safe_mask, repeats, seed)
    method_rows = {}
    observed = slopes[safe_mask]
    for method in METHODS:
        gradient = np.asarray(arrays[f"{method}_latent"], dtype=np.float64).ravel()
        predicted = center_directions[safe_mask] @ gradient
        calibration = float(np.dot(predicted, observed) / max(np.dot(predicted, predicted), 1.0e-30))
        calibrated = calibration * predicted
        top_threshold = np.percentile(observed, 75.0)
        predicted_top = predicted >= np.percentile(predicted, 75.0)
        method_cosine = cosine(predicted, observed)
        method_rows[method] = {
            "cosine": method_cosine,
            "pearson": correlation(predicted, observed),
            "spearman": correlation(predicted, observed, rank=True),
            "sign_rate": float(np.mean(np.sign(predicted) == np.sign(observed))),
            "top_quartile_hit_rate": float(np.mean(observed[predicted_top] >= top_threshold)),
            "calibration_slope": calibration,
            "calibrated_relative_rms": float(
                np.sqrt(np.mean((calibrated - observed) ** 2)) /
                max(np.sqrt(np.mean(observed ** 2)), 1.0e-30)
            ),
            "equivalent_random_k": equivalent_k(method_cosine, random_rows),
            "native_diagnostics": summary[f"{method}_native_diagnostics"],
            "flow_vjp": summary[f"{method}_flow_vjp"],
        }
    return {
        "center_id": center_id,
        "scale": float(summary["scale"]),
        "safe_count": int(np.sum(safe_mask)),
        "safe_fraction": float(np.mean(safe_mask)),
        "reference_kind": "full" if np.all(safe_mask) else "safe_subspace_projection",
        "methods": method_rows,
        "random_k": random_rows,
        "observed_slopes": observed,
        "predicted": {
            method: center_directions[safe_mask] @ np.asarray(arrays[f"{method}_latent"], dtype=np.float64).ravel()
            for method in METHODS
        },
    }


def make_plots(output_dir: Path, reference: dict[str, Any], validations: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    labels = [row["center_id"] for row in validations]
    positions = np.arange(len(labels))
    width = 0.22
    for method_index, method in enumerate(METHODS):
        axis.bar(
            positions + (method_index - 1) * width,
            [row["methods"][method]["cosine"] for row in validations],
            width,
            label=method.upper(),
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    axis.set_ylabel("Cosine with same-branch reference")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "gradient_cosine_by_center.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    for row in validations:
        axis.errorbar(
            [item["k"] for item in row["random_k"]],
            [item["cosine_mean"] for item in row["random_k"]],
            yerr=[item["cosine_std"] for item in row["random_k"]],
            marker="o",
            capsize=2,
            label=row["center_id"],
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Black-box antithetic direction count K")
    axis.set_ylabel("Cosine with reference")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "random_k_cosine.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    for center_id, row in reference.items():
        axis.plot(
            [item["scale"] for item in row["scale_rows"]],
            [item["safe_fraction"] for item in row["scale_rows"]],
            marker="o",
            label=center_id,
        )
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.set_ylim(0.0, 1.03)
    axis.set_xlabel("Latent RMS perturbation h")
    axis.set_ylabel("Same-branch direction fraction")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "same_branch_fraction.png", dpi=180)
    plt.close(figure)

    for row in validations:
        figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharex=True, sharey=True)
        observed = row["observed_slopes"]
        lower = min(float(np.min(observed)), *(float(np.min(row["predicted"][m])) for m in ("g2", "g3")))
        upper = max(float(np.max(observed)), *(float(np.max(row["predicted"][m])) for m in ("g2", "g3")))
        for axis, method in zip(axes, ("g2", "g3"), strict=True):
            axis.scatter(observed, row["predicted"][method], s=12, alpha=0.65)
            axis.plot([lower, upper], [lower, upper], color="black", linewidth=0.8)
            axis.set_title(f"{method.upper()} cosine={row['methods'][method]['cosine']:.3f}")
            axis.set_xlabel("Black-box directional slope")
        axes[0].set_ylabel("Physics-gradient prediction")
        figure.tight_layout()
        figure.savefig(output_dir / f"direction_scatter_{row['center_id']}.png", dpi=180)
        plt.close(figure)


def strip_arrays(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_arrays(item) for item in value]
    if isinstance(value, np.ndarray):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze QH black-box and physics-gradient comparisons.")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--validation", type=parse_validation, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026080429)
    args = parser.parse_args()
    if not args.validation:
        raise ValueError("at least one --validation CENTER_ID=OUTPUT_DIR is required")

    manifest, cases, scored, directions, scales = load_reference(args.reference_dir)
    reference = reference_tables(manifest, cases, scored, directions.shape[1], scales)
    validations = []
    for index, (center_id, validation_dir) in enumerate(args.validation):
        center = reference[center_id]
        validations.append(
            analyze_validation(
                center_id,
                validation_dir,
                center,
                directions[center["center_index"]],
                scales,
                args.bootstrap_repeats,
                args.seed + index,
            )
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_plots(args.output_dir, reference, validations)
    output = {
        "format": "qh_blackbox_gradient_analysis_v1",
        "reference_dir": str(args.reference_dir),
        "reference": strip_arrays(reference),
        "validations": strip_arrays(validations),
        "random_k_note": (
            "Subsets are repeatedly sampled from one frozen 200-direction orthogonal bank; "
            "they are not independent newly scored direction banks."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
