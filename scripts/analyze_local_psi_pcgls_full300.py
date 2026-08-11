from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_gradient_proxy import (
    GRADIENT_OMITTED_COMPONENT,
    SCORE_COMPONENTS,
    SCORE_WEIGHTS,
    coordinate_omitted_gradient_score,
)


COMPONENTS = SCORE_COMPONENTS


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


def gradient_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    error_norm = float(np.linalg.norm(candidate - reference))
    return {
        "cosine": cosine(reference, candidate),
        "reference_norm": reference_norm,
        "candidate_norm": candidate_norm,
        "norm_ratio": candidate_norm / reference_norm if reference_norm > 0.0 else float("nan"),
        "relative_error": error_norm / reference_norm if reference_norm > 0.0 else float("nan"),
    }


def gradients(rows: list[dict], variant: str, field: str, scale: float) -> np.ndarray:
    by_direction: dict[int, dict[int, dict]] = {}
    for row in rows:
        metadata = row["metadata"]
        by_direction.setdefault(int(metadata["direction_index"]), {})[int(metadata["sign"])] = row["variants"][variant]
    output = []
    for direction in sorted(by_direction):
        pair = by_direction[direction]
        if field == "score":
            minus, plus = pair[-1]["score"], pair[1]["score"]
        else:
            minus = pair[-1]["components"][field]
            plus = pair[1]["components"][field]
        output.append((plus - minus) / (2.0 * scale))
    return np.asarray(output, dtype=np.float64)


def gradient_proxy_score(result: dict) -> float:
    return float(result.get(
        "gradient_proxy_score",
        coordinate_omitted_gradient_score(result["score"], result["components"]),
    ))


def component_contribution(result: dict, component: str) -> float:
    full_weight = sum(SCORE_WEIGHTS.values())
    full_average = sum(
        SCORE_WEIGHTS[name] * result["components"][name] for name in COMPONENTS
    ) / full_weight
    gate = result["score"] / full_average if full_average > 0.0 else 0.0
    return gate * SCORE_WEIGHTS[component] * result["components"][component] / full_weight


def derived_gradients(rows: list[dict], variant: str, evaluator, scale: float) -> np.ndarray:
    by_direction: dict[int, dict[int, dict]] = {}
    for row in rows:
        metadata = row["metadata"]
        by_direction.setdefault(int(metadata["direction_index"]), {})[
            int(metadata["sign"])
        ] = row["variants"][variant]
    return np.asarray([
        (evaluator(by_direction[direction][1]) - evaluator(by_direction[direction][-1])) /
        (2.0 * scale)
        for direction in sorted(by_direction)
    ], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.005)
    parser.add_argument("--population-jsonl", type=Path)
    parser.add_argument("--population-variant", default="iota_cubic")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.input_dir.glob("rank_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle)
    if len(rows) != 600:
        raise ValueError(f"expected 600 endpoint rows, found {len(rows)}")
    endpoint_keys = {
        (int(row["metadata"]["direction_index"]), int(row["metadata"]["sign"]))
        for row in rows
    }
    expected_keys = {(direction, sign) for direction in range(300) for sign in (-1, 1)}
    if endpoint_keys != expected_keys:
        missing = sorted(expected_keys - endpoint_keys)
        extra = sorted(endpoint_keys - expected_keys)
        raise ValueError(f"incomplete endpoint coverage: missing={missing[:8]}, extra={extra[:8]}")
    variants = sorted(
        (name for name in rows[0]["variants"] if name.startswith("pcgls")),
        key=lambda name: int(name.removeprefix("pcgls")),
    )
    exact_gradient = gradients(rows, "exact", "score", args.scale)
    exact_without_coordinate = derived_gradients(
        rows, "exact", gradient_proxy_score, args.scale
    )
    exact_coordinate_contribution = derived_gradients(
        rows, "exact", lambda result: component_contribution(result, "coordinate"), args.scale
    )
    exact_other_contribution = exact_gradient - exact_coordinate_contribution
    summary = []
    component_cosines = {}
    coordinate_ablation = {
        "exact_without_coordinate_vs_full": gradient_comparison(
            exact_gradient, exact_without_coordinate
        ),
        "exact_coordinate_contribution": {
            **gradient_comparison(exact_gradient, exact_coordinate_contribution),
            "norm_fraction_of_full": float(
                np.linalg.norm(exact_coordinate_contribution) / np.linalg.norm(exact_gradient)
            ),
        },
        "exact_other_contribution": gradient_comparison(exact_gradient, exact_other_contribution),
        "variants": {},
    }
    for variant in variants:
        gradient = gradients(rows, variant, "score", args.scale)
        without_coordinate = derived_gradients(
            rows, variant, gradient_proxy_score, args.scale,
        )
        exact_scores = np.asarray([row["variants"]["exact"]["score"] for row in rows])
        scores = np.asarray([row["variants"][variant]["score"] for row in rows])
        exact_coordinate = np.asarray([
            row["variants"]["exact"]["components"]["coordinate"] for row in rows
        ])
        candidate_coordinate = np.asarray([
            row["variants"][variant]["components"]["coordinate"] for row in rows
        ])
        component_cosines[variant] = {
            name: gradient_comparison(
                gradients(rows, "exact", name, args.scale),
                gradients(rows, variant, name, args.scale),
            )
            for name in COMPONENTS
        }
        score_comparison = gradient_comparison(exact_gradient, gradient)
        coordinate_ablation["variants"][variant] = {
            "without_coordinate_vs_exact_without_coordinate": gradient_comparison(
                exact_without_coordinate, without_coordinate
            ),
            "without_coordinate_vs_exact_full": gradient_comparison(
                exact_gradient, without_coordinate
            ),
            "coordinate_value_exact_p05_p50_p95": [
                float(value) for value in np.quantile(exact_coordinate, [0.05, 0.5, 0.95])
            ],
            "coordinate_value_candidate_p05_p50_p95": [
                float(value) for value in np.quantile(candidate_coordinate, [0.05, 0.5, 0.95])
            ],
            "coordinate_value_error_rmse": float(np.sqrt(np.mean(np.square(
                candidate_coordinate - exact_coordinate
            )))),
            "coordinate_value_error_max_abs": float(np.max(np.abs(
                candidate_coordinate - exact_coordinate
            ))),
        }
        summary.append({
            "variant": variant,
            "all_status_ok": all(row["variants"][variant]["status"] == "ok" for row in rows),
            "gradient_proxy": gradient_comparison(
                exact_without_coordinate, without_coordinate
            ),
            "score_gradient": score_comparison,
            "score_gradient_sign_fraction": float(np.mean(np.sign(gradient) == np.sign(exact_gradient))),
            "score_rmse": float(np.sqrt(np.mean(np.square(scores - exact_scores)))),
            "score_max_abs_error": float(np.max(np.abs(scores - exact_scores))),
            "psi_fit_s_p50": float(np.median([row["variants"][variant]["psi_fit_s"] for row in rows])),
            "total_s_p50": float(np.median([row["variants"][variant]["total_s"] for row in rows])),
        })
    exact_timing = {
        "all_status_ok": all(row["variants"]["exact"]["status"] == "ok" for row in rows),
        "psi_fit_s_p50": float(np.median([row["variants"]["exact"]["psi_fit_s"] for row in rows])),
        "total_s_p50": float(np.median([row["variants"]["exact"]["total_s"] for row in rows])),
    }
    population = None
    population_scores = np.empty(0)
    population_coordinate = np.empty(0)
    if args.population_jsonl is not None:
        population_rows = [
            json.loads(line)
            for line in args.population_jsonl.read_text(encoding="utf-8").splitlines()
        ]
        population_rows = [
            row for row in population_rows
            if row["variant"] == args.population_variant
            and int(row.get("repeat", 0)) == 0
            and row["result"]["status"] == "ok"
        ]
        population_scores = np.asarray([
            row["result"]["score"] for row in population_rows
        ], dtype=np.float64)
        population_coordinate = np.asarray([
            row["result"]["components"]["coordinate"] for row in population_rows
        ], dtype=np.float64)
        subsets = {}
        for threshold in (None, 80.0, 85.0, 88.0):
            mask = np.ones(len(population_rows), dtype=bool) if threshold is None \
                else population_scores >= threshold
            values = population_coordinate[mask]
            subsets["all" if threshold is None else f"score_ge_{threshold:g}"] = {
                "count": int(np.count_nonzero(mask)),
                "coordinate_min_p10_p50_p90_max": [
                    float(value) for value in np.quantile(values, [0.0, 0.1, 0.5, 0.9, 1.0])
                ],
            }
        population = {
            "source": str(args.population_jsonl),
            "variant": args.population_variant,
            "independent_case_count": len(population_rows),
            "coordinate_score_pearson": float(np.corrcoef(
                population_coordinate, population_scores
            )[0, 1]),
            "subsets": subsets,
        }
    output = {
        "format": "local_psi_pcgls_full300_v2",
        "gradient_objective": {
            "formal_score_unchanged": True,
            "omitted_endpoint_derivative": GRADIENT_OMITTED_COMPONENT,
            "proxy_component_weights": {
                name: weight
                for name, weight in SCORE_WEIGHTS.items()
                if name != GRADIENT_OMITTED_COMPONENT
            },
            "formal_center_and_proposal_acceptance_required": True,
        },
        "scale": args.scale,
        "direction_count": 300,
        "endpoint_count": len(rows),
        "exact_timing": exact_timing,
        "summary": summary,
        "component_gradient_cosines": component_cosines,
        "coordinate_ablation": coordinate_ablation,
        "coordinate_population": population,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, len(variants), figsize=(6.2 * len(variants), 5.2), squeeze=False)
    for axis, variant in zip(axes[0], variants, strict=True):
        gradient = gradients(rows, variant, "score", args.scale)
        axis.scatter(exact_gradient, gradient, s=15, alpha=0.65)
        limit = max(float(np.max(np.abs(exact_gradient))), float(np.max(np.abs(gradient))))
        axis.plot([-limit, limit], [-limit, limit], color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{variant}: cosine={cosine(exact_gradient, gradient):.3f}")
        axis.set_xlabel("endpoint QR score derivative")
        axis.set_ylabel("PCGLS score derivative")
        axis.grid(True, alpha=0.25)
    fig.suptitle("300-coordinate score derivatives at fixed h=0.005")
    fig.tight_layout()
    fig.savefig(args.output_dir / "full300_score_gradient_scatter.png", dpi=180)
    plt.close(fig)

    matrix = np.asarray([
        [component_cosines[variant][name]["cosine"] for variant in variants]
        for name in COMPONENTS
    ])
    fig, axis = plt.subplots(figsize=(6.2, 4.8))
    image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    axis.set_xticks(range(len(variants)), labels=variants)
    axis.set_yticks(range(len(COMPONENTS)), labels=COMPONENTS)
    axis.set_title("Full300 component-gradient cosine vs endpoint QR")
    fig.colorbar(image, ax=axis, label="cosine")
    fig.tight_layout()
    fig.savefig(args.output_dir / "full300_component_gradient_cosines.png", dpi=180)
    plt.close(fig)

    labels = []
    values = []
    for variant in variants:
        labels.extend((f"{variant}\nfull", f"{variant}\nwithout coordinate"))
        values.extend((
            next(row["score_gradient"]["cosine"] for row in summary if row["variant"] == variant),
            coordinate_ablation["variants"][variant][
                "without_coordinate_vs_exact_without_coordinate"
            ]["cosine"],
        ))
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    bars = axis.bar(labels, values, color=["#777777", "#2a788e"] * len(variants))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylim(-1.0, 1.0)
    axis.set_ylabel("gradient cosine vs matching endpoint-QR target")
    axis.set_title("Effect of removing the coordinate component")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "full300_coordinate_ablation.png", dpi=180)
    plt.close(fig)

    if population is not None:
        exact_coordinate = np.asarray([
            row["variants"]["exact"]["components"]["coordinate"] for row in rows
        ])
        pcgls4_coordinate = np.asarray([
            row["variants"]["pcgls4"]["components"]["coordinate"] for row in rows
        ])
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
        axes[0].hist(population_coordinate, bins=16, color="#4c78a8", alpha=0.8)
        axes[0].axvline(np.median(exact_coordinate), color="#d1495b", linestyle="--",
                        label="local exact endpoint median")
        axes[0].axvline(np.median(pcgls4_coordinate), color="#2a9d8f", linestyle=":",
                        label="local PCGLS4 endpoint median")
        axes[0].set_xlabel("coordinate component")
        axes[0].set_ylabel("independent QUASR cases")
        axes[0].set_title("Current cubic-iota calibration distribution")
        axes[0].legend()
        axes[1].scatter(population_scores, population_coordinate, s=24, alpha=0.7,
                        color="#4c78a8")
        axes[1].set_xlabel("total score")
        axes[1].set_ylabel("coordinate component")
        axes[1].set_title("Coordinate quality is not globally saturated")
        axes[1].grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / "coordinate_population_and_local.png", dpi=180)
        plt.close(fig)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
