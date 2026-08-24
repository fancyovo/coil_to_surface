from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon


COLORS = {"flow": "#315A78", "data": "#D17A45"}
LABELS = {"flow": "Flow prior / latent Adam", "data": "Data Gaussian / data Adam"}


def finite(values: list[float] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    return result[np.isfinite(result)]


def distribution(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = finite(values)
    if not array.size:
        return {"count": 0}
    result: dict[str, float | int | None] = {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    for percentile in (5, 10, 25, 50, 75, 90, 95):
        result[f"p{percentile}"] = float(np.percentile(array, percentile))
    return result


def bootstrap_median(values: np.ndarray, seed: int = 20260826) -> list[float] | None:
    array = finite(values)
    if not array.size:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(10000, array.size), replace=True)
    medians = np.median(draws, axis=1)
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]


def read_history(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def running_best(initial: float, history: list[dict[str, Any]]) -> np.ndarray:
    values = np.asarray(
        [initial] + [float(row["best_score"]) for row in history], dtype=np.float64
    )
    return np.maximum.accumulate(values)


def load_screen(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.load(path, allow_pickle=False)
    return (
        np.asarray(arrays["score"], dtype=np.float64),
        np.asarray(arrays["status"]).astype(str),
    )


def paired_test(difference: np.ndarray) -> dict[str, Any]:
    values = finite(difference)
    nonzero = values[np.abs(values) > 1.0e-10]
    return {
        "count": int(values.size),
        "positive": int(np.count_nonzero(values > 1.0e-10)),
        "ties": int(np.count_nonzero(np.abs(values) <= 1.0e-10)),
        "negative": int(np.count_nonzero(values < -1.0e-10)),
        "distribution": distribution(values),
        "median_bootstrap_95": bootstrap_median(values),
        "sign_test_two_sided_p": (
            float(binomtest(int(np.count_nonzero(nonzero > 0)), nonzero.size).pvalue)
            if nonzero.size
            else None
        ),
        "wilcoxon_two_sided_p": (
            float(wilcoxon(nonzero).pvalue) if nonzero.size else None
        ),
    }


def plot_ecdf(axis: Any, values: np.ndarray, *, color: str, label: str) -> None:
    ordered = np.sort(finite(values))
    probability = np.arange(1, ordered.size + 1) / max(ordered.size, 1)
    axis.step(ordered, probability, where="post", color=color, lw=2.2, label=label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(
        (args.run_root / "control_manifest.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    candidate_scores = {"flow": [], "data": []}
    candidate_valid = {"flow": [], "data": []}
    curves: dict[str, list[np.ndarray]] = {"flow": [], "data": []}
    missing: list[str] = []
    for case in manifest["cases"]:
        trajectory_id = case["trajectory_id"]
        result_root = args.run_root / "cases" / trajectory_id
        if not (result_root / "case_manifest.json").is_file():
            missing.append(trajectory_id)
            continue
        reference_path = Path(case["reference_manifest"])
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        control = json.loads(
            (result_root / "case_manifest.json").read_text(encoding="utf-8")
        )

        flow_scores, flow_status = load_screen(
            reference_path.parent / "screening" / "screening_arrays.npz"
        )
        data_scores, data_status = load_screen(
            result_root / "screening" / "screening_arrays.npz"
        )
        candidate_scores["flow"].extend(flow_scores.tolist())
        candidate_scores["data"].extend(data_scores.tolist())
        candidate_valid["flow"].extend((flow_status == "ok").tolist())
        candidate_valid["data"].extend((data_status == "ok").tolist())

        flow_initial = float(reference["optimization"]["initial_score"])
        flow_best = float(reference["optimization"]["best_score"])
        flow_history = read_history(
            reference_path.parent / "optimization" / "history.jsonl"
        )
        curves["flow"].append(running_best(flow_initial, flow_history))

        data_ok = control["status"] == "optimization_ok"
        data_initial = (
            float(control["optimization"]["initial_score"]) if data_ok else 0.0
        )
        data_best = float(control["optimization"]["best_score"]) if data_ok else 0.0
        if data_ok:
            data_history = read_history(result_root / "optimization" / "history.jsonl")
            curves["data"].append(running_best(data_initial, data_history))

        rows.append(
            {
                "trajectory_id": trajectory_id,
                "nfp": int(case["nfp"]),
                "n_base_coils": int(case["n_base_coils"]),
                "data_status": control["status"],
                "flow_valid_candidates": int(np.count_nonzero(flow_status == "ok")),
                "data_valid_candidates": int(np.count_nonzero(data_status == "ok")),
                "flow_initial": flow_initial,
                "data_initial": data_initial,
                "flow_best": flow_best,
                "data_best": data_best,
                "flow_gain": flow_best - flow_initial,
                "data_gain": data_best - data_initial if data_ok else 0.0,
                "flow_minus_data_initial": flow_initial - data_initial,
                "flow_minus_data_best": flow_best - data_best,
                "data_case_wall_s": float(control["timing"]["case_wall_s"]),
            }
        )

    if missing and not args.allow_partial:
        raise RuntimeError(f"{len(missing)} prepared cases are incomplete")
    if not rows:
        raise RuntimeError("no completed control cases are available")

    flow_initial = np.asarray([row["flow_initial"] for row in rows])
    data_initial = np.asarray([row["data_initial"] for row in rows])
    flow_best = np.asarray([row["flow_best"] for row in rows])
    data_best = np.asarray([row["data_best"] for row in rows])
    data_ok = np.asarray([row["data_status"] == "optimization_ok" for row in rows])
    thresholds = (50, 70, 80, 85, 90, 92)
    summary = {
        "format": "qh_data_prior_end_to_end_control_analysis_v1",
        "prepared_case_count": int(manifest["case_count"]),
        "analyzed_case_count": len(rows),
        "missing_case_count": len(missing),
        "missing_cases": missing,
        "data_case_status_counts": dict(
            sorted(Counter(row["data_status"] for row in rows).items())
        ),
        "candidate_count_per_case": 32,
        "candidate_score": {
            key: distribution(candidate_scores[key]) for key in ("flow", "data")
        },
        "candidate_valid_fraction": {
            key: float(np.mean(candidate_valid[key])) for key in ("flow", "data")
        },
        "selected_start_score_end_to_end": {
            "flow": distribution(flow_initial),
            "data": distribution(data_initial),
        },
        "best_score_end_to_end": {
            "flow": distribution(flow_best),
            "data": distribution(data_best),
        },
        "paired_flow_minus_data_initial": paired_test(flow_initial - data_initial),
        "paired_flow_minus_data_best": paired_test(flow_best - data_best),
        "threshold_fraction": {
            str(value): {
                "flow_initial": float(np.mean(flow_initial >= value)),
                "data_initial": float(np.mean(data_initial >= value)),
                "flow_best": float(np.mean(flow_best >= value)),
                "data_best": float(np.mean(data_best >= value)),
            }
            for value in thresholds
        },
        "conditional_on_data_valid_start": {
            "count": int(np.count_nonzero(data_ok)),
            "data_initial": distribution(data_initial[data_ok]),
            "data_best": distribution(data_best[data_ok]),
            "data_gain": distribution(data_best[data_ok] - data_initial[data_ok]),
            "flow_initial_matched": distribution(flow_initial[data_ok]),
            "flow_best_matched": distribution(flow_best[data_ok]),
            "flow_gain_matched": distribution(
                flow_best[data_ok] - flow_initial[data_ok]
            ),
        },
        "data_case_wall_s": distribution(
            [row["data_case_wall_s"] for row in rows]
        ),
        "rows": rows,
        "provenance": {
            "control_manifest": manifest,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "case_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    for key in ("flow", "data"):
        plot_ecdf(
            axes[0, 0],
            np.asarray(candidate_scores[key]),
            color=COLORS[key],
            label=f"{LABELS[key]} (valid {np.mean(candidate_valid[key]):.1%})",
        )
        plot_ecdf(
            axes[0, 1],
            flow_initial if key == "flow" else data_initial,
            color=COLORS[key],
            label=LABELS[key],
        )
        plot_ecdf(
            axes[1, 0],
            flow_best if key == "flow" else data_best,
            color=COLORS[key],
            label=LABELS[key],
        )
    x = np.arange(len(thresholds))
    width = 0.2
    for offset, (key, stage) in enumerate(
        (("flow", "initial"), ("data", "initial"), ("flow", "best"), ("data", "best"))
    ):
        values = [summary["threshold_fraction"][str(t)][f"{key}_{stage}"] for t in thresholds]
        axes[1, 1].bar(
            x + (offset - 1.5) * width,
            values,
            width,
            color=COLORS[key],
            alpha=0.55 if stage == "initial" else 1.0,
            label=f"{key.capitalize()} {stage}",
        )
    axes[0, 0].set(
        xlabel="Candidate score",
        ylabel="Empirical cumulative probability",
        title="All 32 random candidates per matched condition",
    )
    axes[0, 1].set(
        xlabel="Best-of-32 start score",
        ylabel="Empirical cumulative probability",
        title="Initialization quality",
    )
    axes[1, 0].set(
        xlabel="Best score after 200 Adam steps",
        ylabel="Empirical cumulative probability",
        title="End-to-end outcome (failed starts count as zero)",
    )
    axes[1, 1].set(
        xlabel="Score threshold",
        ylabel="Fraction at or above threshold",
        title="Useful-sample yield before and after optimization",
        xticks=x,
        xticklabels=[str(value) for value in thresholds],
        ylim=(0, 1.03),
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(args.output_dir / "prior_and_end_to_end.png", dpi=220, facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for key in ("flow", "data"):
        if not curves[key]:
            continue
        minimum = min(len(curve) for curve in curves[key])
        values = np.stack([curve[:minimum] for curve in curves[key]])
        steps = np.arange(minimum)
        axes[0].fill_between(
            steps,
            np.percentile(values, 25, axis=0),
            np.percentile(values, 75, axis=0),
            color=COLORS[key],
            alpha=0.18,
        )
        axes[0].plot(
            steps,
            np.median(values, axis=0),
            color=COLORS[key],
            lw=2.3,
            label=f"{LABELS[key]} (n={len(values)})",
        )
        gains = values - values[:, :1]
        axes[1].fill_between(
            steps,
            np.percentile(gains, 25, axis=0),
            np.percentile(gains, 75, axis=0),
            color=COLORS[key],
            alpha=0.18,
        )
        axes[1].plot(
            steps,
            np.median(gains, axis=0),
            color=COLORS[key],
            lw=2.3,
            label=f"{LABELS[key]} (n={len(values)})",
        )
    axes[0].set(
        xlabel="Adam step",
        ylabel="Running-best score",
        title="Conditional optimization trajectory (median and IQR)",
    )
    axes[1].set(
        xlabel="Adam step",
        ylabel="Running-best gain from selected start",
        title="Local optimizer contribution after screening",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(args.output_dir / "conditional_optimization.png", dpi=220, facecolor="white")
    plt.close(figure)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}))


if __name__ == "__main__":
    main()
