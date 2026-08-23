from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"latent": "#315A78", "data": "#D17A45"}


def load_history(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def q(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability)) if values else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_root / "pair_manifest.json").read_text(encoding="utf-8"))
    rows = []
    histories: dict[tuple[str, str], list[dict]] = {}
    for case in manifest["cases"]:
        for parameter_space in ("latent", "data"):
            root = args.run_root / case["case_id"] / parameter_space
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            history = load_history(root / "history.jsonl")
            histories[(case["case_id"], parameter_space)] = history
            rows.append(
                {
                    **{key: case[key] for key in ("case_id", "nfp", "n_base_coils", "probe_iteration")},
                    "parameter_space": parameter_space,
                    "initial_score": float(summary["initial_score"]),
                    "best_score": float(summary["best_score"]),
                    "best_iteration": int(summary["best_iteration"]),
                    "completed_iterations": int(summary["completed_iterations"]),
                    "total_wall_s": float(history[-1]["total_wall_s"]),
                    "accepted_updates": int(sum(bool(item["center_update_accepted"]) for item in history)),
                    "invalid_gradient_steps": int(sum(not bool(item["gradient_step_applied"]) for item in history)),
                }
            )

    paired = []
    for case in manifest["cases"]:
        by_space = {row["parameter_space"]: row for row in rows if row["case_id"] == case["case_id"]}
        paired.append(
            {
                "case_id": case["case_id"],
                "latent_best": by_space["latent"]["best_score"],
                "data_best": by_space["data"]["best_score"],
                "latent_gain": by_space["latent"]["best_score"] - by_space["latent"]["initial_score"],
                "data_gain": by_space["data"]["best_score"] - by_space["data"]["initial_score"],
                "best_difference": by_space["latent"]["best_score"] - by_space["data"]["best_score"],
            }
        )
    difference = [row["best_difference"] for row in paired]
    summary = {
        "format": "summary1_flow_parameterization_pairs_analysis_v1",
        "case_count": len(manifest["cases"]),
        "runs": rows,
        "pairs": paired,
        "aggregate": {
            "latent_wins": int(sum(value > 0.0 for value in difference)),
            "ties": int(sum(value == 0.0 for value in difference)),
            "data_wins": int(sum(value < 0.0 for value in difference)),
            "latent_minus_data_best_p10": q(difference, 0.10),
            "latent_minus_data_best_p50": q(difference, 0.50),
            "latent_minus_data_best_p90": q(difference, 0.90),
            "latent_gain_p50": q([row["latent_gain"] for row in paired], 0.50),
            "data_gain_p50": q([row["data_gain"] for row in paired], 0.50),
            "latent_wall_s_p50": q(
                [row["total_wall_s"] for row in rows if row["parameter_space"] == "latent"],
                0.50,
            ),
            "data_wall_s_p50": q(
                [row["total_wall_s"] for row in rows if row["parameter_space"] == "data"],
                0.50,
            ),
            "latent_to_data_wall_ratio_p50": q(
                [
                    next(
                        row["total_wall_s"]
                        for row in rows
                        if row["case_id"] == case["case_id"]
                        and row["parameter_space"] == "latent"
                    )
                    / next(
                        row["total_wall_s"]
                        for row in rows
                        if row["case_id"] == case["case_id"]
                        and row["parameter_space"] == "data"
                    )
                    for case in manifest["cases"]
                ],
                0.50,
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.7), constrained_layout=True)
    iteration_curves: dict[str, list[np.ndarray]] = {"latent": [], "data": []}
    wall_curves: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "latent": [],
        "data": [],
    }
    common_wall_limit = min(
        float(history[-1]["total_wall_s"])
        for history in histories.values()
    ) / 60.0
    for parameter_space in ("latent", "data"):
        for case in manifest["cases"]:
            history = histories[(case["case_id"], parameter_space)]
            initial = next(
                row["initial_score"]
                for row in rows
                if row["case_id"] == case["case_id"]
                and row["parameter_space"] == parameter_space
            )
            score = np.maximum.accumulate(
                [initial] + [float(item["best_score"]) for item in history]
            )
            gain = score - initial
            wall = np.asarray(
                [0.0] + [float(item["total_wall_s"]) for item in history]
            ) / 60.0
            iteration_curves[parameter_space].append(gain)
            wall_curves[parameter_space].append((wall, gain))
            axes[0, 0].plot(
                np.arange(len(gain)), gain, color=COLORS[parameter_space], alpha=0.16, lw=1
            )
        minimum = min(len(value) for value in iteration_curves[parameter_space])
        stacked = np.stack(
            [value[:minimum] for value in iteration_curves[parameter_space]]
        )
        x_iteration = np.arange(minimum)
        axes[0, 0].fill_between(
            x_iteration,
            np.quantile(stacked, 0.25, axis=0),
            np.quantile(stacked, 0.75, axis=0),
            color=COLORS[parameter_space],
            alpha=0.12,
        )
        axes[0, 0].plot(
            x_iteration,
            np.median(stacked, axis=0),
            color=COLORS[parameter_space],
            lw=2.6,
            label="Flow latent" if parameter_space == "latent" else "Normalized data",
        )
        common_wall = np.linspace(
            0.0,
            common_wall_limit,
            160,
        )
        interpolated = np.stack(
            [
                np.interp(common_wall, wall, gain)
                for wall, gain in wall_curves[parameter_space]
            ]
        )
        axes[0, 1].fill_between(
            common_wall,
            np.quantile(interpolated, 0.25, axis=0),
            np.quantile(interpolated, 0.75, axis=0),
            color=COLORS[parameter_space],
            alpha=0.12,
        )
        axes[0, 1].plot(
            common_wall,
            np.median(interpolated, axis=0),
            color=COLORS[parameter_space],
            lw=2.6,
            label="Flow latent" if parameter_space == "latent" else "Normalized data",
        )

    latent_gain = np.asarray([row["latent_gain"] for row in paired])
    data_gain = np.asarray([row["data_gain"] for row in paired])
    axes[1, 0].scatter(data_gain, latent_gain, s=46, color="#4D8B6A", alpha=0.86)
    upper = max(float(np.max(latent_gain)), float(np.max(data_gain)), 1.0)
    axes[1, 0].plot([0.0, upper], [0.0, upper], color="#555555", ls="--", lw=1)
    axes[1, 0].set(
        xlabel="Best gain in normalized data space",
        ylabel="Best gain in Flow latent space",
        title="Paired best gain after 100 steps",
    )
    axes[1, 0].text(
        0.04,
        0.94,
        f"Flow wins {summary['aggregate']['latent_wins']}/{len(paired)} pairs",
        transform=axes[1, 0].transAxes,
        va="top",
    )

    case_axis = np.arange(1, len(manifest["cases"]) + 1)
    latent_wall = np.asarray(
        [
            next(
                row["total_wall_s"]
                for row in rows
                if row["case_id"] == case["case_id"]
                and row["parameter_space"] == "latent"
            )
            / 60.0
            for case in manifest["cases"]
        ]
    )
    data_wall = np.asarray(
        [
            next(
                row["total_wall_s"]
                for row in rows
                if row["case_id"] == case["case_id"]
                and row["parameter_space"] == "data"
            )
            / 60.0
            for case in manifest["cases"]
        ]
    )
    for x, latent_value, data_value in zip(
        case_axis, latent_wall, data_wall, strict=True
    ):
        axes[1, 1].plot(
            [x, x], [data_value, latent_value], color="#B8B8B8", lw=1.2, zorder=1
        )
    axes[1, 1].scatter(
        case_axis, latent_wall, color=COLORS["latent"], s=38, label="Flow latent", zorder=2
    )
    axes[1, 1].scatter(
        case_axis, data_wall, color=COLORS["data"], s=38, label="Normalized data", zorder=2
    )
    axes[1, 1].set(
        xlabel="Matched case",
        ylabel="Wall time for 100 steps [min]",
        title="Cost of the two parameterizations",
        xticks=case_axis,
    )

    axes[0, 0].set(
        xlabel="Adam iteration",
        ylabel="Running-best score gain",
        title="Gain at a fixed iteration budget",
    )
    axes[0, 1].set(
        xlabel="Elapsed GPU time [min]",
        ylabel="Running-best score gain",
        title="Gain at a fixed wall-clock budget",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    axes[1, 1].legend(frameon=False)
    fig.savefig(args.output_dir / "flow_vs_data_optimization.png", dpi=220, facecolor="white")
    plt.close(fig)
    print(json.dumps(summary["aggregate"]))


if __name__ == "__main__":
    main()
