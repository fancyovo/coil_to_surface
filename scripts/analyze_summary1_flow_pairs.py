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
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), constrained_layout=True)
    for parameter_space in ("latent", "data"):
        curves = []
        wall_curves = []
        for case in manifest["cases"]:
            history = histories[(case["case_id"], parameter_space)]
            score = np.maximum.accumulate([float(item["best_score"]) for item in history])
            wall = np.asarray([float(item["total_wall_s"]) for item in history]) / 60.0
            curves.append(score)
            wall_curves.append((wall, score))
            axes[0].plot(np.arange(1, len(score) + 1), score, color=COLORS[parameter_space], alpha=0.18, lw=1)
            axes[1].plot(wall, score, color=COLORS[parameter_space], alpha=0.18, lw=1)
        minimum = min(len(value) for value in curves)
        stacked = np.stack([value[:minimum] for value in curves])
        axes[0].plot(
            np.arange(1, minimum + 1),
            np.median(stacked, axis=0),
            color=COLORS[parameter_space],
            lw=2.6,
            label=parameter_space.capitalize(),
        )
        common_wall = np.linspace(0.0, min(value[0][-1] for value in wall_curves), 160)
        interpolated = np.stack([np.interp(common_wall, wall, score) for wall, score in wall_curves])
        axes[1].plot(
            common_wall,
            np.median(interpolated, axis=0),
            color=COLORS[parameter_space],
            lw=2.6,
            label=parameter_space.capitalize(),
        )
    axes[0].set(xlabel="Adam iteration", ylabel="Running-best formal score", title="Matched starts, fixed iteration budget")
    axes[1].set(xlabel="Elapsed GPU time [min]", ylabel="Running-best formal score", title="Matched starts, fixed wall-clock budget")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    fig.savefig(args.output_dir / "flow_vs_data_optimization.png", dpi=220, facecolor="white")
    plt.close(fig)
    print(json.dumps(summary["aggregate"]))


if __name__ == "__main__":
    main()
