from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def analyze(panel: dict[str, Any], run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    checkpoints = (1, 5, 10, 20, 30, 40)
    for start in panel["starts"]:
        output = run_root / f"start_{int(start['start_id']):02d}"
        summary = load_json(output / "summary.json")
        history = load_jsonl(output / "history.jsonl")
        if not history:
            raise ValueError(f"empty history for {output}")
        initial = float(summary["initial_score"])
        row = {
            **start,
            "optimizer_initial_score": initial,
            "initial_score_discrepancy": initial - float(start["recorded_score"]),
            "final_score": float(summary["final_score"]),
            "best_score": float(summary["best_score"]),
            "final_gain": float(summary["final_score"] - initial),
            "best_gain": float(summary["best_score"] - initial),
            "best_iteration": int(summary["best_iteration"]),
            "completed_iterations": int(summary["completed_iterations"]),
            "total_wall_s": float(summary["total_wall_s"]),
            "mean_iteration_wall_s": float(summary["mean_iteration_wall_s"]),
            "valid_endpoint_fraction_mean": float(np.mean([item["valid_endpoint_fraction"] for item in history])),
            "checkpoints": {},
            "history": history,
        }
        for checkpoint in checkpoints:
            eligible = [item for item in history if int(item["iteration"]) <= checkpoint]
            if eligible:
                item = eligible[-1]
                row["checkpoints"][str(checkpoint)] = {
                    "current_score": float(item["current_score"]),
                    "best_score": float(item["best_score"]),
                    "current_gain": float(item["current_score"] - initial),
                    "best_gain": float(item["best_score"] - initial),
                }
        rows.append(row)

    initial = np.asarray([row["optimizer_initial_score"] for row in rows])
    best_gain = np.asarray([row["best_gain"] for row in rows])
    final_gain = np.asarray([row["final_gain"] for row in rows])
    best = np.asarray([row["best_score"] for row in rows])
    strata = {}
    for name in dict.fromkeys(row["stratum"] for row in rows):
        selected = [row for row in rows if row["stratum"] == name]
        strata[name] = {
            "count": len(selected),
            "initial_score": [row["optimizer_initial_score"] for row in selected],
            "best_score": [row["best_score"] for row in selected],
            "best_gain": [row["best_gain"] for row in selected],
            "final_gain": [row["final_gain"] for row in selected],
            "best_gain_mean": float(np.mean([row["best_gain"] for row in selected])),
            "best_score_mean": float(np.mean([row["best_score"] for row in selected])),
        }
    summary = {
        "format": "qh_iid_score_adam_start_sweep_summary_v1",
        "count": len(rows),
        "completed_iterations": sorted({row["completed_iterations"] for row in rows}),
        "initial_score_vs_best_gain": correlation(initial, best_gain),
        "initial_score_vs_final_gain": correlation(initial, final_gain),
        "initial_score_vs_best_score": correlation(initial, best),
        "score": {
            "initial": initial.tolist(),
            "final": [row["final_score"] for row in rows],
            "best": best.tolist(),
            "best_gain": best_gain.tolist(),
        },
        "strata": strata,
        "runtime": {
            "trajectory_wall_s_sum": float(np.sum([row["total_wall_s"] for row in rows])),
            "trajectory_wall_s_mean": float(np.mean([row["total_wall_s"] for row in rows])),
        },
    }
    return summary, rows


def plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    initial = np.asarray([row["optimizer_initial_score"] for row in rows])
    norm = Normalize(vmin=float(initial.min()), vmax=float(initial.max()))
    cmap = plt.get_cmap("viridis")
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for row in rows:
        color = cmap(norm(row["optimizer_initial_score"]))
        iterations = [0] + [item["iteration"] for item in row["history"]]
        current = [row["optimizer_initial_score"]] + [item["current_score"] for item in row["history"]]
        best = [row["optimizer_initial_score"]] + [item["best_score"] for item in row["history"]]
        axes[0, 0].plot(iterations, current, color=color, alpha=0.6)
        axes[0, 1].plot(iterations, best, color=color, alpha=0.75)
    axes[0, 0].set(xlabel="Adam iteration", ylabel="current score", title="Current score trajectories")
    axes[0, 1].set(xlabel="Adam iteration", ylabel="best score", title="Best-so-far trajectories")

    axes[1, 0].scatter(initial, [row["best_gain"] for row in rows], c=initial, cmap=cmap, norm=norm, s=55)
    axes[1, 0].axhline(0.0, color="#777777", ls="--")
    axes[1, 0].set(xlabel="initial score", ylabel="best score gain", title="Optimization gain vs start quality")
    axes[1, 1].scatter(initial, [row["best_score"] for row in rows], c=initial, cmap=cmap, norm=norm, s=55, label="best")
    axes[1, 1].scatter(initial, [row["final_score"] for row in rows], c=initial, cmap=cmap, norm=norm, marker="x", s=55, label="final")
    low = min(float(initial.min()), min(row["final_score"] for row in rows))
    high = max(max(row["best_score"] for row in rows), float(initial.max()))
    axes[1, 1].plot((low, high), (low, high), color="#777777", ls="--")
    axes[1, 1].set(xlabel="initial score", ylabel="optimized score", title="Initial vs optimized score")
    axes[1, 1].legend()
    figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="initial score", shrink=0.8)
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze native-score Adam trajectories across IID start scores.")
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--start-ids",
        help="Optional comma-separated subset of panel start IDs to analyze.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = load_json(args.panel_manifest)
    if args.start_ids:
        selected_ids = {int(value) for value in args.start_ids.split(",")}
        panel = {
            **panel,
            "starts": [
                start for start in panel["starts"] if int(start["start_id"]) in selected_ids
            ],
        }
        found_ids = {int(start["start_id"]) for start in panel["starts"]}
        if found_ids != selected_ids:
            raise ValueError(f"requested start IDs are absent from panel: {selected_ids - found_ids}")
    summary, rows = analyze(panel, args.run_root)
    (args.run_root / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.run_root / "trajectory_summary.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            compact = {name: value for name, value in row.items() if name != "history"}
            stream.write(json.dumps(compact, separators=(",", ":")) + "\n")
    plot(rows, args.run_root / "initial_score_vs_adam_outcome.png")
    print(json.dumps({"event": "complete", **summary}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
