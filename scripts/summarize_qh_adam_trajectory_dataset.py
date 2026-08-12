from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return {"count": 0, "mean": None, "p05": None, "p50": None, "p95": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def summarize(root: Path) -> dict[str, Any]:
    manifests = []
    for path in sorted((root / "trajectories").glob("*/trajectory_manifest.json")):
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    streams = []
    for path in sorted((root / "streams").glob("*/progress.json")):
        streams.append(json.loads(path.read_text(encoding="utf-8")))

    conditions = Counter(row["condition"]["group"] for row in manifests)
    start_scores = [float(row["optimization"]["initial_score"]) for row in manifests]
    final_scores = [float(row["optimization"]["final_score"]) for row in manifests]
    best_scores = [float(row["optimization"]["best_score"]) for row in manifests]
    durations = [float(row["timing"]["trajectory_wall_s"]) for row in manifests]
    completed_streams = sum(int(row.get("completed_trajectories", 0)) for row in streams)
    elapsed_stream_s = sum(float(row.get("elapsed_s", 0.0)) for row in streams)
    projected_six_gpu = (
        completed_streams * 86400.0 * 6.0 / elapsed_stream_s
        if completed_streams and elapsed_stream_s
        else None
    )
    return {
        "dataset_root": str(root.resolve()),
        "completed_trajectories": len(manifests),
        "failure_directories": len(list((root / "failures").glob("*"))),
        "incomplete_directories": len(list((root / "incomplete").glob("*"))),
        "condition_counts": dict(sorted(conditions.items())),
        "start_score": quantiles(start_scores),
        "final_score": quantiles(final_scores),
        "best_score": quantiles(best_scores),
        "score_gain_best_minus_start": quantiles(
            [best - start for best, start in zip(best_scores, start_scores, strict=True)]
        ),
        "trajectory_wall_s": quantiles(durations),
        "stream_count": len(streams),
        "completed_trajectories_reported_by_streams": completed_streams,
        "aggregate_stream_elapsed_s": elapsed_stream_s,
        "projected_trajectories_per_day_six_gpu": projected_six_gpu,
        "streams": streams,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a QH Adam trajectory corpus.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = summarize(args.dataset_root)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
