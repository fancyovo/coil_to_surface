from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


from scripts.run_axis_surface_prior_axisflip_stream import (
    PROTOCOL_ID,
    classify_iota_interval,
)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    data = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "min": float(np.min(data)),
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.9)),
        "max": float(np.max(data)),
    }


def trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = [float(row["best_score"]) for row in rows]
    return {
        "count": len(rows),
        "initial_score": distribution([float(row["initial_score"]) for row in rows]),
        "best_score": distribution(best),
        "gain": distribution(
            [float(row["best_score"]) - float(row["initial_score"]) for row in rows]
        ),
        "best_ge_50": sum(value >= 50.0 for value in best),
        "best_ge_60": sum(value >= 60.0 for value in best),
        "best_ge_70": sum(value >= 70.0 for value in best),
        "initial_iota_sign_counts": dict(
            Counter(str(row["initial_iota_sign"]) for row in rows)
        ),
        "best_iota_sign_counts": dict(
            Counter(str(row["best_iota_sign"]) for row in rows)
        ),
        "trajectory_wall_s": distribution(
            [float(row["trajectory_wall_s"]) for row in rows]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the axis-flipped stream-to-Adam200 experiment."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run_root / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("protocol ID mismatch")

    output_dir = args.run_root / "analysis"
    output_dir.mkdir(exist_ok=True)
    screening: list[dict[str, Any]] = []
    for path in sorted((args.run_root / "screening").glob("worker_*.jsonl")):
        screening.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    case_ids = [int(row["case_id"]) for row in screening]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate screened case IDs")
    screening.sort(key=lambda row: int(row["case_id"]))

    trajectories: list[dict[str, Any]] = []
    for path in sorted((args.run_root / "trajectories").glob("*/trajectory_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"trajectory protocol mismatch in {path}")
        case = manifest["case"]
        endpoints = manifest["endpoints"]
        summary = manifest["optimization"]
        trajectories.append(
            {
                "trajectory_id": manifest["trajectory_id"],
                "case_id": int(case["case_id"]),
                "worker_index": int(case["worker_index"]),
                "nfp": int(case["nfp"]),
                "n_base_coils": int(case["n_base_coils"]),
                "initial_score": float(summary["initial_score"]),
                "initial_iota_sign": endpoints["initial"]["iota_sign"],
                "best_score": float(summary["best_score"]),
                "best_iteration": int(summary["best_iteration"]),
                "best_iota_sign": endpoints["best"]["iota_sign"],
                "final_score": float(summary["final_score"]),
                "final_iota_sign": endpoints["final"]["iota_sign"],
                "trajectory_wall_s": float(manifest["timing"]["trajectory_wall_s"]),
            }
        )
    trajectories.sort(key=lambda row: int(row["case_id"]))

    failure_case_ids: set[int] = set()
    for path in sorted((args.run_root / "failures").glob("*/failure.json")):
        failure = json.loads(path.read_text(encoding="utf-8"))
        failure_case_ids.add(int(failure["case"]["case_id"]))
    incomplete_case_ids: set[int] = set()
    for path in sorted((args.run_root / "incomplete").glob("*.partial/case.json")):
        incomplete_case_ids.add(int(json.loads(path.read_text(encoding="utf-8"))["case_id"]))

    valid_rows = [
        row
        for row in screening
        if isinstance(row.get("native"), dict)
        and row["native"].get("status") == "ok"
    ]
    valid_case_ids = {int(row["case_id"]) for row in valid_rows}
    completed_case_ids = {int(row["case_id"]) for row in trajectories}
    accounted = completed_case_ids | failure_case_ids | incomplete_case_ids
    unaccounted = sorted(valid_case_ids - accounted)

    screen_by_nc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectory_by_nc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectory_by_nfp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in screening:
        screen_by_nc[str(row["n_base_coils"])].append(row)
    for row in trajectories:
        trajectory_by_nc[str(row["n_base_coils"])].append(row)
        trajectory_by_nfp[str(row["nfp"])].append(row)

    def screen_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        valid_group = [
            row
            for row in rows
            if isinstance(row.get("native"), dict)
            and row["native"].get("status") == "ok"
        ]
        sign_counts = Counter()
        for row in valid_group:
            diagnostics = row["native"].get("diagnostics") or {}
            sign_counts[
                classify_iota_interval(
                    diagnostics.get("iota_min"), diagnostics.get("iota_max")
                )
            ] += 1
        return {
            "count": len(rows),
            "status_counts": dict(
                Counter(
                    str((row.get("native") or {}).get("status", "score_exception"))
                    for row in rows
                )
            ),
            "valid_count": len(valid_group),
            "valid_rate": len(valid_group) / len(rows) if rows else None,
            "valid_iota_sign_counts": dict(sign_counts),
        }

    worker_states = []
    for path in sorted((args.run_root / "workers").glob("worker_*/progress.json")):
        worker_states.append(json.loads(path.read_text(encoding="utf-8")))
    overall_trajectories = trajectory_summary(trajectories)
    best_record = (
        max(trajectories, key=lambda row: float(row["best_score"]))
        if trajectories
        else None
    )
    summary = {
        "format": "axis_surface_prior_axisflip_stream_summary_v2",
        "protocol_id": PROTOCOL_ID,
        "screening": screen_group(screening),
        "by_n_base_coils_screening": {
            key: screen_group(value) for key, value in sorted(screen_by_nc.items())
        },
        "adam200": overall_trajectories,
        "by_n_base_coils_adam200": {
            key: trajectory_summary(value)
            for key, value in sorted(trajectory_by_nc.items())
        },
        "by_nfp_adam200": {
            key: trajectory_summary(value)
            for key, value in sorted(trajectory_by_nfp.items())
        },
        "best_record": best_record,
        "accounting": {
            "valid_discovered": len(valid_case_ids),
            "completed": len(completed_case_ids),
            "failed": len(failure_case_ids),
            "incomplete": len(incomplete_case_ids),
            "unaccounted_case_ids": unaccounted,
        },
        "worker_stage_counts": dict(
            Counter(str(worker.get("stage", "missing")) for worker in worker_states)
        ),
        "worker_elapsed_s": distribution(
            [float(worker["elapsed_s"]) for worker in worker_states]
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if screening:
        screening_rows = [
            {
                "case_id": row["case_id"],
                "worker_index": row["worker_index"],
                "nfp": row["nfp"],
                "n_base_coils": row["n_base_coils"],
                "status": (row.get("native") or {}).get("status", "score_exception"),
                "score": (row.get("native") or {}).get("score"),
                "iota_min": ((row.get("native") or {}).get("diagnostics") or {}).get("iota_min"),
                "iota_max": ((row.get("native") or {}).get("diagnostics") or {}).get("iota_max"),
                "score_wall_s": row["score_wall_s"],
            }
            for row in screening
        ]
        with (output_dir / "screening.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(screening_rows[0]))
            writer.writeheader()
            writer.writerows(screening_rows)
    if trajectories:
        with (output_dir / "trajectories.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trajectories[0]))
            writer.writeheader()
            writer.writerows(trajectories)
    print(json.dumps(summary, indent=2), flush=True)
    if unaccounted:
        raise RuntimeError(f"{len(unaccounted)} valid cases are unaccounted for")


if __name__ == "__main__":
    main()
