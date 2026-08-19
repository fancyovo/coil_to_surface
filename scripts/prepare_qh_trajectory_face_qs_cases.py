from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

FORMAT = "qh_trajectory_face_qs_probe_v1"
DEFAULT_ITERATIONS = (0, 10, 25, 50, 75, 100, 150, 200)
COEFFICIENT_COUNT = 33


def token_case(tokens: np.ndarray, *, nfp: int) -> dict[str, Any]:
    values = np.atleast_2d(np.asarray(tokens, dtype=float))
    if values.shape[1] != 100:
        raise ValueError(f"expected 100 values per coil token, got {values.shape[1]}")
    return {
        "nfp": int(nfp),
        "raw": {
            "x": values[:, :COEFFICIENT_COUNT].tolist(),
            "y": values[:, COEFFICIENT_COUNT : 2 * COEFFICIENT_COUNT].tolist(),
            "z": values[:, 2 * COEFFICIENT_COUNT : 3 * COEFFICIENT_COUNT].tolist(),
            "current": values[:, -1].tolist(),
            "current_unit": "A",
            "nfp": int(nfp),
            "metadata": {"helicity": 1, "native_score_target": "QH"},
        },
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def condition_quotas(rows: list[dict[str, str]], count: int) -> dict[tuple[int, int], int]:
    groups: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (int(row["nfp"]), int(row["n_base_coils"]))
        groups.setdefault(key, []).append(row)
    if count >= len(rows):
        return {key: len(values) for key, values in groups.items()}

    exact = {key: count * len(values) / len(rows) for key, values in groups.items()}
    quotas = {key: min(len(groups[key]), int(math.floor(value))) for key, value in exact.items()}
    for key in sorted(groups, key=lambda item: (len(groups[item]), item)):
        if quotas[key] == 0 and sum(quotas.values()) < count:
            quotas[key] = 1
    while sum(quotas.values()) < count:
        candidates = [key for key in groups if quotas[key] < len(groups[key])]
        key = max(candidates, key=lambda item: (exact[item] - quotas[item], len(groups[item]), item))
        quotas[key] += 1
    while sum(quotas.values()) > count:
        candidates = [key for key in groups if quotas[key] > 1]
        key = min(candidates, key=lambda item: (exact[item] - quotas[item], -len(groups[item]), item))
        quotas[key] -= 1
    return quotas


def evenly_spaced_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: (float(row["best_global_score"]), row["trajectory_id"]))
    if count >= len(ordered):
        return ordered
    positions = np.linspace(0, len(ordered) - 1, count)
    indices = np.rint(positions).astype(int)
    return [ordered[index] for index in indices]


def select_trajectories(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    quotas = condition_quotas(rows, count)
    selected: list[dict[str, str]] = []
    for key, quota in sorted(quotas.items()):
        group = [
            row
            for row in rows
            if (int(row["nfp"]), int(row["n_base_coils"])) == key
        ]
        selected.extend(evenly_spaced_rows(group, quota))
    if len(selected) != min(count, len(rows)):
        raise RuntimeError("trajectory selection did not produce the requested count")
    return sorted(selected, key=lambda row: row["trajectory_id"])


def read_center_results(path: Path) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            results[int(row["iteration"])] = row["center_after_native_score"]
    return results


def read_history_wall_times(path: Path) -> dict[int, float]:
    values = {0: 0.0}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            values[int(row["iteration"])] = float(row["total_wall_s"])
    return values


def finite_diagnostic(result: dict[str, Any], key: str) -> float:
    value = float(result.get("diagnostics", {}).get(key, float("nan")))
    if not math.isfinite(value):
        raise ValueError(f"native score diagnostic {key!r} is not finite")
    return value


def token_at_iteration(trace: np.lib.npyio.NpzFile, iteration: int) -> np.ndarray:
    if iteration == 0:
        return np.asarray(trace["initial_tokens"], dtype=np.float32)
    return np.asarray(trace["center_after_tokens"][iteration - 1], dtype=np.float32)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_iterations(text: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in text.split(",") if value.strip()}))
    if not values or values[0] < 0 or values[-1] > 200:
        raise ValueError("iterations must be a non-empty subset of [0, 200]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and materialize sparse QH trajectory centers for face-QS calibration.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trajectory-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trajectory-count", type=int, default=96)
    parser.add_argument("--iterations", default=",".join(str(value) for value in DEFAULT_ITERATIONS))
    parser.add_argument("--fixed-probe-rho", type=float, default=0.8)
    parser.add_argument("--source-a", type=float, default=0.05)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if not 0.0 < args.fixed_probe_rho <= 1.0:
        raise ValueError("fixed-probe-rho must be in (0, 1]")
    iterations = parse_iterations(args.iterations)
    rows = read_csv(args.trajectory_summary)
    selected = select_trajectories(rows, args.trajectory_count)
    args.output_root.mkdir(parents=True)
    cases_dir = args.output_root / "cases"
    cases_dir.mkdir()

    case_records = []
    trajectory_records = []
    for summary_row in selected:
        trajectory_id = summary_row["trajectory_id"]
        trajectory_dir = args.dataset_root / "trajectories" / trajectory_id
        optimization_dir = trajectory_dir / "optimization"
        center_results = read_center_results(optimization_dir / "center_native_results.jsonl.gz")
        wall_times = read_history_wall_times(optimization_dir / "history.jsonl")
        missing = [iteration for iteration in iterations if iteration not in center_results or iteration not in wall_times]
        if missing:
            raise ValueError(f"{trajectory_id} is missing iterations {missing}")
        initial_level = finite_diagnostic(center_results[0], "surface_effective_level")
        fixed_level = initial_level * args.fixed_probe_rho**2
        with np.load(optimization_dir / "training_trace.npz") as trace:
            for iteration in iterations:
                native = center_results[iteration]
                adaptive_level = finite_diagnostic(native, "surface_effective_level")
                levels = {
                    "fixed_probe": float(fixed_level),
                    "adaptive_edge": float(adaptive_level),
                }
                maximum_level = max(levels.values())
                targets = [
                    {
                        "name": name,
                        "s_level": level,
                        "rho_in_alpha_fit": math.sqrt(level / maximum_level),
                    }
                    for name, level in levels.items()
                ]
                case_id = f"{trajectory_id}_step{iteration:04d}"
                case_dir = cases_dir / case_id
                case_dir.mkdir()
                tokens = token_at_iteration(trace, iteration)
                write_json(case_dir / "case.json", token_case(tokens, nfp=int(summary_row["nfp"])))
                metadata = {
                    "format": FORMAT,
                    "case_id": case_id,
                    "trajectory_id": trajectory_id,
                    "iteration": iteration,
                    "optimizer_wall_s": wall_times[iteration],
                    "nfp": int(summary_row["nfp"]),
                    "n_base_coils": int(summary_row["n_base_coils"]),
                    "source_a_m": float(args.source_a),
                    "alpha_s_edge": float(maximum_level),
                    "surface_targets": targets,
                    "native_score": native,
                    "trajectory_summary": {
                        key: float(summary_row[key])
                        for key in (
                            "initial_online_score",
                            "best_online_score",
                            "best_global_score",
                            "optimization_wall_s",
                        )
                    },
                }
                write_json(case_dir / "metadata.json", metadata)
                case_records.append(
                    {
                        "case_id": case_id,
                        "trajectory_id": trajectory_id,
                        "iteration": iteration,
                        "optimizer_wall_s": wall_times[iteration],
                        "nfp": int(summary_row["nfp"]),
                        "n_base_coils": int(summary_row["n_base_coils"]),
                        "native_score": float(native.get("score", float("nan"))),
                        "fixed_probe_s": fixed_level,
                        "adaptive_edge_s": adaptive_level,
                    }
                )
        trajectory_records.append(
            {
                "trajectory_id": trajectory_id,
                "nfp": int(summary_row["nfp"]),
                "n_base_coils": int(summary_row["n_base_coils"]),
                "initial_score": float(summary_row["initial_online_score"]),
                "best_global_score": float(summary_row["best_global_score"]),
            }
        )

    write_json(args.output_root / "cases.json", case_records)
    write_json(
        args.output_root / "experiment_manifest.json",
        {
            "format": FORMAT,
            "source_dataset": str(args.dataset_root.resolve()),
            "trajectory_summary": str(args.trajectory_summary.resolve()),
            "trajectory_count": len(selected),
            "iterations": list(iterations),
            "case_count": len(case_records),
            "surface_count": 2 * len(case_records),
            "fixed_probe_rho": float(args.fixed_probe_rho),
            "source_a_m": float(args.source_a),
            "selection": "condition-proportional, score-quantile-spaced within each (nfp,n_base_coils)",
            "trajectories": trajectory_records,
        },
    )
    shutil.copy2(args.trajectory_summary, args.output_root / "source_trajectory_summary.csv")
    print(json.dumps({"trajectories": len(selected), "cases": len(case_records), "surfaces": 2 * len(case_records)}))


if __name__ == "__main__":
    main()
