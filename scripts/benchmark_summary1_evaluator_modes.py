from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stellarator_eval.experimental import NeighborhoodEvaluator, NeighborhoodSettings
from stellarator_eval.native_evaluator import CoilSet, EvaluationMode, Evaluator


def token_coils(tokens: np.ndarray, nfp: int) -> CoilSet:
    values = np.asarray(tokens, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 100:
        raise ValueError("trajectory tokens must have shape (coils, 100)")
    return CoilSet(
        values[:, :33],
        values[:, 33:66],
        values[:, 66:99],
        values[:, 99],
        nfp,
    )


def quantiles(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not finite.size:
        return {"count": 0, "p50": math.nan, "p95": math.nan, "mean": math.nan}
    return {
        "count": int(finite.size),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "mean": float(np.mean(finite)),
    }


def correlation(left: list[float], right: list[float]) -> dict[str, float | int]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 3:
        return {"count": int(np.count_nonzero(keep)), "spearman": math.nan, "mae": math.nan}
    return {
        "count": int(np.count_nonzero(keep)),
        "spearman": float(spearmanr(x[keep], y[keep]).statistic),
        "mae": float(np.mean(np.abs(x[keep] - y[keep]))),
    }


def discover_cases(
    root: Path,
    *,
    case_count: int,
    probe_iterations: tuple[int, ...],
    seed: int,
) -> list[dict[str, Any]]:
    manifests = sorted(root.glob("trajectories/*/trajectory_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no trajectories found below {root}")
    grouped: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition = payload["condition"]
        grouped[(int(condition["nfp"]), int(condition["n_base_coils"]))].append(path)
    rng = np.random.default_rng(seed)
    for paths in grouped.values():
        rng.shuffle(paths)
    ordered_groups = sorted(grouped, key=lambda key: (-len(grouped[key]), key))
    selected: list[Path] = []
    round_index = 0
    while len(selected) < min(case_count, len(manifests)):
        added = False
        for key in ordered_groups:
            paths = grouped[key]
            if round_index < len(paths):
                selected.append(paths[round_index])
                added = True
                if len(selected) == min(case_count, len(manifests)):
                    break
        if not added:
            break
        round_index += 1

    cases: list[dict[str, Any]] = []
    for index, manifest_path in enumerate(selected):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trace_path = manifest_path.parent / "optimization" / "training_trace.npz"
        with np.load(trace_path, allow_pickle=False) as trace:
            iterations = np.asarray(trace["iteration"], dtype=int)
            requested = int(probe_iterations[index % len(probe_iterations)])
            trace_index = int(np.argmin(np.abs(iterations - requested)))
            center_tokens = np.asarray(trace["probe_tokens"][trace_index], dtype=np.float64)
            endpoint_tokens = np.asarray(trace["endpoint_tokens"][trace_index], dtype=np.float64)
            actual_iteration = int(iterations[trace_index])
        condition = manifest["condition"]
        cases.append(
            {
                "case_id": f"{manifest['trajectory_id']}_probe{actual_iteration:04d}",
                "trajectory_id": manifest["trajectory_id"],
                "probe_iteration": actual_iteration,
                "nfp": int(condition["nfp"]),
                "n_base_coils": int(condition["n_base_coils"]),
                "center_tokens": center_tokens,
                "endpoint_tokens": endpoint_tokens,
                "source_manifest": str(manifest_path),
            }
        )
    return cases


def result_row(
    *,
    case: dict[str, Any],
    mode: str,
    repeat: int,
    candidate_index: int | None,
    wall_s: float,
    result,
    batch_size: int | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "trajectory_id": case["trajectory_id"],
        "probe_iteration": case["probe_iteration"],
        "nfp": case["nfp"],
        "n_base_coils": case["n_base_coils"],
        "mode": mode,
        "repeat": repeat,
        "candidate_index": candidate_index,
        "batch_size": batch_size,
        "wall_s": float(wall_s),
        "status": result.status,
        "score": result.score,
        "native_score": result.native_score,
        "components": dict(result.components),
        "timing": dict(result.timing),
        "diagnostics": dict(result.diagnostics),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    modes = sorted(
        {row["mode"] for row in records if row["mode"] != "neighborhood_batch"}
    )
    mode_summary: dict[str, Any] = {}
    for mode in modes:
        rows = [row for row in records if row["mode"] == mode]
        mode_summary[mode] = {
            "count": len(rows),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "wall_s": quantiles([row["wall_s"] for row in rows]),
            "native_total_s": quantiles(
                [float(row["timing"].get("total_s", math.nan)) for row in rows]
            ),
        }

    batch_summary: dict[str, Any] = {}
    for size in sorted(
        {int(row["batch_size"]) for row in records if row["mode"] == "neighborhood_batch"}
    ):
        rows = [
            row
            for row in records
            if row["mode"] == "neighborhood_batch" and int(row["batch_size"]) == size
        ]
        batch_summary[str(size)] = {
            "batch_wall_s": quantiles([row["wall_s"] for row in rows]),
            "per_candidate_wall_s": quantiles([row["wall_s"] / size for row in rows]),
            "ok_fraction": quantiles([float(row["ok_fraction"]) for row in rows]),
        }

    first: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in records:
        candidate = row.get("candidate_index")
        if candidate is None or row["mode"] not in {
            "independent",
            "strict_continuation",
            "neighborhood_proxy",
        }:
            continue
        key = (row["case_id"], int(candidate), row["mode"])
        if key not in first or int(row["repeat"]) < int(first[key]["repeat"]):
            first[key] = row
    pair_keys = sorted({(key[0], key[1]) for key in first})
    comparisons: dict[str, Any] = {}
    for left_mode, right_mode, name in (
        ("independent", "strict_continuation", "independent_vs_strict"),
        ("strict_continuation", "neighborhood_proxy", "strict_vs_proxy"),
    ):
        left: list[float] = []
        right: list[float] = []
        status_equal: list[bool] = []
        for case_id, candidate in pair_keys:
            left_row = first.get((case_id, candidate, left_mode))
            right_row = first.get((case_id, candidate, right_mode))
            if left_row is None or right_row is None:
                continue
            status_equal.append(left_row["status"] == right_row["status"])
            if left_row["status"] == right_row["status"] == "ok":
                left.append(float(left_row["score"]))
                right.append(float(right_row["score"]))
        comparisons[name] = {
            **correlation(left, right),
            "status_pair_count": len(status_equal),
            "status_agreement": float(np.mean(status_equal)) if status_equal else math.nan,
        }
    return {
        "mode_summary": mode_summary,
        "batch_scaling": batch_summary,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=12)
    parser.add_argument("--probe-iterations", default="1,50,150,200")
    parser.add_argument("--formal-reference-count", type=int, default=8)
    parser.add_argument("--formal-repeats", type=int, default=2)
    parser.add_argument("--batch-sizes", default="2,8,32,64,128")
    parser.add_argument("--batch-repeats", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=2026082401)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    probe_iterations = tuple(int(value) for value in args.probe_iterations.split(","))
    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(","))
    all_cases = discover_cases(
        args.trajectory_root,
        case_count=args.case_count,
        probe_iterations=probe_iterations,
        seed=args.selection_seed,
    )
    cases = [
        case for index, case in enumerate(all_cases) if index % args.shard_count == args.shard_index
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(args.lib)
    neighborhood = NeighborhoodEvaluator(
        args.lib,
        settings=NeighborhoodSettings(
            psi_grid=48,
            psi_iterations=4,
            alpha_iterations=4,
            iota_degree=3,
        ),
    )
    records: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for case in cases:
        center_coils = token_coils(case["center_tokens"], case["nfp"])
        endpoints = [token_coils(value, case["nfp"]) for value in case["endpoint_tokens"]]
        maximum_batch = min(max(batch_sizes), len(endpoints))
        reference_indices = np.linspace(
            0,
            maximum_batch - 1,
            min(args.formal_reference_count, maximum_batch),
            dtype=int,
        )
        started = time.perf_counter()
        center = evaluator.evaluate(center_coils)
        center_wall_s = time.perf_counter() - started
        records.append(
            result_row(
                case=case,
                mode="center_independent",
                repeat=0,
                candidate_index=None,
                wall_s=center_wall_s,
                result=center,
            )
        )
        if not center.ok:
            selection_rows.append(
                {
                    "case_id": case["case_id"],
                    "status": "center_failed",
                    "center_status": center.status,
                }
            )
            continue
        continuation = center.continuation_state()
        for repeat in range(args.formal_repeats):
            mode_order = (
                (EvaluationMode.INDEPENDENT, EvaluationMode.STRICT_CONTINUATION)
                if (repeat + case["probe_iteration"]) % 2 == 0
                else (EvaluationMode.STRICT_CONTINUATION, EvaluationMode.INDEPENDENT)
            )
            for candidate_index in reference_indices:
                for mode in mode_order:
                    started = time.perf_counter()
                    result = evaluator.evaluate(
                        endpoints[int(candidate_index)],
                        mode=mode,
                        continuation=(
                            continuation if mode is EvaluationMode.STRICT_CONTINUATION else None
                        ),
                    )
                    records.append(
                        result_row(
                            case=case,
                            mode=mode.value,
                            repeat=repeat,
                            candidate_index=int(candidate_index),
                            wall_s=time.perf_counter() - started,
                            result=result,
                        )
                    )

        proxy_by_size: dict[int, Any] = {}
        for size in batch_sizes:
            actual_size = min(size, len(endpoints))
            for repeat in range(args.batch_repeats):
                started = time.perf_counter()
                batch = neighborhood.evaluate(
                    center_coils,
                    endpoints[:actual_size],
                    center,
                )
                wall_s = time.perf_counter() - started
                records.append(
                    {
                        "case_id": case["case_id"],
                        "trajectory_id": case["trajectory_id"],
                        "probe_iteration": case["probe_iteration"],
                        "nfp": case["nfp"],
                        "n_base_coils": case["n_base_coils"],
                        "mode": "neighborhood_batch",
                        "repeat": repeat,
                        "candidate_index": None,
                        "batch_size": actual_size,
                        "wall_s": wall_s,
                        "ok_fraction": batch.ok_fraction,
                        "timing": dict(batch.timing),
                        "diagnostics": dict(batch.diagnostics),
                    }
                )
                proxy_by_size[actual_size] = batch
        largest = proxy_by_size[maximum_batch]
        for candidate_index in reference_indices:
            records.append(
                result_row(
                    case=case,
                    mode="neighborhood_proxy",
                    repeat=0,
                    candidate_index=int(candidate_index),
                    batch_size=maximum_batch,
                    wall_s=largest.timing["total_s"] / maximum_batch,
                    result=largest.candidates[int(candidate_index)],
                )
            )
        selection_rows.append(
            {
                "case_id": case["case_id"],
                "trajectory_id": case["trajectory_id"],
                "probe_iteration": case["probe_iteration"],
                "nfp": case["nfp"],
                "n_base_coils": case["n_base_coils"],
                "endpoint_count": len(endpoints),
                "reference_indices": reference_indices.tolist(),
                "source_manifest": case["source_manifest"],
                "status": "ok",
            }
        )
        print(json.dumps({"event": "case_complete", **selection_rows[-1]}), flush=True)

    record_path = args.output_dir / f"records_shard_{args.shard_index:02d}.jsonl"
    with record_path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n")
    summary = {
        "format": "summary1_evaluator_modes_benchmark_v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "library": str(args.lib.resolve()),
        "trajectory_root": str(args.trajectory_root.resolve()),
        "selection": selection_rows,
        "protocol": {
            "case_count_total": len(all_cases),
            "case_count_shard": len(cases),
            "probe_iterations": probe_iterations,
            "formal_reference_count": args.formal_reference_count,
            "formal_repeats": args.formal_repeats,
            "batch_sizes": batch_sizes,
            "batch_repeats": args.batch_repeats,
            "selection_seed": args.selection_seed,
        },
        **summarize(records),
    }
    (args.output_dir / f"summary_shard_{args.shard_index:02d}.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", "records": len(records)}), flush=True)


if __name__ == "__main__":
    main()
