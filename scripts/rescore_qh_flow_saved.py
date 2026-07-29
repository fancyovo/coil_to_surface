from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
for path in (REPO_ROOT, GPU_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.optimize_native_score_cem import token_case


QUASR_V3_P10 = 41.31
QUASR_V3_MEDIAN = 51.44


def load_rows(path: Path, pattern: str) -> list[dict[str, Any]]:
    rows = []
    for input_path in sorted(path.glob(pattern)):
        with input_path.open("r", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    rows.sort(key=lambda row: int(row["candidate_id"]))
    candidate_ids = [int(row["candidate_id"]) for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"duplicate candidate IDs in {path}/{pattern}")
    return rows


def parse_candidate_ids(value: str) -> set[int] | None:
    if not value.strip():
        return None
    return {int(item) for item in value.split(",") if item.strip()}


def status_of(row: dict[str, Any], score_key: str) -> str:
    if not row.get("geometry_eligible", False):
        return "geometry_rejected"
    native = row.get(score_key)
    if native is None:
        return "error"
    return str(native["status"])


def score_of(row: dict[str, Any], score_key: str) -> float:
    native = row.get(score_key)
    return float(native["score"]) if native is not None else 0.0


def score_partition(args: argparse.Namespace) -> None:
    from stellarator_gpu import score_coils_native

    rows = load_rows(args.input_dir, "rank_*.jsonl")
    selected_ids = parse_candidate_ids(args.candidate_ids)
    if selected_ids is not None:
        available = {int(row["candidate_id"]) for row in rows}
        missing = selected_ids - available
        if missing:
            raise ValueError(f"candidate IDs not found: {sorted(missing)}")
    rows = [
        row
        for row in rows
        if int(row["candidate_id"]) % args.world_size == args.rank
        and (selected_ids is None or int(row["candidate_id"]) in selected_ids)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"rescore_rank_{args.rank:02d}.jsonl"
    started = time.perf_counter()
    scored = 0
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            result = None
            error = None
            elapsed = 0.0
            if row["geometry_eligible"]:
                case = token_case(
                    np.asarray(row["tokens"], dtype=np.float64),
                    nfp=int(row["nfp"]),
                    target="QH",
                    metadata={"flow_candidate_id": int(row["candidate_id"])},
                )
                raw = case["raw"]
                score_started = time.perf_counter()
                try:
                    result = score_coils_native(
                        args.lib,
                        raw["x"],
                        raw["y"],
                        raw["z"],
                        raw["current"],
                        int(row["nfp"]),
                        device_id=args.rank,
                        target_helicity=(1, int(row["nfp"])),
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - score_started
                scored += 1
            output = {
                "candidate_id": int(row["candidate_id"]),
                "nfp": int(row["nfp"]),
                "n_coils": int(row["n_coils"]),
                "geometry_eligible": bool(row["geometry_eligible"]),
                "old_status": status_of(row, "native_score"),
                "old_score": score_of(row, "native_score"),
                "native_score_v3": result,
                "score_error": error,
                "score_wall_s": elapsed,
            }
            stream.write(json.dumps(output, separators=(",", ":"), allow_nan=True) + "\n")
            stream.flush()
            if scored and scored % 10 == 0:
                print(
                    json.dumps(
                        {
                            "event": "rescore_progress",
                            "rank": args.rank,
                            "scored": scored,
                            "assigned": len(rows),
                            "elapsed_s": time.perf_counter() - started,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    runtime = {
        "rank": args.rank,
        "assigned": len(rows),
        "scored": scored,
        "wall_s": time.perf_counter() - started,
    }
    (args.output_dir / f"runtime_rank_{args.rank:02d}.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def minimum_absolute_iota(diagnostics: dict[str, Any]) -> float:
    lower = float(diagnostics["iota_min"])
    upper = float(diagnostics["iota_max"])
    if lower <= 0.0 <= upper:
        return 0.0
    return min(abs(lower), abs(upper))


def compact_row(old: dict[str, Any], rescored: dict[str, Any]) -> dict[str, Any]:
    native = rescored.get("native_score_v3")
    output = {
        "candidate_id": int(old["candidate_id"]),
        "nfp": int(old["nfp"]),
        "n_coils": int(old["n_coils"]),
        "old_status": status_of(old, "native_score"),
        "old_score": score_of(old, "native_score"),
        "new_status": status_of(
            {**rescored, "geometry_eligible": old["geometry_eligible"]},
            "native_score_v3",
        ),
        "new_score": float(native["score"]) if native is not None else 0.0,
        "score_wall_s": float(rescored["score_wall_s"]),
    }
    if native is None or str(native["status"]) != "ok":
        return output
    diagnostics = native["diagnostics"]
    output["components"] = native["components"]
    output["diagnostics"] = {
        "iota_star": minimum_absolute_iota(diagnostics),
        "surface_inverse_aspect_ratio": float(
            diagnostics["surface_inverse_aspect_ratio"]
        ),
        "surface_volume": float(diagnostics["surface_volume"]),
        "surface_one_period_drift_relative_p95": float(
            diagnostics["surface_one_period_drift_relative_p95"]
        ),
        "surface_long_drift_relative_p95": float(
            diagnostics["surface_drift_relative_p95"]
        ),
        "qh_error": float(diagnostics["qs_global_error"]),
        "qh_error_per_helicity": float(diagnostics["qs_global_error"])
        / math.hypot(1.0, int(old["nfp"])),
        "qa_error": float(diagnostics["qs_qa_global_error"]),
        "qp_error": float(diagnostics["qs_qp_global_error"]),
        "helicity_advantage": float(
            diagnostics["score_qh_helicity_advantage"]
        ),
        "helicity_quality": float(diagnostics["score_qh_helicity_quality"]),
        "helicity_factor": float(
            diagnostics["score_qh_total_helicity_factor"]
        ),
        "iota_factor": float(diagnostics["score_qh_total_iota_factor"]),
        "score_before_gates": float(diagnostics["score_before_qh_iota_gate"]),
    }
    return output


def plot_comparison(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = [row for row in rows if "diagnostics" in row]
    old = np.asarray([row["old_score"] for row in scored])
    new = np.asarray([row["new_score"] for row in scored])
    advantage = np.asarray(
        [row["diagnostics"]["helicity_advantage"] for row in scored]
    )
    iota = np.asarray([row["diagnostics"]["iota_star"] for row in scored])
    qh = np.asarray(
        [row["diagnostics"]["qh_error_per_helicity"] for row in scored]
    )
    competitor = np.asarray(
        [
            min(row["diagnostics"]["qa_error"], row["diagnostics"]["qp_error"])
            for row in scored
        ]
    )

    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.1))
    axes[0, 0].scatter(old, new, s=9, alpha=0.35, color="#277c83")
    limit = max(float(np.max(old)), float(np.max(new)), 1.0)
    axes[0, 0].plot([0, limit], [0, limit], color="#555555", linestyle=":")
    fake = [row for row in scored if row["candidate_id"] == 1439]
    if fake:
        axes[0, 0].scatter(
            [fake[0]["old_score"]], [fake[0]["new_score"]], marker="*", s=180,
            color="#b43b2f", edgecolor="black", linewidth=0.6,
            label="old false positive 1439",
        )
        axes[0, 0].legend(fontsize=8)
    axes[0, 0].set(xlabel="old score", ylabel="score v3", title="Matched samples")

    axes[0, 1].hist(old, bins=35, alpha=0.55, color="#4f5d75", label="old")
    axes[0, 1].hist(new, bins=35, alpha=0.65, color="#b43b2f", label="v3")
    axes[0, 1].axvline(QUASR_V3_P10, color="#555555", linestyle=":", label="QUASR v3 P10")
    axes[0, 1].set(xlabel="score", ylabel="count", title="Successful native evaluations")
    axes[0, 1].legend(fontsize=8)

    scatter = axes[1, 0].scatter(
        advantage, new, c=np.clip(iota, 0.0, 1.6), s=9, alpha=0.45,
        cmap="viridis", rasterized=True,
    )
    axes[1, 0].axvline(0.10, color="#555555", linestyle=":", label="calibrated bad")
    axes[1, 0].axvline(0.30, color="#555555", linestyle="--", label="calibrated good")
    axes[1, 0].set(
        xlabel="QH relative helicity advantage", ylabel="score v3",
        title="Target selectivity",
    )
    axes[1, 0].legend(fontsize=8)
    figure.colorbar(scatter, ax=axes[1, 0], label=r"minimum $|\iota|$")

    scatter = axes[1, 1].scatter(
        qh, competitor, c=new, s=9, alpha=0.4, cmap="plasma", rasterized=True
    )
    positive = np.concatenate([qh[qh > 0.0], competitor[competitor > 0.0]])
    lower = float(np.min(positive))
    upper = float(np.max(positive))
    axes[1, 1].plot([lower, upper], [lower, upper], color="#555555", linestyle=":")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(
        xlabel="QH error per helicity", ylabel="best QA/QP competitor error",
        title="Target versus competing symmetry",
    )
    figure.colorbar(scatter, ax=axes[1, 1], label="score v3")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("Saved QH flow samples rescored with score v3")
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def analyze(args: argparse.Namespace) -> None:
    old_rows = load_rows(args.input_dir, "rank_*.jsonl")
    rescored_rows = load_rows(args.output_dir, "rescore_rank_*.jsonl")
    old_by_id = {int(row["candidate_id"]): row for row in old_rows}
    compact = []
    for rescored in rescored_rows:
        candidate_id = int(rescored["candidate_id"])
        if candidate_id not in old_by_id:
            raise ValueError(f"rescored candidate {candidate_id} is missing from input")
        compact.append(compact_row(old_by_id[candidate_id], rescored))
    compact.sort(key=lambda row: row["candidate_id"])
    if not compact:
        raise RuntimeError("no rescored candidates found")

    statuses = Counter((row["old_status"], row["new_status"]) for row in compact)
    old_ok = [row for row in compact if row["old_status"] == "ok"]
    new_ok = [row for row in compact if row["new_status"] == "ok"]
    diagnostics = [row for row in new_ok if "diagnostics" in row]
    best_new = max(compact, key=lambda row: row["new_score"])
    best_old = max(compact, key=lambda row: row["old_score"])
    top_new = sorted(compact, key=lambda row: row["new_score"], reverse=True)[:50]
    fake = next((row for row in compact if row["candidate_id"] == 1439), None)
    old_high = [row for row in compact if row["old_score"] >= 55.0]
    runtimes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.output_dir.glob("runtime_rank_*.json"))
    ]
    summary = {
        "input_count": len(old_rows),
        "rescored_count": len(compact),
        "status_transitions": {
            f"{old}->{new}": count for (old, new), count in sorted(statuses.items())
        },
        "old_score_all": distribution([row["old_score"] for row in compact]),
        "new_score_all": distribution([row["new_score"] for row in compact]),
        "old_score_ok": distribution([row["old_score"] for row in old_ok]),
        "new_score_ok": distribution([row["new_score"] for row in new_ok]),
        "new_reference_counts": {
            "score_ge_quasr_v3_p10": sum(
                row["new_score"] >= QUASR_V3_P10 for row in compact
            ),
            "score_ge_quasr_v3_median": sum(
                row["new_score"] >= QUASR_V3_MEDIAN for row in compact
            ),
            "helicity_advantage_ge_0_10": sum(
                row["diagnostics"]["helicity_advantage"] >= 0.10
                for row in diagnostics
            ),
            "helicity_advantage_ge_0_20": sum(
                row["diagnostics"]["helicity_advantage"] >= 0.20
                for row in diagnostics
            ),
            "helicity_advantage_ge_0_30": sum(
                row["diagnostics"]["helicity_advantage"] >= 0.30
                for row in diagnostics
            ),
            "iota_ge_1_advantage_ge_0_10_size_ge_0_02": sum(
                row["diagnostics"]["iota_star"] >= 1.0
                and row["diagnostics"]["helicity_advantage"] >= 0.10
                and row["diagnostics"]["surface_inverse_aspect_ratio"] >= 0.02
                for row in diagnostics
            ),
        },
        "old_high_score_ge_55": {
            "count": len(old_high),
            "new_score": distribution([row["new_score"] for row in old_high]),
        },
        "old_false_positive_1439": fake,
        "best_old": best_old,
        "best_new": best_new,
        "top_new": top_new,
        "runtime": {
            "rank": runtimes,
            "max_rank_wall_s": max((row["wall_s"] for row in runtimes), default=None),
            "sum_score_wall_s": sum(row["score_wall_s"] for row in compact),
        },
        "library_sha256": hashlib.sha256(args.lib.read_bytes()).hexdigest(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    top_dir = args.output_dir / "top_cases"
    top_dir.mkdir(exist_ok=True)
    for rank, row in enumerate(top_new[:10], start=1):
        old = old_by_id[row["candidate_id"]]
        case = token_case(
            np.asarray(old["tokens"], dtype=np.float64),
            nfp=int(old["nfp"]),
            target="QH",
            metadata={"flow_candidate_id": row["candidate_id"], "score_v3_rank": rank},
        )
        case["flow_rescore"] = row
        (top_dir / f"rank_{rank:03d}_id_{row['candidate_id']:06d}.json").write_text(
            json.dumps(case, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    plot_comparison(compact, args.output_dir / "score_v3_comparison.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore saved QH flow samples with the current native score.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--candidate-ids", default="")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    if args.analyze_only:
        analyze(args)
    else:
        score_partition(args)


if __name__ == "__main__":
    main()
