from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def first_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    row = next((item for item in rows if float(item["best_score"]) >= threshold), None)
    if row is None:
        return None
    return {
        "iteration": int(row["iteration"]),
        "best_score": float(row["best_score"]),
        "wall_s": float(row["total_wall_s"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.run_dir / "history.jsonl")
    baseline = read_jsonl(args.baseline_history)
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    best_payload = json.loads((args.run_dir / "best.json").read_text(encoding="utf-8"))
    best_metadata = best_payload["flow_prior_local_full_gradient_adam"]
    start_payload = json.loads(
        (args.run_dir / "trajectory" / "step_0000.json").read_text(encoding="utf-8")
    )
    start_result = start_payload["flow_prior_local_full_gradient_adam"]["native_score"]
    best_result = best_metadata["native_score"]

    iterations = [int(row["iteration"]) for row in rows]
    history_best = max(rows, key=lambda row: float(row["current_score"]))
    endpoint_valid = [
        row["gradient_endpoint_statuses"] == {"ok": 600} for row in rows
    ]
    checks = {
        "contiguous_iterations": iterations == list(range(1, len(rows) + 1)),
        "summary_iteration_matches": int(summary["completed_iterations"]) == len(rows),
        "summary_best_matches_history": bool(
            np.isclose(summary["best_score"], history_best["current_score"], atol=1.0e-12)
            and int(summary["best_iteration"]) == int(history_best["iteration"])
        ),
        "best_artifact_matches_history": bool(
            np.isclose(best_metadata["best_score"], history_best["current_score"], atol=1.0e-12)
            and int(best_metadata["iteration"]) == int(history_best["iteration"])
        ),
        "all_formal_centers_ok": all(row["current_status"] == "ok" for row in rows),
        "all_600_endpoints_ok": all(endpoint_valid),
        "all_gradient_steps_applied": all(row["gradient_step_applied"] for row in rows),
        "all_centers_accepted": all(row["center_update_accepted"] for row in rows),
        "all_updates_full_fraction": all(
            float(row["center_acceptance_fraction"]) == 1.0 for row in rows
        ),
        "no_temporal_outliers": all(
            not row["temporal_gradient_outlier"] and not row["temporal_update_outlier"]
            for row in rows
        ),
        "stderr_empty": (args.run_dir / "slurm.err").stat().st_size == 0,
        "clean_gpu_postflight": all(
            int(line.split(",")[3].strip()) == 0
            and int(line.split(",")[4].strip()) <= 2
            for line in (args.run_dir / "gpu_postflight.csv").read_text().splitlines()
            if line.strip()
        ),
    }

    matched_wall = []
    full_wall = np.asarray([row["total_wall_s"] for row in rows], dtype=np.float64)
    for baseline_step in (200, 500, 750, 1000):
        baseline_row = baseline[baseline_step - 1]
        index = int(np.searchsorted(full_wall, baseline_row["total_wall_s"], side="right") - 1)
        full_row = rows[index]
        matched_wall.append(
            {
                "wall_s": float(baseline_row["total_wall_s"]),
                "baseline_iteration": baseline_step,
                "baseline_best_score": float(baseline_row["best_score"]),
                "full_gradient_iteration": int(full_row["iteration"]),
                "full_gradient_best_score": float(full_row["best_score"]),
                "full_minus_baseline": float(
                    full_row["best_score"] - baseline_row["best_score"]
                ),
            }
        )

    thresholds = {}
    for threshold in (90.0, 91.0, 92.0, 92.5, 93.0, 93.5, 93.6):
        thresholds[str(threshold)] = {
            "full_gradient": first_threshold(rows, threshold),
            "spsa": first_threshold(baseline, threshold),
        }

    component_delta = {
        name: float(best_result["components"][name] - start_result["components"][name])
        for name in start_result["components"]
    }
    output = {
        "format": "local_full_gradient_adam_acceptance_v1",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "run": {
            "stop_reason": summary["stop_reason"],
            "completed_iterations": len(rows),
            "completed_adam_steps": int(summary["completed_adam_steps"]),
            "total_wall_s": float(summary["total_wall_s"]),
            "initial_score": float(summary["initial_score"]),
            "final_score": float(summary["final_score"]),
            "best_score": float(summary["best_score"]),
            "best_iteration": int(summary["best_iteration"]),
            "formal_score_gain": float(summary["best_score"] - summary["initial_score"]),
        },
        "stability": {
            "valid_endpoint_iterations": int(sum(endpoint_valid)),
            "accepted_center_iterations": int(
                sum(bool(row["center_update_accepted"]) for row in rows)
            ),
            "backtracked_iterations": int(
                sum(float(row["center_acceptance_fraction"]) < 1.0 for row in rows)
            ),
            "temporal_gradient_outliers": int(
                sum(bool(row["temporal_gradient_outlier"]) for row in rows)
            ),
            "temporal_update_outliers": int(
                sum(bool(row["temporal_update_outlier"]) for row in rows)
            ),
        },
        "timing_s": {
            "iteration": quantiles([row["iteration_wall_s"] for row in rows]),
            "gradient": quantiles([row["gradient_wall_s"] for row in rows]),
            "endpoint_flow": quantiles([row["endpoint_decode_wall_s"] for row in rows]),
            "formal_proposal": quantiles(
                [row["proposal_decode_wall_s"] + row["proposal_score_wall_s"] for row in rows]
            ),
        },
        "physics": {
            "start_components": start_result["components"],
            "best_components": best_result["components"],
            "component_delta": component_delta,
            "start_qh_error": float(start_result["diagnostics"]["qs_global_error"]),
            "best_qh_error": float(best_result["diagnostics"]["qs_global_error"]),
            "qh_error_reduction_factor": float(
                start_result["diagnostics"]["qs_global_error"]
                / best_result["diagnostics"]["qs_global_error"]
            ),
            "best_qa_error": float(best_result["diagnostics"]["qs_qa_global_error"]),
            "best_qp_error": float(best_result["diagnostics"]["qs_qp_global_error"]),
            "best_iota_min": float(best_result["diagnostics"]["iota_min"]),
            "best_iota_max": float(best_result["diagnostics"]["iota_max"]),
            "best_surface_level": float(best_result["diagnostics"]["surface_level"]),
            "best_surface_inverse_aspect_ratio": float(
                best_result["diagnostics"]["surface_inverse_aspect_ratio"]
            ),
            "best_one_period_drift_p95": float(
                best_result["diagnostics"]["surface_one_period_drift_relative_p95"]
            ),
        },
        "spsa_comparison": {
            "baseline_iterations": len(baseline),
            "baseline_best_score": float(baseline[-1]["best_score"]),
            "baseline_total_wall_s": float(baseline[-1]["total_wall_s"]),
            "same_iteration": [
                {
                    "iteration": step,
                    "full_gradient_best_score": float(rows[step - 1]["best_score"]),
                    "spsa_best_score": float(baseline[step - 1]["best_score"]),
                    "full_minus_spsa": float(
                        rows[step - 1]["best_score"] - baseline[step - 1]["best_score"]
                    ),
                }
                for step in (50, 100, 200, 300, 500, 750, 1000)
            ],
            "same_wall_time": matched_wall,
            "thresholds": thresholds,
        },
        "sha256": {
            name: sha256(args.run_dir / name)
            for name in ("best.json", "history.jsonl", "state_latest.npz", "summary.json")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
