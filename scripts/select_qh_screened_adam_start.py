from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("candidate score pool is empty")
    return max(rows, key=lambda row: (float(row["score"]), -int(row["case_id"])))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the highest native-score latent from an IID candidate pool."
    )
    parser.add_argument("--scored-cases", type=Path, required=True)
    parser.add_argument("--random-latents", type=Path, required=True)
    parser.add_argument("--output-start", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--nfp", type=int, required=True)
    parser.add_argument("--n-base-coils", type=int, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.scored_cases)
    with np.load(args.random_latents, allow_pickle=False) as payload:
        latent = np.asarray(payload["latent"], dtype=np.float32)
    expected_shape = (args.expected_count, args.n_base_coils, 100)
    if latent.shape != expected_shape:
        raise ValueError(f"latent shape {latent.shape} != {expected_shape}")
    if len(rows) != args.expected_count:
        raise ValueError(f"score row count {len(rows)} != {args.expected_count}")
    case_ids = [int(row["case_id"]) for row in rows]
    if sorted(case_ids) != list(range(args.expected_count)):
        raise ValueError("candidate case IDs must be a permutation of the latent rows")

    selected = select_best_row(rows)
    case_id = int(selected["case_id"])
    start = {
        "format": "qh_screened_adam_start_v1",
        "flow_prior_start": {
            "noise": latent[case_id].tolist(),
            "source": "best_of_iid_random_candidate_pool",
            "source_case_id": case_id,
            "recorded_score": float(selected["score"]),
            "recorded_status": str(selected["status"]),
            "candidate_count": args.expected_count,
            "candidate_seed": args.seed,
            "nfp": args.nfp,
            "n_base_coils": args.n_base_coils,
        },
    }
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    statuses = np.asarray([str(row["status"]) for row in rows], dtype="U32")
    summary = {
        "format": "qh_screened_adam_selection_v1",
        "candidate_seed": args.seed,
        "candidate_count": args.expected_count,
        "nfp": args.nfp,
        "n_base_coils": args.n_base_coils,
        "selected_case_id": case_id,
        "selected_score": float(selected["score"]),
        "selected_status": str(selected["status"]),
        "selected_latent_rms": float(
            np.sqrt(np.mean(latent[case_id].astype(np.float64) ** 2))
        ),
        "candidate_score": {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "p90": float(np.percentile(scores, 90)),
            "p95": float(np.percentile(scores, 95)),
            "p99": float(np.percentile(scores, 99)),
            "max": float(np.max(scores)),
        },
        "candidate_status_counts": {
            status: int(np.sum(statuses == status)) for status in np.unique(statuses)
        },
    }
    args.output_start.write_text(json.dumps(start, indent=2) + "\n", encoding="utf-8")
    args.output_summary.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "selected", **summary}, separators=(",", ":")))


if __name__ == "__main__":
    main()
