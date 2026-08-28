from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, REPO_ROOT / "gpu_backend" / "python"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from flow_matching.data import file_sha256  # noqa: E402
from flow_matching.trajectory_dataset import atomic_write_json  # noqa: E402
from scripts.qh_data_space_random_survey import (  # noqa: E402
    STANDALONE_SCORE_OVERRIDES,
    case_tokens,
    score_tokens_standalone,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a native score library against a frozen QH reference case.")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-case-sha", required=True)
    parser.add_argument("--expected-lib-sha")
    parser.add_argument("--expected-score", type=float, required=True)
    parser.add_argument("--score-atol", type=float, default=1.0e-5)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    case_sha = file_sha256(args.case)
    library_sha = file_sha256(args.lib)
    if case_sha != args.expected_case_sha:
        raise ValueError(f"reference case SHA-256 mismatch: {case_sha}")
    if args.expected_lib_sha and library_sha != args.expected_lib_sha:
        raise ValueError(f"score-library SHA-256 mismatch: {library_sha}")
    tokens, nfp = case_tokens(json.loads(args.case.read_text(encoding="utf-8")))
    started = time.perf_counter()
    measured = [
        score_tokens_standalone(
            args.lib,
            tokens,
            nfp=nfp,
            device=args.device,
        )
        for _ in range(3)
    ]
    results = [item[0] for item in measured]
    measured_scores = [float(result["score"]) for result in results]
    measured_wall_s = [float(item[1]) for item in measured]
    score_spread = max(measured_scores) - min(measured_scores)
    abi_values = [int(result["diagnostics"]["abi_version"]) for result in results]
    passed = (
        all(result["status"] == "ok" for result in results)
        and abi_values == [10, 10, 10]
        and all(
            abs(score - args.expected_score) <= args.score_atol
            for score in measured_scores
        )
        and score_spread <= args.score_atol
    )
    payload = {
        "format": "native_score_reference_validation_v3",
        "passed": passed,
        "case": str(args.case.resolve()),
        "case_sha256": case_sha,
        "library": str(args.lib.resolve()),
        "library_sha256": library_sha,
        "expected_score": args.expected_score,
        "score_atol": args.score_atol,
        "evaluation_count": len(results),
        "observed_scores": measured_scores,
        "score_deltas": [score - args.expected_score for score in measured_scores],
        "score_spread": score_spread,
        "statuses": [result["status"] for result in results],
        "abi_values": abi_values,
        "score_mode": {
            "base": "standalone library defaults",
            "explicit_overrides": dict(STANDALONE_SCORE_OVERRIDES),
        },
        "score_wall_s": measured_wall_s,
        "process_wall_s": time.perf_counter() - started,
        "components": [result["components"] for result in results],
    }
    atomic_write_json(args.output, payload, allow_nan=True)
    print(json.dumps(payload, indent=2), flush=True)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
