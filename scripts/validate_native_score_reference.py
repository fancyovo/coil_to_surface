from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, REPO_ROOT / "gpu_backend" / "python"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from flow_matching.data import file_sha256  # noqa: E402
from flow_matching.trajectory_dataset import atomic_write_json  # noqa: E402
from scripts.qh_data_space_random_survey import score_tokens_standalone  # noqa: E402


def case_tokens(payload: dict) -> tuple[np.ndarray, int]:
    raw = payload["raw"]
    unit = str(raw.get("current_unit", "A")).lower()
    if unit in {"a", "amp", "amps"}:
        current_scale = 1.0
    elif unit in {"ma", "megaamp", "megaamps"}:
        current_scale = 1.0e6
    else:
        raise ValueError(f"unsupported current unit {unit!r}")
    x = np.asarray(raw["x"], dtype=np.float64)
    y = np.asarray(raw["y"], dtype=np.float64)
    z = np.asarray(raw["z"], dtype=np.float64)
    current = np.asarray(raw["current"], dtype=np.float64) * current_scale
    if x.shape != y.shape or x.shape != z.shape or x.shape[1] != 33:
        raise ValueError("reference case has inconsistent Fourier arrays")
    tokens = np.concatenate((x, y, z, current[:, None]), axis=1)
    return tokens, int(payload.get("nfp", raw["nfp"]))


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
    result, score_wall_s = score_tokens_standalone(
        args.lib,
        tokens,
        nfp=nfp,
        device=args.device,
    )
    score = float(result["score"])
    abi = int(result["diagnostics"]["abi_version"])
    passed = (
        result["status"] == "ok"
        and abi == 10
        and abs(score - args.expected_score) <= args.score_atol
    )
    payload = {
        "format": "native_score_reference_validation_v1",
        "passed": passed,
        "case": str(args.case.resolve()),
        "case_sha256": case_sha,
        "library": str(args.lib.resolve()),
        "library_sha256": library_sha,
        "expected_score": args.expected_score,
        "score_atol": args.score_atol,
        "observed_score": score,
        "score_delta": score - args.expected_score,
        "status": result["status"],
        "abi": abi,
        "score_mode": "standalone library defaults; no Python overrides",
        "score_wall_s": score_wall_s,
        "process_wall_s": time.perf_counter() - started,
        "components": result["components"],
    }
    atomic_write_json(args.output, payload, allow_nan=True)
    print(json.dumps(payload, indent=2), flush=True)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
