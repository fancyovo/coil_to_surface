from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(
    selection: dict[str, Any],
    adam: dict[str, Any],
    *,
    run_started_ns: int,
    selection_finished_ns: int,
    adam_started_ns: int,
    run_finished_ns: int,
    optimizer_seed: int,
) -> dict[str, Any]:
    timestamps = (
        run_started_ns,
        selection_finished_ns,
        adam_started_ns,
        run_finished_ns,
    )
    if any(value < 0 for value in timestamps):
        raise ValueError("timestamps must be nonnegative")
    if not run_started_ns <= selection_finished_ns <= adam_started_ns <= run_finished_ns:
        raise ValueError("timestamps are not monotone")
    selected_score = float(selection["selected_score"])
    adam_initial = float(adam["initial_score"])
    if not math.isclose(selected_score, adam_initial, rel_tol=0.0, abs_tol=1.0e-4):
        raise ValueError(
            f"selected score {selected_score} does not match Adam initial {adam_initial}"
        )
    best_score = float(adam["best_score"])
    return {
        "format": "qh_screened_start_adam_run_v1",
        "candidate_seed": int(selection["candidate_seed"]),
        "optimizer_seed": optimizer_seed,
        "candidate_count": int(selection["candidate_count"]),
        "nfp": int(selection["nfp"]),
        "n_base_coils": int(selection["n_base_coils"]),
        "selected_case_id": int(selection["selected_case_id"]),
        "selected_status": str(selection["selected_status"]),
        "initial_score": adam_initial,
        "final_score": float(adam["final_score"]),
        "best_score": best_score,
        "best_iteration": int(adam["best_iteration"]),
        "gain": best_score - adam_initial,
        "completed_iterations": int(adam["completed_iterations"]),
        "completed_adam_steps": int(adam["completed_adam_steps"]),
        "stop_reason": str(adam["stop_reason"]),
        "crossed_40": best_score >= 40.0,
        "crossed_50": best_score >= 50.0,
        "timing_s": {
            "candidate_selection": (selection_finished_ns - run_started_ns) / 1.0e9,
            "selection_to_adam_overhead": (adam_started_ns - selection_finished_ns)
            / 1.0e9,
            "adam_process": (run_finished_ns - adam_started_ns) / 1.0e9,
            "adam_internal": float(adam["total_wall_s"]),
            "end_to_end": (run_finished_ns - run_started_ns) / 1.0e9,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the exact end-to-end summary for one screened-start Adam run."
    )
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--adam-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-started-ns", type=int, required=True)
    parser.add_argument("--selection-finished-ns", type=int, required=True)
    parser.add_argument("--adam-started-ns", type=int, required=True)
    parser.add_argument("--run-finished-ns", type=int, required=True)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    args = parser.parse_args()
    summary = build_summary(
        read_json(args.selection_summary),
        read_json(args.adam_summary),
        run_started_ns=args.run_started_ns,
        selection_finished_ns=args.selection_finished_ns,
        adam_started_ns=args.adam_started_ns,
        run_finished_ns=args.run_finished_ns,
        optimizer_seed=args.optimizer_seed,
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **summary}, separators=(",", ":")))


if __name__ == "__main__":
    main()
