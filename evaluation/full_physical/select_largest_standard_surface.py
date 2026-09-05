#!/usr/bin/env python3
"""Select the largest independently validated standard LS/Newton surface."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-open-upper-bound", action="store_true")
    return parser.parse_args()


def _final_abs_volume(summary: dict) -> float | None:
    for stage in ("newton", "least_squares"):
        value = (
            summary.get(stage, {})
            .get("state", {})
            .get("geometry", {})
            .get("signed_volume_m3")
        )
        if value is not None and math.isfinite(float(value)):
            return abs(float(value))
    return None


def load_candidate_rows(candidate_root: Path) -> list[dict]:
    rows = []
    for candidate_dir in sorted(candidate_root.glob("s_*")):
        if not candidate_dir.is_dir():
            continue
        try:
            directory_level = float(
                candidate_dir.name.removeprefix("s_").replace("p", ".")
            )
        except ValueError:
            continue
        summary_path = candidate_dir / "standard_rho_1" / "summary.json"
        if not summary_path.is_file():
            rejection_path = candidate_dir / "alpha" / "rejection.json"
            rejection = None
            if rejection_path.is_file():
                try:
                    candidate = json.loads(rejection_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    candidate = None
                if candidate is not None and candidate.get("status") == "rejected":
                    rejection = candidate
            rows.append(
                {
                    "target_s": directory_level,
                    "solver_accepted": False,
                    "accepted": False,
                    "evaluation_complete": rejection is not None,
                    "summary": None,
                    "surface": None,
                    "final_abs_volume_m3": None,
                    "acceptance_checks": {},
                    "branch_diagnostics": {},
                    "failure_stage": (
                        rejection.get("stage")
                        if rejection is not None
                        else "missing_standard_summary"
                    ),
                    "rejection": rejection,
                    "rejection_record": (
                        str(rejection_path.resolve()) if rejection is not None else None
                    ),
                }
            )
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        level = float(summary["target_s"])
        solver_accepted = bool(summary.get("accepted_for_downstream", False))
        surface = summary_path.parent / "boozer_standard.npz"
        final_abs_volume = _final_abs_volume(summary)
        provenance_ok = (
            summary.get("source_surface_kind") == "alpha_nu"
            and summary.get("output_surface_kind")
            == "alpha_nu_standard_ls_newton"
        )
        evaluation_complete = not solver_accepted or (
            surface.is_file() and provenance_ok
        )
        rows.append(
            {
                "target_s": level,
                "solver_accepted": solver_accepted,
                "accepted": solver_accepted and surface.is_file() and provenance_ok,
                "evaluation_complete": evaluation_complete,
                "summary": str(summary_path.resolve()),
                "surface": str(surface.resolve()),
                "final_abs_volume_m3": final_abs_volume,
                "acceptance_checks": summary.get("acceptance_checks", {}),
                "branch_diagnostics": summary.get("branch_diagnostics", {}),
                "surface_provenance": {
                    "source_surface_kind": summary.get("source_surface_kind"),
                    "output_surface_kind": summary.get("output_surface_kind"),
                    "valid": provenance_ok,
                },
                "failure_stage": (
                    None
                    if solver_accepted and surface.is_file() and provenance_ok
                    else "standard_surface_acceptance"
                    if not solver_accepted
                    else "invalid_surface_provenance"
                    if surface.is_file()
                    else "missing_accepted_surface_artifact"
                ),
            }
        )
    return sorted(rows, key=lambda row: row["target_s"])


def apply_nested_volume_check(rows: list[dict]) -> None:
    """Reject solver successes that jump to a smaller enclosed-volume branch."""

    largest_nested_volume = None
    for row in sorted(rows, key=lambda item: item["target_s"]):
        row["branch_consistency"] = {"nested_volume_increasing": None}
        if not row["accepted"]:
            continue
        volume = row["final_abs_volume_m3"]
        if volume is None:
            row["accepted"] = False
            row["branch_consistency"] = {
                "nested_volume_increasing": False,
                "rejection_reason": "missing_final_signed_volume",
                "previous_largest_abs_volume_m3": largest_nested_volume,
            }
            continue
        if largest_nested_volume is not None and volume <= largest_nested_volume:
            row["accepted"] = False
            row["branch_consistency"] = {
                "nested_volume_increasing": False,
                "rejection_reason": "non_increasing_enclosed_volume",
                "previous_largest_abs_volume_m3": largest_nested_volume,
            }
            continue
        row["branch_consistency"] = {
            "nested_volume_increasing": True,
            "previous_largest_abs_volume_m3": largest_nested_volume,
        }
        largest_nested_volume = volume


def main() -> int:
    args = parse_args()
    rows = load_candidate_rows(args.candidate_root)
    incomplete = [row for row in rows if not row.get("evaluation_complete", False)]
    if incomplete:
        levels = ", ".join(str(row["target_s"]) for row in incomplete)
        print(
            f"ERROR: incomplete surface candidates without a structured rejection: {levels}",
            file=sys.stderr,
        )
        return 3
    apply_nested_volume_check(rows)

    accepted_rows = [row for row in rows if row["accepted"]]
    if not accepted_rows:
        print("ERROR: no accepted standard LS/Newton surface", file=sys.stderr)
        return 1
    selected = max(accepted_rows, key=lambda row: row["target_s"])
    outer = [row for row in rows if row["target_s"] > selected["target_s"]]
    outer_failures = [row for row in outer if not row["accepted"]]
    if not outer_failures and not args.allow_open_upper_bound:
        print(
            "ERROR: largest tested candidate passed; extend S_EDGES outward or use "
            "--allow-open-upper-bound explicitly",
            file=sys.stderr,
        )
        return 2

    output = {
        "selected": selected,
        "nearest_outer_failure": (
            min(outer_failures, key=lambda row: row["target_s"])
            if outer_failures
            else None
        ),
        "candidates": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(selected["surface"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
