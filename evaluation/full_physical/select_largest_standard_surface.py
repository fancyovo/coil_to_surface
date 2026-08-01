#!/usr/bin/env python3
"""Select the largest independently validated standard LS/Newton surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-open-upper-bound", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    pattern = "*/standard_rho_1/summary.json"
    for summary_path in sorted(args.candidate_root.glob(pattern)):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        level = float(summary["target_s"])
        accepted = bool(summary.get("accepted_for_downstream", False))
        surface = summary_path.parent / "boozer_standard.npz"
        rows.append(
            {
                "target_s": level,
                "accepted": accepted and surface.is_file(),
                "summary": str(summary_path.resolve()),
                "surface": str(surface.resolve()),
                "acceptance_checks": summary.get("acceptance_checks", {}),
                "branch_diagnostics": summary.get("branch_diagnostics", {}),
            }
        )

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
