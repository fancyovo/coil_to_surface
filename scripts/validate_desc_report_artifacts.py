#!/usr/bin/env python3
"""Fail when a physical-evaluation report omits generated DESC figures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--desc-dir", required=True, type=Path)
    parser.add_argument(
        "--target-helicity",
        default="QH",
        choices=("QA", "QH", "QP"),
        help="Target QS channel required among the five core figures.",
    )
    return parser.parse_args()


def successful_plot_names(summary: dict) -> set[str]:
    names: set[str] = set()
    for group_name in ("plots_initial", "plots_final"):
        group = summary.get(group_name, {})
        if not isinstance(group, dict):
            continue
        for result in group.values():
            if not isinstance(result, dict) or not result.get("success"):
                continue
            path = result.get("path")
            if path:
                names.add(Path(path).name)
    return names


def main() -> int:
    args = parse_args()
    report = args.report.resolve()
    desc_dir = args.desc_dir.resolve()
    summary_path = desc_dir / "summary.json"

    errors: list[str] = []
    if not report.is_file():
        errors.append(f"report does not exist: {report}")
    if not summary_path.is_file():
        errors.append(f"DESC summary does not exist: {summary_path}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    report_text = report.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    generated = {path.name for path in desc_dir.glob("*.png")}
    successful = successful_plot_names(summary)
    core = {
        "boundary.png",
        "boozer_B.png",
        "boozer_modes.png",
        "iota.png",
        f"qs_{args.target_helicity}.png",
    }

    for name in sorted(core - generated):
        errors.append(f"missing core DESC figure: {desc_dir / name}")
    for name in sorted(successful - generated):
        errors.append(f"summary marks a missing figure successful: {name}")

    required = generated | successful | core
    for name in sorted(required & generated):
        figure = desc_dir / name
        relative = Path(os.path.relpath(figure, report.parent)).as_posix()
        if relative not in report_text:
            errors.append(f"report does not reference generated figure: {relative}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(
        f"PASS: report references all {len(generated)} DESC PNGs "
        f"({len(core)} core, {len(successful)} marked successful)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
