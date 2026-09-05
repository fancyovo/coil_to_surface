#!/usr/bin/env python3
"""Select the source-psi fit with the largest verified physical screen radius."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-axis-residual", type=float, default=1e-6)
    parser.add_argument("--max-validation-rms", type=float, default=5e-4)
    parser.add_argument("--max-validation-train-ratio", type=float, default=2.0)
    parser.add_argument("--allow-open-upper-bound", action="store_true")
    return parser.parse_args()


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def inspect_candidate(
    candidate_dir: Path,
    *,
    max_axis_residual: float,
    max_validation_rms: float,
    max_validation_train_ratio: float,
    allow_open_upper_bound: bool,
) -> dict[str, object]:
    summary_path = candidate_dir / "summary.json"
    row: dict[str, object] = {
        "candidate_dir": str(candidate_dir.resolve()),
        "summary": str(summary_path.resolve()),
        "eligible": False,
        "rejection_reasons": [],
    }
    if not summary_path.is_file():
        row["rejection_reasons"] = ["missing_summary"]
        return row

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    axis = summary.get("axis", {})
    psi = summary.get("psi", {})
    fit = psi.get("fit_info", {})
    axis_residual = _finite_float(axis.get("best_residual"))
    train_rms = _finite_float(fit.get("train_rms"))
    validation_rms = _finite_float(fit.get("validation_rms"))
    a_value = _finite_float(psi.get("a"))
    ratio = None
    if train_rms is not None and train_rms > 0 and validation_rms is not None:
        ratio = validation_rms / train_rms

    levels = sorted(
        summary.get("surface_screen", {}).get("levels", []),
        key=lambda item: float(item.get("psi_level", -math.inf)),
    )
    verified = [
        level
        for level in levels
        if level.get("ok") is True
        and level.get("verify_ok") is True
        and _finite_float(level.get("radius_mean")) is not None
    ]
    largest_verified = max(
        verified,
        key=lambda level: float(level["radius_mean"]),
        default=None,
    )
    outer_failures = []
    if largest_verified is not None:
        selected_level = float(largest_verified["psi_level"])
        outer_failures = [
            level
            for level in levels
            if float(level.get("psi_level", -math.inf)) > selected_level
            and level.get("ok") is not True
        ]

    reasons: list[str] = []
    if axis.get("has_axis") is not True:
        reasons.append("missing_axis")
    if axis_residual is None or axis_residual > max_axis_residual:
        reasons.append("axis_residual_above_limit")
    if train_rms is None or validation_rms is None:
        reasons.append("missing_fit_rms")
    elif validation_rms > max_validation_rms:
        reasons.append("validation_rms_above_limit")
    if ratio is None or ratio > max_validation_train_ratio:
        reasons.append("validation_train_ratio_above_limit")
    if largest_verified is None:
        reasons.append("missing_verified_screen_level")
    elif not outer_failures and not allow_open_upper_bound:
        reasons.append("missing_outer_screen_failure")

    row.update(
        {
            "a_m": a_value,
            "axis_residual_m": axis_residual,
            "train_rms": train_rms,
            "validation_rms": validation_rms,
            "validation_train_ratio": ratio,
            "largest_verified_level": largest_verified,
            "nearest_outer_failure": outer_failures[0] if outer_failures else None,
            "eligible": not reasons,
            "rejection_reasons": reasons,
        }
    )
    return row


def select_source_candidate(
    candidate_root: Path,
    *,
    max_axis_residual: float = 1e-6,
    max_validation_rms: float = 5e-4,
    max_validation_train_ratio: float = 2.0,
    allow_open_upper_bound: bool = False,
) -> dict[str, object]:
    rows = [
        inspect_candidate(
            path,
            max_axis_residual=max_axis_residual,
            max_validation_rms=max_validation_rms,
            max_validation_train_ratio=max_validation_train_ratio,
            allow_open_upper_bound=allow_open_upper_bound,
        )
        for path in sorted(candidate_root.glob("a_*"))
        if path.is_dir()
    ]
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("no source-psi candidate passed the fixed selection gates")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["largest_verified_level"]["radius_mean"]),
            -float(row["validation_rms"]),
            float(row["a_m"]),
        ),
    )
    return {
        "schema_version": 1,
        "selection_rule": "largest_verified_physical_radius_then_fit_error",
        "thresholds": {
            "max_axis_residual": max_axis_residual,
            "max_validation_rms": max_validation_rms,
            "max_validation_train_ratio": max_validation_train_ratio,
            "require_outer_screen_failure": not allow_open_upper_bound,
        },
        "selected": selected,
        "candidates": rows,
    }


def main() -> int:
    args = parse_args()
    try:
        result = select_source_candidate(
            args.candidate_root,
            max_axis_residual=args.max_axis_residual,
            max_validation_rms=args.max_validation_rms,
            max_validation_train_ratio=args.max_validation_train_ratio,
            allow_open_upper_bound=args.allow_open_upper_bound,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["selected"]["candidate_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
