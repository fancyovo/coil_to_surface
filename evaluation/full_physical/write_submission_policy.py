#!/usr/bin/env python3
"""Write an auditable parallel or serial candidate-submission policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ALLOWED_POOLS = {"p107", "students"}


def build_policy(
    *,
    stage: str,
    serial: bool,
    serial_reason: str,
    candidate_count: int,
    candidate_pools: list[str],
) -> dict[str, object]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if len(candidate_pools) != candidate_count:
        raise ValueError("candidate_pools must contain one entry per candidate")
    invalid = sorted(set(candidate_pools) - ALLOWED_POOLS)
    if invalid:
        raise ValueError(f"unsupported candidate pools: {', '.join(invalid)}")

    reason = serial_reason.strip()
    if serial and not reason:
        raise ValueError("SERIAL_REASON is required when SERIAL_CANDIDATES=1")
    if not serial and reason:
        raise ValueError("SERIAL_REASON must be empty when candidates are parallel")

    return {
        "schema_version": 1,
        "stage": stage,
        "mode": "serial" if serial else "parallel",
        "dependency_policy": "afterany_chain" if serial else "independent_jobs",
        "candidate_count": candidate_count,
        "candidate_pools": candidate_pools,
        "serial_reason": reason or None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--serial", required=True, type=int, choices=(0, 1))
    parser.add_argument("--serial-reason", default="")
    parser.add_argument("--candidate-count", required=True, type=int)
    parser.add_argument("--candidate-pool", action="append", dest="candidate_pools", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = build_policy(
        stage=args.stage,
        serial=bool(args.serial),
        serial_reason=args.serial_reason,
        candidate_count=args.candidate_count,
        candidate_pools=args.candidate_pools,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(policy, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    print(f"PASS: wrote {policy['mode']} submission policy to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
