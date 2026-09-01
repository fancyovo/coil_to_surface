from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]

from flow_matching.data import CoilNormalizer, file_sha256
from scripts.native_score_runtime import token_case, write_json
from scripts.prepare_axis_surface_prior_adam200 import (
    assign_workers,
    exact_standardized_parameters,
    load_scored_rows,
    row_sha256,
    select_valid_rows,
)


SOURCE_PROTOCOL_ID = "qh-axis-surface-contour-compact-flexible-score-abi11-v3"
PROTOCOL_ID = (
    "qh-axis-surface-contour-compact-flexible-random-ok-adam200-64d-abi11-v1"
)
EXPECTED_ROW_COUNT = 3600
ARTIFACT_FORMATS = {
    "selection": "axis_surface_prior_compact_flexible_v3_adam200_selection_v1",
    "start": "axis_surface_prior_compact_flexible_v3_exact_data_start_v1",
    "trajectory": "axis_surface_prior_compact_flexible_v3_adam200_trajectory_v1",
    "failure": "axis_surface_prior_compact_flexible_v3_adam200_failure_v1",
    "worker": "axis_surface_prior_compact_flexible_v3_adam200_worker_v1",
    "summary": "axis_surface_prior_compact_flexible_v3_adam200_summary_v1",
}


def validate_source_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_ROW_COUNT:
        raise ValueError(f"expected {EXPECTED_ROW_COUNT} scored rows, found {len(rows)}")
    protocols = {row.get("protocol_id") for row in rows}
    if protocols != {SOURCE_PROTOCOL_ID}:
        raise ValueError(f"source protocol mismatch: {sorted(map(str, protocols))}")
    unsupported = sorted(
        {int(row["n_base_coils"]) for row in rows if int(row["n_base_coils"]) > 4}
    )
    if unsupported:
        raise ValueError(f"source contains explicitly excluded nc values: {unsupported}")


def validate_source_shards(input_dir: Path, expected_lib_sha: str) -> list[Path]:
    row_paths = sorted(input_dir.glob("shard_*.jsonl"))
    done_paths = sorted(input_dir.glob("shard_*.done.json"))
    if len(row_paths) != 6 or len(done_paths) != 6:
        raise ValueError(
            f"expected six source row shards and six done markers, found "
            f"{len(row_paths)} and {len(done_paths)}"
        )
    for path in done_paths:
        done = json.loads(path.read_text(encoding="utf-8"))
        if done.get("protocol_id") != SOURCE_PROTOCOL_ID:
            raise ValueError(f"source done-marker protocol mismatch in {path.name}")
        if int(done.get("errors", -1)) != 0:
            raise ValueError(f"source shard reports scoring errors in {path.name}")
        if done.get("score_library_sha256") != expected_lib_sha:
            raise ValueError(f"source score-library hash mismatch in {path.name}")
        if int(done.get("total_count", -1)) != EXPECTED_ROW_COUNT:
            raise ValueError(f"source population size mismatch in {path.name}")
    return done_paths


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Prepare compact-flexible-v3 random valid starts for direct-data Adam200."
    )
    value.add_argument("--input-dir", type=Path, required=True)
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--score-lib", type=Path, required=True)
    value.add_argument("--sample-count", type=int, default=120)
    value.add_argument("--worker-count", type=int, default=6)
    value.add_argument("--seed", type=int, default=20260904)
    value.add_argument("--expected-commit", required=True)
    value.add_argument("--expected-lib-sha", required=True)
    value.add_argument("--expected-checkpoint-sha", required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if commit != args.expected_commit:
        raise RuntimeError(f"repository commit {commit} != {args.expected_commit}")
    if file_sha256(args.score_lib) != args.expected_lib_sha:
        raise RuntimeError("score-library hash mismatch")
    if file_sha256(args.checkpoint) != args.expected_checkpoint_sha:
        raise RuntimeError("checkpoint hash mismatch")
    if (args.run_root / "selection_manifest.json").exists():
        raise FileExistsError("selection manifest already exists")

    done_paths = validate_source_shards(args.input_dir, args.expected_lib_sha)
    rows, source_by_case = load_scored_rows(args.input_dir)
    validate_source_rows(rows)
    selected = select_valid_rows(rows, count=args.sample_count, seed=args.seed)
    assignment = assign_workers(selected, worker_count=args.worker_count)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])
    starts_dir = args.run_root / "starts"
    starts_dir.mkdir(parents=True, exist_ok=False)
    cases: list[dict[str, Any]] = []
    for rank, row in enumerate(selected):
        case_id = int(row["case_id"])
        nfp = int(row["nfp"])
        n_base_coils = int(row["n_base_coils"])
        tokens = np.asarray(row["tokens"], dtype=np.float32)
        parameters, current_l1_a, roundtrip = exact_standardized_parameters(
            tokens, normalizer, condition=(nfp, n_base_coils)
        )
        if roundtrip["geometry_relative_rms"] > 2.0e-6:
            raise RuntimeError(f"case {case_id} geometry roundtrip exceeds tolerance")
        if roundtrip["current_relative_rms"] > 2.0e-6:
            raise RuntimeError(f"case {case_id} current roundtrip exceeds tolerance")
        start = token_case(tokens, nfp=nfp, target="QH")
        start["data_prior_screening"] = {
            "format": ARTIFACT_FORMATS["start"],
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "normalized_coil_tokens": parameters.tolist(),
            "current_l1_a": current_l1_a,
            "native_score": row["native"],
            "source_case_id": case_id,
            "source_shard": source_by_case[case_id],
            "source_row_sha256": row_sha256(row),
            "selection_seed": args.seed,
            "selection_rank": rank,
            "roundtrip": roundtrip,
        }
        start_path = starts_dir / f"case_{case_id:05d}.json"
        write_json(start_path, start)
        cases.append(
            {
                "trajectory_id": f"axisv3_case_{case_id:05d}",
                "case_id": case_id,
                "selection_rank": rank,
                "worker_index": assignment[rank],
                "nfp": nfp,
                "n_base_coils": n_base_coils,
                "initial_score": float(row["native"]["score"]),
                "initial_coil_component": float(row["native"]["components"]["coil"]),
                "initial_status": row["native"]["status"],
                "optimizer_seed": 20280000 + case_id,
                "start": str(start_path.resolve()),
                "start_sha256": file_sha256(start_path),
                "source_shard": source_by_case[case_id],
                "roundtrip": roundtrip,
            }
        )

    valid_rows = [row for row in rows if (row.get("native") or {}).get("status") == "ok"]
    manifest = {
        "format": ARTIFACT_FORMATS["selection"],
        "artifact_formats": ARTIFACT_FORMATS,
        "protocol_id": PROTOCOL_ID,
        "status": "prepared",
        "code_commit": commit,
        "input": {
            "protocol_id": SOURCE_PROTOCOL_ID,
            "run_root": str(args.input_dir.resolve()),
            "row_count": len(rows),
            "valid_count": len(valid_rows),
            "valid_rate": len(valid_rows) / len(rows),
            "shards": [
                {"file": path.name, "sha256": file_sha256(path)}
                for path in sorted(args.input_dir.glob("shard_*.jsonl"))
            ],
            "done_markers": [
                {"file": path.name, "sha256": file_sha256(path)}
                for path in done_paths
            ],
        },
        "selection": {
            "population": "native status == ok",
            "method": "simple random sample without replacement",
            "seed": args.seed,
            "sample_count": args.sample_count,
            "worker_count": args.worker_count,
            "samples_per_worker": args.sample_count // args.worker_count,
            "selected_by_nc": dict(
                sorted(Counter(str(case["n_base_coils"]) for case in cases).items())
            ),
        },
        "optimizer": {
            "parameter_space": "exact-unclipped standardized coil coefficients",
            "iterations": 200,
            "directions": 64,
            "difference": "centered",
            "perturbation": 0.0025,
            "learning_rate": 0.01,
            "beta": [0.7, 0.999],
            "flow_calls": 0,
            "success_threshold": 50.0,
        },
        "artifacts": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": args.expected_checkpoint_sha,
            "score_library": str(args.score_lib.resolve()),
            "score_library_abi": 11,
            "score_library_sha256": args.expected_lib_sha,
        },
        "cases": sorted(
            cases, key=lambda case: (case["worker_index"], case["selection_rank"])
        ),
    }
    write_json(args.run_root / "selection_manifest.json", manifest)
    print(json.dumps(manifest["selection"], indent=2), flush=True)


if __name__ == "__main__":
    main()
