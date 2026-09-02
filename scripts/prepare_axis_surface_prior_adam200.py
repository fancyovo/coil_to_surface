from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]

from flow_matching.data import CoilNormalizer, canonicalize_currents, file_sha256
from scripts.native_score_runtime import token_case, write_json


PROTOCOL_ID = "qh-axis-surface-contour-balanced-random-ok-adam200-64d-abi11-v1"
RUNTIME_WEIGHT = {1: 0.78, 2: 0.92, 3: 1.08, 4: 1.22}


def row_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_scored_rows(input_dir: Path) -> tuple[list[dict[str, Any]], dict[int, str]]:
    rows: list[dict[str, Any]] = []
    source_by_case: dict[int, str] = {}
    for path in sorted(input_dir.glob("shard_*.jsonl")):
        if path.name.endswith(".done.json"):
            continue
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                case_id = int(row["case_id"])
                if case_id in source_by_case:
                    raise ValueError(f"duplicate case_id {case_id}")
                source_by_case[case_id] = path.name
                rows.append(row)
    rows.sort(key=lambda row: int(row["case_id"]))
    return rows, source_by_case


def select_valid_rows(
    rows: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    valid = [row for row in rows if (row.get("native") or {}).get("status") == "ok"]
    if count > len(valid):
        raise ValueError(f"requested {count} rows from only {len(valid)} valid rows")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(valid), size=count, replace=False)
    return [valid[int(index)] for index in indices]


def assign_workers(
    rows: list[dict[str, Any]], *, worker_count: int
) -> dict[int, int]:
    if len(rows) % worker_count:
        raise ValueError("sample count must be divisible by worker count")
    capacity = len(rows) // worker_count
    loads = [0.0] * worker_count
    counts = [0] * worker_count
    assignment: dict[int, int] = {}
    ordered = sorted(
        enumerate(rows),
        key=lambda item: (-RUNTIME_WEIGHT[int(item[1]["n_base_coils"])], item[0]),
    )
    for selection_rank, row in ordered:
        candidates = [index for index in range(worker_count) if counts[index] < capacity]
        worker = min(candidates, key=lambda index: (loads[index], counts[index], index))
        assignment[selection_rank] = worker
        counts[worker] += 1
        loads[worker] += RUNTIME_WEIGHT[int(row["n_base_coils"])]
    return assignment


def exact_standardized_start(
    tokens: np.ndarray,
    normalizer: CoilNormalizer,
    *,
    condition: tuple[int, int],
) -> tuple[np.ndarray, float, np.ndarray, dict[str, float]]:
    source = np.asarray(tokens, dtype=np.float64)
    values = source.astype(np.float32)
    current_l1_a = float(np.sum(np.abs(values[:, -1]), dtype=np.float64))
    canonical = canonicalize_currents(values[None], current_l1_a)[0]
    parameters = ((canonical - normalizer.mean) / normalizer.std).astype(np.float32)
    exact = CoilNormalizer(
        mean=normalizer.mean.copy(),
        std=normalizer.std.copy(),
        current_l1_a={f"{condition[0]}:{condition[1]}": current_l1_a},
        clip=float("inf"),
    )
    reconstructed = exact.inverse(parameters[None], condition)[0]
    geometry = values[:, :-1]
    geometry_reconstructed = reconstructed[:, :-1]
    geometry_relative_rms = float(
        np.linalg.norm(geometry_reconstructed - geometry)
        / max(np.linalg.norm(geometry), 1.0e-30)
    )
    current_relative_rms = float(
        np.linalg.norm(reconstructed[:, -1] - values[:, -1])
        / max(np.linalg.norm(values[:, -1]), 1.0e-30)
    )
    diagnostics = {
        "geometry_relative_rms": geometry_relative_rms,
        "current_relative_rms": current_relative_rms,
        "source_to_start_geometry_relative_rms": float(
            np.linalg.norm(
                reconstructed[:, :-1].astype(np.float64) - source[:, :-1]
            )
            / max(np.linalg.norm(source[:, :-1]), 1.0e-30)
        ),
        "source_to_start_current_relative_rms": float(
            np.linalg.norm(reconstructed[:, -1].astype(np.float64) - source[:, -1])
            / max(np.linalg.norm(source[:, -1]), 1.0e-30)
        ),
        "max_abs_parameter": float(np.max(np.abs(parameters))),
    }
    return parameters, current_l1_a, reconstructed.astype(np.float64), diagnostics


def exact_standardized_parameters(
    tokens: np.ndarray,
    normalizer: CoilNormalizer,
    *,
    condition: tuple[int, int],
) -> tuple[np.ndarray, float, dict[str, float]]:
    parameters, current_l1_a, _, diagnostics = exact_standardized_start(
        tokens, normalizer, condition=condition
    )
    return parameters, current_l1_a, diagnostics


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Prepare balanced-v2 random valid starts for direct-data Adam200.")
    value.add_argument("--input-dir", type=Path, required=True)
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--score-lib", type=Path, required=True)
    value.add_argument("--sample-count", type=int, default=84)
    value.add_argument("--worker-count", type=int, default=6)
    value.add_argument("--seed", type=int, default=20260902)
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

    rows, source_by_case = load_scored_rows(args.input_dir)
    if len(rows) != 6000:
        raise ValueError(f"expected 6000 scored rows, found {len(rows)}")
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
            "format": "axis_surface_prior_balanced_v2_exact_data_start_v1",
            "protocol_id": PROTOCOL_ID,
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
                "trajectory_id": f"axisv2_case_{case_id:05d}",
                "case_id": case_id,
                "selection_rank": rank,
                "worker_index": assignment[rank],
                "nfp": nfp,
                "n_base_coils": n_base_coils,
                "initial_score": float(row["native"]["score"]),
                "initial_coil_component": float(row["native"]["components"]["coil"]),
                "initial_status": row["native"]["status"],
                "optimizer_seed": 20270000 + case_id,
                "start": str(start_path.resolve()),
                "start_sha256": file_sha256(start_path),
                "source_shard": source_by_case[case_id],
                "roundtrip": roundtrip,
            }
        )

    valid_rows = [row for row in rows if (row.get("native") or {}).get("status") == "ok"]
    manifest = {
        "format": "axis_surface_prior_balanced_v2_adam200_selection_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "prepared",
        "code_commit": commit,
        "input": {
            "run_root": str(args.input_dir.resolve()),
            "row_count": len(rows),
            "valid_count": len(valid_rows),
            "valid_rate": len(valid_rows) / len(rows),
            "shards": [
                {"file": path.name, "sha256": file_sha256(path)}
                for path in sorted(args.input_dir.glob("shard_*.jsonl"))
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
        "cases": sorted(cases, key=lambda case: (case["worker_index"], case["selection_rank"])),
    }
    write_json(args.run_root / "selection_manifest.json", manifest)
    print(json.dumps(manifest["selection"], indent=2), flush=True)


if __name__ == "__main__":
    main()
