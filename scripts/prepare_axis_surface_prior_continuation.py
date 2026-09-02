from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

from flow_matching.axis_surface_prior import TOKEN_DIM
from flow_matching.data import file_sha256
from scripts.native_score_runtime import write_json


def continuation_payload(
    best: dict[str, Any],
    source_start: dict[str, Any],
    *,
    protocol_id: str,
    target_helicity: tuple[int, int],
) -> dict[str, Any]:
    optimizer = best.get("original_space_local_gradient_adam")
    if not isinstance(optimizer, dict) or optimizer.get("parameter_space") != "data":
        raise ValueError("best case is not a direct-data Adam result")
    parameters = np.asarray(optimizer.get("normalized_coil_tokens"), dtype=np.float32)
    raw = best.get("raw")
    n_base_coils = len(raw.get("current", [])) if isinstance(raw, dict) else 0
    if parameters.shape != (n_base_coils, TOKEN_DIM):
        raise ValueError("best case has invalid normalized-coil-token shape")
    source = source_start.get("data_prior_screening")
    if not isinstance(source, dict):
        raise ValueError("source start lacks data_prior_screening metadata")
    current_l1_a = float(source.get("current_l1_a", 0.0))
    if not np.isfinite(current_l1_a) or current_l1_a <= 0.0:
        raise ValueError("source start has invalid current_l1_a")
    native_score = optimizer.get("native_score")
    if not isinstance(native_score, dict) or native_score.get("status") != "ok":
        raise ValueError("best case lacks a valid native score")
    source_target = (
        optimizer.get("manifest", {}).get("target_helicity", [1, int(best["nfp"])])
    )
    if source_target != [1, int(best["nfp"])]:
        raise ValueError("source best was not scored with the expected positive-hand target")

    payload = copy.deepcopy(best)
    payload["data_prior_screening"] = {
        "format": "axis_surface_prior_direct_data_continuation_start_v1",
        "protocol_id": protocol_id,
        "normalized_coil_tokens": parameters.tolist(),
        "current_l1_a": current_l1_a,
        "native_score": native_score,
        "native_score_target_helicity": source_target,
        "continuation_target_helicity": list(target_helicity),
        "source_best_iteration": int(optimizer["best_iteration"]),
    }
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Prepare selected balanced-v2 best cases for direct-data continuation.")
    value.add_argument("--source-run-root", type=Path, required=True)
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--case-id", type=int, action="append", required=True)
    value.add_argument("--protocol-id", required=True)
    value.add_argument("--expected-commit", required=True)
    value.add_argument("--optimizer-seed-base", type=int, default=20290901)
    value.add_argument("--target-helicity-sign", type=int, choices=(-1, 1), default=1)
    return value


def main() -> None:
    args = parser().parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if commit != args.expected_commit:
        raise RuntimeError(f"repository commit {commit} != {args.expected_commit}")
    if args.run_root.exists():
        raise FileExistsError(args.run_root)
    source_manifest_path = args.source_run_root / "selection_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_cases = {int(case["case_id"]): case for case in source_manifest["cases"]}
    args.run_root.mkdir(parents=True)
    starts_dir = args.run_root / "starts"
    starts_dir.mkdir()

    cases = []
    for worker_index, case_id in enumerate(args.case_id):
        source_case = source_cases.get(case_id)
        if source_case is None:
            raise ValueError(f"case {case_id} is absent from source selection")
        trajectory_id = str(source_case["trajectory_id"])
        best_path = args.source_run_root / "trajectories" / trajectory_id / "optimization" / "best.json"
        source_start_path = Path(source_case["start"])
        best = json.loads(best_path.read_text(encoding="utf-8"))
        source_start = json.loads(source_start_path.read_text(encoding="utf-8"))
        nfp = int(source_case["nfp"])
        target_helicity = (1, args.target_helicity_sign * nfp)
        prepared = continuation_payload(
            best,
            source_start,
            protocol_id=args.protocol_id,
            target_helicity=target_helicity,
        )
        start_path = starts_dir / f"case_{case_id:05d}_best.json"
        write_json(start_path, prepared)
        native_score = prepared["data_prior_screening"]["native_score"]
        cases.append(
            {
                "trajectory_id": f"{trajectory_id}_continue_adam200",
                "case_id": case_id,
                "selection_rank": worker_index,
                "worker_index": worker_index,
                "nfp": nfp,
                "n_base_coils": int(source_case["n_base_coils"]),
                "target_helicity": list(target_helicity),
                "source_score": float(native_score["score"]),
                "source_score_target_helicity": [1, nfp],
                "target_score_at_prepare": (
                    float(native_score["score"])
                    if args.target_helicity_sign == 1
                    else None
                ),
                "source_coil_component": float(native_score["components"]["coil"]),
                "source_status": str(native_score["status"]),
                "optimizer_seed": args.optimizer_seed_base + worker_index,
                "start": str(start_path.resolve()),
                "start_sha256": file_sha256(start_path),
                "source_best": str(best_path.resolve()),
                "source_best_sha256": file_sha256(best_path),
            }
        )

    manifest = {
        "format": "axis_surface_prior_top2_continue_selection_v1",
        "protocol_id": args.protocol_id,
        "status": "prepared",
        "code_commit": commit,
        "target_helicity_sign": args.target_helicity_sign,
        "target_helicity_contract": "(M,N)=(1,target_helicity_sign*nfp)",
        "source": {
            "run_root": str(args.source_run_root.resolve()),
            "selection_manifest": str(source_manifest_path.resolve()),
            "selection_manifest_sha256": file_sha256(source_manifest_path),
        },
        "artifacts": source_manifest["artifacts"],
        "cases": cases,
    }
    write_json(args.run_root / "selection_manifest.json", manifest)
    print(json.dumps({"event": "prepared", "run_root": str(args.run_root), "case_count": len(cases)}))


if __name__ == "__main__":
    main()
