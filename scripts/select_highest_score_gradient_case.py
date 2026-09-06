"""Select the highest valid native-score center from completed RL rounds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from flow_matching.axis_prior_rl import inverse_tokens
from flow_matching.data import CoilNormalizer, file_sha256
from scripts.native_score_runtime import token_case


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row_digest(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select(run_root: Path, output_dir: Path) -> dict:
    progress = load_json(run_root / "progress.json")
    boundary = int(progress["next_round"])
    manifest_path = run_root / "manifest.json"
    manifest = load_json(manifest_path)
    candidates = []
    for round_dir in sorted(run_root.joinpath("rounds").glob("round_*")):
        try:
            round_index = int(round_dir.name.split("_")[-1])
        except ValueError:
            continue
        if round_index >= boundary or not (round_dir / "training_summary.json").is_file():
            continue
        for rank_json in sorted(round_dir.glob("rank_*.json")):
            rank = int(rank_json.stem.split("_")[-1])
            records = load_json(rank_json)["records"]
            npz_path = rank_json.with_suffix(".npz")
            with np.load(npz_path, allow_pickle=False) as arrays:
                if len(records) != len(arrays["current"]):
                    raise ValueError(f"record/array count mismatch: {rank_json}")
                for index, record in enumerate(records):
                    score = float(record["initial"]["score"])
                    if record.get("valid") is True and np.isfinite(score):
                        candidates.append((score, round_index, rank, index, record, np.asarray(arrays["current"][index])))
    if not candidates:
        raise RuntimeError("no valid completed-round score records found")
    score, round_index, rank, index, record, current = max(candidates, key=lambda row: row[0])
    checkpoint_path = run_root / "checkpoints" / f"round_{round_index:04d}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])
    physical = inverse_tokens(current[None], normalizer)[0]
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "selection_format": "score_gradient_completed_round_max_v1",
        "selected_at_unix_s": time.time(),
        "selection_boundary_next_round": boundary,
        "source_run_root": str(run_root.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_round": round_index,
        "source_rank": rank,
        "source_sample_index": index,
        "source_sample_id": record["sample_id"],
        "source_record_sha256": row_digest(record),
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_score": score,
        "source_components": record["initial"].get("components", {}),
        "source_valid": bool(record["valid"]),
        "source_gradient_ok": bool(record.get("gradient_ok", False)),
        "flow_current_shape": list(current.shape),
        "conversion": "inverse the round checkpoint Flow normalizer to physical A coefficients before token_case",
    }
    case = token_case(physical, nfp=8, target="QH", metadata={"rl_selection": metadata})
    case_path = output_dir / "selected_case.json"
    case_path.write_text(json.dumps(case, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metadata["case_file"] = str(case_path.resolve())
    metadata["case_sha256"] = file_sha256(case_path)
    (output_dir / "selection.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    select(args.run_root, args.output_dir)
