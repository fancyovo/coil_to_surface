from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
for path in (REPO_ROOT, GPU_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flow_matching.vjp import decode_physical_vjp
from scripts.optimize_flow_prior_zo_adam import load_flow_checkpoint
from stellarator_gpu import (
    score_coils_g1_gradient_native,
    score_coils_g2_gradient_native,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def token_cotangent(gradient: dict[str, np.ndarray]) -> np.ndarray:
    shape = gradient["x"].shape
    output = np.empty((shape[0], 100), dtype=np.float32)
    output[:, :33] = gradient["x"]
    output[:, 33:66] = gradient["y"]
    output[:, 66:99] = gradient["z"]
    output[:, 99] = gradient["current"]
    return output


def score_arguments(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tokens[:, :33], tokens[:, 33:66], tokens[:, 66:99], tokens[:, 99]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate physical and latent G1/G2 gradients on a frozen reference bank.")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gradient-lib", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--center-id", default="main_nfp6_step200")
    parser.add_argument("--scale", type=float, default=0.005)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--rk4-steps", type=int, default=256)
    parser.add_argument("--checkpoint-steps", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    manifest = json.loads((args.reference_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "complete":
        raise RuntimeError("reference job must be complete before latent validation")
    center_index = next(
        index for index, center in enumerate(manifest["centers"])
        if center["center_id"] == args.center_id
    )
    center = manifest["centers"][center_index]
    nfp = int(center["nfp"])
    cases = load_jsonl(args.reference_dir / "cases.jsonl")
    scored: dict[int, dict[str, Any]] = {}
    for path in sorted(args.reference_dir.glob("scores_rank_*.jsonl")):
        for row in load_jsonl(path):
            scored[int(row["case_id"])] = row
    raw = np.load(args.reference_dir / "raw_tokens.npy", mmap_mode="r")
    banks = np.load(args.reference_dir / "latent_banks.npz")
    noises = np.asarray(banks["centers"], dtype=np.float32)
    directions = np.asarray(banks["directions"], dtype=np.float32)
    center_case = next(
        row for row in cases
        if int(row["center_index"]) == center_index and row["kind"] == "center"
    )
    center_tokens = np.asarray(raw[int(center_case["case_id"])], dtype=np.float64)
    x, y, z, current = score_arguments(center_tokens)
    g1 = score_coils_g1_gradient_native(
        args.gradient_lib, x, y, z, current, nfp, target_helicity=(1, nfp)
    )
    g2 = score_coils_g2_gradient_native(
        args.gradient_lib, x, y, z, current, nfp, target_helicity=(1, nfp)
    )
    g1_physical = token_cotangent(g1["gradient"])
    g2_physical = token_cotangent(g2["gradient"])
    device = torch.device(args.device)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
    decoded_g1, latent_g1, g1_vjp = decode_physical_vjp(
        model,
        normalizer,
        noises[center_index],
        g1_physical,
        nfp=nfp,
        device=device,
        rk4_steps=args.rk4_steps,
        checkpoint_steps=args.checkpoint_steps,
    )
    decoded_g2, latent_g2, g2_vjp = decode_physical_vjp(
        model,
        normalizer,
        noises[center_index],
        g2_physical,
        nfp=nfp,
        device=device,
        rk4_steps=args.rk4_steps,
        checkpoint_steps=args.checkpoint_steps,
    )
    decode_relative_l2 = float(
        np.linalg.norm(decoded_g2[0].astype(np.float64) - center_tokens) /
        max(np.linalg.norm(center_tokens), 1.0e-30)
    )

    selected_directions = min(args.directions, directions.shape[1])
    direction_rows = []
    for direction_index in range(selected_directions):
        endpoints = {}
        for sign in (-1, 1):
            case = next(
                row for row in cases
                if int(row["center_index"]) == center_index
                and row["kind"] == "endpoint"
                and int(row["direction_index"]) == direction_index
                and int(row["sign"]) == sign
                and math.isclose(float(row["scale"]), args.scale, rel_tol=0.0, abs_tol=1.0e-12)
            )
            endpoints[sign] = {
                "raw": np.asarray(raw[int(case["case_id"])], dtype=np.float64),
                "score": scored[int(case["case_id"])]["result"],
            }
        direction = directions[center_index, direction_index].reshape(center["n_coils"], 100)
        blackbox_slope = (
            float(endpoints[1]["score"]["score"]) -
            float(endpoints[-1]["score"]["score"])
        ) / (2.0 * args.scale)
        physical_delta = (endpoints[1]["raw"] - endpoints[-1]["raw"]) / (2.0 * args.scale)
        direction_rows.append(
            {
                "direction": direction_index,
                "minus_status": endpoints[-1]["score"]["status"],
                "plus_status": endpoints[1]["score"]["status"],
                "minus_fingerprint": endpoints[-1]["score"]["branch_fingerprint"],
                "plus_fingerprint": endpoints[1]["score"]["branch_fingerprint"],
                "blackbox_slope": blackbox_slope,
                "g1_latent_prediction": float(np.sum(latent_g1[0] * direction)),
                "g2_latent_prediction": float(np.sum(latent_g2[0] * direction)),
                "g1_flow_objective_slope": float(np.sum(g1_physical * physical_delta)),
                "g2_flow_objective_slope": float(np.sum(g2_physical * physical_delta)),
            }
        )

    reference = np.load(args.reference_dir / "reference_gradients.npz")
    scales = np.asarray(reference["scales"], dtype=np.float64)
    scale_index = int(np.flatnonzero(np.isclose(scales, args.scale, rtol=0.0, atol=1.0e-12))[0])
    reference_gradient = np.asarray(reference["gradients"][center_index, scale_index], dtype=np.float64)
    output = {
        "format": "qh_latent_gradient_validation_v1",
        "reference_dir": str(args.reference_dir),
        "center_id": args.center_id,
        "center_index": center_index,
        "nfp": nfp,
        "scale": args.scale,
        "direction_count": selected_directions,
        "checkpoint_step": int(checkpoint["step"]),
        "decoded_center_relative_l2": decode_relative_l2,
        "g1_score": g1["score_result"],
        "g2_score": g2["score_result"],
        "g1_native_diagnostics": g1["gradient_diagnostics"],
        "g2_native_diagnostics": g2["gradient_diagnostics"],
        "g1_flow_vjp": asdict(g1_vjp),
        "g2_flow_vjp": asdict(g2_vjp),
        "g1_reference_cosine": cosine(latent_g1, reference_gradient),
        "g2_reference_cosine": cosine(latent_g2, reference_gradient),
        "direction_rows": direction_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "gradients.npz",
        g1_physical=g1_physical,
        g2_physical=g2_physical,
        g1_latent=latent_g1,
        g2_latent=latent_g2,
        reference_latent=reference_gradient,
    )


if __name__ == "__main__":
    main()
