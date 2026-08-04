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
from flow_matching.model import compile_flow_transformer
from scripts.optimize_flow_prior_zo_adam import load_flow_checkpoint
from stellarator_gpu import (
    score_coils_g1_gradient_native,
    score_coils_g2_gradient_native,
    score_coils_g3_gradient_native,
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
    parser = argparse.ArgumentParser(description="Validate physical and latent G1/G2/G3 gradients on a frozen reference bank.")
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
    parser.add_argument("--compile-flow-model", action="store_true")
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
    center_result = scored[int(center_case["case_id"])]["result"]
    center_fingerprint = tuple(center_result["branch_fingerprint"])
    x, y, z, current = score_arguments(center_tokens)
    g1 = score_coils_g1_gradient_native(
        args.gradient_lib, x, y, z, current, nfp, target_helicity=(1, nfp)
    )
    g2 = score_coils_g2_gradient_native(
        args.gradient_lib, x, y, z, current, nfp, target_helicity=(1, nfp)
    )
    g3 = score_coils_g3_gradient_native(
        args.gradient_lib, x, y, z, current, nfp, target_helicity=(1, nfp)
    )
    g1_physical = token_cotangent(g1["gradient"])
    g2_physical = token_cotangent(g2["gradient"])
    g3_physical = token_cotangent(g3["gradient"])
    device = torch.device(args.device)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
    if args.compile_flow_model:
        model = compile_flow_transformer(model)
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
    decoded_g3, latent_g3, g3_vjp = decode_physical_vjp(
        model,
        normalizer,
        noises[center_index],
        g3_physical,
        nfp=nfp,
        device=device,
        rk4_steps=args.rk4_steps,
        checkpoint_steps=args.checkpoint_steps,
    )
    decode_relative_l2 = float(
        np.linalg.norm(decoded_g3[0].astype(np.float64) - center_tokens) /
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
                "g3_latent_prediction": float(np.sum(latent_g3[0] * direction)),
                "g1_flow_objective_slope": float(np.sum(g1_physical * physical_delta)),
                "g2_flow_objective_slope": float(np.sum(g2_physical * physical_delta)),
                "g3_flow_objective_slope": float(np.sum(g3_physical * physical_delta)),
            }
        )

    endpoint_index = {
        (
            int(row["center_index"]),
            int(row["direction_index"]),
            int(row["sign"]),
            float(row["scale"]),
        ): row
        for row in cases
        if row["kind"] == "endpoint"
    }
    all_slopes = np.full(directions.shape[1], np.nan, dtype=np.float64)
    safe_mask = np.zeros(directions.shape[1], dtype=bool)
    for direction_index in range(directions.shape[1]):
        endpoint_results = []
        for sign in (-1, 1):
            case = endpoint_index[(center_index, direction_index, sign, args.scale)]
            endpoint_results.append(scored[int(case["case_id"])]["result"])
        if all(
            result is not None
            and result["status"] == "ok"
            and tuple(result["branch_fingerprint"]) == center_fingerprint
            for result in endpoint_results
        ):
            safe_mask[direction_index] = True
            all_slopes[direction_index] = (
                float(endpoint_results[1]["score"]) - float(endpoint_results[0]["score"])
            ) / (2.0 * args.scale)
    if not np.any(safe_mask):
        raise RuntimeError("selected center and scale have no same-branch reference directions")
    reference_gradient = np.mean(
        all_slopes[safe_mask, None] * directions[center_index, safe_mask], axis=0
    ).reshape(center["n_coils"], 100)
    output = {
        "format": "qh_latent_gradient_validation_v1",
        "reference_dir": str(args.reference_dir),
        "center_id": args.center_id,
        "center_index": center_index,
        "nfp": nfp,
        "scale": args.scale,
        "direction_count": selected_directions,
        "reference_safe_direction_count": int(np.sum(safe_mask)),
        "reference_safe_fraction": float(np.mean(safe_mask)),
        "reference_kind": "full" if np.all(safe_mask) else "safe_subspace_projection",
        "checkpoint_step": int(checkpoint["step"]),
        "decoded_center_relative_l2": decode_relative_l2,
        "g1_score": g1["score_result"],
        "g2_score": g2["score_result"],
        "g3_score": g3["score_result"],
        "g1_native_diagnostics": g1["gradient_diagnostics"],
        "g2_native_diagnostics": g2["gradient_diagnostics"],
        "g3_native_diagnostics": g3["gradient_diagnostics"],
        "g1_flow_vjp": asdict(g1_vjp),
        "g2_flow_vjp": asdict(g2_vjp),
        "g3_flow_vjp": asdict(g3_vjp),
        "g1_reference_cosine": cosine(latent_g1, reference_gradient),
        "g2_reference_cosine": cosine(latent_g2, reference_gradient),
        "g3_reference_cosine": cosine(latent_g3, reference_gradient),
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
        g3_physical=g3_physical,
        g1_latent=latent_g1,
        g2_latent=latent_g2,
        g3_latent=latent_g3,
        reference_latent=reference_gradient,
        reference_safe_mask=safe_mask,
        reference_slopes=all_slopes,
    )


if __name__ == "__main__":
    main()
