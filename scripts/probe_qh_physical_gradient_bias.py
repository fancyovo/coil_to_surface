from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
for path in (REPO_ROOT, GPU_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flow_matching.vjp import decode_physical_vjp
from scripts.optimize_flow_prior_zo_adam import decode_noise_rk4, load_flow_checkpoint
from scripts.qh_blackbox_gradient_reference import compact_result, file_sha256, write_json
from stellarator_gpu import (
    score_coils_g2_gradient_native,
    score_coils_g3_gradient_native,
    score_coils_native,
)


def parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result or any(item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("steps must be positive")
    return result


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def score_arguments(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(tokens, dtype=np.float64)
    return values[:, :33], values[:, 33:66], values[:, 66:99], values[:, 99]


def token_cotangent(gradient: dict[str, np.ndarray]) -> np.ndarray:
    shape = np.asarray(gradient["x"]).shape
    result = np.empty((shape[0], 100), dtype=np.float32)
    result[:, :33] = gradient["x"]
    result[:, 33:66] = gradient["y"]
    result[:, 66:99] = gradient["z"]
    result[:, 99] = gradient["current"]
    return result


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def normalized(value: np.ndarray) -> np.ndarray:
    scale = rms(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("direction has invalid RMS")
    return np.asarray(value, dtype=np.float64) / scale


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else float("nan")


def evaluate_score(lib: Path, tokens: np.ndarray, nfp: int) -> dict[str, Any]:
    x, y, z, current = score_arguments(tokens)
    return score_coils_native(
        lib,
        x,
        y,
        z,
        current,
        nfp,
        device_id=0,
        target_helicity=(1, nfp),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe exact score around saved G2-Adam trajectory states.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gradient-lib", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=parse_floats, default=(0.0003125, 0.000625, 0.00125))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    manifest = read_json(args.run_dir / "manifest.json")
    nfp = int(manifest["nfp"])
    rk4_steps = int(manifest["rk4_steps"])
    state_path = args.run_dir / "trajectory" / f"step_{args.iteration:04d}.json"
    next_path = args.run_dir / "trajectory" / f"step_{args.iteration + 1:04d}.json"
    state = read_json(state_path)
    next_state = read_json(next_path)
    noise = np.asarray(state["noise"], dtype=np.float32)
    saved_tokens = np.asarray(state["tokens"], dtype=np.float64)
    x, y, z, current = score_arguments(saved_tokens)

    g2_native = score_coils_g2_gradient_native(
        args.gradient_lib, x, y, z, current, nfp, device_id=0, target_helicity=(1, nfp)
    )
    g3_native = score_coils_g3_gradient_native(
        args.gradient_lib, x, y, z, current, nfp, device_id=0, target_helicity=(1, nfp)
    )
    g2_physical = token_cotangent(g2_native["gradient"])
    g3_physical = token_cotangent(g3_native["gradient"])

    device = torch.device(args.device)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
    decoded_g2, latent_g2, g2_flow = decode_physical_vjp(
        model,
        normalizer,
        noise,
        g2_physical,
        nfp=nfp,
        device=device,
        rk4_steps=rk4_steps,
        checkpoint_steps=8,
        use_checkpoint=False,
    )
    decoded_g3, latent_g3, g3_flow = decode_physical_vjp(
        model,
        normalizer,
        noise,
        g3_physical,
        nfp=nfp,
        device=device,
        rk4_steps=rk4_steps,
        checkpoint_steps=8,
        use_checkpoint=False,
    )
    latent_g2 = np.asarray(latent_g2[0], dtype=np.float64)
    latent_g3 = np.asarray(latent_g3[0], dtype=np.float64)
    adam_displacement = np.asarray(next_state["noise"], dtype=np.float64) - noise.astype(np.float64)
    directions = {
        "g2": normalized(latent_g2),
        "g3": normalized(latent_g3),
        "adam": normalized(adam_displacement),
    }

    candidate_metadata: list[dict[str, Any]] = []
    candidate_noise: list[np.ndarray] = []
    for name, direction in directions.items():
        for step in args.steps:
            for sign in (-1, 1):
                candidate_metadata.append({"method": name, "step": float(step), "sign": int(sign)})
                candidate_noise.append((noise.astype(np.float64) + sign * step * direction).astype(np.float32))

    decoded_candidates, decode_wall_s = decode_noise_rk4(
        model,
        normalizer,
        np.asarray(candidate_noise, dtype=np.float32),
        nfp=nfp,
        steps=rk4_steps,
        device=device,
    )
    center_score = evaluate_score(args.gradient_lib, decoded_g2[0], nfp)
    score_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for metadata, tokens in zip(candidate_metadata, decoded_candidates, strict=True):
        result = evaluate_score(args.gradient_lib, tokens, nfp)
        rows.append({**metadata, "result": compact_result(result)})
    score_wall_s = time.perf_counter() - score_started

    pairs: list[dict[str, Any]] = []
    center_value = float(center_score["score"])
    center_fingerprint = tuple(compact_result(center_score)["branch_fingerprint"])
    for name, direction in directions.items():
        for step in args.steps:
            selected = [row for row in rows if row["method"] == name and row["step"] == step]
            by_sign = {int(row["sign"]): row["result"] for row in selected}
            minus = by_sign[-1]
            plus = by_sign[1]
            pairs.append({
                "method": name,
                "step": float(step),
                "minus_status": minus["status"],
                "plus_status": plus["status"],
                "minus_same_branch": tuple(minus["branch_fingerprint"]) == center_fingerprint,
                "plus_same_branch": tuple(plus["branch_fingerprint"]) == center_fingerprint,
                "minus_delta": float(minus["score"] - center_value),
                "plus_delta": float(plus["score"] - center_value),
                "centered_slope": float((plus["score"] - minus["score"]) / (2.0 * step)),
                "g2_prediction": float(np.sum(latent_g2 * direction)),
                "g3_prediction": float(np.sum(latent_g3 * direction)),
            })

    output = {
        "format": "qh_physical_gradient_bias_probe_v1",
        "run_dir": str(args.run_dir.resolve()),
        "iteration": int(args.iteration),
        "nfp": nfp,
        "rk4_steps": rk4_steps,
        "steps": list(args.steps),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "gradient_lib": str(args.gradient_lib.resolve()),
        "gradient_lib_sha256": file_sha256(args.gradient_lib),
        "saved_score": state["score"],
        "reevaluated_center": compact_result(center_score),
        "decoded_g2_saved_token_relative_l2": float(np.linalg.norm(decoded_g2[0] - saved_tokens) / max(np.linalg.norm(saved_tokens), 1.0e-30)),
        "decoded_g3_saved_token_relative_l2": float(np.linalg.norm(decoded_g3[0] - saved_tokens) / max(np.linalg.norm(saved_tokens), 1.0e-30)),
        "decoded_g2_g3_relative_l2": float(np.linalg.norm(decoded_g2 - decoded_g3) / max(np.linalg.norm(decoded_g2), 1.0e-30)),
        "g2_g3_latent_cosine": cosine(latent_g2, latent_g3),
        "g2_adam_cosine": cosine(latent_g2, adam_displacement),
        "g3_adam_cosine": cosine(latent_g3, adam_displacement),
        "g2_latent_rms": rms(latent_g2),
        "g3_latent_rms": rms(latent_g3),
        "adam_displacement_rms": rms(adam_displacement),
        "g2_native": g2_native["gradient_diagnostics"],
        "g3_native": g3_native["gradient_diagnostics"],
        "g2_flow": asdict(g2_flow),
        "g3_flow": asdict(g3_flow),
        "candidate_decode_wall_s": float(decode_wall_s),
        "candidate_score_wall_s": float(score_wall_s),
        "pairs": pairs,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", output)
    print(json.dumps({"iteration": args.iteration, "pairs": pairs}, indent=2))


if __name__ == "__main__":
    main()
