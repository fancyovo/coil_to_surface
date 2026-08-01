from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_adam_proposal import (
    read_jsonl,
    score_token_direct,
)
from scripts.optimize_flow_prior_zo_adam import (
    decode_noise_rk4,
    gradient_from_pairs,
    load_flow_checkpoint,
    load_initial_noise,
    orthogonal_directions,
    rms,
)
from scripts.optimize_native_score_cem import write_json


def replay_invalid_endpoints(
    initial_noise: np.ndarray,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[list[np.ndarray], list[dict[str, Any]], float, float]:
    beta1, beta2 = (float(value) for value in manifest["betas"])
    epsilon = float(manifest["adam_epsilon"])
    learning_rate = float(manifest["learning_rate"])
    perturbation = float(manifest["perturbation"])
    direction_count = int(manifest["directions"])
    rng = np.random.default_rng(int(manifest["seed"]))
    noise = np.asarray(initial_noise, dtype=np.float32).copy()
    first = np.zeros_like(noise, dtype=np.float64)
    second = np.zeros_like(noise, dtype=np.float64)
    states: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    maximum_noise_error = 0.0
    maximum_update_error = 0.0

    for row in rows:
        iteration = int(row["iteration"])
        directions = orthogonal_directions(rng, noise.shape, direction_count)
        statuses = list(row["pair_statuses"])
        scores = np.asarray(row["pair_scores"], dtype=np.float64)
        for endpoint, status in enumerate(statuses):
            if status == "ok":
                continue
            direction = endpoint % direction_count
            sign = 1.0 if endpoint < direction_count else -1.0
            states.append(
                (noise.astype(np.float64) + sign * perturbation * directions[direction]).astype(
                    np.float32
                )
            )
            metadata.append(
                {
                    "iteration": iteration,
                    "direction": direction,
                    "sign": int(sign),
                    "endpoint_index": endpoint,
                    "recorded_status": status,
                    "recorded_score": float(scores[endpoint]),
                    "recorded_pair_scores": [
                        float(scores[direction]),
                        float(scores[direction + direction_count]),
                    ],
                    "recorded_direction_delta": float(
                        scores[direction] - scores[direction + direction_count]
                    ),
                }
            )

        used_delta = np.asarray(
            row.get("used_direction_deltas", row["raw_direction_deltas"]),
            dtype=np.float64,
        )
        gradient, _ = gradient_from_pairs(
            0.5 * used_delta,
            -0.5 * used_delta,
            directions,
            perturbation,
            delta_clip=None,
        )
        gradient *= float(row.get("gradient_clip_scale", 1.0))
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient * gradient
        first_hat = first / (1.0 - beta1**iteration)
        second_hat = second / (1.0 - beta2**iteration)
        update = learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        noise = (noise.astype(np.float64) + update).astype(np.float32)
        maximum_update_error = max(
            maximum_update_error, abs(rms(update) - float(row["update_rms"]))
        )
        maximum_noise_error = max(
            maximum_noise_error, abs(rms(noise) - float(row["noise_rms"]))
        )
    return states, metadata, maximum_noise_error, maximum_update_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate historical invalid Adam endpoints without the axis safety margin."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--initial-case", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()

    rows = read_jsonl(args.history)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    initial_noise, _ = load_initial_noise(args.initial_case)
    states, endpoint_metadata, noise_error, update_error = replay_invalid_endpoints(
        initial_noise, rows, manifest
    )
    if max(noise_error, update_error) > 2.0e-7:
        raise RuntimeError(
            f"trajectory replay mismatch: noise={noise_error}, update={update_error}"
        )
    if not states:
        raise ValueError("history contains no invalid endpoints")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model, normalizer, _ = load_flow_checkpoint(args.checkpoint, device)
    tokens, decode_wall_s = decode_noise_rk4(
        model,
        normalizer,
        np.asarray(states, dtype=np.float32),
        nfp=int(manifest["nfp"]),
        steps=int(manifest["flow_steps"]),
        device=device,
    )
    gpu_ids = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("at least one GPU is required")
    default_scored = [
        score_token_direct(
            args.lib,
            token,
            nfp=int(manifest["nfp"]),
            device_id=gpu_ids[0],
        )
        for token in tokens
    ]
    margin_zero_scored = [
        score_token_direct(
            args.lib,
            token,
            nfp=int(manifest["nfp"]),
            device_id=gpu_ids[0],
            config_overrides={"axis_topology_margin": 0.0},
        )
        for token in tokens
    ]

    endpoints = []
    for metadata, default_item, margin_zero_item in zip(
        endpoint_metadata, default_scored, margin_zero_scored, strict=True
    ):
        pair_scores = list(metadata["recorded_pair_scores"])
        pair_slot = 0 if metadata["sign"] > 0 else 1
        pair_scores[pair_slot] = float(margin_zero_item["score"])
        corrected_delta = pair_scores[0] - pair_scores[1]
        endpoints.append(
            {
                **metadata,
                "default_recheck": default_item,
                "margin_zero": margin_zero_item,
                "margin_zero_direction_delta": corrected_delta,
                "direction_delta_reduction_factor": abs(metadata["recorded_direction_delta"])
                / max(abs(corrected_delta), 1.0e-30),
            }
        )

    output = {
        "history": str(args.history.resolve()),
        "manifest": str(args.manifest.resolve()),
        "invalid_endpoint_count": len(endpoints),
        "maximum_noise_rms_replay_error": noise_error,
        "maximum_update_rms_replay_error": update_error,
        "decode_wall_s": decode_wall_s,
        "restored_ok_count": sum(
            endpoint["margin_zero"]["status"] == "ok" for endpoint in endpoints
        ),
        "endpoints": endpoints,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "dirty_endpoint_axis_audit.json", output)
    print(
        json.dumps(
            {
                "invalid_endpoint_count": output["invalid_endpoint_count"],
                "restored_ok_count": output["restored_ok_count"],
                "maximum_noise_rms_replay_error": noise_error,
                "maximum_update_rms_replay_error": update_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
