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

from scripts.optimize_flow_prior_zo_adam import (
    decode_noise_rk4,
    gradient_from_pairs,
    load_flow_checkpoint,
    load_initial_noise,
    orthogonal_directions,
    result_score,
    rms,
    score_tokens,
)
from scripts.optimize_native_score_cem import NativeScorePool, write_json


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def replay_proposal(
    initial_noise: np.ndarray,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    target_iteration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    beta1, beta2 = (float(value) for value in manifest["betas"])
    epsilon = float(manifest["adam_epsilon"])
    learning_rate = float(manifest["learning_rate"])
    perturbation = float(manifest["perturbation"])
    direction_count = int(manifest["directions"])
    rng = np.random.default_rng(int(manifest["seed"]))
    noise = np.asarray(initial_noise, dtype=np.float32).copy()
    first = np.zeros_like(noise, dtype=np.float64)
    second = np.zeros_like(noise, dtype=np.float64)
    adam_step = 0
    replay = []

    for row in rows:
        iteration = int(row["iteration"])
        directions = orthogonal_directions(rng, noise.shape, direction_count)
        update = np.zeros_like(noise, dtype=np.float64)
        if bool(row.get("gradient_step_applied", True)):
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
            adam_step += 1
            first = beta1 * first + (1.0 - beta1) * gradient
            second = beta2 * second + (1.0 - beta2) * gradient * gradient
            first_hat = first / (1.0 - beta1**adam_step)
            second_hat = second / (1.0 - beta2**adam_step)
            update = learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        previous = noise.copy()
        noise = (noise.astype(np.float64) + update).astype(np.float32)
        replay.append(
            {
                "iteration": iteration,
                "replayed_noise_rms": rms(noise),
                "recorded_noise_rms": float(row["noise_rms"]),
                "noise_rms_error": abs(rms(noise) - float(row["noise_rms"])),
                "replayed_update_rms": rms(update),
                "recorded_update_rms": float(row["update_rms"]),
                "update_rms_error": abs(rms(update) - float(row["update_rms"])),
            }
        )
        if iteration == target_iteration:
            return previous, update.astype(np.float32), directions, replay
    raise ValueError(f"target iteration {target_iteration} is absent from history")


def compact_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {"status": None, "score": 0.0, "diagnostics": None}
    return {
        "status": result.get("status"),
        "score": result_score(result),
        "diagnostics": result.get("diagnostics"),
    }


def score_token_direct(
    lib_path: Path,
    token: np.ndarray,
    *,
    nfp: int,
    device_id: int,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gpu_python = REPO_ROOT / "gpu_backend" / "python"
    if str(gpu_python) not in sys.path:
        sys.path.insert(0, str(gpu_python))
    from stellarator_gpu import score_coils_native

    value = np.atleast_2d(np.asarray(token, dtype=np.float64))
    result = score_coils_native(
        lib_path,
        value[:, :33],
        value[:, 33:66],
        value[:, 66:99],
        value[:, 99],
        nfp,
        device_id=device_id,
        target_helicity=(1, nfp),
        config_overrides=config_overrides,
    )
    return compact_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently replay and rescore an Adam proposal.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--initial-case", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()

    rows = read_jsonl(args.history)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    initial_noise, _ = load_initial_noise(args.initial_case)
    previous, update, directions, replay = replay_proposal(
        initial_noise, rows, manifest, args.iteration
    )
    if max(row["noise_rms_error"] for row in replay) > 2.0e-7:
        raise RuntimeError(f"noise replay mismatch: {replay}")
    if max(row["update_rms_error"] for row in replay) > 2.0e-7:
        raise RuntimeError(f"update replay mismatch: {replay}")

    alphas = np.asarray([0.0, 0.25, 0.5, 0.75, 0.875, 0.9375, 1.0, 1.0625, 1.25])
    line_states = np.asarray(
        [previous.astype(np.float64) + alpha * update for alpha in alphas],
        dtype=np.float32,
    )
    perturbation = float(manifest["perturbation"])
    pair_states = np.concatenate(
        [previous[None] + perturbation * directions, previous[None] - perturbation * directions]
    ).astype(np.float32)

    gpu_ids = [int(value) for value in args.gpus.split(",") if value.strip()]
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model, normalizer, _ = load_flow_checkpoint(args.checkpoint, device)
    evaluations = {}
    decoded_by_steps: dict[int, np.ndarray] = {}
    with NativeScorePool(args.lib, gpu_ids) as pool:
        for flow_steps in (256, 512):
            tokens, decode_wall = decode_noise_rk4(
                model,
                normalizer,
                line_states,
                nfp=int(manifest["nfp"]),
                steps=flow_steps,
                device=device,
            )
            decoded_by_steps[flow_steps] = tokens
            results, _, errors, score_wall = score_tokens(
                pool,
                tokens,
                nfp=int(manifest["nfp"]),
                target=str(manifest["target"]),
                timeout_s=300.0,
                metadata={"phase": "proposal_line", "flow_steps": flow_steps},
            )
            if any(error is not None for error in errors):
                raise RuntimeError(f"score errors at RK4-{flow_steps}: {errors}")
            evaluations[str(flow_steps)] = {
                "decode_wall_s": decode_wall,
                "score_wall_s": score_wall,
                "line": [
                    {"alpha": float(alpha), **compact_result(result)}
                    for alpha, result in zip(alphas, results, strict=True)
                ],
            }
        pair_tokens, pair_decode_wall = decode_noise_rk4(
            model,
            normalizer,
            pair_states,
            nfp=int(manifest["nfp"]),
            steps=256,
            device=device,
        )
        pair_results, _, pair_errors, pair_score_wall = score_tokens(
            pool,
            pair_tokens,
            nfp=int(manifest["nfp"]),
            target=str(manifest["target"]),
            timeout_s=300.0,
            metadata={"phase": "proposal_pair_recheck"},
        )
        if any(error is not None for error in pair_errors):
            raise RuntimeError(f"pair score errors: {pair_errors}")

    proposal_index = int(np.flatnonzero(alphas == 1.0)[0])
    proposal_token = decoded_by_steps[512][proposal_index]
    device_repeats = [
        {
            "device_id": device_id,
            **score_token_direct(
                args.lib,
                proposal_token,
                nfp=int(manifest["nfp"]),
                device_id=device_id,
            ),
        }
        for device_id in gpu_ids
    ]
    topology_margin_scan = [
        {
            "axis_topology_margin": margin,
            **score_token_direct(
                args.lib,
                proposal_token,
                nfp=int(manifest["nfp"]),
                device_id=gpu_ids[0],
                config_overrides={"axis_topology_margin": margin},
            ),
        }
        for margin in (0.0, 0.005, 0.01, 0.015, 0.019, 0.02, 0.025)
    ]
    dense_fallback = score_token_direct(
        args.lib,
        proposal_token,
        nfp=int(manifest["nfp"]),
        device_id=gpu_ids[0],
        config_overrides={
            "axis_fallback_grid": 96,
            "axis_fallback_max_candidates": 96,
            "axis_fallback_newton_iters": 8,
        },
    )

    output = {
        "target_iteration": args.iteration,
        "replay": replay,
        "previous_noise_rms": rms(previous),
        "update_rms": rms(update),
        "perturbation": perturbation,
        "evaluations": evaluations,
        "pair_recheck": {
            "decode_wall_s": pair_decode_wall,
            "score_wall_s": pair_score_wall,
            "results": [compact_result(result) for result in pair_results],
        },
        "proposal_axis_audit": {
            "flow_steps": 512,
            "device_repeats": device_repeats,
            "topology_margin_scan": topology_margin_scan,
            "dense_fallback": dense_fallback,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "proposal_diagnosis.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
