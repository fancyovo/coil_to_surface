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
from scripts.qh_blackbox_gradient_reference import file_sha256, rms_orthogonal_basis, write_json
from stellarator_gpu import (
    score_coils_g1_gradient_native,
    score_coils_g2_frozen_batch_native,
    score_coils_g2_gradient_native,
    score_coils_g3_frozen_batch_native,
    score_coils_g3_gradient_native,
    score_coils_g4_fixed_branch_batch_native,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_floats(value: str) -> tuple[float, ...]:
    output = tuple(float(item) for item in value.split(",") if item.strip())
    if not output or any(item <= 0.0 for item in output):
        raise argparse.ArgumentTypeError("scales must be positive")
    return output


def score_arguments(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(tokens, dtype=np.float64)
    return values[..., :33], values[..., 33:66], values[..., 66:99], values[..., 99]


def token_cotangent(gradient: dict[str, np.ndarray]) -> np.ndarray:
    shape = np.asarray(gradient["x"]).shape
    output = np.empty((shape[0], 100), dtype=np.float32)
    output[:, :33] = gradient["x"]
    output[:, 33:66] = gradient["y"]
    output[:, 66:99] = gradient["z"]
    output[:, 99] = gradient["current"]
    return output


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Close the native G2 gradient against its exact frozen-front scalar."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gradient-lib", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-directions", type=int, default=32)
    parser.add_argument(
        "--scales",
        type=parse_floats,
        default=(0.0003125, 0.000625, 0.00125, 0.0025),
    )
    parser.add_argument("--seed", type=int, default=2026080407)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.random_directions < 1:
        raise ValueError("random-directions must be positive")

    manifest = read_json(args.run_dir / "manifest.json")
    state = read_json(args.run_dir / "trajectory" / f"step_{args.iteration:04d}.json")
    next_state = read_json(args.run_dir / "trajectory" / f"step_{args.iteration + 1:04d}.json")
    nfp = int(manifest["nfp"])
    rk4_steps = int(manifest["rk4_steps"])
    noise = np.asarray(state["noise"], dtype=np.float32)
    saved_tokens = np.asarray(state["tokens"], dtype=np.float64)
    x, y, z, current = score_arguments(saved_tokens)

    native = {}
    native["g1"] = score_coils_g1_gradient_native(
        args.gradient_lib, x, y, z, current, nfp, device_id=0, target_helicity=(1, nfp)
    )
    native["g2"] = score_coils_g2_gradient_native(
        args.gradient_lib, x, y, z, current, nfp, device_id=0, target_helicity=(1, nfp)
    )
    native["g3"] = score_coils_g3_gradient_native(
        args.gradient_lib, x, y, z, current, nfp, device_id=0, target_helicity=(1, nfp)
    )
    physical_gradients = {
        name: token_cotangent(payload["gradient"]) for name, payload in native.items()
    }

    device = torch.device(args.device)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
    latent_gradients: dict[str, np.ndarray] = {}
    flow_diagnostics: dict[str, Any] = {}
    decoded_center = None
    for name, physical_gradient in physical_gradients.items():
        decoded, latent, diagnostics = decode_physical_vjp(
            model,
            normalizer,
            noise,
            physical_gradient,
            nfp=nfp,
            device=device,
            rk4_steps=rk4_steps,
            checkpoint_steps=8,
            use_checkpoint=False,
        )
        if decoded_center is None:
            decoded_center = np.asarray(decoded[0], dtype=np.float64)
        latent_gradients[name] = np.asarray(latent[0], dtype=np.float64)
        flow_diagnostics[name] = asdict(diagnostics)
    assert decoded_center is not None

    direction_rows: list[dict[str, Any]] = []
    directions: list[np.ndarray] = []
    named = {
        "g2": normalized(latent_gradients["g2"]),
        "g3": normalized(latent_gradients["g3"]),
        "adam": normalized(
            np.asarray(next_state["noise"], dtype=np.float64) - noise.astype(np.float64)
        ),
    }
    for name, direction in named.items():
        direction_rows.append({"name": name, "kind": "named"})
        directions.append(direction)
    random_basis = rms_orthogonal_basis(noise.size, args.seed + args.iteration)
    for index, direction in enumerate(random_basis[: args.random_directions]):
        direction_rows.append({"name": f"random_{index:03d}", "kind": "random"})
        directions.append(np.asarray(direction, dtype=np.float64).reshape(noise.shape))

    candidate_metadata: list[dict[str, Any]] = [{"kind": "center"}]
    candidate_noise: list[np.ndarray] = [noise.copy()]
    for direction_index, direction in enumerate(directions):
        for scale in args.scales:
            for sign in (-1, 1):
                candidate_metadata.append(
                    {
                        "kind": "endpoint",
                        "direction_index": direction_index,
                        "scale": float(scale),
                        "sign": sign,
                    }
                )
                candidate_noise.append(
                    (noise.astype(np.float64) + sign * scale * direction).astype(np.float32)
                )
    decoded_candidates, candidate_decode_wall_s = decode_noise_rk4(
        model,
        normalizer,
        np.asarray(candidate_noise, dtype=np.float32),
        nfp=nfp,
        steps=rk4_steps,
        device=device,
    )
    query_x, query_y, query_z, query_current = score_arguments(decoded_candidates)
    frozen_started = time.perf_counter()
    frozen = score_coils_g2_frozen_batch_native(
        args.gradient_lib,
        x,
        y,
        z,
        current,
        query_x,
        query_y,
        query_z,
        query_current,
        nfp,
        device_id=0,
        target_helicity=(1, nfp),
    )
    frozen_wall_s = time.perf_counter() - frozen_started
    g3_frozen_started = time.perf_counter()
    g3_frozen = score_coils_g3_frozen_batch_native(
        args.gradient_lib,
        x,
        y,
        z,
        current,
        query_x,
        query_y,
        query_z,
        query_current,
        nfp,
        device_id=0,
        target_helicity=(1, nfp),
    )
    g3_frozen_wall_s = time.perf_counter() - g3_frozen_started
    g4_branch_started = time.perf_counter()
    g4_branch = score_coils_g4_fixed_branch_batch_native(
        args.gradient_lib,
        x,
        y,
        z,
        current,
        query_x,
        query_y,
        query_z,
        query_current,
        nfp,
        device_id=0,
        target_helicity=(1, nfp),
    )
    g4_branch_wall_s = time.perf_counter() - g4_branch_started

    rows: list[dict[str, Any]] = []
    for index, metadata in enumerate(candidate_metadata):
        g4_result = g4_branch["query_score_results"][index]
        rows.append(
            {
                **metadata,
                **{
                    name: float(np.asarray(frozen[name])[index])
                    for name in (
                        "frozen_score",
                        "volume_qs_component",
                        "coil_component",
                        "target_error",
                        "qa_error",
                        "qp_error",
                    )
                },
                **{
                    f"g3_{name}": float(np.asarray(g3_frozen[name])[index])
                    for name in (
                        "frozen_score",
                        "volume_qs_component",
                        "coordinate_component",
                        "iota_component",
                        "coil_component",
                        "target_error",
                        "qa_error",
                        "qp_error",
                        "iota_min",
                        "iota_max",
                    )
                },
                "g4_status": str(g4_result["status"]),
                "g4_score": float(g4_result["score"]),
                "g4_components": {
                    name: float(value)
                    for name, value in g4_result["components"].items()
                },
                "g4_diagnostics": {
                    name: g4_result["diagnostics"].get(name)
                    for name in (
                        "psi_angle_p95",
                        "surface_level",
                        "surface_inverse_aspect_ratio",
                        "flux_edge",
                        "iota_min",
                        "iota_max",
                        "qs_global_error",
                        "qs_qa_global_error",
                        "qs_qp_global_error",
                        "volume_candidate_count",
                        "volume_available_count",
                    )
                },
                "g4_total_wall_s": float(g4_result["timing"]["total"]),
            }
        )

    pairs: list[dict[str, Any]] = []
    for direction_index, (metadata, direction) in enumerate(zip(direction_rows, directions, strict=True)):
        predictions = {
            name: float(np.sum(gradient * direction))
            for name, gradient in latent_gradients.items()
        }
        for scale in args.scales:
            selected = [
                (index, row)
                for index, row in enumerate(rows)
                if row.get("kind") == "endpoint"
                and row.get("direction_index") == direction_index
                and row.get("scale") == scale
            ]
            by_sign = {int(row["sign"]): (index, row) for index, row in selected}
            minus_index, minus = by_sign[-1]
            plus_index, plus = by_sign[1]
            physical_secant = (
                np.asarray(decoded_candidates[plus_index], dtype=np.float64)
                - np.asarray(decoded_candidates[minus_index], dtype=np.float64)
            ) / (2.0 * scale)
            pair = {
                **metadata,
                "direction_index": direction_index,
                "scale": float(scale),
                "g1_prediction": predictions["g1"],
                "g2_prediction": predictions["g2"],
                "g3_prediction": predictions["g3"],
                "g2_physical_secant_prediction": float(
                    np.sum(physical_gradients["g2"] * physical_secant)
                ),
                "g3_physical_secant_prediction": float(
                    np.sum(physical_gradients["g3"] * physical_secant)
                ),
            }
            for key in (
                "frozen_score",
                "volume_qs_component",
                "coil_component",
                "target_error",
                "qa_error",
                "qp_error",
            ):
                pair[f"{key}_slope"] = (plus[key] - minus[key]) / (2.0 * scale)
            for key in (
                "g3_frozen_score",
                "g3_volume_qs_component",
                "g3_coordinate_component",
                "g3_iota_component",
                "g3_coil_component",
                "g3_target_error",
                "g3_qa_error",
                "g3_qp_error",
                "g3_iota_min",
                "g3_iota_max",
            ):
                pair[f"{key}_slope"] = (plus[key] - minus[key]) / (2.0 * scale)
            if minus["g4_status"] == "ok" and plus["g4_status"] == "ok":
                pair["g4_score_slope"] = (
                    plus["g4_score"] - minus["g4_score"]
                ) / (2.0 * scale)
                for component in (
                    "psi", "surface", "coordinate", "volume_qs", "iota", "coil"
                ):
                    pair[f"g4_{component}_slope"] = (
                        plus["g4_components"][component]
                        - minus["g4_components"][component]
                    ) / (2.0 * scale)
                for diagnostic_name in (
                    "iota_min", "qs_global_error", "qs_qa_global_error", "qs_qp_global_error"
                ):
                    pair[f"g4_{diagnostic_name}_slope"] = (
                        float(plus["g4_diagnostics"][diagnostic_name])
                        - float(minus["g4_diagnostics"][diagnostic_name])
                    ) / (2.0 * scale)
            else:
                pair["g4_score_slope"] = float("nan")
            pairs.append(pair)

    scale_summaries = []
    for scale in args.scales:
        selected = [row for row in pairs if row["kind"] == "random" and row["scale"] == scale]
        observed = np.asarray([row["frozen_score_slope"] for row in selected])
        physical = np.asarray([row["g2_physical_secant_prediction"] for row in selected])
        latent = np.asarray([row["g2_prediction"] for row in selected])
        g3_observed = np.asarray([row["g3_frozen_score_slope"] for row in selected])
        g3_physical = np.asarray([row["g3_physical_secant_prediction"] for row in selected])
        g3_latent = np.asarray([row["g3_prediction"] for row in selected])
        g4_observed = np.asarray([row["g4_score_slope"] for row in selected])
        g4_finite = np.isfinite(g4_observed)
        scale_summaries.append(
            {
                "scale": float(scale),
                "random_direction_count": len(selected),
                "frozen_vs_physical_cosine": cosine(observed, physical),
                "frozen_vs_latent_cosine": cosine(observed, latent),
                "physical_vs_latent_cosine": cosine(physical, latent),
                "frozen_slope_rms": rms(observed),
                "physical_prediction_rms": rms(physical),
                "latent_prediction_rms": rms(latent),
                "g3_frozen_vs_physical_cosine": cosine(g3_observed, g3_physical),
                "g3_frozen_vs_latent_cosine": cosine(g3_observed, g3_latent),
                "g3_physical_vs_latent_cosine": cosine(g3_physical, g3_latent),
                "g3_frozen_slope_rms": rms(g3_observed),
                "g3_physical_prediction_rms": rms(g3_physical),
                "g3_latent_prediction_rms": rms(g3_latent),
                "g4_valid_direction_count": int(np.sum(g4_finite)),
                "g4_vs_g2_latent_cosine": cosine(g4_observed[g4_finite], latent[g4_finite]),
                "g4_vs_g3_latent_cosine": cosine(g4_observed[g4_finite], g3_latent[g4_finite]),
                "g4_slope_rms": rms(g4_observed[g4_finite]),
            }
        )

    output = {
        "format": "qh_g2_g3_fixed_geometry_closure_v2",
        "iteration": int(args.iteration),
        "nfp": nfp,
        "n_coils": int(noise.shape[0]),
        "rk4_steps": rk4_steps,
        "scales": list(args.scales),
        "random_direction_count": int(args.random_directions),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "gradient_lib": str(args.gradient_lib),
        "gradient_lib_sha256": file_sha256(args.gradient_lib),
        "saved_score": state["score"],
        "center_score_result": frozen["center_score_result"],
        "decoded_saved_token_relative_l2": float(
            np.linalg.norm(decoded_center - saved_tokens)
            / max(np.linalg.norm(saved_tokens), 1.0e-30)
        ),
        "frozen_center_score": float(frozen["frozen_score"][0]),
        "frozen_center_score_delta": float(
            frozen["frozen_score"][0] - frozen["center_score_result"]["score"]
        ),
        "g3_frozen_center_score": float(g3_frozen["frozen_score"][0]),
        "g3_frozen_center_score_delta": float(
            g3_frozen["frozen_score"][0] - g3_frozen["center_score_result"]["score"]
        ),
        "g4_branch_center_score": float(g4_branch["query_score_results"][0]["score"]),
        "g4_branch_center_score_delta": float(
            g4_branch["query_score_results"][0]["score"]
            - g4_branch["center_score_result"]["score"]
        ),
        "native_gradient_diagnostics": {
            name: payload["gradient_diagnostics"] for name, payload in native.items()
        },
        "flow_diagnostics": flow_diagnostics,
        "latent_gradient_cosines": {
            "g1_g2": cosine(latent_gradients["g1"], latent_gradients["g2"]),
            "g2_g3": cosine(latent_gradients["g2"], latent_gradients["g3"]),
        },
        "latent_gradients": {
            name: gradient.tolist() for name, gradient in latent_gradients.items()
        },
        "candidate_decode_wall_s": float(candidate_decode_wall_s),
        "frozen_batch_wall_s": float(frozen_wall_s),
        "g3_frozen_batch_wall_s": float(g3_frozen_wall_s),
        "g4_branch_batch_wall_s": float(g4_branch_wall_s),
        "g4_branch_query_mean_wall_s": float(
            np.mean([row["g4_total_wall_s"] for row in rows])
        ),
        "scale_summaries": scale_summaries,
        "pairs": pairs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", output)
    print(json.dumps({key: output[key] for key in (
        "iteration", "frozen_center_score_delta", "g3_frozen_center_score_delta",
        "g4_branch_center_score_delta", "g4_branch_query_mean_wall_s",
        "scale_summaries"
    )}, indent=2))


if __name__ == "__main__":
    main()
