from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "gpu_backend" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flow_matching.data import CoilNormalizer
from flow_matching.flow import integrate_flow
from flow_matching.model import CoilFlowTransformer
from scripts.optimize_native_score_cem import (
    NativeScorePool,
    append_jsonl,
    file_sha256,
    token_case,
    write_json,
)
from scripts.qh_score_noise_sensitivity import perturbation_metrics


TOKEN_DIM = 100


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def orthogonal_directions(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    count: int,
) -> np.ndarray:
    dimension = int(np.prod(shape))
    if not 1 <= count <= dimension:
        raise ValueError("direction count must be in [1, latent dimension]")
    matrix = rng.standard_normal((dimension, count))
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    directions = basis.T.reshape(count, *shape) * math.sqrt(dimension)
    return directions.astype(np.float32)


def gradient_from_pairs(
    plus_scores: np.ndarray,
    minus_scores: np.ndarray,
    directions: np.ndarray,
    perturbation: float,
    *,
    delta_clip: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if perturbation <= 0.0:
        raise ValueError("perturbation must be positive")
    delta = np.asarray(plus_scores, dtype=np.float64) - np.asarray(
        minus_scores, dtype=np.float64
    )
    used_delta = (
        delta
        if delta_clip is None
        else np.clip(delta, -float(delta_clip), float(delta_clip))
    )
    gradient = np.mean(
        used_delta.reshape((-1,) + (1,) * (directions.ndim - 1))
        * directions.astype(np.float64),
        axis=0,
    ) / (2.0 * perturbation)
    return gradient, delta


def prior_penalty_and_gradient(
    noise: np.ndarray,
    *,
    rms_soft: float,
    coordinate_soft: float,
    rms_weight: float,
    coordinate_weight: float,
) -> tuple[float, np.ndarray]:
    value = np.asarray(noise, dtype=np.float64)
    dimension = value.size
    value_rms = rms(value)
    gradient = np.zeros_like(value)
    penalty = 0.0
    if value_rms > rms_soft:
        excess = value_rms - rms_soft
        penalty += rms_weight * excess * excess
        gradient += (
            rms_weight
            * 2.0
            * excess
            * value
            / (dimension * max(value_rms, 1.0e-30))
        )
    coordinate_excess = np.maximum(np.abs(value) - coordinate_soft, 0.0)
    if np.any(coordinate_excess > 0.0):
        penalty += coordinate_weight * float(np.mean(coordinate_excess**2))
        gradient += (
            coordinate_weight
            * 2.0
            * coordinate_excess
            * np.sign(value)
            / dimension
        )
    return float(penalty), gradient


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=np.float64).ravel()
    right_flat = np.asarray(right, dtype=np.float64).ravel()
    denominator = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    if denominator <= 1.0e-30:
        return float("nan")
    return float(np.dot(left_flat, right_flat) / denominator)


def result_score(result: dict[str, Any] | None) -> float:
    if result is None:
        return 0.0
    score = float(result.get("score", 0.0))
    return score if math.isfinite(score) else 0.0


def result_valid(result: dict[str, Any] | None) -> bool:
    return result is not None and result.get("status") == "ok"


def load_flow_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[CoilFlowTransformer, CoilNormalizer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"ema", "model_config", "normalizer", "step"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"flow checkpoint is missing keys: {sorted(missing)}")
    model = CoilFlowTransformer(**checkpoint["model_config"]).to(
        device=device, dtype=torch.float32
    )
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    return model, CoilNormalizer.from_dict(checkpoint["normalizer"]), checkpoint


def load_initial_noise(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "flow_prior_start" in payload:
        noise = payload["flow_prior_start"]["noise"]
    elif "flow_prior_screening" in payload:
        noise = payload["flow_prior_screening"]["noise"]
    elif "flow_prior_zo_adam" in payload:
        noise = payload["flow_prior_zo_adam"]["noise"]
    elif "flow_prior_standard_adam" in payload:
        noise = payload["flow_prior_standard_adam"]["noise"]
    elif "flow_prior_local_full_gradient_adam" in payload:
        noise = payload["flow_prior_local_full_gradient_adam"]["noise"]
    elif "flow_prior_local_full_gradient_bfgs" in payload:
        noise = payload["flow_prior_local_full_gradient_bfgs"]["noise"]
    elif "flow_prior_subspace_bfgs" in payload:
        noise = payload["flow_prior_subspace_bfgs"]["noise"]
    elif "flow_prior_g3_informed_subspace_adam" in payload:
        noise = payload["flow_prior_g3_informed_subspace_adam"]["noise"]
    elif "flow_prior_cem" in payload:
        noise = payload["flow_prior_cem"]["noise"]
    elif "noise" in payload:
        noise = payload["noise"]
    else:
        raise ValueError("initial case does not contain flow-prior noise")
    value = np.asarray(noise, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != TOKEN_DIM:
        raise ValueError(f"initial noise must have shape (coils, {TOKEN_DIM})")
    return value, payload


@torch.inference_mode()
def decode_noise_rk4(
    model: CoilFlowTransformer,
    normalizer: CoilNormalizer,
    noise: np.ndarray,
    *,
    nfp: int,
    steps: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    values = np.asarray(noise, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != TOKEN_DIM:
        raise ValueError(f"noise must have shape (batch, coils, {TOKEN_DIM})")
    started = time.perf_counter()
    state = torch.from_numpy(values).to(device=device, dtype=torch.float32)
    nfp_tensor = torch.full(
        (len(values),), int(nfp), dtype=torch.long, device=device
    )
    decoded = integrate_flow(
        model,
        state,
        nfp_tensor,
        start_time=0.0,
        end_time=1.0,
        steps=steps,
        method="rk4",
    )
    torch.cuda.synchronize(device)
    normalized = decoded.cpu().numpy()
    key = (int(nfp), int(values.shape[1]))
    raw = normalizer.inverse(normalized, key).astype(np.float64, copy=False)
    return raw, float(time.perf_counter() - started)


def score_tokens(
    pool: NativeScorePool,
    tokens: np.ndarray,
    *,
    nfp: int,
    target: str,
    timeout_s: float,
    metadata: dict[str, Any],
    config_overrides: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any] | None], list[float], list[str | None], float]:
    cases = [
        token_case(
            value,
            nfp=nfp,
            target=target,
            metadata={**metadata, "batch_index": index},
        )
        for index, value in enumerate(tokens)
    ]
    started = time.perf_counter()
    evaluated = pool.map(
        cases,
        target=target,
        timeout_s=timeout_s,
        config_overrides=config_overrides,
    )
    wall_s = time.perf_counter() - started
    return (
        [item[0] for item in evaluated],
        [float(item[1]) for item in evaluated],
        [item[2] for item in evaluated],
        float(wall_s),
    )


def direction_sign_agreement(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    keep = np.abs(reference) > 0.02
    if not np.any(keep):
        return 1.0
    return float(np.mean(np.sign(candidate[keep]) == np.sign(reference[keep])))


def calibrate_integrator(
    model: CoilFlowTransformer,
    normalizer: CoilNormalizer,
    pool: NativeScorePool,
    center: np.ndarray,
    directions: np.ndarray,
    *,
    perturbation: float,
    steps_to_test: tuple[int, ...],
    nfp: int,
    target: str,
    device: torch.device,
    timeout_s: float,
) -> tuple[dict[str, Any], int, np.ndarray, dict[str, Any]]:
    states = np.concatenate(
        [
            center[None],
            center[None] + perturbation * directions,
            center[None] - perturbation * directions,
        ],
        axis=0,
    ).astype(np.float32)
    records: list[dict[str, Any]] = []
    raw_by_steps: dict[int, np.ndarray] = {}
    results_by_steps: dict[int, list[dict[str, Any] | None]] = {}
    gradients: dict[int, np.ndarray] = {}
    deltas: dict[int, np.ndarray] = {}
    count = len(directions)
    for steps in steps_to_test:
        raw, decode_wall_s = decode_noise_rk4(
            model,
            normalizer,
            states,
            nfp=nfp,
            steps=steps,
            device=device,
        )
        results, elapsed, errors, score_wall_s = score_tokens(
            pool,
            raw,
            nfp=nfp,
            target=target,
            timeout_s=timeout_s,
            metadata={"phase": "integrator_calibration", "rk4_steps": steps},
        )
        if any(error is not None for error in errors):
            raise RuntimeError(f"integrator calibration score errors at {steps}: {errors}")
        scores = np.asarray([result_score(result) for result in results])
        gradient, delta = gradient_from_pairs(
            scores[1 : 1 + count],
            scores[1 + count :],
            directions,
            perturbation,
            delta_clip=None,
        )
        raw_by_steps[steps] = raw
        results_by_steps[steps] = results
        gradients[steps] = gradient
        deltas[steps] = delta
        records.append(
            {
                "steps": steps,
                "decode_wall_s": decode_wall_s,
                "score_wall_s": score_wall_s,
                "score_elapsed_s": elapsed,
                "center_score": scores[0],
                "scores": scores.tolist(),
                "statuses": [
                    None if result is None else str(result.get("status"))
                    for result in results
                ],
                "direction_deltas": delta.tolist(),
            }
        )

    reference_steps = steps_to_test[-1]
    reference_raw = raw_by_steps[reference_steps]
    reference_delta = deltas[reference_steps]
    reference_gradient = gradients[reference_steps]
    reference_displacements = [
        perturbation_metrics(reference_raw[index], reference_raw[0])[
            "position_delta_rms_m"
        ]
        for index in range(1, len(reference_raw))
    ]
    perturbation_position_rms = float(np.median(reference_displacements))
    selected_steps = reference_steps
    for record in records:
        steps = int(record["steps"])
        raw = raw_by_steps[steps]
        integration_errors = [
            perturbation_metrics(raw[index], reference_raw[index])[
                "position_delta_rms_m"
            ]
            for index in range(len(raw))
        ]
        ratio = float(
            max(integration_errors) / max(perturbation_position_rms, 1.0e-30)
        )
        delta = deltas[steps]
        delta_tolerance = np.maximum(0.02, 0.1 * np.abs(reference_delta))
        delta_ok = bool(np.all(np.abs(delta - reference_delta) <= delta_tolerance))
        gradient_cosine = cosine_similarity(gradients[steps], reference_gradient)
        sign_agreement = direction_sign_agreement(delta, reference_delta)
        status_match = float(
            np.mean(
                [
                    (result_valid(left) == result_valid(right))
                    for left, right in zip(
                        results_by_steps[steps], results_by_steps[reference_steps]
                    )
                ]
            )
        )
        passed = bool(
            ratio <= 0.01
            and delta_ok
            and (math.isnan(gradient_cosine) or gradient_cosine >= 0.98)
            and sign_agreement >= 1.0
            and status_match >= 1.0
        )
        record.update(
            {
                "reference_steps": reference_steps,
                "max_position_error_m": float(max(integration_errors)),
                "perturbation_position_rms_m": perturbation_position_rms,
                "position_error_to_perturbation": ratio,
                "max_direction_delta_error": float(
                    np.max(np.abs(delta - reference_delta))
                ),
                "gradient_cosine_to_reference": gradient_cosine,
                "direction_sign_agreement": sign_agreement,
                "status_match_fraction": status_match,
                "passed": passed,
            }
        )
        if passed and selected_steps == reference_steps:
            selected_steps = steps

    selected_results = results_by_steps[selected_steps]
    center_result = selected_results[0]
    if not result_valid(center_result):
        raise RuntimeError(
            f"selected RK4 center is not valid: {None if center_result is None else center_result.get('status')}"
        )
    calibration = {
        "method": "rk4_fp32",
        "perturbation": perturbation,
        "direction_count": len(directions),
        "reference_steps": reference_steps,
        "selected_steps": selected_steps,
        "records": records,
    }
    return calibration, selected_steps, raw_by_steps[selected_steps][0], center_result


def make_best_case(
    tokens: np.ndarray,
    noise: np.ndarray,
    result: dict[str, Any],
    *,
    nfp: int,
    target: str,
    iteration: int,
    seed: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    case = token_case(tokens, nfp=nfp, target=target)
    case["flow_prior_zo_adam"] = {
        "target": target,
        "seed": seed,
        "iteration": iteration,
        "best_score": result_score(result),
        "noise": np.asarray(noise, dtype=np.float32).tolist(),
        "native_score": result,
        "manifest": manifest,
    }
    return case


def save_state(
    path: Path,
    *,
    current_noise: np.ndarray,
    best_noise: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    iteration: int,
    adam_step: int,
    learning_rate: float,
    perturbation: float,
    rng: np.random.Generator,
) -> None:
    np.savez_compressed(
        path,
        current_noise=current_noise,
        best_noise=best_noise,
        first_moment=first_moment,
        second_moment=second_moment,
        iteration=np.asarray(iteration, dtype=np.int64),
        adam_step=np.asarray(adam_step, dtype=np.int64),
        learning_rate=np.asarray(learning_rate, dtype=np.float64),
        perturbation=np.asarray(perturbation, dtype=np.float64),
        rng_state=np.asarray(json.dumps(rng.bit_generator.state)),
    )


def plot_progress(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    iterations = [row["iteration"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(iterations, [row["current_score"] for row in rows], label="current")
    axes[0, 0].plot(iterations, [row["best_score"] for row in rows], label="best")
    axes[0, 0].set(ylabel="native score", title="Optimization")
    axes[0, 0].legend()
    axes[0, 1].plot(iterations, [row["current_qh_error"] for row in rows], label="QH")
    axes[0, 1].plot(iterations, [row["current_qa_error"] for row in rows], label="QA")
    axes[0, 1].plot(iterations, [row["current_qp_error"] for row in rows], label="QP")
    axes[0, 1].set(ylabel="volume residual", title="Helicity diagnostics")
    axes[0, 1].legend()
    axes[1, 0].plot(iterations, [row["learning_rate"] for row in rows], label="lr")
    axes[1, 0].plot(iterations, [row["perturbation"] for row in rows], label="c")
    axes[1, 0].plot(iterations, [row["update_rms"] for row in rows], label="update RMS")
    axes[1, 0].set(yscale="log", ylabel="latent scale", xlabel="iteration")
    axes[1, 0].legend()
    axes[1, 1].plot(iterations, [row["valid_endpoint_fraction"] for row in rows], label="valid endpoints")
    axes[1, 1].plot(iterations, [row["noise_rms"] for row in rows], label="noise RMS")
    axes[1, 1].set(ylabel="fraction / RMS", xlabel="iteration")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def diagnostics_value(result: dict[str, Any], name: str) -> float:
    return float(result.get("diagnostics", {}).get(name, float("nan")))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize flow-prior noise with orthogonal antithetic gradients and Adam."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-case", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--lib",
        type=Path,
        default=REPO_ROOT / "gpu_backend" / "build_native_score" / "libstellarator_gpu.so",
    )
    parser.add_argument("--target", choices=("QA", "QH"), default="QH")
    parser.add_argument("--nfp", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--calibration-steps", default="64,128,256")
    parser.add_argument("--perturbation", type=float, default=0.01)
    parser.add_argument("--min-perturbation", type=float, default=0.003)
    parser.add_argument("--max-perturbation", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--min-learning-rate", type=float, default=0.0001)
    parser.add_argument("--max-learning-rate", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--max-update-rms", type=float, default=0.0075)
    parser.add_argument("--backtrack-scales", default="1,0.5,0.25")
    parser.add_argument("--accept-drop", type=float, default=0.1)
    parser.add_argument("--delta-clip", type=float, default=15.0)
    parser.add_argument("--noise-limit", type=float, default=6.0)
    parser.add_argument("--prior-rms-soft", type=float, default=2.0)
    parser.add_argument("--prior-coordinate-soft", type=float, default=4.0)
    parser.add_argument("--prior-rms-weight", type=float, default=5.0)
    parser.add_argument("--prior-coordinate-weight", type=float, default=5.0)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-timeout-s", type=float, default=300.0)
    parser.add_argument("--max-wall-s", type=float, default=7200.0)
    parser.add_argument("--plot-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026073002)
    args = parser.parse_args()

    calibration_steps = parse_ints(args.calibration_steps)
    backtrack_scales = parse_floats(args.backtrack_scales)
    gpu_ids = parse_ints(args.gpus)
    if not torch.cuda.is_available():
        raise RuntimeError("flow-prior zeroth-order Adam requires CUDA")
    if not calibration_steps or tuple(sorted(set(calibration_steps))) != calibration_steps:
        raise ValueError("calibration steps must be positive, unique, and increasing")
    if any(value < 1 for value in calibration_steps):
        raise ValueError("calibration steps must be positive")
    if not gpu_ids:
        raise ValueError("at least one score GPU is required")
    if args.directions < 1:
        raise ValueError("invalid direction count")
    if not backtrack_scales or any(value <= 0.0 for value in backtrack_scales):
        raise ValueError("backtrack scales must be positive")
    if not 0.0 < args.beta1 < 1.0 or not 0.0 < args.beta2 < 1.0:
        raise ValueError("Adam betas must be in (0, 1)")
    if not 0.0 < args.min_learning_rate <= args.learning_rate <= args.max_learning_rate:
        raise ValueError("learning-rate bounds are inconsistent")
    if not 0.0 < args.min_perturbation <= args.perturbation <= args.max_perturbation:
        raise ValueError("perturbation bounds are inconsistent")
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if args.plot_every < 1:
        raise ValueError("plot-every must be positive")
    if not args.checkpoint.is_file() or not args.initial_case.is_file() or not args.lib.is_file():
        raise FileNotFoundError("checkpoint, initial case, and score library must exist")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(
        (args.out_dir / name).exists()
        for name in ("manifest.json", "history.jsonl", "summary.json")
    ):
        raise FileExistsError(f"refusing to overwrite existing run {args.out_dir}")
    rng = np.random.default_rng(args.seed)
    current_noise, initial_payload = load_initial_noise(args.initial_case)
    if current_noise.shape[0] < 1:
        raise ValueError("initial noise must contain at least one coil")
    if args.directions > current_noise.size:
        raise ValueError("directions exceed latent dimension")
    current_noise = np.clip(
        current_noise, -args.noise_limit, args.noise_limit
    ).astype(np.float32)

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
    normalizer_key = f"{args.nfp}:{current_noise.shape[0]}"
    if normalizer_key not in normalizer.current_l1_a:
        raise ValueError(f"condition {normalizer_key} is absent from normalizer")

    manifest = {
        "algorithm": "orthogonal_antithetic_zo_adam_over_flow_prior_noise",
        "target": args.target,
        "nfp": args.nfp,
        "n_base_coils": int(current_noise.shape[0]),
        "noise_shape": list(current_noise.shape),
        "seed": args.seed,
        "iterations": args.iterations,
        "directions": args.directions,
        "calibration_steps": list(calibration_steps),
        "perturbation": args.perturbation,
        "learning_rate": args.learning_rate,
        "betas": [args.beta1, args.beta2],
        "max_update_rms": args.max_update_rms,
        "backtrack_scales": list(backtrack_scales),
        "accept_drop": args.accept_drop,
        "delta_clip": args.delta_clip,
        "noise_limit": args.noise_limit,
        "flow_dtype": "torch.float32",
        "flow_method": "rk4",
        "flow_autocast": False,
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "native_lib_sha256": file_sha256(args.lib),
        "initial_case": str(args.initial_case.resolve()),
        "initial_recorded_score": initial_payload.get("flow_prior_cem", {}).get(
            "best_score"
        ),
        "gpu_ids": list(gpu_ids),
        "max_wall_s": args.max_wall_s,
    }
    write_json(args.out_dir / "manifest.json", manifest)

    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    history_path = args.out_dir / "history.jsonl"
    stop_reason = "completed_iterations"
    learning_rate = float(args.learning_rate)
    perturbation = float(args.perturbation)
    first_moment = np.zeros_like(current_noise, dtype=np.float64)
    second_moment = np.zeros_like(current_noise, dtype=np.float64)
    adam_step = 0

    with NativeScorePool(args.lib, list(gpu_ids)) as pool:
        calibration_directions = orthogonal_directions(
            rng, current_noise.shape, args.directions
        )
        calibration, flow_steps, current_tokens, current_result = calibrate_integrator(
            model,
            normalizer,
            pool,
            current_noise,
            calibration_directions,
            perturbation=perturbation,
            steps_to_test=calibration_steps,
            nfp=args.nfp,
            target=args.target,
            device=device,
            timeout_s=args.batch_timeout_s,
        )
        write_json(args.out_dir / "integration_calibration.json", calibration)
        manifest["selected_flow_steps"] = flow_steps
        write_json(args.out_dir / "manifest.json", manifest)

        current_penalty, _ = prior_penalty_and_gradient(
            current_noise,
            rms_soft=args.prior_rms_soft,
            coordinate_soft=args.prior_coordinate_soft,
            rms_weight=args.prior_rms_weight,
            coordinate_weight=args.prior_coordinate_weight,
        )
        current_merit = result_score(current_result) - current_penalty
        initial_score = result_score(current_result)
        best_score = initial_score
        best_noise = current_noise.copy()
        best_tokens = current_tokens.copy()
        best_result = current_result
        best_iteration = 0
        write_json(
            args.out_dir / "best.json",
            make_best_case(
                best_tokens,
                best_noise,
                best_result,
                nfp=args.nfp,
                target=args.target,
                iteration=0,
                seed=args.seed,
                manifest=manifest,
            ),
        )

        recent_walls: list[float] = []
        for iteration in range(1, args.iterations + 1):
            elapsed_before = time.perf_counter() - started
            if recent_walls and args.max_wall_s > 0.0:
                projected = 1.2 * float(np.mean(recent_walls[-5:]))
                if elapsed_before + projected >= args.max_wall_s:
                    stop_reason = "wall_budget"
                    break

            iteration_started = time.perf_counter()
            directions = orthogonal_directions(
                rng, current_noise.shape, args.directions
            )
            pair_states = np.concatenate(
                [
                    current_noise[None] + perturbation * directions,
                    current_noise[None] - perturbation * directions,
                ],
                axis=0,
            )
            pair_states = np.clip(
                pair_states, -args.noise_limit, args.noise_limit
            ).astype(np.float32)
            pair_tokens, pair_decode_wall_s = decode_noise_rk4(
                model,
                normalizer,
                pair_states,
                nfp=args.nfp,
                steps=flow_steps,
                device=device,
            )
            pair_results, pair_elapsed, pair_errors, pair_score_wall_s = score_tokens(
                pool,
                pair_tokens,
                nfp=args.nfp,
                target=args.target,
                timeout_s=args.batch_timeout_s,
                metadata={"phase": "gradient", "iteration": iteration},
            )
            if any(error is not None for error in pair_errors):
                raise RuntimeError(
                    f"score worker error at iteration {iteration}: {pair_errors}"
                )
            pair_scores = np.asarray(
                [result_score(result) for result in pair_results], dtype=np.float64
            )
            pair_valid = np.asarray(
                [result_valid(result) for result in pair_results], dtype=bool
            )
            valid_fraction = float(np.mean(pair_valid))
            gradient, raw_delta = gradient_from_pairs(
                pair_scores[: args.directions],
                pair_scores[args.directions :],
                directions,
                perturbation,
                delta_clip=args.delta_clip,
            )
            _, prior_gradient = prior_penalty_and_gradient(
                current_noise,
                rms_soft=args.prior_rms_soft,
                coordinate_soft=args.prior_coordinate_soft,
                rms_weight=args.prior_rms_weight,
                coordinate_weight=args.prior_coordinate_weight,
            )
            gradient -= prior_gradient
            gradient_rms = rms(gradient)
            accepted = False
            selected_proposal = None
            proposal_rows: list[dict[str, Any]] = []
            update_rms = 0.0
            proposal_decode_wall_s = 0.0
            proposal_score_wall_s = 0.0
            proposal_elapsed: list[float] = []

            enough_valid = np.count_nonzero(pair_valid) >= args.directions
            if enough_valid and math.isfinite(gradient_rms) and gradient_rms > 0.0:
                next_adam_step = adam_step + 1
                tentative_first = (
                    args.beta1 * first_moment + (1.0 - args.beta1) * gradient
                )
                tentative_second = (
                    args.beta2 * second_moment
                    + (1.0 - args.beta2) * gradient * gradient
                )
                first_hat = tentative_first / (1.0 - args.beta1**next_adam_step)
                second_hat = tentative_second / (1.0 - args.beta2**next_adam_step)
                update = (
                    learning_rate
                    * first_hat
                    / (np.sqrt(second_hat) + args.adam_epsilon)
                )
                update_rms = rms(update)
                if update_rms > args.max_update_rms:
                    update *= args.max_update_rms / update_rms
                    update_rms = args.max_update_rms
                proposal_noise = np.stack(
                    [
                        np.clip(
                            current_noise + scale * update,
                            -args.noise_limit,
                            args.noise_limit,
                        )
                        for scale in backtrack_scales
                    ]
                ).astype(np.float32)
                proposal_tokens, proposal_decode_wall_s = decode_noise_rk4(
                    model,
                    normalizer,
                    proposal_noise,
                    nfp=args.nfp,
                    steps=flow_steps,
                    device=device,
                )
                (
                    proposal_results,
                    proposal_elapsed,
                    proposal_errors,
                    proposal_score_wall_s,
                ) = score_tokens(
                    pool,
                    proposal_tokens,
                    nfp=args.nfp,
                    target=args.target,
                    timeout_s=args.batch_timeout_s,
                    metadata={"phase": "proposal", "iteration": iteration},
                )
                if any(error is not None for error in proposal_errors):
                    raise RuntimeError(
                        f"proposal score error at iteration {iteration}: {proposal_errors}"
                    )
                for index, (scale, noise, tokens, result) in enumerate(
                    zip(
                        backtrack_scales,
                        proposal_noise,
                        proposal_tokens,
                        proposal_results,
                    )
                ):
                    penalty, _ = prior_penalty_and_gradient(
                        noise,
                        rms_soft=args.prior_rms_soft,
                        coordinate_soft=args.prior_coordinate_soft,
                        rms_weight=args.prior_rms_weight,
                        coordinate_weight=args.prior_coordinate_weight,
                    )
                    merit = result_score(result) - penalty
                    proposal_rows.append(
                        {
                            "index": index,
                            "scale": scale,
                            "score": result_score(result),
                            "status": None if result is None else result.get("status"),
                            "penalty": penalty,
                            "merit": merit,
                        }
                    )
                valid_indices = [
                    index
                    for index, result in enumerate(proposal_results)
                    if result_valid(result)
                ]
                if valid_indices:
                    selected_index = max(
                        valid_indices,
                        key=lambda index: proposal_rows[index]["merit"],
                    )
                    candidate_merit = float(proposal_rows[selected_index]["merit"])
                    if candidate_merit >= current_merit - args.accept_drop:
                        accepted = True
                        selected_proposal = selected_index
                        current_noise = proposal_noise[selected_index].copy()
                        current_tokens = proposal_tokens[selected_index].copy()
                        current_result = proposal_results[selected_index]
                        current_penalty = float(
                            proposal_rows[selected_index]["penalty"]
                        )
                        current_merit = candidate_merit
                        first_moment = tentative_first
                        second_moment = tentative_second
                        adam_step = next_adam_step
                        if result_score(current_result) > best_score:
                            best_score = result_score(current_result)
                            best_noise = current_noise.copy()
                            best_tokens = current_tokens.copy()
                            best_result = current_result
                            best_iteration = iteration
                            write_json(
                                args.out_dir / "best.json",
                                make_best_case(
                                    best_tokens,
                                    best_noise,
                                    best_result,
                                    nfp=args.nfp,
                                    target=args.target,
                                    iteration=best_iteration,
                                    seed=args.seed,
                                    manifest=manifest,
                                ),
                            )
                        learning_rate = min(
                            args.max_learning_rate, learning_rate * 1.01
                        )
                    else:
                        learning_rate = max(
                            args.min_learning_rate, learning_rate * 0.5
                        )
                else:
                    learning_rate = max(
                        args.min_learning_rate, learning_rate * 0.5
                    )
            else:
                if gradient_rms <= 1.0e-12:
                    perturbation = min(
                        args.max_perturbation, perturbation * 1.25
                    )
                else:
                    perturbation = max(
                        args.min_perturbation, perturbation * 0.8
                    )
                learning_rate = max(
                    args.min_learning_rate, learning_rate * 0.5
                )

            if accepted and valid_fraction < 0.75:
                perturbation = max(
                    args.min_perturbation, perturbation * 0.9
                )
            elif accepted and iteration % 25 == 0:
                perturbation = max(
                    args.min_perturbation, perturbation * 0.95
                )

            iteration_wall_s = time.perf_counter() - iteration_started
            recent_walls.append(iteration_wall_s)
            row = {
                "iteration": iteration,
                "accepted": accepted,
                "selected_proposal": selected_proposal,
                "current_score": result_score(current_result),
                "current_status": current_result.get("status"),
                "current_merit": current_merit,
                "current_penalty": current_penalty,
                "best_score": best_score,
                "best_iteration": best_iteration,
                "current_qh_error": diagnostics_value(
                    current_result, "qs_global_error"
                ),
                "current_qa_error": diagnostics_value(
                    current_result, "qs_qa_global_error"
                ),
                "current_qp_error": diagnostics_value(
                    current_result, "qs_qp_global_error"
                ),
                "current_iota": diagnostics_value(current_result, "iota_min"),
                "valid_endpoint_fraction": valid_fraction,
                "pair_scores": pair_scores.tolist(),
                "pair_statuses": [
                    None if result is None else result.get("status")
                    for result in pair_results
                ],
                "raw_direction_deltas": raw_delta.tolist(),
                "gradient_rms": gradient_rms,
                "update_rms": update_rms,
                "learning_rate": learning_rate,
                "perturbation": perturbation,
                "noise_rms": rms(current_noise),
                "noise_abs_max": float(np.max(np.abs(current_noise))),
                "proposals": proposal_rows,
                "pair_score_elapsed_s": pair_elapsed,
                "proposal_score_elapsed_s": proposal_elapsed,
                "pair_decode_wall_s": pair_decode_wall_s,
                "pair_score_wall_s": pair_score_wall_s,
                "proposal_decode_wall_s": proposal_decode_wall_s,
                "proposal_score_wall_s": proposal_score_wall_s,
                "iteration_wall_s": iteration_wall_s,
                "total_wall_s": time.perf_counter() - started,
            }
            history.append(row)
            append_jsonl(history_path, row)
            write_json(
                args.out_dir / "progress.json",
                {
                    "manifest": manifest,
                    "calibration": calibration,
                    "initial_score": initial_score,
                    "best_score": best_score,
                    "best_iteration": best_iteration,
                    "iterations": history,
                },
            )
            save_state(
                args.out_dir / "state_latest.npz",
                current_noise=current_noise,
                best_noise=best_noise,
                first_moment=first_moment,
                second_moment=second_moment,
                iteration=iteration,
                adam_step=adam_step,
                learning_rate=learning_rate,
                perturbation=perturbation,
                rng=rng,
            )
            if iteration % args.plot_every == 0 or iteration == 1:
                plot_progress(history, args.out_dir / "progress.png")
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "accepted": accepted,
                        "current_score": result_score(current_result),
                        "best_score": best_score,
                        "valid_endpoint_fraction": valid_fraction,
                        "gradient_rms": gradient_rms,
                        "update_rms": update_rms,
                        "learning_rate": learning_rate,
                        "perturbation": perturbation,
                        "wall_s": iteration_wall_s,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

    plot_progress(history, args.out_dir / "progress.png")
    summary = {
        "manifest": manifest,
        "calibration": calibration,
        "stop_reason": stop_reason,
        "completed_iterations": len(history),
        "initial_score": initial_score,
        "final_score": result_score(current_result),
        "best_score": best_score,
        "best_iteration": best_iteration,
        "best_components": best_result["components"],
        "best_diagnostics": best_result["diagnostics"],
        "accepted_steps": int(sum(row["accepted"] for row in history)),
        "total_wall_s": time.perf_counter() - started,
    }
    write_json(args.out_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "stop_reason": stop_reason,
                "completed_iterations": len(history),
                "initial_score": initial_score,
                "best_score": best_score,
                "best_iteration": best_iteration,
                "total_wall_s": summary["total_wall_s"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
