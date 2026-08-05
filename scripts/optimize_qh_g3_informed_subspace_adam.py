from __future__ import annotations

import argparse
import json
import math
import os
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

from scripts.optimize_flow_prior_zo_adam import (
    decode_noise_rk4,
    load_flow_checkpoint,
    load_initial_noise,
    parse_floats,
    parse_ints,
    result_score,
    result_valid,
    score_tokens,
)
from scripts.optimize_native_score_cem import NativeScorePool, token_case
from scripts.optimize_qh_physical_gradient_adam import (
    COMPONENT_NAMES,
    Evaluation,
    accept_exact_score_candidate,
    array_sha256,
    diagnostic,
    evaluate,
    rms,
)
from scripts.qh_blackbox_gradient_reference import (
    append_jsonl,
    branch_fingerprint,
    compact_result,
    file_sha256,
    write_json,
)


def decode_noise_rk4_independent(
    model,
    normalizer,
    values: np.ndarray,
    *,
    nfp: int,
    steps: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    states = np.asarray(values, dtype=np.float32)
    if states.ndim != 3 or not len(states):
        raise ValueError("independent flow decode requires a nonempty state batch")
    started = time.perf_counter()
    decoded: list[np.ndarray] = []
    for state in states:
        raw, _ = decode_noise_rk4(
            model,
            normalizer,
            state[None],
            nfp=nfp,
            steps=steps,
            device=device,
        )
        decoded.append(raw[0])
    return np.stack(decoded), float(time.perf_counter() - started)


def informed_orthogonal_directions(
    rng: np.random.Generator,
    reference_gradient: np.ndarray,
    random_count: int,
) -> np.ndarray:
    gradient = np.asarray(reference_gradient, dtype=np.float64)
    dimension = gradient.size
    if not 0 <= random_count < dimension:
        raise ValueError("random_count must be in [0, latent dimension)")
    scale = rms(gradient)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("reference gradient must have positive finite RMS")
    informed = gradient.ravel() / scale
    if random_count == 0:
        return informed.reshape((1, *gradient.shape)).astype(np.float32)
    matrix = rng.standard_normal((dimension, random_count))
    matrix -= informed[:, None] * (informed @ matrix)[None] / dimension
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    random_directions = basis.T * math.sqrt(dimension)
    directions = np.concatenate((informed[None], random_directions), axis=0)
    return directions.reshape((-1, *gradient.shape)).astype(np.float32)


def exact_subspace_gradient(
    center_result: dict[str, Any],
    plus_results: list[dict[str, Any] | None],
    minus_results: list[dict[str, Any] | None],
    directions: np.ndarray,
    perturbation: float,
) -> tuple[np.ndarray | None, list[dict[str, Any]], float]:
    if perturbation <= 0.0 or len(plus_results) != len(directions) or len(minus_results) != len(directions):
        raise ValueError("directional score inputs are inconsistent")
    center_branch = branch_fingerprint(center_result)
    rows: list[dict[str, Any]] = []
    gradient = np.zeros_like(directions[0], dtype=np.float64)
    accepted_slopes: list[float] = []
    dimension = directions[0].size
    for index, (plus, minus, direction) in enumerate(
        zip(plus_results, minus_results, directions, strict=True)
    ):
        plus_branch = branch_fingerprint(plus)
        minus_branch = branch_fingerprint(minus)
        plus_same_branch = bool(result_valid(plus) and plus_branch == center_branch)
        minus_same_branch = bool(result_valid(minus) and minus_branch == center_branch)
        valid = plus_same_branch or minus_same_branch
        slope = float("nan")
        difference_scheme = "invalid"
        if plus_same_branch and minus_same_branch:
            slope = (result_score(plus) - result_score(minus)) / (2.0 * perturbation)
            difference_scheme = "centered"
        elif plus_same_branch:
            slope = (result_score(plus) - result_score(center_result)) / perturbation
            difference_scheme = "forward"
        elif minus_same_branch:
            slope = (result_score(center_result) - result_score(minus)) / perturbation
            difference_scheme = "backward"
        if valid:
            if math.isfinite(slope):
                gradient += slope * np.asarray(direction, dtype=np.float64) / dimension
                accepted_slopes.append(slope)
            else:
                valid = False
        rows.append(
            {
                "index": index,
                "kind": "g3" if index == 0 else "random",
                "valid": valid,
                "slope": slope,
                "difference_scheme": difference_scheme,
                "plus_status": None if plus is None else str(plus.get("status")),
                "minus_status": None if minus is None else str(minus.get("status")),
                "plus_score": result_score(plus),
                "minus_score": result_score(minus),
                "plus_same_branch": plus_branch == center_branch,
                "minus_same_branch": minus_branch == center_branch,
                "center_branch": list(center_branch),
                "plus_branch": list(plus_branch),
                "minus_branch": list(minus_branch),
            }
        )
    if not accepted_slopes:
        return None, rows, 0.0
    predicted_gain = float(np.sum(np.square(accepted_slopes)) / dimension)
    return gradient, rows, predicted_gain


def projected_trust_update(
    projected_gradient: np.ndarray,
    *,
    step_rms: float,
) -> np.ndarray | None:
    if step_rms <= 0.0:
        raise ValueError("step_rms must be positive")
    gradient = np.asarray(projected_gradient, dtype=np.float64)
    scale = rms(gradient)
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return gradient * (step_rms / scale)


def diagonal_quadratic_trust_update(
    center_result: dict[str, Any],
    direction_rows: list[dict[str, Any]],
    directions: np.ndarray,
    *,
    perturbation: float,
    max_step_rms: float,
) -> tuple[np.ndarray | None, list[dict[str, Any]], float]:
    if perturbation <= 0.0 or max_step_rms <= 0.0:
        raise ValueError("quadratic trust radii must be positive")
    if len(direction_rows) != len(directions):
        raise ValueError("quadratic direction inputs are inconsistent")
    center_score = result_score(center_result)
    coefficients: list[float] = []
    model_rows: list[dict[str, Any]] = []
    for row in direction_rows:
        scheme = str(row["difference_scheme"])
        plus_score = float(row["plus_score"])
        minus_score = float(row["minus_score"])
        linear = float("nan")
        quadratic = float("nan")
        coefficient = 0.0
        if scheme == "centered":
            linear = (plus_score - minus_score) / (2.0 * perturbation)
            quadratic = (
                plus_score + minus_score - 2.0 * center_score
            ) / (2.0 * perturbation * perturbation)
            candidates = [0.0, -perturbation, perturbation]
            if quadratic < 0.0 and math.isfinite(quadratic):
                stationary = -linear / (2.0 * quadratic)
                candidates.append(float(np.clip(stationary, -perturbation, perturbation)))

            def modeled_gain(value: float) -> float:
                return linear * value + quadratic * value * value

            coefficient = max(candidates, key=modeled_gain)
        elif scheme == "forward" and plus_score > center_score:
            coefficient = perturbation
        elif scheme == "backward" and minus_score > center_score:
            coefficient = -perturbation
        if scheme == "centered":
            unscaled_predicted_gain = (
                linear * coefficient + quadratic * coefficient * coefficient
            )
        elif coefficient > 0.0:
            unscaled_predicted_gain = max(plus_score - center_score, 0.0)
        elif coefficient < 0.0:
            unscaled_predicted_gain = max(minus_score - center_score, 0.0)
        else:
            unscaled_predicted_gain = 0.0
        coefficients.append(coefficient)
        model_rows.append(
            {
                "index": int(row["index"]),
                "kind": str(row["kind"]),
                "difference_scheme": scheme,
                "linear": linear,
                "quadratic": quadratic,
                "unscaled_coefficient": coefficient,
                "unscaled_predicted_gain": unscaled_predicted_gain,
            }
        )
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    if not np.any(coefficient_array):
        return None, model_rows, 0.0
    # The directions are orthonormal under the RMS inner product, so the RMS
    # norm of their linear combination is the Euclidean norm of coefficients.
    coefficient_norm = float(np.linalg.norm(coefficient_array))
    if coefficient_norm > max_step_rms:
        coefficient_array *= max_step_rms / coefficient_norm
    update = np.tensordot(
        coefficient_array,
        np.asarray(directions, dtype=np.float64),
        axes=(0, 0),
    )
    predicted_gain = 0.0
    for coefficient, row in zip(coefficient_array, model_rows, strict=True):
        row["coefficient"] = float(coefficient)
        if row["difference_scheme"] == "centered":
            predicted_gain += (
                float(row["linear"]) * coefficient
                + float(row["quadratic"]) * coefficient * coefficient
            )
        elif coefficient > 0.0:
            predicted_gain += max(
                float(direction_rows[int(row["index"])]["plus_score"]) - center_score,
                0.0,
            ) * coefficient / perturbation
        elif coefficient < 0.0:
            predicted_gain += max(
                float(direction_rows[int(row["index"])]["minus_score"]) - center_score,
                0.0,
            ) * (-coefficient) / perturbation
    if not math.isfinite(predicted_gain) or predicted_gain <= 0.0:
        return None, model_rows, predicted_gain
    return update, model_rows, float(predicted_gain)


def best_improving_branch_endpoint(
    center_result: dict[str, Any],
    endpoint_results: list[dict[str, Any] | None],
    *,
    minimum_gain: float,
) -> int | None:
    center_branch = branch_fingerprint(center_result)
    threshold = result_score(center_result) + minimum_gain
    candidates = [
        index
        for index, result in enumerate(endpoint_results)
        if result_valid(result)
        and branch_fingerprint(result) != center_branch
        and result_score(result) > threshold
    ]
    return max(candidates, key=lambda index: result_score(endpoint_results[index])) if candidates else None


def best_improving_endpoint(
    center_result: dict[str, Any],
    endpoint_results: list[dict[str, Any] | None],
    *,
    same_branch_minimum_gain: float,
    branch_minimum_gain: float,
) -> int | None:
    center_branch = branch_fingerprint(center_result)
    center_score = result_score(center_result)
    candidates = [
        index
        for index, result in enumerate(endpoint_results)
        if result_valid(result)
        and result_score(result)
        > center_score
        + (
            same_branch_minimum_gain
            if branch_fingerprint(result) == center_branch
            else branch_minimum_gain
        )
    ]
    return max(candidates, key=lambda index: result_score(endpoint_results[index])) if candidates else None


def score_improves_by(
    candidate_result: dict[str, Any] | None,
    incumbent_result: dict[str, Any] | None,
    *,
    minimum_gain: float,
) -> bool:
    return bool(
        result_valid(candidate_result)
        and result_valid(incumbent_result)
        and result_score(candidate_result) > result_score(incumbent_result) + minimum_gain
    )


def make_best_case(
    evaluation: Evaluation,
    *,
    nfp: int,
    iteration: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    case = token_case(evaluation.tokens, nfp=nfp, target="QH")
    case["flow_prior_g3_informed_subspace_adam"] = {
        "iteration": int(iteration),
        "score": result_score(evaluation.score_result),
        "noise": evaluation.noise.tolist(),
        "native_score": compact_result(evaluation.score_result),
        "manifest": manifest,
    }
    return case


def save_trajectory(
    directory: Path,
    *,
    iteration: int,
    evaluation: Evaluation,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    adam_step: int,
    transition: dict[str, Any] | None,
) -> None:
    write_json(
        directory / f"step_{iteration:04d}.json",
        {
            "iteration": int(iteration),
            "adam_step": int(adam_step),
            "noise": evaluation.noise.tolist(),
            "tokens": evaluation.tokens.tolist(),
            "score": compact_result(evaluation.score_result),
            "g3_latent_gradient": (
                None if evaluation.latent_gradient is None else evaluation.latent_gradient.tolist()
            ),
            "g3_physical_gradient": evaluation.physical_gradient.tolist(),
            "first_moment": first_moment.tolist(),
            "second_moment": second_moment.tolist(),
            "gradient_diagnostics": evaluation.gradient_diagnostics,
            "flow_diagnostics": evaluation.flow_diagnostics,
            "evaluation_wall_s": evaluation.wall_s,
            "transition": transition,
        },
    )


def plot_progress(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    iteration = [row["iteration"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(iteration, [row["current_score"] for row in rows], label="current")
    axes[0, 0].plot(iteration, [row["best_score"] for row in rows], label="best")
    axes[0, 0].set(title="Exact ABI-9 score", ylabel="score")
    axes[0, 0].legend()
    for name in COMPONENT_NAMES:
        axes[0, 1].plot(iteration, [row["components"][name] for row in rows], label=name)
    axes[0, 1].set(title="Score components", ylabel="component")
    axes[0, 1].legend(ncol=2, fontsize=8)
    axes[1, 0].plot(iteration, [row["g3_slope"] for row in rows], label="exact slope on G3")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(title="G3 direction verification", xlabel="iteration", ylabel="score / latent RMS")
    axes[1, 0].legend()
    axes[1, 1].plot(iteration, [row["valid_directions"] for row in rows], label="valid directions")
    axes[1, 1].plot(iteration, [row["iteration_wall_s"] for row in rows], label="wall seconds")
    axes[1, 1].set(title="Cost and branch validity", xlabel="iteration")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize QH score in a full-score-verified G3-informed latent subspace."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gradient-lib", type=Path, required=True)
    parser.add_argument("--initial-case", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--nfp", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rk4-steps", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--random-directions", type=int, default=4)
    parser.add_argument("--perturbation", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--proposal-mode",
        choices=("adam", "projected", "quadratic", "quadratic_axis"),
        default="adam",
    )
    parser.add_argument(
        "--projected-step-rms",
        type=float,
        default=None,
        help="RMS trust radius for projected mode; defaults to --perturbation.",
    )
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--backtrack-fractions", type=parse_floats, default=(0.5, 0.25, 0.125))
    parser.add_argument("--accept-drop", type=float, default=0.0)
    parser.add_argument(
        "--probe-accept-minimum-gain",
        type=float,
        default=0.001,
    )
    parser.add_argument("--branch-accept-minimum-gain", type=float, default=0.01)
    parser.add_argument("--noise-limit", type=float, default=6.0)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-timeout-s", type=float, default=180.0)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026080504)
    args = parser.parse_args()
    projected_step_rms = (
        float(args.perturbation)
        if args.projected_step_rms is None
        else float(args.projected_step_rms)
    )
    gpu_ids = parse_ints(args.gpus)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.iterations < 1 or args.perturbation <= 0.0 or args.learning_rate <= 0.0:
        raise ValueError("iterations, perturbation, and learning rate must be positive")
    if not 0.0 <= args.beta1 < 1.0 or not 0.0 <= args.beta2 < 1.0:
        raise ValueError("Adam betas must be in [0, 1)")
    if (
        args.accept_drop < 0.0
        or args.probe_accept_minimum_gain < 0.0
        or args.branch_accept_minimum_gain < 0.0
        or args.noise_limit <= 0.0
        or projected_step_rms <= 0.0
    ):
        raise ValueError("accept-drop and noise-limit are invalid")
    if not gpu_ids:
        raise ValueError("at least one score GPU is required")
    if any(path.exists() for path in (args.out_dir / "manifest.json", args.out_dir / "history.jsonl")):
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.out_dir / "trajectory"
    trajectory_dir.mkdir(exist_ok=True)
    current_noise, initial_payload = load_initial_noise(args.initial_case)
    current_noise = np.clip(current_noise, -args.noise_limit, args.noise_limit).astype(np.float32)
    if args.random_directions >= current_noise.size:
        raise ValueError("too many random directions for latent dimension")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
    normalizer_key = f"{int(args.nfp)}:{int(current_noise.shape[0])}"
    if normalizer_key not in normalizer.current_l1_a:
        raise ValueError(f"condition {normalizer_key} is absent from normalizer")

    manifest = {
        "format": "qh_g3_informed_subspace_adam_v1",
        "algorithm": (
            "exact_secant_projection_over_g3_plus_random_orthogonal_subspace_then_"
            + args.proposal_mode
        ),
        "score_path": "native_cpp_cuda_abi9_exact_forward",
        "gradient_reference": "native_fixed_geometry_g3_vjp_through_flow",
        "nfp": int(args.nfp),
        "n_coils": int(current_noise.shape[0]),
        "iterations": int(args.iterations),
        "rk4_steps": int(args.rk4_steps),
        "random_directions": int(args.random_directions),
        "total_secant_directions": int(args.random_directions + 1),
        "perturbation": float(args.perturbation),
        "learning_rate": float(args.learning_rate),
        "proposal_mode": str(args.proposal_mode),
        "projected_step_rms": projected_step_rms,
        "betas": [float(args.beta1), float(args.beta2)],
        "adam_epsilon": float(args.adam_epsilon),
        "backtrack_fractions": list(args.backtrack_fractions),
        "accept_drop": float(args.accept_drop),
        "probe_accept_minimum_gain": float(args.probe_accept_minimum_gain),
        "branch_accept_minimum_gain": float(args.branch_accept_minimum_gain),
        "acceptance": (
            "centered or feasible one-sided same-branch secants plus monotone exact ABI-9 "
            "gate; improving probe endpoints compete with the smooth Adam candidate"
        ),
        "secant_flow_decode": "independent_batch1",
        "secant_center_score": "native_score_api_in_same_worker_pool_batch",
        "smooth_candidate_evaluation": (
            "independent_batch1_score_batch_then_single_selected_g3"
        ),
        "noise_limit": float(args.noise_limit),
        "seed": int(args.seed),
        "gpu_ids": list(gpu_ids),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "gradient_lib": str(args.gradient_lib.resolve()),
        "gradient_lib_sha256": file_sha256(args.gradient_lib),
        "initial_case": str(args.initial_case.resolve()),
        "initial_case_sha256": file_sha256(args.initial_case),
        "initial_noise_float32_sha256": array_sha256(current_noise),
        "initial_metadata": initial_payload.get("flow_prior_start", {}),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
    write_json(args.out_dir / "manifest.json", manifest)

    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    current = evaluate(
        model,
        normalizer,
        current_noise,
        nfp=args.nfp,
        rk4_steps=args.rk4_steps,
        gradient_lib=args.gradient_lib,
        device=device,
        gradient_group=3,
    )
    if not current.valid:
        raise RuntimeError(f"initial G3 state is invalid: {current.score_result.get('status')}")
    initial_score = result_score(current.score_result)
    first_moment = np.zeros_like(current.noise, dtype=np.float64)
    second_moment = np.zeros_like(current.noise, dtype=np.float64)
    adam_step = 0
    best = current
    best_iteration = 0
    history: list[dict[str, Any]] = []
    history_path = args.out_dir / "history.jsonl"
    save_trajectory(
        trajectory_dir,
        iteration=0,
        evaluation=current,
        first_moment=first_moment,
        second_moment=second_moment,
        adam_step=adam_step,
        transition=None,
    )
    write_json(args.out_dir / "best.json", make_best_case(best, nfp=args.nfp, iteration=0, manifest=manifest))

    with NativeScorePool(args.gradient_lib, list(gpu_ids)) as pool:
        for iteration in range(1, args.iterations + 1):
            iteration_started = time.perf_counter()
            center_score_before = result_score(current.score_result)
            g3_gradient = np.asarray(current.latent_gradient, dtype=np.float64)
            directions = informed_orthogonal_directions(
                rng, g3_gradient, args.random_directions
            )
            pair_states = np.concatenate(
                [current.noise[None] + args.perturbation * directions,
                 current.noise[None] - args.perturbation * directions],
                axis=0,
            )
            pair_states = np.clip(pair_states, -args.noise_limit, args.noise_limit).astype(np.float32)
            pair_tokens, pair_decode_wall_s = decode_noise_rk4_independent(
                model,
                normalizer,
                pair_states,
                nfp=args.nfp,
                steps=args.rk4_steps,
                device=device,
            )
            secant_tokens = np.concatenate((current.tokens[None], pair_tokens), axis=0)
            (
                secant_results,
                secant_elapsed_s,
                secant_errors,
                pair_score_wall_s,
            ) = score_tokens(
                pool,
                secant_tokens,
                nfp=args.nfp,
                target="QH",
                timeout_s=args.batch_timeout_s,
                metadata={"phase": "informed_subspace", "iteration": iteration},
            )
            if any(error is not None for error in secant_errors):
                raise RuntimeError(
                    f"score worker failed at iteration {iteration}: {secant_errors}"
                )
            secant_center_result = secant_results[0]
            if not result_valid(secant_center_result):
                raise RuntimeError(
                    "batch-1 center score disagrees with the valid optimizer center: "
                    f"{None if secant_center_result is None else secant_center_result.get('status')}"
                )
            secant_center_score_delta = (
                result_score(secant_center_result) - center_score_before
            )
            pair_results = secant_results[1:]
            pair_elapsed_s = secant_elapsed_s[1:]
            count = len(directions)
            projected, direction_rows, predicted_gain = exact_subspace_gradient(
                secant_center_result,
                pair_results[:count],
                pair_results[count:],
                directions,
                args.perturbation,
            )
            accepted: Evaluation | None = None
            accepted_fraction = 0.0
            accepted_mode = "rejected"
            candidate_rows: list[dict[str, Any]] = []
            full_update = np.zeros_like(current.noise, dtype=np.float64)
            gradient = np.zeros_like(current.noise, dtype=np.float64)
            quadratic_rows: list[dict[str, Any]] = []
            quadratic_predicted_gain = 0.0
            quadratic_axis_rows: list[dict[str, Any]] = []
            quadratic_axis_decode_wall_s = 0.0
            quadratic_axis_score_wall_s = 0.0
            smooth_candidate_decode_wall_s = 0.0
            smooth_candidate_score_wall_s = 0.0
            smooth_candidate_score_elapsed_s: list[float] = []
            projected_rms = 0.0 if projected is None else rms(projected)
            previous_noise = current.noise.copy()
            tentative_first: np.ndarray | None = None
            tentative_second: np.ndarray | None = None
            next_adam_step = adam_step
            probe_endpoint_index = best_improving_endpoint(
                current.score_result,
                pair_results,
                same_branch_minimum_gain=args.probe_accept_minimum_gain,
                branch_minimum_gain=args.branch_accept_minimum_gain,
            )
            probe_endpoint_same_branch = bool(
                probe_endpoint_index is not None
                and branch_fingerprint(pair_results[probe_endpoint_index])
                == branch_fingerprint(current.score_result)
            )
            probe_endpoint_minimum_gain = (
                args.probe_accept_minimum_gain
                if probe_endpoint_same_branch
                else args.branch_accept_minimum_gain
            )
            if (
                projected is not None
                and projected_rms > 0.0
                and math.isfinite(projected_rms)
            ):
                if args.proposal_mode == "adam":
                    gradient = projected * (rms(g3_gradient) / projected_rms)
                    next_adam_step = adam_step + 1
                    tentative_first = args.beta1 * first_moment + (1.0 - args.beta1) * gradient
                    tentative_second = (
                        args.beta2 * second_moment
                        + (1.0 - args.beta2) * gradient * gradient
                    )
                    first_hat = tentative_first / (1.0 - args.beta1**next_adam_step)
                    second_hat = tentative_second / (1.0 - args.beta2**next_adam_step)
                    full_update = args.learning_rate * first_hat / (
                        np.sqrt(second_hat) + args.adam_epsilon
                    )
                elif args.proposal_mode == "projected":
                    gradient = projected
                    trust_update = projected_trust_update(
                        projected,
                        step_rms=projected_step_rms,
                    )
                    assert trust_update is not None
                    full_update = trust_update
                elif args.proposal_mode == "quadratic":
                    gradient = projected
                    quadratic_update, quadratic_rows, quadratic_predicted_gain = (
                        diagonal_quadratic_trust_update(
                            secant_center_result,
                            direction_rows,
                            directions,
                            perturbation=args.perturbation,
                            max_step_rms=projected_step_rms,
                        )
                    )
                    if quadratic_update is None:
                        full_update = np.zeros_like(current.noise, dtype=np.float64)
                    else:
                        full_update = quadratic_update
                else:
                    gradient = projected
                    _, quadratic_rows, quadratic_predicted_gain = (
                        diagonal_quadratic_trust_update(
                            secant_center_result,
                            direction_rows,
                            directions,
                            perturbation=args.perturbation,
                            max_step_rms=projected_step_rms,
                        )
                    )
                candidate_fractions = (
                    (1.0, *args.backtrack_fractions) if np.any(full_update) else ()
                )
                candidate_states = [
                    np.clip(
                        current.noise.astype(np.float64) + fraction * full_update,
                        -args.noise_limit,
                        args.noise_limit,
                    ).astype(np.float32)
                    for fraction in candidate_fractions
                ]
                if candidate_states:
                    candidate_tokens, smooth_candidate_decode_wall_s = (
                        decode_noise_rk4_independent(
                            model,
                            normalizer,
                            np.stack(candidate_states),
                            nfp=args.nfp,
                            steps=args.rk4_steps,
                            device=device,
                        )
                    )
                    (
                        candidate_score_results,
                        smooth_candidate_score_elapsed_s,
                        candidate_score_errors,
                        smooth_candidate_score_wall_s,
                    ) = score_tokens(
                        pool,
                        candidate_tokens,
                        nfp=args.nfp,
                        target="QH",
                        timeout_s=args.batch_timeout_s,
                        metadata={"phase": "smooth_candidates", "iteration": iteration},
                    )
                    if any(error is not None for error in candidate_score_errors):
                        raise RuntimeError(
                            f"smooth score worker failed at iteration {iteration}: "
                            f"{candidate_score_errors}"
                        )
                else:
                    candidate_score_results = []
                for fraction, proposal, score_only_result in zip(
                    candidate_fractions,
                    candidate_states,
                    candidate_score_results,
                    strict=True,
                ):
                    score_only_accepted = bool(
                        result_valid(score_only_result)
                        and accept_exact_score_candidate(
                            current.score_result,
                            score_only_result,
                            accept_drop=args.accept_drop,
                        )
                    )
                    candidate_row = {
                        "mode": args.proposal_mode,
                        "fraction": float(fraction),
                        "status": str(score_only_result.get("status")),
                        "score": result_score(score_only_result),
                        "score_only_accepted": score_only_accepted,
                        "g3_evaluated": False,
                        "valid_g3": None,
                        "exact_score_accepted": False,
                    }
                    candidate_rows.append(candidate_row)
                    if not score_only_accepted:
                        continue
                    candidate = evaluate(
                        model,
                        normalizer,
                        proposal,
                        nfp=args.nfp,
                        rk4_steps=args.rk4_steps,
                        gradient_lib=args.gradient_lib,
                        device=device,
                        gradient_group=3,
                    )
                    exact_accepted = bool(
                        candidate.valid
                        and accept_exact_score_candidate(
                            current.score_result,
                            candidate.score_result,
                            accept_drop=args.accept_drop,
                        )
                    )
                    candidate_row.update(
                        status=str(candidate.score_result.get("status")),
                        score=result_score(candidate.score_result),
                        score_only_score=result_score(score_only_result),
                        score_repeat_delta=(
                            result_score(candidate.score_result)
                            - result_score(score_only_result)
                        ),
                        g3_evaluated=True,
                        valid_g3=bool(candidate.valid),
                        exact_score_accepted=exact_accepted,
                        wall_s=float(candidate.wall_s),
                    )
                    if exact_accepted:
                        accepted = candidate
                        accepted_fraction = float(fraction)
                        accepted_mode = args.proposal_mode
                        break

            incumbent = current if accepted is None else accepted
            if args.proposal_mode == "quadratic_axis":
                axis_entries: list[tuple[int, float, np.ndarray]] = []
                endpoint_tolerance = args.perturbation * 1.0e-6
                for row, direction in zip(quadratic_rows, directions, strict=True):
                    coefficient = float(row["unscaled_coefficient"])
                    if (
                        coefficient == 0.0
                        or abs(abs(coefficient) - args.perturbation) <= endpoint_tolerance
                    ):
                        continue
                    state = np.clip(
                        current.noise.astype(np.float64) + coefficient * direction,
                        -args.noise_limit,
                        args.noise_limit,
                    ).astype(np.float32)
                    axis_entries.append((int(row["index"]), coefficient, state))
                if axis_entries:
                    axis_states = np.stack([entry[2] for entry in axis_entries])
                    axis_tokens, quadratic_axis_decode_wall_s = decode_noise_rk4_independent(
                        model,
                        normalizer,
                        axis_states,
                        nfp=args.nfp,
                        steps=args.rk4_steps,
                        device=device,
                    )
                    (
                        axis_results,
                        _,
                        axis_errors,
                        quadratic_axis_score_wall_s,
                    ) = score_tokens(
                        pool,
                        axis_tokens,
                        nfp=args.nfp,
                        target="QH",
                        timeout_s=args.batch_timeout_s,
                        metadata={"phase": "quadratic_axis", "iteration": iteration},
                    )
                    if any(error is not None for error in axis_errors):
                        raise RuntimeError(
                            f"quadratic-axis score worker failed at iteration {iteration}: "
                            f"{axis_errors}"
                        )
                    for entry, result in zip(axis_entries, axis_results, strict=True):
                        model_index, coefficient, _ = entry
                        quadratic_axis_rows.append(
                            {
                                "direction_index": model_index,
                                "coefficient": coefficient,
                                "status": str(result.get("status")),
                                "score": result_score(result),
                                "same_branch": branch_fingerprint(result)
                                == branch_fingerprint(current.score_result),
                            }
                        )
                    axis_index = best_improving_endpoint(
                        current.score_result,
                        axis_results,
                        same_branch_minimum_gain=args.probe_accept_minimum_gain,
                        branch_minimum_gain=args.branch_accept_minimum_gain,
                    )
                    if axis_index is not None:
                        model_index, coefficient, axis_state = axis_entries[axis_index]
                        axis_candidate = evaluate(
                            model,
                            normalizer,
                            axis_state,
                            nfp=args.nfp,
                            rk4_steps=args.rk4_steps,
                            gradient_lib=args.gradient_lib,
                            device=device,
                            gradient_group=3,
                        )
                        axis_same_branch = (
                            branch_fingerprint(axis_candidate.score_result)
                            == branch_fingerprint(current.score_result)
                        )
                        axis_minimum_gain = (
                            args.probe_accept_minimum_gain
                            if axis_same_branch
                            else args.branch_accept_minimum_gain
                        )
                        axis_wins = bool(
                            axis_candidate.valid
                            and score_improves_by(
                                axis_candidate.score_result,
                                incumbent.score_result,
                                minimum_gain=axis_minimum_gain,
                            )
                        )
                        candidate_rows.append(
                            {
                                "mode": "quadratic_axis",
                                "direction_index": model_index,
                                "coefficient": coefficient,
                                "batch_score": result_score(axis_results[axis_index]),
                                "same_branch": axis_same_branch,
                                "status": str(axis_candidate.score_result.get("status")),
                                "score": result_score(axis_candidate.score_result),
                                "valid_g3": bool(axis_candidate.valid),
                                "exact_score_accepted": axis_wins,
                                "wall_s": float(axis_candidate.wall_s),
                            }
                        )
                        if axis_wins:
                            accepted = axis_candidate
                            accepted_fraction = abs(coefficient) / args.perturbation
                            accepted_mode = "quadratic_axis"
                            incumbent = accepted

            probe_pair_competitive = bool(
                probe_endpoint_index is not None
                and score_improves_by(
                    pair_results[probe_endpoint_index],
                    incumbent.score_result,
                    minimum_gain=probe_endpoint_minimum_gain,
                )
            )
            if probe_pair_competitive:
                probe_candidate = evaluate(
                    model,
                    normalizer,
                    pair_states[probe_endpoint_index],
                    nfp=args.nfp,
                    rk4_steps=args.rk4_steps,
                    gradient_lib=args.gradient_lib,
                    device=device,
                    gradient_group=3,
                )
                wins_competition = bool(
                    probe_candidate.valid
                    and score_improves_by(
                        probe_candidate.score_result,
                        incumbent.score_result,
                        minimum_gain=probe_endpoint_minimum_gain,
                    )
                )
                same_branch = branch_fingerprint(probe_candidate.score_result) == branch_fingerprint(
                    current.score_result
                )
                endpoint_mode = "probe_endpoint" if same_branch else "branch_endpoint"
                direction_index = probe_endpoint_index % count
                sign = 1 if probe_endpoint_index < count else -1
                candidate_rows.append(
                    {
                        "mode": endpoint_mode,
                        "endpoint_index": int(probe_endpoint_index),
                        "direction_index": int(direction_index),
                        "direction_kind": "g3" if direction_index == 0 else "random",
                        "sign": sign,
                        "pair_score": result_score(pair_results[probe_endpoint_index]),
                        "same_branch": same_branch,
                        "status": str(probe_candidate.score_result.get("status")),
                        "score": result_score(probe_candidate.score_result),
                        "valid_g3": bool(probe_candidate.valid),
                        "exact_score_accepted": wins_competition,
                        "wins_competition": wins_competition,
                        "wall_s": float(probe_candidate.wall_s),
                    }
                )
                if wins_competition:
                    accepted = probe_candidate
                    accepted_fraction = 0.0
                    accepted_mode = endpoint_mode

            if accepted is not None:
                current = accepted
                if accepted_mode == "adam":
                    assert tentative_first is not None and tentative_second is not None
                    first_moment = tentative_first
                    second_moment = tentative_second
                    adam_step = next_adam_step
                else:
                    first_moment = np.zeros_like(first_moment)
                    second_moment = np.zeros_like(second_moment)
                    adam_step = 0

            if result_score(current.score_result) > result_score(best.score_result):
                best = current
                best_iteration = iteration
                write_json(
                    args.out_dir / "best.json",
                    make_best_case(best, nfp=args.nfp, iteration=best_iteration, manifest=manifest),
                )
            applied_update_rms = rms(current.noise.astype(np.float64) - previous_noise)
            accepted_score_gain = result_score(current.score_result) - center_score_before
            transition = {
                "directions": directions.tolist(),
                "direction_rows": direction_rows,
                "predicted_local_gain": predicted_gain,
                "secant_center_score": result_score(secant_center_result),
                "secant_center_score_delta": secant_center_score_delta,
                "accepted_score_gain": accepted_score_gain,
                "quadratic_model_rows": quadratic_rows,
                "quadratic_predicted_gain": quadratic_predicted_gain,
                "quadratic_axis_rows": quadratic_axis_rows,
                "quadratic_axis_decode_wall_s": quadratic_axis_decode_wall_s,
                "quadratic_axis_score_wall_s": quadratic_axis_score_wall_s,
                "smooth_candidate_decode_wall_s": smooth_candidate_decode_wall_s,
                "smooth_candidate_score_wall_s": smooth_candidate_score_wall_s,
                "smooth_candidate_score_elapsed_s": smooth_candidate_score_elapsed_s,
                "g3_gradient_rms": rms(g3_gradient),
                "projected_gradient_rms": projected_rms,
                "used_gradient_rms": rms(gradient),
                "full_update_rms": rms(full_update),
                "applied_update_rms": applied_update_rms,
                "accepted_fraction": accepted_fraction,
                "accepted_mode": accepted_mode,
                "probe_endpoint_index": probe_endpoint_index,
                "branch_endpoint_index": (
                    probe_endpoint_index
                    if probe_endpoint_index is not None and not probe_endpoint_same_branch
                    else None
                ),
                "pair_decode_wall_s": pair_decode_wall_s,
                "pair_score_wall_s": pair_score_wall_s,
                "pair_score_elapsed_s": pair_elapsed_s,
                "candidate_trials": candidate_rows,
            }
            iteration_wall_s = time.perf_counter() - iteration_started
            g3_slope = float(direction_rows[0]["slope"])
            row = {
                "iteration": iteration,
                "adam_step": adam_step,
                "current_score": result_score(current.score_result),
                "best_score": result_score(best.score_result),
                "best_iteration": best_iteration,
                "status": str(current.score_result.get("status")),
                "components": {
                    name: float(current.score_result["components"][name]) for name in COMPONENT_NAMES
                },
                "qh_error": diagnostic(current.score_result, "qs_global_error"),
                "qa_error": diagnostic(current.score_result, "qs_qa_global_error"),
                "qp_error": diagnostic(current.score_result, "qs_qp_global_error"),
                "iota": diagnostic(current.score_result, "iota_min"),
                "surface_level": diagnostic(current.score_result, "surface_level"),
                "g3_slope": g3_slope,
                "secant_center_score": result_score(secant_center_result),
                "secant_center_score_delta": secant_center_score_delta,
                "accepted_score_gain": accepted_score_gain,
                "valid_directions": int(sum(item["valid"] for item in direction_rows)),
                "accepted_fraction": accepted_fraction,
                "accepted_mode": accepted_mode,
                "update_rms": applied_update_rms,
                "pair_decode_wall_s": pair_decode_wall_s,
                "pair_score_wall_s": pair_score_wall_s,
                "smooth_candidate_decode_wall_s": smooth_candidate_decode_wall_s,
                "smooth_candidate_score_wall_s": smooth_candidate_score_wall_s,
                "iteration_wall_s": iteration_wall_s,
                "total_wall_s": time.perf_counter() - started,
            }
            history.append(row)
            append_jsonl(history_path, row)
            save_trajectory(
                trajectory_dir,
                iteration=iteration,
                evaluation=current,
                first_moment=first_moment,
                second_moment=second_moment,
                adam_step=adam_step,
                transition=transition,
            )
            np.savez_compressed(
                args.out_dir / "state_latest.npz",
                current_noise=current.noise,
                best_noise=best.noise,
                first_moment=first_moment,
                second_moment=second_moment,
                iteration=np.asarray(iteration),
                adam_step=np.asarray(adam_step),
                best_iteration=np.asarray(best_iteration),
            )
            write_json(
                args.out_dir / "progress.json",
                {
                    "completed_iterations": iteration,
                    "current_score": row["current_score"],
                    "best_score": row["best_score"],
                    "best_iteration": best_iteration,
                    "last_row": row,
                },
            )
            if iteration == 1 or iteration % args.plot_every == 0 or iteration == args.iterations:
                plot_progress(history, args.out_dir / "progress.png")
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "score": row["current_score"],
                        "best": row["best_score"],
                        "g3_slope": g3_slope,
                        "valid_directions": row["valid_directions"],
                        "accepted_fraction": accepted_fraction,
                        "accepted_mode": accepted_mode,
                        "wall_s": iteration_wall_s,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

    plot_progress(history, args.out_dir / "progress.png")
    summary = {
        "manifest": manifest,
        "completed_iterations": int(args.iterations),
        "initial_score": float(initial_score),
        "final_score": result_score(current.score_result),
        "best_score": result_score(best.score_result),
        "best_iteration": int(best_iteration),
        "accepted_steps": int(sum(row["accepted_mode"] != "rejected" for row in history)),
        "adam_accepted_steps": int(sum(row["accepted_mode"] == "adam" for row in history)),
        "projected_accepted_steps": int(
            sum(row["accepted_mode"] == "projected" for row in history)
        ),
        "quadratic_accepted_steps": int(
            sum(row["accepted_mode"] == "quadratic" for row in history)
        ),
        "quadratic_axis_accepted_steps": int(
            sum(row["accepted_mode"] == "quadratic_axis" for row in history)
        ),
        "branch_accepted_steps": int(
            sum(row["accepted_mode"] == "branch_endpoint" for row in history)
        ),
        "probe_accepted_steps": int(
            sum(row["accepted_mode"] == "probe_endpoint" for row in history)
        ),
        "rejected_steps": int(sum(row["accepted_mode"] == "rejected" for row in history)),
        "best_components": best.score_result["components"],
        "best_diagnostics": compact_result(best.score_result)["diagnostics"],
        "mean_iteration_wall_s": float(np.mean([row["iteration_wall_s"] for row in history])),
        "total_wall_s": float(time.perf_counter() - started),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
