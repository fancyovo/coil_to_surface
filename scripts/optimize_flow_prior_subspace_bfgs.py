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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.optimize_flow_prior_zo_adam import (
    TOKEN_DIM,
    decode_noise_rk4,
    diagnostics_value,
    load_flow_checkpoint,
    load_initial_noise,
    result_score,
    result_valid,
    rms,
    score_tokens,
)
from scripts.optimize_native_score_cem import (
    NativeScorePool,
    append_jsonl,
    file_sha256,
    token_case,
    write_json,
)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def rms_orthonormal_basis(
    candidates: list[np.ndarray],
    *,
    shape: tuple[int, ...],
    rank: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dimension = int(np.prod(shape))
    if not 1 <= rank <= dimension:
        raise ValueError("rank must be in [1, latent dimension]")
    vectors: list[np.ndarray] = []
    pending = [np.asarray(value, dtype=np.float64).reshape(-1) for value in candidates]
    while len(vectors) < rank:
        if pending:
            vector = pending.pop(0).copy()
            if vector.size != dimension:
                raise ValueError("basis candidate has the wrong dimension")
        else:
            vector = rng.standard_normal(dimension)
        for basis_vector in vectors:
            vector -= np.dot(vector, basis_vector) * basis_vector / dimension
        vector_rms = float(np.sqrt(np.mean(vector * vector)))
        if not math.isfinite(vector_rms) or vector_rms <= 1.0e-10:
            continue
        vectors.append(vector / vector_rms)
    return np.stack(vectors, axis=0).reshape(rank, *shape).astype(np.float32)


def projected_central_gradient(
    plus_scores: np.ndarray,
    minus_scores: np.ndarray,
    perturbation: float,
) -> np.ndarray:
    if perturbation <= 0.0:
        raise ValueError("perturbation must be positive")
    plus = np.asarray(plus_scores, dtype=np.float64)
    minus = np.asarray(minus_scores, dtype=np.float64)
    if plus.shape != minus.shape or plus.ndim != 1:
        raise ValueError("plus/minus scores must be matching vectors")
    return (plus - minus) / (2.0 * perturbation)


def gradient_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 1.0e-14:
        return 1.0 if np.linalg.norm(left - right) <= 1.0e-10 else 0.0
    return float(np.dot(left, right) / denominator)


def initial_inverse_hessian(curvature_f: np.ndarray) -> tuple[np.ndarray, float]:
    curvature_phi = -np.asarray(curvature_f, dtype=np.float64)
    positive = curvature_phi[np.isfinite(curvature_phi) & (curvature_phi > 1.0e-8)]
    hessian_scale = float(np.median(positive)) if positive.size else 1.0
    hessian_scale = float(np.clip(hessian_scale, 1.0e-4, 1.0e4))
    inverse = np.eye(curvature_phi.size, dtype=np.float64) / hessian_scale
    return inverse, hessian_scale


def damped_inverse_bfgs(
    inverse_hessian: np.ndarray,
    step: np.ndarray,
    objective_gradient_delta: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    inverse = np.asarray(inverse_hessian, dtype=np.float64)
    step = np.asarray(step, dtype=np.float64)
    delta = np.asarray(objective_gradient_delta, dtype=np.float64)
    if inverse.shape != (step.size, step.size) or delta.shape != step.shape:
        raise ValueError("incompatible BFGS shapes")
    try:
        hessian_step = np.linalg.solve(inverse, step)
    except np.linalg.LinAlgError:
        return np.eye(step.size), {"updated": False, "reason": "singular_inverse"}
    step_hessian_step = float(np.dot(step, hessian_step))
    step_delta = float(np.dot(step, delta))
    if not math.isfinite(step_hessian_step) or step_hessian_step <= 1.0e-14:
        return np.eye(step.size), {"updated": False, "reason": "invalid_model"}
    damped = False
    if step_delta < 0.2 * step_hessian_step:
        denominator = step_hessian_step - step_delta
        if denominator <= 1.0e-14:
            return inverse.copy(), {"updated": False, "reason": "invalid_damping"}
        theta = 0.8 * step_hessian_step / denominator
        delta = theta * delta + (1.0 - theta) * hessian_step
        step_delta = float(np.dot(step, delta))
        damped = True
    if not math.isfinite(step_delta) or step_delta <= 1.0e-14:
        return inverse.copy(), {"updated": False, "reason": "nonpositive_curvature"}
    rho = 1.0 / step_delta
    identity = np.eye(step.size)
    transform = identity - rho * np.outer(step, delta)
    updated = transform @ inverse @ transform.T + rho * np.outer(step, step)
    updated = 0.5 * (updated + updated.T)
    eigenvalues = np.linalg.eigvalsh(updated)
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[0]) <= 0.0:
        return inverse.copy(), {"updated": False, "reason": "non_spd_result"}
    return updated, {
        "updated": True,
        "damped": damped,
        "step_gradient_delta": step_delta,
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
    }


def choose_line_search_candidate(
    scores: np.ndarray,
    valid: np.ndarray,
    *,
    current_score: float,
    min_improvement: float,
) -> int | None:
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    eligible = np.flatnonzero(valid & (scores >= current_score + min_improvement))
    if eligible.size == 0:
        return None
    return int(eligible[np.argmax(scores[eligible])])


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
    case["flow_prior_subspace_bfgs"] = {
        "target": target,
        "seed": seed,
        "iteration": iteration,
        "best_score": result_score(result),
        "noise": np.asarray(noise, dtype=np.float32).tolist(),
        "native_score": result,
        "manifest": manifest,
    }
    return case


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
    axes[0, 0].set(ylabel="native score", title="Fixed-subspace BFGS")
    axes[0, 0].legend()
    axes[0, 1].plot(iterations, [row["gradient_norm"] for row in rows])
    axes[0, 1].set(yscale="log", ylabel="projected gradient norm")
    axes[1, 0].plot(iterations, [row["trust_radius"] for row in rows], label="trust radius")
    axes[1, 0].plot(iterations, [row["accepted_step_norm"] for row in rows], label="step norm")
    axes[1, 0].set(yscale="log", xlabel="iteration", ylabel="latent RMS")
    axes[1, 0].legend()
    axes[1, 1].plot(iterations, [row["valid_gradient_fraction"] for row in rows])
    axes[1, 1].set(xlabel="iteration", ylabel="valid gradient endpoints")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locally maximize native score with fixed-subspace finite-difference BFGS."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-case", type=Path, required=True)
    parser.add_argument("--adam-state", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--lib",
        type=Path,
        default=REPO_ROOT / "gpu_backend" / "build_native_score" / "libstellarator_gpu.so",
    )
    parser.add_argument("--target", choices=("QA", "QH"), default="QH")
    parser.add_argument("--nfp", type=int, default=4)
    parser.add_argument("--n-base-coils", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=256)
    parser.add_argument("--perturbation", type=float, default=0.005)
    parser.add_argument("--min-perturbation", type=float, default=0.000625)
    parser.add_argument("--min-gradient-cosine", type=float, default=0.5)
    parser.add_argument("--trust-radius", type=float, default=0.002)
    parser.add_argument("--min-trust-radius", type=float, default=0.00002)
    parser.add_argument("--max-trust-radius", type=float, default=0.01)
    parser.add_argument("--line-alphas", default="1,0.5,0.25,0.125")
    parser.add_argument("--min-improvement", type=float, default=1.0e-5)
    parser.add_argument("--max-rejections", type=int, default=3)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-timeout-s", type=float, default=300.0)
    parser.add_argument("--max-wall-s", type=float, default=1500.0)
    parser.add_argument("--seed", type=int, default=2026080101)
    args = parser.parse_args()

    gpu_ids = parse_ints(args.gpus)
    line_alphas = tuple(float(value) for value in args.line_alphas.split(","))
    if not torch.cuda.is_available():
        raise RuntimeError("subspace BFGS requires CUDA")
    for path in (args.checkpoint, args.initial_case, args.lib):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.adam_state is not None and not args.adam_state.is_file():
        raise FileNotFoundError(args.adam_state)
    if args.iterations < 1 or args.rank < 1 or args.flow_steps < 1:
        raise ValueError("iterations, rank, and flow-steps must be positive")
    if args.rank > args.n_base_coils * TOKEN_DIM:
        raise ValueError("rank exceeds latent dimension")
    if not 0.0 < args.min_perturbation <= args.perturbation:
        raise ValueError("invalid perturbation bounds")
    if not 0.0 < args.min_trust_radius <= args.trust_radius <= args.max_trust_radius:
        raise ValueError("invalid trust-radius bounds")
    if not line_alphas or any(not 0.0 < value <= 1.0 for value in line_alphas):
        raise ValueError("line-search alphas must be in (0, 1]")
    if sorted(line_alphas, reverse=True) != list(line_alphas):
        raise ValueError("line-search alphas must be descending")
    if not gpu_ids:
        raise ValueError("at least one score GPU is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    protected = ("manifest.json", "history.jsonl", "progress.json", "best.json", "summary.json")
    if any((args.out_dir / name).exists() for name in protected):
        raise FileExistsError(f"refusing to overwrite existing run {args.out_dir}")

    initial_noise, initial_payload = load_initial_noise(args.initial_case)
    expected_shape = (args.n_base_coils, TOKEN_DIM)
    if initial_noise.shape != expected_shape:
        raise ValueError(f"initial noise shape {initial_noise.shape} != {expected_shape}")
    rng = np.random.default_rng(args.seed)
    basis_candidates: list[np.ndarray] = []
    adam_state_metadata: dict[str, Any] | None = None
    if args.adam_state is not None:
        state = np.load(args.adam_state, allow_pickle=False)
        state_best = np.asarray(state["best_noise"], dtype=np.float32)
        if state_best.shape != expected_shape or not np.allclose(
            state_best, initial_noise, rtol=0.0, atol=2.0e-6
        ):
            raise ValueError("Adam state best_noise does not match initial case")
        first_moment = np.asarray(state["first_moment"], dtype=np.float64)
        second_moment = np.asarray(state["second_moment"], dtype=np.float64)
        current_noise = np.asarray(state["current_noise"], dtype=np.float64)
        basis_candidates.extend(
            [
                first_moment / (np.sqrt(np.maximum(second_moment, 0.0)) + 1.0e-8),
                first_moment,
                current_noise - initial_noise.astype(np.float64),
            ]
        )
        adam_state_metadata = {
            "path": str(args.adam_state.resolve()),
            "sha256": file_sha256(args.adam_state),
            "iteration": int(state["iteration"]),
        }
    basis_candidates.append(initial_noise.astype(np.float64))
    basis = rms_orthonormal_basis(
        basis_candidates, shape=expected_shape, rank=args.rank, rng=rng
    )
    np.savez_compressed(args.out_dir / "basis.npz", basis=basis)

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
    normalizer_key = f"{args.nfp}:{args.n_base_coils}"
    if normalizer_key not in normalizer.current_l1_a:
        raise ValueError(f"condition {normalizer_key} is absent from normalizer")

    recorded_metadata = initial_payload.get("flow_prior_standard_adam") or initial_payload.get(
        "flow_prior_subspace_bfgs"
    ) or initial_payload.get("flow_prior_zo_adam") or initial_payload.get("flow_prior_cem") or {}
    manifest = {
        "algorithm": "fixed_subspace_damped_bfgs_with_trust_cap",
        "objective": "maximize_native_qh_score",
        "initial_case": {
            "path": str(args.initial_case.resolve()),
            "sha256": file_sha256(args.initial_case),
            "recorded_score": recorded_metadata.get("best_score"),
        },
        "adam_state": adam_state_metadata,
        "target": args.target,
        "nfp": args.nfp,
        "n_base_coils": args.n_base_coils,
        "noise_shape": list(initial_noise.shape),
        "seed": args.seed,
        "iterations": args.iterations,
        "rank": args.rank,
        "perturbation": args.perturbation,
        "min_perturbation": args.min_perturbation,
        "min_gradient_cosine": args.min_gradient_cosine,
        "initial_trust_radius": args.trust_radius,
        "min_trust_radius": args.min_trust_radius,
        "max_trust_radius": args.max_trust_radius,
        "line_alphas": list(line_alphas),
        "min_improvement": args.min_improvement,
        "max_rejections": args.max_rejections,
        "flow_dtype": "torch.float32",
        "flow_method": "rk4",
        "flow_steps": args.flow_steps,
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "native_lib_sha256": file_sha256(args.lib),
        "gpu_ids": list(gpu_ids),
        "max_wall_s": args.max_wall_s,
    }
    write_json(args.out_dir / "manifest.json", manifest)

    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    trust_radius = args.trust_radius
    perturbation = args.perturbation
    stop_reason = "completed_iterations"
    rejected_steps = 0

    def evaluate_states(
        pool: NativeScorePool,
        states: np.ndarray,
        *,
        phase: str,
        iteration: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, dict[str, Any]]:
        tokens, decode_wall_s = decode_noise_rk4(
            model,
            normalizer,
            np.asarray(states, dtype=np.float32),
            nfp=args.nfp,
            steps=args.flow_steps,
            device=device,
        )
        results, elapsed, errors, score_wall_s = score_tokens(
            pool,
            tokens,
            nfp=args.nfp,
            target=args.target,
            timeout_s=args.batch_timeout_s,
            metadata={"phase": phase, "iteration": iteration},
        )
        if any(error is not None for error in errors) or any(result is None for result in results):
            raise RuntimeError(f"native-score failure in {phase}: {errors}")
        concrete_results = [result for result in results if result is not None]
        scores = np.asarray([result_score(result) for result in concrete_results])
        timing = {
            "decode_wall_s": decode_wall_s,
            "score_wall_s": score_wall_s,
            "score_elapsed_s": elapsed,
        }
        return tokens, concrete_results, scores, timing

    def gradient_at(
        pool: NativeScorePool,
        center_noise: np.ndarray,
        center_score: float,
        *,
        requested_h: float,
        phase: str,
        iteration: int,
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
        local_h = requested_h
        while True:
            states = np.concatenate(
                [center_noise[None] + local_h * basis, center_noise[None] - local_h * basis],
                axis=0,
            )
            _, results, scores, timing = evaluate_states(
                pool, states, phase=phase, iteration=iteration
            )
            valid = np.asarray([result_valid(result) for result in results])
            if np.all(valid) or local_h <= args.min_perturbation * (1.0 + 1.0e-12):
                plus = scores[: args.rank]
                minus = scores[args.rank :]
                gradient = projected_central_gradient(plus, minus, local_h)
                curvature = (plus - 2.0 * center_score + minus) / (local_h * local_h)
                return gradient, curvature, local_h, {
                    "scores": scores.tolist(),
                    "statuses": [result.get("status") for result in results],
                    "valid_fraction": float(np.mean(valid)),
                    **timing,
                }
            local_h = max(args.min_perturbation, 0.5 * local_h)

    with NativeScorePool(args.lib, list(gpu_ids)) as pool:
        center_tokens_batch, center_results, center_scores, center_timing = evaluate_states(
            pool, initial_noise[None], phase="initial_center", iteration=0
        )
        current_noise = initial_noise.copy()
        current_tokens = center_tokens_batch[0]
        current_result = center_results[0]
        current_score = float(center_scores[0])
        if not result_valid(current_result):
            raise RuntimeError(f"initial case is not status=ok: {current_result.get('status')}")
        recorded_score = manifest["initial_case"]["recorded_score"]
        if recorded_score is not None and not math.isclose(
            current_score, float(recorded_score), abs_tol=1.0e-5
        ):
            raise RuntimeError(
                f"initial score differs from recorded score: {current_score} != {recorded_score}"
            )
        initial_score = current_score
        best_score = current_score
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

        coarse_gradient, _, coarse_h, coarse_info = gradient_at(
            pool,
            current_noise,
            current_score,
            requested_h=perturbation,
            phase="smoothness_coarse",
            iteration=0,
        )
        fine_gradient, fine_curvature, fine_h, fine_info = gradient_at(
            pool,
            current_noise,
            current_score,
            requested_h=max(args.min_perturbation, 0.5 * coarse_h),
            phase="smoothness_fine",
            iteration=0,
        )
        consistency = gradient_cosine(coarse_gradient, fine_gradient)
        norm_ratio = float(
            np.linalg.norm(fine_gradient) / max(np.linalg.norm(coarse_gradient), 1.0e-14)
        )
        smoothness = {
            "coarse_h": coarse_h,
            "fine_h": fine_h,
            "gradient_cosine": consistency,
            "gradient_norm_ratio_fine_over_coarse": norm_ratio,
            "coarse": coarse_info,
            "fine": fine_info,
        }
        write_json(args.out_dir / "smoothness.json", smoothness)
        perturbation = fine_h
        projected_gradient = fine_gradient
        inverse_hessian, initial_hessian_scale = initial_inverse_hessian(fine_curvature)
        if (
            coarse_info["valid_fraction"] < 1.0
            or fine_info["valid_fraction"] < 1.0
            or consistency < args.min_gradient_cosine
        ):
            stop_reason = "smoothness_rejected"
        else:
            coordinates = np.zeros(args.rank, dtype=np.float64)
            for iteration in range(1, args.iterations + 1):
                if time.perf_counter() - started >= args.max_wall_s:
                    stop_reason = "wall_budget"
                    break
                iteration_started = time.perf_counter()
                direction = inverse_hessian @ projected_gradient
                predicted_slope = float(np.dot(projected_gradient, direction))
                if not math.isfinite(predicted_slope) or predicted_slope <= 0.0:
                    inverse_hessian = np.eye(args.rank) / initial_hessian_scale
                    direction = projected_gradient.copy()
                    predicted_slope = float(np.dot(projected_gradient, direction))
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm <= 1.0e-14:
                    stop_reason = "projected_stationary"
                    break
                if direction_norm > trust_radius:
                    direction *= trust_radius / direction_norm
                line_steps = np.stack([alpha * direction for alpha in line_alphas], axis=0)
                line_states = current_noise[None] + np.tensordot(line_steps, basis, axes=(1, 0))
                line_tokens, line_results, line_scores, line_timing = evaluate_states(
                    pool, line_states, phase="line_search", iteration=iteration
                )
                line_valid = np.asarray([result_valid(result) for result in line_results])
                selected = choose_line_search_candidate(
                    line_scores,
                    line_valid,
                    current_score=current_score,
                    min_improvement=args.min_improvement,
                )
                previous_score = current_score
                bfgs_info: dict[str, Any] = {"updated": False, "reason": "no_step"}
                accepted_step = np.zeros(args.rank, dtype=np.float64)
                gradient_info: dict[str, Any] = {
                    "valid_fraction": 1.0,
                    "statuses": [],
                    "scores": [],
                    "decode_wall_s": 0.0,
                    "score_wall_s": 0.0,
                    "score_elapsed_s": [],
                }
                if selected is None:
                    rejected_steps += 1
                    trust_radius = max(args.min_trust_radius, 0.5 * trust_radius)
                    if rejected_steps >= args.max_rejections or trust_radius <= args.min_trust_radius:
                        stop_reason = "line_search_stalled"
                else:
                    rejected_steps = 0
                    accepted_step = line_steps[selected]
                    coordinates += accepted_step
                    current_noise = line_states[selected].astype(np.float32)
                    current_tokens = line_tokens[selected]
                    current_result = line_results[selected]
                    current_score = float(line_scores[selected])
                    new_gradient, _, used_h, gradient_info = gradient_at(
                        pool,
                        current_noise,
                        current_score,
                        requested_h=perturbation,
                        phase="accepted_gradient",
                        iteration=iteration,
                    )
                    perturbation = used_h
                    if gradient_info["valid_fraction"] < 1.0:
                        stop_reason = "left_smooth_region"
                    inverse_hessian, bfgs_info = damped_inverse_bfgs(
                        inverse_hessian,
                        accepted_step,
                        projected_gradient - new_gradient,
                    )
                    projected_gradient = new_gradient
                    alpha = line_alphas[selected]
                    if alpha == 1.0:
                        trust_radius = min(args.max_trust_radius, 1.25 * trust_radius)
                    elif alpha <= 0.25:
                        trust_radius = max(args.min_trust_radius, 0.5 * trust_radius)
                    if current_score > best_score:
                        best_score = current_score
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
                                iteration=iteration,
                                seed=args.seed,
                                manifest=manifest,
                            ),
                        )

                row = {
                    "iteration": iteration,
                    "current_score": current_score,
                    "best_score": best_score,
                    "best_iteration": best_iteration,
                    "score_improvement": current_score - previous_score,
                    "current_status": current_result.get("status"),
                    "current_qh_error": diagnostics_value(current_result, "qs_global_error"),
                    "current_iota": diagnostics_value(current_result, "iota_min"),
                    "gradient_norm": float(np.linalg.norm(projected_gradient)),
                    "perturbation": perturbation,
                    "trust_radius": trust_radius,
                    "accepted": selected is not None,
                    "accepted_alpha": None if selected is None else line_alphas[selected],
                    "accepted_step_norm": float(np.linalg.norm(accepted_step)),
                    "line_scores": line_scores.tolist(),
                    "line_statuses": [result.get("status") for result in line_results],
                    "valid_gradient_fraction": gradient_info["valid_fraction"],
                    "gradient_statuses": gradient_info["statuses"],
                    "bfgs": bfgs_info,
                    "line_decode_wall_s": line_timing["decode_wall_s"],
                    "line_score_wall_s": line_timing["score_wall_s"],
                    "gradient_decode_wall_s": gradient_info["decode_wall_s"],
                    "gradient_score_wall_s": gradient_info["score_wall_s"],
                    "iteration_wall_s": time.perf_counter() - iteration_started,
                    "total_wall_s": time.perf_counter() - started,
                }
                history.append(row)
                append_jsonl(args.out_dir / "history.jsonl", row)
                write_json(
                    args.out_dir / "progress.json",
                    {
                        "manifest": manifest,
                        "smoothness": smoothness,
                        "initial_score": initial_score,
                        "best_score": best_score,
                        "best_iteration": best_iteration,
                        "iterations": history,
                    },
                )
                plot_progress(history, args.out_dir / "progress.png")
                print(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "score": current_score,
                            "best": best_score,
                            "accepted": selected is not None,
                            "trust_radius": trust_radius,
                            "gradient_norm": float(np.linalg.norm(projected_gradient)),
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                if stop_reason in ("line_search_stalled", "left_smooth_region"):
                    break

    summary = {
        "status": "ok",
        "stop_reason": stop_reason,
        "initial_score": initial_score,
        "final_score": current_score,
        "best_score": best_score,
        "best_iteration": best_iteration,
        "completed_iterations": len(history),
        "accepted_iterations": int(sum(row["accepted"] for row in history)),
        "total_wall_s": time.perf_counter() - started,
        "initial_center_timing": center_timing,
        "smoothness": smoothness,
        "initial_hessian_scale": initial_hessian_scale,
        "final_trust_radius": trust_radius,
        "final_noise_rms": rms(current_noise),
        "best_case": str((args.out_dir / "best.json").resolve()),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
