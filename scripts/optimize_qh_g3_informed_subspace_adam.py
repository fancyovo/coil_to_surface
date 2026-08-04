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
        valid = bool(
            result_valid(plus)
            and result_valid(minus)
            and plus_branch == center_branch
            and minus_branch == center_branch
        )
        slope = float("nan")
        if valid:
            slope = (result_score(plus) - result_score(minus)) / (2.0 * perturbation)
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
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--backtrack-fractions", type=parse_floats, default=(0.5, 0.25, 0.125))
    parser.add_argument("--accept-drop", type=float, default=0.0)
    parser.add_argument("--branch-accept-minimum-gain", type=float, default=0.01)
    parser.add_argument("--noise-limit", type=float, default=6.0)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-timeout-s", type=float, default=180.0)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026080504)
    args = parser.parse_args()
    gpu_ids = parse_ints(args.gpus)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.iterations < 1 or args.perturbation <= 0.0 or args.learning_rate <= 0.0:
        raise ValueError("iterations, perturbation, and learning rate must be positive")
    if not 0.0 <= args.beta1 < 1.0 or not 0.0 <= args.beta2 < 1.0:
        raise ValueError("Adam betas must be in [0, 1)")
    if args.accept_drop < 0.0 or args.branch_accept_minimum_gain < 0.0 or args.noise_limit <= 0.0:
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
        "algorithm": "exact_secant_projection_over_g3_plus_random_orthogonal_subspace_then_adam",
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
        "betas": [float(args.beta1), float(args.beta2)],
        "adam_epsilon": float(args.adam_epsilon),
        "backtrack_fractions": list(args.backtrack_fractions),
        "accept_drop": float(args.accept_drop),
        "branch_accept_minimum_gain": float(args.branch_accept_minimum_gain),
        "acceptance": (
            "same-branch secants plus monotone exact ABI-9 gate; improving discrete "
            "endpoints compete with the smooth Adam candidate"
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
            pair_tokens, pair_decode_wall_s = decode_noise_rk4(
                model,
                normalizer,
                pair_states,
                nfp=args.nfp,
                steps=args.rk4_steps,
                device=device,
            )
            pair_results, pair_elapsed_s, pair_errors, pair_score_wall_s = score_tokens(
                pool,
                pair_tokens,
                nfp=args.nfp,
                target="QH",
                timeout_s=args.batch_timeout_s,
                metadata={"phase": "informed_subspace", "iteration": iteration},
            )
            if any(error is not None for error in pair_errors):
                raise RuntimeError(f"score worker failed at iteration {iteration}: {pair_errors}")
            count = len(directions)
            projected, direction_rows, predicted_gain = exact_subspace_gradient(
                current.score_result,
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
            projected_rms = 0.0 if projected is None else rms(projected)
            previous_noise = current.noise.copy()
            tentative_first: np.ndarray | None = None
            tentative_second: np.ndarray | None = None
            next_adam_step = adam_step
            branch_endpoint_index = best_improving_branch_endpoint(
                current.score_result,
                pair_results,
                minimum_gain=args.branch_accept_minimum_gain,
            )
            if (
                projected is not None
                and projected_rms > 0.0
                and math.isfinite(projected_rms)
            ):
                gradient = projected * (rms(g3_gradient) / projected_rms)
                next_adam_step = adam_step + 1
                tentative_first = args.beta1 * first_moment + (1.0 - args.beta1) * gradient
                tentative_second = args.beta2 * second_moment + (1.0 - args.beta2) * gradient * gradient
                first_hat = tentative_first / (1.0 - args.beta1**next_adam_step)
                second_hat = tentative_second / (1.0 - args.beta2**next_adam_step)
                full_update = args.learning_rate * first_hat / (
                    np.sqrt(second_hat) + args.adam_epsilon
                )
                for fraction in (1.0, *args.backtrack_fractions):
                    proposal = np.clip(
                        current.noise.astype(np.float64) + fraction * full_update,
                        -args.noise_limit,
                        args.noise_limit,
                    ).astype(np.float32)
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
                    candidate_rows.append(
                        {
                            "mode": "adam",
                            "fraction": float(fraction),
                            "status": str(candidate.score_result.get("status")),
                            "score": result_score(candidate.score_result),
                            "valid_g3": bool(candidate.valid),
                            "exact_score_accepted": exact_accepted,
                            "wall_s": float(candidate.wall_s),
                        }
                    )
                    if exact_accepted:
                        accepted = candidate
                        accepted_fraction = float(fraction)
                        accepted_mode = "adam"
                        break

            incumbent = current if accepted is None else accepted
            branch_pair_competitive = bool(
                branch_endpoint_index is not None
                and score_improves_by(
                    pair_results[branch_endpoint_index],
                    incumbent.score_result,
                    minimum_gain=args.branch_accept_minimum_gain,
                )
            )
            if branch_pair_competitive:
                branch_candidate = evaluate(
                    model,
                    normalizer,
                    pair_states[branch_endpoint_index],
                    nfp=args.nfp,
                    rk4_steps=args.rk4_steps,
                    gradient_lib=args.gradient_lib,
                    device=device,
                    gradient_group=3,
                )
                wins_competition = bool(
                    branch_candidate.valid
                    and score_improves_by(
                        branch_candidate.score_result,
                        incumbent.score_result,
                        minimum_gain=args.branch_accept_minimum_gain,
                    )
                )
                direction_index = branch_endpoint_index % count
                sign = 1 if branch_endpoint_index < count else -1
                candidate_rows.append(
                    {
                        "mode": "branch_endpoint",
                        "endpoint_index": int(branch_endpoint_index),
                        "direction_index": int(direction_index),
                        "direction_kind": "g3" if direction_index == 0 else "random",
                        "sign": sign,
                        "pair_score": result_score(pair_results[branch_endpoint_index]),
                        "status": str(branch_candidate.score_result.get("status")),
                        "score": result_score(branch_candidate.score_result),
                        "valid_g3": bool(branch_candidate.valid),
                        "exact_score_accepted": wins_competition,
                        "wins_competition": wins_competition,
                        "wall_s": float(branch_candidate.wall_s),
                    }
                )
                if wins_competition:
                    accepted = branch_candidate
                    accepted_fraction = 0.0
                    accepted_mode = "branch_endpoint"

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
            transition = {
                "directions": directions.tolist(),
                "direction_rows": direction_rows,
                "predicted_local_gain": predicted_gain,
                "g3_gradient_rms": rms(g3_gradient),
                "projected_gradient_rms": projected_rms,
                "used_gradient_rms": rms(gradient),
                "full_update_rms": rms(full_update),
                "applied_update_rms": applied_update_rms,
                "accepted_fraction": accepted_fraction,
                "accepted_mode": accepted_mode,
                "branch_endpoint_index": branch_endpoint_index,
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
                "valid_directions": int(sum(item["valid"] for item in direction_rows)),
                "accepted_fraction": accepted_fraction,
                "accepted_mode": accepted_mode,
                "update_rms": applied_update_rms,
                "pair_decode_wall_s": pair_decode_wall_s,
                "pair_score_wall_s": pair_score_wall_s,
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
        "branch_accepted_steps": int(
            sum(row["accepted_mode"] == "branch_endpoint" for row in history)
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
