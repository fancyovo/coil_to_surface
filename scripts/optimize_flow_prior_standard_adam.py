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
    gradient_from_pairs,
    load_flow_checkpoint,
    load_initial_noise,
    orthogonal_directions,
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
    case["flow_prior_standard_adam"] = {
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
    rng: np.random.Generator,
) -> None:
    np.savez_compressed(
        path,
        current_noise=current_noise,
        best_noise=best_noise,
        first_moment=first_moment,
        second_moment=second_moment,
        iteration=np.asarray(iteration, dtype=np.int64),
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
    axes[0, 0].set(ylabel="native score", title="Standard Adam optimization")
    axes[0, 0].legend()
    axes[0, 1].plot(iterations, [row["current_qh_error"] for row in rows], label="QH")
    axes[0, 1].plot(iterations, [row["current_qa_error"] for row in rows], label="QA")
    axes[0, 1].plot(iterations, [row["current_qp_error"] for row in rows], label="QP")
    axes[0, 1].set(ylabel="volume residual", title="Helicity diagnostics")
    axes[0, 1].legend()
    axes[1, 0].plot(iterations, [row["gradient_rms"] for row in rows], label="gradient RMS")
    axes[1, 0].plot(iterations, [row["update_rms"] for row in rows], label="update RMS")
    axes[1, 0].set(yscale="log", ylabel="latent scale", xlabel="iteration")
    axes[1, 0].legend()
    axes[1, 1].plot(
        iterations,
        [row["valid_endpoint_fraction"] for row in rows],
        label="valid endpoints",
    )
    axes[1, 1].plot(iterations, [row["noise_rms"] for row in rows], label="noise RMS")
    axes[1, 1].set(ylabel="fraction / RMS", xlabel="iteration")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Maximize native score from a flow-prior latent with a fixed-step, "
            "standard Adam update and orthogonal antithetic zeroth-order gradients."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-case", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--lib",
        type=Path,
        default=REPO_ROOT / "gpu_backend" / "build_native_score" / "libstellarator_gpu.so",
    )
    parser.add_argument("--target", choices=("QA", "QH"), default="QH")
    parser.add_argument("--nfp", type=int, default=4)
    parser.add_argument("--n-base-coils", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=256)
    parser.add_argument("--perturbation", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-timeout-s", type=float, default=300.0)
    parser.add_argument("--max-wall-s", type=float, default=1500.0)
    parser.add_argument("--plot-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026073004)
    args = parser.parse_args()

    gpu_ids = parse_ints(args.gpus)
    if not torch.cuda.is_available():
        raise RuntimeError("flow-prior optimization requires CUDA")
    if not args.checkpoint.is_file() or not args.lib.is_file():
        raise FileNotFoundError("checkpoint and native score library must exist")
    if args.initial_case is not None and not args.initial_case.is_file():
        raise FileNotFoundError(f"initial case does not exist: {args.initial_case}")
    if args.nfp < 1 or args.n_base_coils < 1:
        raise ValueError("nfp and n-base-coils must be positive")
    if args.iterations < 1 or args.directions < 1 or args.flow_steps < 1:
        raise ValueError("iterations, directions, and flow-steps must be positive")
    if args.directions > args.n_base_coils * TOKEN_DIM:
        raise ValueError("directions exceed latent dimension")
    if args.learning_rate <= 0.0 or args.perturbation <= 0.0:
        raise ValueError("learning rate and perturbation must be positive")
    if not 0.0 < args.beta1 < 1.0 or not 0.0 < args.beta2 < 1.0:
        raise ValueError("Adam betas must be in (0, 1)")
    if args.adam_epsilon <= 0.0 or args.plot_every < 1:
        raise ValueError("Adam epsilon and plot-every must be positive")
    if not gpu_ids:
        raise ValueError("at least one score GPU is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(
        (args.out_dir / name).exists()
        for name in ("manifest.json", "history.jsonl", "summary.json")
    ):
        raise FileExistsError(f"refusing to overwrite existing run {args.out_dir}")

    rng = np.random.default_rng(args.seed)
    if args.initial_case is None:
        current_noise = rng.standard_normal(
            (args.n_base_coils, TOKEN_DIM), dtype=np.float32
        )
        initialization = "independent_standard_normal_flow_prior"
        initial_case_metadata = None
    else:
        current_noise, initial_payload = load_initial_noise(args.initial_case)
        if current_noise.shape != (args.n_base_coils, TOKEN_DIM):
            raise ValueError(
                "initial case noise shape does not match n-base-coils: "
                f"{current_noise.shape} != {(args.n_base_coils, TOKEN_DIM)}"
            )
        initialization = "provided_flow_prior_noise_with_zero_adam_moments"
        generic_start = initial_payload.get("flow_prior_start", {})
        initial_case_metadata = {
            "path": str(args.initial_case.resolve()),
            "source": generic_start.get("source"),
            "source_case_id": generic_start.get("source_case_id"),
            "recorded_input_score": generic_start.get("recorded_score"),
            "recorded_input_status": generic_start.get("recorded_status"),
            "recorded_cem_score": initial_payload.get("flow_prior_cem", {}).get(
                "best_score"
            ),
            "recorded_standard_adam_score": initial_payload.get(
                "flow_prior_standard_adam", {}
            ).get("best_score"),
        }
    first_moment = np.zeros_like(current_noise, dtype=np.float64)
    second_moment = np.zeros_like(current_noise, dtype=np.float64)

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
    normalizer_key = f"{args.nfp}:{args.n_base_coils}"
    if normalizer_key not in normalizer.current_l1_a:
        raise ValueError(f"condition {normalizer_key} is absent from normalizer")

    manifest = {
        "algorithm": "standard_adam_with_orthogonal_antithetic_zo_gradient",
        "objective": "maximize_native_qh_score",
        "initialization": initialization,
        "initial_case": initial_case_metadata,
        "target": args.target,
        "nfp": args.nfp,
        "n_base_coils": args.n_base_coils,
        "noise_shape": list(current_noise.shape),
        "seed": args.seed,
        "iterations": args.iterations,
        "directions": args.directions,
        "perturbation": args.perturbation,
        "learning_rate": args.learning_rate,
        "betas": [args.beta1, args.beta2],
        "adam_epsilon": args.adam_epsilon,
        "learning_rate_schedule": "constant",
        "weight_decay": 0.0,
        "gradient_delta_clip": None,
        "update_clip": None,
        "parameter_clip": None,
        "proposal_search": None,
        "accept_reject": None,
        "flow_dtype": "torch.float32",
        "flow_method": "rk4",
        "flow_steps": args.flow_steps,
        "flow_autocast": False,
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "native_lib_sha256": file_sha256(args.lib),
        "gpu_ids": list(gpu_ids),
        "max_wall_s": args.max_wall_s,
    }
    write_json(args.out_dir / "manifest.json", manifest)

    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    history_path = args.out_dir / "history.jsonl"
    stop_reason = "completed_iterations"

    with NativeScorePool(args.lib, list(gpu_ids)) as pool:
        initial_tokens, initial_decode_wall_s = decode_noise_rk4(
            model,
            normalizer,
            current_noise[None],
            nfp=args.nfp,
            steps=args.flow_steps,
            device=device,
        )
        initial_results, initial_elapsed, initial_errors, initial_score_wall_s = score_tokens(
            pool,
            initial_tokens,
            nfp=args.nfp,
            target=args.target,
            timeout_s=args.batch_timeout_s,
            metadata={"phase": "initial", "iteration": 0},
        )
        if any(error is not None for error in initial_errors) or initial_results[0] is None:
            raise RuntimeError(f"initial native-score failure: {initial_errors}")
        current_tokens = initial_tokens[0]
        current_result = initial_results[0]
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
                iteration=best_iteration,
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
                    current_noise[None] + args.perturbation * directions,
                    current_noise[None] - args.perturbation * directions,
                ],
                axis=0,
            ).astype(np.float32)
            pair_tokens, pair_decode_wall_s = decode_noise_rk4(
                model,
                normalizer,
                pair_states,
                nfp=args.nfp,
                steps=args.flow_steps,
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
            gradient, raw_delta = gradient_from_pairs(
                pair_scores[: args.directions],
                pair_scores[args.directions :],
                directions,
                args.perturbation,
                delta_clip=None,
            )
            gradient_rms = rms(gradient)
            if not math.isfinite(gradient_rms):
                raise RuntimeError(f"non-finite gradient at iteration {iteration}")

            first_moment = (
                args.beta1 * first_moment + (1.0 - args.beta1) * gradient
            )
            second_moment = (
                args.beta2 * second_moment
                + (1.0 - args.beta2) * gradient * gradient
            )
            first_hat = first_moment / (1.0 - args.beta1**iteration)
            second_hat = second_moment / (1.0 - args.beta2**iteration)
            update = (
                args.learning_rate
                * first_hat
                / (np.sqrt(second_hat) + args.adam_epsilon)
            )
            update_rms = rms(update)
            current_noise = (current_noise.astype(np.float64) + update).astype(np.float32)

            current_batch, center_decode_wall_s = decode_noise_rk4(
                model,
                normalizer,
                current_noise[None],
                nfp=args.nfp,
                steps=args.flow_steps,
                device=device,
            )
            center_results, center_elapsed, center_errors, center_score_wall_s = score_tokens(
                pool,
                current_batch,
                nfp=args.nfp,
                target=args.target,
                timeout_s=args.batch_timeout_s,
                metadata={"phase": "updated_center", "iteration": iteration},
            )
            if any(error is not None for error in center_errors) or center_results[0] is None:
                raise RuntimeError(
                    f"updated-center score failure at iteration {iteration}: {center_errors}"
                )
            current_tokens = current_batch[0]
            current_result = center_results[0]
            current_score = result_score(current_result)
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
                        iteration=best_iteration,
                        seed=args.seed,
                        manifest=manifest,
                    ),
                )

            iteration_wall_s = time.perf_counter() - iteration_started
            recent_walls.append(iteration_wall_s)
            row = {
                "iteration": iteration,
                "current_score": current_score,
                "best_score": best_score,
                "best_iteration": best_iteration,
                "current_status": current_result.get("status"),
                "current_qh_error": diagnostics_value(current_result, "qs_global_error"),
                "current_qa_error": diagnostics_value(current_result, "qs_qa_global_error"),
                "current_qp_error": diagnostics_value(current_result, "qs_qp_global_error"),
                "current_iota": diagnostics_value(current_result, "iota_min"),
                "valid_endpoint_fraction": float(
                    np.mean([result_valid(result) for result in pair_results])
                ),
                "pair_scores": pair_scores.tolist(),
                "pair_statuses": [
                    None if result is None else result.get("status")
                    for result in pair_results
                ],
                "raw_direction_deltas": raw_delta.tolist(),
                "gradient_rms": gradient_rms,
                "update_rms": update_rms,
                "learning_rate": args.learning_rate,
                "perturbation": args.perturbation,
                "noise_rms": rms(current_noise),
                "noise_abs_max": float(np.max(np.abs(current_noise))),
                "pair_score_elapsed_s": pair_elapsed,
                "center_score_elapsed_s": center_elapsed,
                "pair_decode_wall_s": pair_decode_wall_s,
                "pair_score_wall_s": pair_score_wall_s,
                "center_decode_wall_s": center_decode_wall_s,
                "center_score_wall_s": center_score_wall_s,
                "iteration_wall_s": iteration_wall_s,
                "total_wall_s": time.perf_counter() - started,
            }
            history.append(row)
            append_jsonl(history_path, row)
            write_json(
                args.out_dir / "progress.json",
                {
                    "manifest": manifest,
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
                rng=rng,
            )
            if iteration == 1 or iteration % args.plot_every == 0:
                plot_progress(history, args.out_dir / "progress.png")
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "score": current_score,
                        "best": best_score,
                        "gradient_rms": gradient_rms,
                        "update_rms": update_rms,
                        "noise_rms": rms(current_noise),
                        "wall_s": iteration_wall_s,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

    total_wall_s = time.perf_counter() - started
    if history:
        plot_progress(history, args.out_dir / "progress.png")
        final_score = history[-1]["current_score"]
        completed_iterations = history[-1]["iteration"]
    else:
        final_score = initial_score
        completed_iterations = 0
    summary = {
        "status": "ok",
        "stop_reason": stop_reason,
        "initial_score": initial_score,
        "final_score": final_score,
        "best_score": best_score,
        "best_iteration": best_iteration,
        "completed_iterations": completed_iterations,
        "total_wall_s": total_wall_s,
        "mean_iteration_wall_s": (
            float(np.mean([row["iteration_wall_s"] for row in history]))
            if history
            else float("nan")
        ),
        "initial_decode_wall_s": initial_decode_wall_s,
        "initial_score_wall_s": initial_score_wall_s,
        "initial_score_elapsed_s": initial_elapsed,
        "final_noise_rms": rms(current_noise),
        "final_noise_abs_max": float(np.max(np.abs(current_noise))),
        "best_case": str((args.out_dir / "best.json").resolve()),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
