from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
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
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
for path in (REPO_ROOT, GPU_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flow_matching.model import compile_flow_transformer
from flow_matching.vjp import decode_physical_vjp_with_provider
from scripts.optimize_flow_prior_zo_adam import load_flow_checkpoint, load_initial_noise
from scripts.optimize_native_score_cem import token_case
from scripts.qh_blackbox_gradient_reference import (
    append_jsonl,
    compact_result,
    file_sha256,
    write_json,
)
from stellarator_gpu import score_coils_g2_gradient_native, score_coils_g3_gradient_native


COMPONENT_NAMES = ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")


def parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if any(not 0.0 < item < 1.0 for item in result):
        raise argparse.ArgumentTypeError("backtrack fractions must be in (0, 1)")
    return result


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()


def score_arguments(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(tokens, dtype=np.float64)
    return values[:, :33], values[:, 33:66], values[:, 66:99], values[:, 99]


def token_cotangent(gradient: dict[str, np.ndarray]) -> np.ndarray:
    shape = np.asarray(gradient["x"]).shape
    output = np.empty((shape[0], 100), dtype=np.float32)
    output[:, :33] = gradient["x"]
    output[:, 33:66] = gradient["y"]
    output[:, 66:99] = gradient["z"]
    output[:, 99] = gradient["current"]
    return output


def result_score(result: dict[str, Any]) -> float:
    value = float(result.get("score", 0.0))
    return value if math.isfinite(value) else 0.0


def accept_exact_score_candidate(
    current_result: dict[str, Any],
    candidate_result: dict[str, Any],
    *,
    accept_drop: float,
) -> bool:
    if candidate_result.get("status") != "ok":
        return False
    return result_score(candidate_result) >= result_score(current_result) - accept_drop


def diagnostic(result: dict[str, Any], name: str) -> float:
    return float(result.get("diagnostics", {}).get(name, float("nan")))


@dataclass
class Evaluation:
    noise: np.ndarray
    tokens: np.ndarray
    score_result: dict[str, Any]
    physical_gradient: np.ndarray
    latent_gradient: np.ndarray | None
    gradient_diagnostics: dict[str, Any]
    flow_diagnostics: dict[str, Any]
    wall_s: float

    @property
    def valid(self) -> bool:
        return (
            self.score_result.get("status") == "ok"
            and self.latent_gradient is not None
            and np.all(np.isfinite(self.latent_gradient))
        )


def evaluate(
    model,
    normalizer,
    noise: np.ndarray,
    *,
    nfp: int,
    rk4_steps: int,
    gradient_lib: Path,
    device: torch.device,
    gradient_group: int = 2,
) -> Evaluation:
    started = time.perf_counter()
    providers = {
        2: score_coils_g2_gradient_native,
        3: score_coils_g3_gradient_native,
    }
    if gradient_group not in providers:
        raise ValueError("gradient_group must be 2 or 3")
    gradient_provider = providers[gradient_group]

    def native_provider(physical: np.ndarray):
        tokens = np.asarray(physical[0], dtype=np.float64)
        x, y, z, current = score_arguments(tokens)
        native = gradient_provider(
            gradient_lib,
            x,
            y,
            z,
            current,
            nfp,
            device_id=0,
            target_helicity=(1, nfp),
        )
        message = str(native["gradient_diagnostics"].get("error_message", ""))
        if message:
            raise RuntimeError(f"native G{gradient_group} gradient failed: {message}")
        physical_gradient = token_cotangent(native["gradient"])
        score_result = native["score_result"]
        cotangent = None
        if score_result.get("status") == "ok" and np.all(np.isfinite(physical_gradient)):
            cotangent = physical_gradient[None]
        return cotangent, (score_result, physical_gradient, native["gradient_diagnostics"])

    physical, latent_gradient, payload, flow_diagnostics = (
        decode_physical_vjp_with_provider(
            model,
            normalizer,
            noise,
            native_provider,
            nfp=nfp,
            device=device,
            rk4_steps=rk4_steps,
            checkpoint_steps=8,
            use_checkpoint=False,
        )
    )
    score_result, physical_gradient, gradient_diagnostics = payload
    return Evaluation(
        noise=np.asarray(noise, dtype=np.float32).copy(),
        tokens=np.asarray(physical[0], dtype=np.float64).copy(),
        score_result=score_result,
        physical_gradient=np.asarray(physical_gradient, dtype=np.float32).copy(),
        latent_gradient=(
            None
            if latent_gradient is None
            else np.asarray(latent_gradient[0], dtype=np.float32).copy()
        ),
        gradient_diagnostics=gradient_diagnostics,
        flow_diagnostics=asdict(flow_diagnostics),
        wall_s=float(time.perf_counter() - started),
    )


def optimizer_case(
    evaluation: Evaluation,
    *,
    nfp: int,
    iteration: int,
    best_score: float,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    case = token_case(evaluation.tokens, nfp=nfp, target="QH")
    case["flow_prior_physical_gradient_adam"] = {
        "iteration": int(iteration),
        "score": result_score(evaluation.score_result),
        "best_score": float(best_score),
        "noise": evaluation.noise.tolist(),
        "native_score": compact_result(evaluation.score_result),
        "gradient_method": manifest["gradient_method"],
        "manifest": manifest,
    }
    return case


def save_state(
    path: Path,
    *,
    evaluation: Evaluation,
    best: Evaluation,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    iteration: int,
    adam_step: int,
    best_iteration: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            current_noise=evaluation.noise,
            best_noise=best.noise,
            first_moment=first_moment,
            second_moment=second_moment,
            iteration=np.asarray(iteration, dtype=np.int64),
            adam_step=np.asarray(adam_step, dtype=np.int64),
            best_iteration=np.asarray(best_iteration, dtype=np.int64),
            current_score=np.asarray(result_score(evaluation.score_result)),
            best_score=np.asarray(result_score(best.score_result)),
        )
    temporary.replace(path)


def save_trajectory_state(
    directory: Path,
    *,
    evaluation: Evaluation,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    iteration: int,
    adam_step: int,
    transition: dict[str, Any] | None,
) -> None:
    payload = {
        "iteration": int(iteration),
        "adam_step": int(adam_step),
        "noise": evaluation.noise.tolist(),
        "tokens": evaluation.tokens.tolist(),
        "score": compact_result(evaluation.score_result),
        "latent_gradient": (
            None if evaluation.latent_gradient is None else evaluation.latent_gradient.tolist()
        ),
        "physical_gradient": evaluation.physical_gradient.tolist(),
        "first_moment": np.asarray(first_moment, dtype=np.float64).tolist(),
        "second_moment": np.asarray(second_moment, dtype=np.float64).tolist(),
        "gradient_diagnostics": evaluation.gradient_diagnostics,
        "flow_diagnostics": evaluation.flow_diagnostics,
        "evaluation_wall_s": evaluation.wall_s,
        "transition": transition,
    }
    write_json(directory / f"step_{iteration:04d}.json", payload)


def history_row(
    evaluation: Evaluation,
    *,
    iteration: int,
    adam_step: int,
    best_score: float,
    best_iteration: int,
    gradient_rms: float,
    update_rms: float,
    accepted_fraction: float,
    iteration_wall_s: float,
    total_wall_s: float,
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    result = evaluation.score_result
    return {
        "iteration": int(iteration),
        "adam_step": int(adam_step),
        "current_score": result_score(result),
        "best_score": float(best_score),
        "best_iteration": int(best_iteration),
        "status": str(result.get("status")),
        "components": {
            name: float(result.get("components", {}).get(name, float("nan")))
            for name in COMPONENT_NAMES
        },
        "qh_error": diagnostic(result, "qs_global_error"),
        "qa_error": diagnostic(result, "qs_qa_global_error"),
        "qp_error": diagnostic(result, "qs_qp_global_error"),
        "iota": diagnostic(result, "iota_min"),
        "surface_level": diagnostic(result, "surface_level"),
        "gradient_rms": float(gradient_rms),
        "physical_gradient_rms": rms(evaluation.physical_gradient),
        "update_rms": float(update_rms),
        "noise_rms": rms(evaluation.noise),
        "noise_abs_max": float(np.max(np.abs(evaluation.noise))),
        "accepted_fraction": float(accepted_fraction),
        "trials": trials,
        "flow_decode_wall_s": float(evaluation.flow_diagnostics["decode_wall_s"]),
        "native_provider_wall_s": float(evaluation.flow_diagnostics["provider_wall_s"]),
        "flow_backward_wall_s": float(evaluation.flow_diagnostics["backward_wall_s"]),
        "evaluation_wall_s": float(evaluation.wall_s),
        "iteration_wall_s": float(iteration_wall_s),
        "total_wall_s": float(total_wall_s),
    }


def plot_progress(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    iteration = np.asarray([row["iteration"] for row in rows])
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    axes[0, 0].plot(iteration, [row["current_score"] for row in rows], label="current")
    axes[0, 0].plot(iteration, [row["best_score"] for row in rows], label="best")
    axes[0, 0].set(title="Native score", ylabel="score")
    axes[0, 0].legend()
    for name in COMPONENT_NAMES:
        axes[0, 1].plot(iteration, [row["components"][name] for row in rows], label=name)
    axes[0, 1].set(title="Score components", ylabel="component")
    axes[0, 1].legend(ncol=2, fontsize=8)
    for key, label in (("qh_error", "QH"), ("qa_error", "QA"), ("qp_error", "QP")):
        axes[1, 0].plot(iteration, [row[key] for row in rows], label=label)
    axes[1, 0].set(title="Volume QS residual", ylabel="residual", yscale="log")
    axes[1, 0].legend()
    axes[1, 1].plot(iteration, [row["iota"] for row in rows], label="iota")
    axes[1, 1].plot(iteration, [row["surface_level"] for row in rows], label="surface level")
    axes[1, 1].set(title="Physical state")
    axes[1, 1].legend()
    for key, label in (("gradient_rms", "latent gradient"), ("update_rms", "update")):
        axes[2, 0].plot(iteration, [max(float(row[key]), 1.0e-30) for row in rows], label=label)
    axes[2, 0].set(title="Adam scales", xlabel="iteration", yscale="log")
    axes[2, 0].legend()
    for key, label in (
        ("flow_decode_wall_s", "flow forward"),
        ("native_provider_wall_s", "native gradient"),
        ("flow_backward_wall_s", "flow VJP"),
    ):
        axes[2, 1].plot(iteration, [row[key] for row in rows], label=label)
    axes[2, 1].set(title="Evaluation timing", xlabel="iteration", ylabel="seconds")
    axes[2, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize QH score with native fixed-front G2 and flow VJP."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gradient-lib", type=Path, required=True)
    parser.add_argument("--gradient-group", type=int, choices=(2, 3), default=2)
    parser.add_argument("--initial-case", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--nfp", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rk4-steps", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--backtrack-fractions", type=parse_floats, default=(0.5, 0.25, 0.125))
    parser.add_argument(
        "--accept-drop",
        type=float,
        default=0.0,
        help="Maximum exact ABI-9 score decrease accepted after backtracking.",
    )
    parser.add_argument("--noise-limit", type=float, default=6.0)
    parser.add_argument("--plot-every", type=int, default=10)
    parser.add_argument("--compile-flow-model", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.iterations < 1 or args.learning_rate <= 0.0:
        raise ValueError("iterations and learning rate must be positive")
    if not 0.0 <= args.beta1 < 1.0 or not 0.0 <= args.beta2 < 1.0:
        raise ValueError("Adam betas must be in [0, 1)")
    if args.noise_limit <= 0.0:
        raise ValueError("noise limit must be positive")
    if args.accept_drop < 0.0:
        raise ValueError("accept-drop must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.out_dir / "trajectory"
    trajectory_dir.mkdir(exist_ok=True)
    current_noise, initial_payload = load_initial_noise(args.initial_case)
    if current_noise.shape != (3, 100):
        raise ValueError(f"the fixed score93 start must have shape (3, 100), got {current_noise.shape}")
    device = torch.device(args.device)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
    compile_warmup_s = 0.0
    if args.compile_flow_model:
        model = compile_flow_transformer(model)
        warmup_started = time.perf_counter()
        decode_physical_vjp_with_provider(
            model,
            normalizer,
            current_noise,
            lambda physical: (np.zeros_like(physical, dtype=np.float32), None),
            nfp=args.nfp,
            device=device,
            rk4_steps=8,
            checkpoint_steps=8,
            use_checkpoint=False,
        )
        compile_warmup_s = time.perf_counter() - warmup_started

    manifest = {
        "format": "qh_physical_gradient_adam_v1",
        "gradient_method": f"native_fixed_front_g{args.gradient_group}_vjp_through_flow",
        "score_path": "native_cpp_cuda_abi9_exact_forward",
        "nfp": int(args.nfp),
        "n_coils": int(current_noise.shape[0]),
        "iterations": int(args.iterations),
        "rk4_steps": int(args.rk4_steps),
        "learning_rate": float(args.learning_rate),
        "betas": [float(args.beta1), float(args.beta2)],
        "adam_epsilon": float(args.adam_epsilon),
        "backtrack_fractions": list(args.backtrack_fractions),
        "acceptance": "first differentiable candidate passing exact ABI-9 score gate",
        "accept_drop": float(args.accept_drop),
        "noise_limit": float(args.noise_limit),
        "flow_dtype": "torch.float32",
        "flow_method": "rk4_retained_activations",
        "flow_model_compiled": bool(args.compile_flow_model),
        "compile_warmup_s": float(compile_warmup_s),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "gradient_lib": str(args.gradient_lib.resolve()),
        "gradient_lib_sha256": file_sha256(args.gradient_lib),
        "initial_case": str(args.initial_case.resolve()),
        "initial_case_sha256": file_sha256(args.initial_case),
        "initial_noise_float32_sha256": array_sha256(current_noise),
        "initial_source_case_id": initial_payload.get("flow_prior_start", {}).get("source_case_id"),
        "initial_recorded_score": initial_payload.get("flow_prior_start", {}).get("recorded_score"),
        "historical_baseline": {
            "job_id": 31058,
            "method": "four_direction_antithetic_spsa_adam",
            "rk4_steps": 256,
            "initial_score": 85.8832483,
            "best_score": 93.1655597,
            "best_iteration": 197,
            "final_score": 93.1601644,
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
    write_json(args.out_dir / "manifest.json", manifest)

    started = time.perf_counter()
    current = evaluate(
        model,
        normalizer,
        current_noise,
        nfp=args.nfp,
        rk4_steps=args.rk4_steps,
        gradient_lib=args.gradient_lib,
        device=device,
        gradient_group=args.gradient_group,
    )
    if not current.valid:
        raise RuntimeError(f"initial state is not differentiable: {current.score_result.get('status')}")
    first_moment = np.zeros_like(current.noise, dtype=np.float64)
    second_moment = np.zeros_like(current.noise, dtype=np.float64)
    adam_step = 0
    best = current
    best_iteration = 0
    history: list[dict[str, Any]] = []
    history_path = args.out_dir / "history.jsonl"
    initial_row = history_row(
        current,
        iteration=0,
        adam_step=0,
        best_score=result_score(best.score_result),
        best_iteration=0,
        gradient_rms=rms(current.latent_gradient),
        update_rms=0.0,
        accepted_fraction=0.0,
        iteration_wall_s=current.wall_s,
        total_wall_s=time.perf_counter() - started,
        trials=[],
    )
    history.append(initial_row)
    append_jsonl(history_path, initial_row)
    save_trajectory_state(
        trajectory_dir,
        evaluation=current,
        first_moment=first_moment,
        second_moment=second_moment,
        iteration=0,
        adam_step=0,
        transition=None,
    )
    write_json(
        args.out_dir / "best.json",
        optimizer_case(
            best,
            nfp=args.nfp,
            iteration=0,
            best_score=result_score(best.score_result),
            manifest=manifest,
        ),
    )

    for iteration in range(1, args.iterations + 1):
        iteration_started = time.perf_counter()
        gradient = np.asarray(current.latent_gradient, dtype=np.float64)
        next_adam_step = adam_step + 1
        tentative_first = args.beta1 * first_moment + (1.0 - args.beta1) * gradient
        tentative_second = args.beta2 * second_moment + (1.0 - args.beta2) * gradient * gradient
        first_hat = tentative_first / (1.0 - args.beta1**next_adam_step)
        second_hat = tentative_second / (1.0 - args.beta2**next_adam_step)
        full_update = args.learning_rate * first_hat / (np.sqrt(second_hat) + args.adam_epsilon)
        fractions = (1.0,) + args.backtrack_fractions
        trials: list[dict[str, Any]] = []
        accepted: Evaluation | None = None
        accepted_fraction = 0.0
        accepted_clipped_fraction = 0.0
        for fraction in fractions:
            unbounded_noise = current.noise.astype(np.float64) + fraction * full_update
            candidate_noise = np.clip(
                unbounded_noise,
                -args.noise_limit,
                args.noise_limit,
            ).astype(np.float32)
            clipped_fraction = float(
                np.mean(np.abs(unbounded_noise) > args.noise_limit)
            )
            candidate = evaluate(
                model,
                normalizer,
                candidate_noise,
                nfp=args.nfp,
                rk4_steps=args.rk4_steps,
                gradient_lib=args.gradient_lib,
                device=device,
                gradient_group=args.gradient_group,
            )
            trials.append(
                {
                    "fraction": float(fraction),
                    "status": str(candidate.score_result.get("status")),
                    "score": result_score(candidate.score_result),
                    "valid_gradient": bool(candidate.valid),
                    "clipped_fraction": clipped_fraction,
                    "wall_s": float(candidate.wall_s),
                    "exact_score_accepted": bool(
                        candidate.valid and accept_exact_score_candidate(
                            current.score_result,
                            candidate.score_result,
                            accept_drop=args.accept_drop,
                        )
                    ),
                }
            )
            if candidate.valid and accept_exact_score_candidate(
                current.score_result,
                candidate.score_result,
                accept_drop=args.accept_drop,
            ):
                accepted = candidate
                accepted_fraction = float(fraction)
                accepted_clipped_fraction = clipped_fraction
                break

        previous_score = result_score(current.score_result)
        if accepted is not None:
            current = accepted
            first_moment = tentative_first
            second_moment = tentative_second
            adam_step = next_adam_step
        applied_update_rms = rms(accepted_fraction * full_update)
        if result_score(current.score_result) > result_score(best.score_result):
            best = current
            best_iteration = iteration
            write_json(
                args.out_dir / "best.json",
                optimizer_case(
                    best,
                    nfp=args.nfp,
                    iteration=best_iteration,
                    best_score=result_score(best.score_result),
                    manifest=manifest,
                ),
            )
        transition = {
            "previous_score": previous_score,
            "current_score": result_score(current.score_result),
            "gradient_rms": rms(gradient),
            "full_update_rms": rms(full_update),
            "applied_update_rms": applied_update_rms,
            "accepted_fraction": accepted_fraction,
            "accepted_clipped_fraction": accepted_clipped_fraction,
            "trials": trials,
        }
        iteration_wall_s = time.perf_counter() - iteration_started
        row = history_row(
            current,
            iteration=iteration,
            adam_step=adam_step,
            best_score=result_score(best.score_result),
            best_iteration=best_iteration,
            gradient_rms=rms(current.latent_gradient),
            update_rms=applied_update_rms,
            accepted_fraction=accepted_fraction,
            iteration_wall_s=iteration_wall_s,
            total_wall_s=time.perf_counter() - started,
            trials=trials,
        )
        history.append(row)
        append_jsonl(history_path, row)
        save_trajectory_state(
            trajectory_dir,
            evaluation=current,
            first_moment=first_moment,
            second_moment=second_moment,
            iteration=iteration,
            adam_step=adam_step,
            transition=transition,
        )
        save_state(
            args.out_dir / "state_latest.npz",
            evaluation=current,
            best=best,
            first_moment=first_moment,
            second_moment=second_moment,
            iteration=iteration,
            adam_step=adam_step,
            best_iteration=best_iteration,
        )
        write_json(
            args.out_dir / "progress.json",
            {
                "manifest": manifest,
                "initial_score": history[0]["current_score"],
                "current_score": row["current_score"],
                "best_score": row["best_score"],
                "best_iteration": best_iteration,
                "completed_iterations": iteration,
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
                    "gradient_rms": row["gradient_rms"],
                    "update_rms": row["update_rms"],
                    "accepted_fraction": accepted_fraction,
                    "wall_s": iteration_wall_s,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    plot_progress(history, args.out_dir / "progress.png")
    summary = {
        "manifest": manifest,
        "stop_reason": "completed_iterations",
        "completed_iterations": int(args.iterations),
        "initial_score": float(history[0]["current_score"]),
        "final_score": result_score(current.score_result),
        "best_score": result_score(best.score_result),
        "best_iteration": int(best_iteration),
        "accepted_steps": int(adam_step),
        "backtracked_steps": int(sum(row["accepted_fraction"] not in (0.0, 1.0) for row in history[1:])),
        "rejected_steps": int(sum(row["accepted_fraction"] == 0.0 for row in history[1:])),
        "final_components": current.score_result["components"],
        "best_components": best.score_result["components"],
        "best_diagnostics": compact_result(best.score_result)["diagnostics"],
        "total_wall_s": float(time.perf_counter() - started),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
