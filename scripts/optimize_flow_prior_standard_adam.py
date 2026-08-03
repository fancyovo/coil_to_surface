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
    cosine_similarity,
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


def parse_backtracking_fractions(value: str) -> tuple[float, ...]:
    fractions = tuple(float(item) for item in value.split(",") if item.strip())
    if any(not 0.0 < fraction < 1.0 for fraction in fractions):
        raise argparse.ArgumentTypeError("backtracking fractions must be in (0, 1)")
    if any(right >= left for left, right in zip(fractions, fractions[1:])):
        raise argparse.ArgumentTypeError(
            "backtracking fractions must be strictly decreasing"
        )
    return fractions


def robust_direction_deltas(
    raw_delta: np.ndarray,
    pair_statuses: list[str | None],
    *,
    outlier_ratio: float,
    mad_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    raw_delta = np.asarray(raw_delta, dtype=np.float64)
    count = raw_delta.size
    if len(pair_statuses) != 2 * count:
        raise ValueError("pair status count does not match directional deltas")
    valid = np.asarray(
        [
            pair_statuses[index] == "ok" and pair_statuses[index + count] == "ok"
            for index in range(count)
        ],
        dtype=bool,
    )
    used = raw_delta.copy()
    used[~valid] = 0.0
    outlier = np.zeros(count, dtype=bool)
    adaptive_limit = None
    valid_count = int(np.count_nonzero(valid))
    if valid_count >= 3:
        magnitudes = np.abs(raw_delta[valid])
        median = float(np.median(magnitudes))
        mad = float(np.median(np.abs(magnitudes - median)))
        adaptive_limit = max(
            outlier_ratio * max(median, np.finfo(np.float64).eps),
            median + mad_factor * 1.4826 * mad,
        )
        outlier = valid & (np.abs(raw_delta) > adaptive_limit)
        used[outlier] = np.sign(used[outlier]) * adaptive_limit
    return used, ~valid, outlier, adaptive_limit


def rolling_robust_limit(
    values: list[float],
    *,
    window: int,
    min_history: int,
    ratio: float,
    mad_factor: float,
) -> float | None:
    finite_positive = np.asarray(
        [value for value in values if math.isfinite(value) and value > 0.0],
        dtype=np.float64,
    )
    if finite_positive.size < min_history:
        return None
    recent = finite_positive[-window:]
    median = float(np.median(recent))
    mad = float(np.median(np.abs(recent - median)))
    return max(
        ratio * max(median, np.finfo(np.float64).eps),
        median + mad_factor * 1.4826 * mad,
    )


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


def write_trajectory_case(
    directory: Path,
    tokens: np.ndarray,
    noise: np.ndarray,
    result: dict[str, Any],
    *,
    nfp: int,
    target: str,
    iteration: int,
    optimizer_state: dict[str, Any],
) -> Path:
    case = token_case(tokens, nfp=nfp, target=target)
    case["flow_prior_standard_adam_trajectory"] = {
        "format": "qh_standard_adam_trajectory_v1",
        "iteration": int(iteration),
        "noise": np.asarray(noise, dtype=np.float32).tolist(),
        "native_score": result,
        "optimizer_state": optimizer_state,
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{iteration:04d}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(case, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def save_state(
    path: Path,
    *,
    current_noise: np.ndarray,
    best_noise: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    adam_step: int,
    iteration: int,
    rng: np.random.Generator,
) -> None:
    np.savez_compressed(
        path,
        current_noise=current_noise,
        best_noise=best_noise,
        first_moment=first_moment,
        second_moment=second_moment,
        adam_step=np.asarray(adam_step, dtype=np.int64),
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
    axes[1, 0].plot(
        iterations,
        [row.get("raw_gradient_rms", row["gradient_rms"]) for row in rows],
        label="raw gradient RMS",
    )
    axes[1, 0].plot(iterations, [row["gradient_rms"] for row in rows], label="used gradient RMS")
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
    parser.add_argument("--robust-direction-filter", action="store_true")
    parser.add_argument("--reject-invalid-center", action="store_true")
    parser.add_argument(
        "--invalid-center-backtracking",
        type=parse_backtracking_fractions,
        default=(),
        help=(
            "Comma-separated decreasing update fractions tried after a full "
            "proposal is invalid; requires --reject-invalid-center."
        ),
    )
    parser.add_argument("--direction-outlier-ratio", type=float, default=8.0)
    parser.add_argument("--direction-outlier-mad-factor", type=float, default=8.0)
    parser.add_argument(
        "--no-temporal-scale-guard",
        action="store_false",
        dest="temporal_scale_guard",
        help="Disable the rolling cross-step gradient/update outlier guard.",
    )
    parser.set_defaults(temporal_scale_guard=True)
    parser.add_argument("--temporal-guard-window", type=int, default=20)
    parser.add_argument("--temporal-guard-min-history", type=int, default=20)
    parser.add_argument("--temporal-gradient-ratio", type=float, default=8.0)
    parser.add_argument("--temporal-update-ratio", type=float, default=8.0)
    parser.add_argument("--temporal-guard-mad-factor", type=float, default=8.0)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-timeout-s", type=float, default=300.0)
    parser.add_argument("--max-wall-s", type=float, default=1500.0)
    parser.add_argument("--plot-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026073004)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run in --out-dir without resetting Adam state.",
    )
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
    if args.direction_outlier_ratio <= 1.0 or args.direction_outlier_mad_factor <= 0.0:
        raise ValueError("direction outlier controls must be conservative and positive")
    if (
        args.temporal_guard_window < 1
        or args.temporal_guard_min_history < 1
        or args.temporal_guard_min_history > args.temporal_guard_window
    ):
        raise ValueError("temporal guard history must satisfy 1 <= min <= window")
    if (
        args.temporal_gradient_ratio <= 1.0
        or args.temporal_update_ratio <= 1.0
        or args.temporal_guard_mad_factor <= 0.0
    ):
        raise ValueError("temporal guard controls must be conservative and positive")
    if args.adam_epsilon <= 0.0 or args.plot_every < 1:
        raise ValueError("Adam epsilon and plot-every must be positive")
    if args.invalid_center_backtracking and not args.reject_invalid_center:
        raise ValueError(
            "invalid-center backtracking requires --reject-invalid-center"
        )
    if not gpu_ids:
        raise ValueError("at least one score GPU is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.out_dir / "trajectory"
    run_paths = {
        name: args.out_dir / name
        for name in (
            "manifest.json",
            "history.jsonl",
            "progress.json",
            "best.json",
            "state_latest.npz",
            "summary.json",
        )
    }
    if args.resume:
        required = [path for name, path in run_paths.items() if name != "summary.json"]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"resume artifacts are missing: {missing}")
        if run_paths["summary.json"].exists():
            raise FileExistsError("refusing to resume a run that already has summary.json")
        if not trajectory_dir.is_dir():
            raise FileNotFoundError(
                f"resume trajectory directory is missing: {trajectory_dir}"
            )
        saved_state = np.load(run_paths["state_latest.npz"], allow_pickle=False)
        current_noise = np.asarray(saved_state["current_noise"], dtype=np.float32)
        best_noise = np.asarray(saved_state["best_noise"], dtype=np.float32)
        first_moment = np.asarray(saved_state["first_moment"], dtype=np.float64)
        second_moment = np.asarray(saved_state["second_moment"], dtype=np.float64)
        start_iteration = int(saved_state["iteration"])
        adam_step = (
            int(saved_state["adam_step"])
            if "adam_step" in saved_state.files
            else start_iteration
        )
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(saved_state["rng_state"].item()))
        initialization = "resumed_saved_standard_adam_state"
        initial_case_metadata = None
    else:
        if any(path.exists() for path in run_paths.values()) or trajectory_dir.exists():
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
        best_noise = current_noise.copy()
        first_moment = np.zeros_like(current_noise, dtype=np.float64)
        second_moment = np.zeros_like(current_noise, dtype=np.float64)
        start_iteration = 0
        adam_step = 0
    expected_shape = (args.n_base_coils, TOKEN_DIM)
    for name, value in (
        ("current_noise", current_noise),
        ("best_noise", best_noise),
        ("first_moment", first_moment),
        ("second_moment", second_moment),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} shape {value.shape} != {expected_shape}")
    if args.iterations < start_iteration:
        raise ValueError(
            f"requested iterations {args.iterations} precede saved step {start_iteration}"
        )

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
    normalizer_key = f"{args.nfp}:{args.n_base_coils}"
    if normalizer_key not in normalizer.current_l1_a:
        raise ValueError(f"condition {normalizer_key} is absent from normalizer")

    requested_manifest = {
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
        "robust_direction_filter": args.robust_direction_filter,
        "invalid_direction_policy": (
            "skip_entire_step" if args.robust_direction_filter else "use_all"
        ),
        "reject_invalid_center": args.reject_invalid_center,
        "invalid_center_backtracking": list(args.invalid_center_backtracking),
        "direction_outlier_ratio": args.direction_outlier_ratio,
        "direction_outlier_mad_factor": args.direction_outlier_mad_factor,
        "temporal_scale_guard": args.temporal_scale_guard,
        "temporal_guard_window": args.temporal_guard_window,
        "temporal_guard_min_history": args.temporal_guard_min_history,
        "temporal_gradient_ratio": args.temporal_gradient_ratio,
        "temporal_update_ratio": args.temporal_update_ratio,
        "temporal_guard_mad_factor": args.temporal_guard_mad_factor,
        "temporal_guard_policy": (
            "rolling_accepted_step_median_mad_v1"
            if args.temporal_scale_guard
            else None
        ),
        "update_clip": None,
        "parameter_clip": None,
        "proposal_search": (
            "validity_backtracking" if args.invalid_center_backtracking else None
        ),
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
        "trajectory_artifact": (
            "one atomic JSON case for step 0 and every completed iteration; "
            "each case stores latent noise, decoded coil coefficients/current, "
            "the complete native score, and compact optimizer state"
        ),
    }
    if args.resume:
        manifest = json.loads(run_paths["manifest.json"].read_text(encoding="utf-8"))
        stable_keys = (
            "algorithm",
            "objective",
            "target",
            "nfp",
            "n_base_coils",
            "noise_shape",
            "seed",
            "directions",
            "perturbation",
            "learning_rate",
            "betas",
            "adam_epsilon",
            "robust_direction_filter",
            "invalid_direction_policy",
            "reject_invalid_center",
            "invalid_center_backtracking",
            "direction_outlier_ratio",
            "direction_outlier_mad_factor",
            "temporal_scale_guard",
            "temporal_guard_window",
            "temporal_guard_min_history",
            "temporal_gradient_ratio",
            "temporal_update_ratio",
            "temporal_guard_mad_factor",
            "temporal_guard_policy",
            "flow_method",
            "flow_steps",
            "checkpoint_sha256",
            "native_lib_sha256",
            "gpu_ids",
        )
        mismatches = {
            key: {"saved": manifest.get(key), "requested": requested_manifest.get(key)}
            for key in stable_keys
            if manifest.get(key) != requested_manifest.get(key)
        }
        if mismatches:
            raise ValueError(f"resume configuration mismatch: {mismatches}")
    else:
        manifest = requested_manifest
        write_json(run_paths["manifest.json"], manifest)

    started = time.perf_counter()
    history_path = run_paths["history.jsonl"]
    if args.resume:
        history = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not history or int(history[-1]["iteration"]) != start_iteration:
            raise ValueError("history tail does not match saved Adam iteration")
        prior_wall_s = float(history[-1]["total_wall_s"])
        progress = json.loads(run_paths["progress.json"].read_text(encoding="utf-8"))
        initial_score = float(progress["initial_score"])
        best_payload = json.loads(run_paths["best.json"].read_text(encoding="utf-8"))
        best_metadata = best_payload["flow_prior_standard_adam"]
        best_score = float(best_metadata["best_score"])
        best_iteration = int(best_metadata["iteration"])
        best_result = best_metadata["native_score"]
        append_jsonl(
            args.out_dir / "resume_events.jsonl",
            {
                "saved_iteration": start_iteration,
                "requested_iterations": args.iterations,
                "prior_total_wall_s": prior_wall_s,
                "time_unix_s": time.time(),
            },
        )
    else:
        history = []
        prior_wall_s = 0.0
    accepted_gradient_scales = [
        float(row["gradient_rms"])
        for row in history
        if row.get("gradient_step_applied", False)
        and row.get("center_update_accepted", False)
        and not row.get("temporal_step_rejected", False)
    ]
    accepted_update_scales = [
        float(row["update_rms"])
        for row in history
        if row.get("gradient_step_applied", False)
        and row.get("center_update_accepted", False)
        and not row.get("temporal_step_rejected", False)
    ]
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
        resumed_center_score = result_score(current_result)
        if args.resume:
            previous_score = float(history[-1]["current_score"])
            if not math.isclose(resumed_center_score, previous_score, abs_tol=1e-5):
                raise RuntimeError(
                    "resumed center score differs from saved history: "
                    f"{resumed_center_score} != {previous_score}"
                )
            best_batch, _ = decode_noise_rk4(
                model,
                normalizer,
                best_noise[None],
                nfp=args.nfp,
                steps=args.flow_steps,
                device=device,
            )
            best_tokens = best_batch[0]
            saved_trajectory = trajectory_dir / f"step_{start_iteration:04d}.json"
            if not saved_trajectory.is_file():
                raise FileNotFoundError(
                    f"resume trajectory tail is missing: {saved_trajectory}"
                )
        else:
            initial_score = resumed_center_score
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
            write_trajectory_case(
                trajectory_dir,
                current_tokens,
                current_noise,
                current_result,
                nfp=args.nfp,
                target=args.target,
                iteration=0,
                optimizer_state={
                    "adam_step": 0,
                    "current_score": initial_score,
                    "best_score": best_score,
                    "best_iteration": best_iteration,
                    "gradient_step_applied": False,
                    "center_update_accepted": True,
                },
            )

        recent_walls: list[float] = []
        for iteration in range(start_iteration + 1, args.iterations + 1):
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
            raw_gradient, raw_delta = gradient_from_pairs(
                pair_scores[: args.directions],
                pair_scores[args.directions :],
                directions,
                args.perturbation,
                delta_clip=None,
            )
            pair_statuses = [
                None if result is None else result.get("status")
                for result in pair_results
            ]
            if args.robust_direction_filter:
                used_delta, invalid_direction, outlier_direction, adaptive_delta_limit = (
                    robust_direction_deltas(
                        raw_delta,
                        pair_statuses,
                        outlier_ratio=args.direction_outlier_ratio,
                        mad_factor=args.direction_outlier_mad_factor,
                    )
                )
            else:
                used_delta = raw_delta
                invalid_direction = np.zeros(args.directions, dtype=bool)
                outlier_direction = np.zeros(args.directions, dtype=bool)
                adaptive_delta_limit = None
            previous_first_moment = first_moment.copy()
            previous_second_moment = second_moment.copy()
            previous_adam_step = adam_step
            previous_noise = current_noise.copy()
            previous_tokens = current_tokens.copy()
            previous_result = current_result
            gradient_step_applied = not (
                args.robust_direction_filter and np.any(invalid_direction)
            )
            temporal_gradient_limit = None
            temporal_update_limit = None
            temporal_gradient_outlier = False
            temporal_update_outlier = False
            temporal_step_rejected = False
            if gradient_step_applied:
                gradient, _ = gradient_from_pairs(
                    0.5 * used_delta,
                    -0.5 * used_delta,
                    directions,
                    args.perturbation,
                    delta_clip=None,
                )
                gradient_rms = rms(gradient)
                if not math.isfinite(gradient_rms):
                    raise RuntimeError(f"non-finite gradient at iteration {iteration}")
                candidate_adam_step = adam_step + 1
                candidate_first_moment = (
                    args.beta1 * first_moment + (1.0 - args.beta1) * gradient
                )
                candidate_second_moment = (
                    args.beta2 * second_moment
                    + (1.0 - args.beta2) * gradient * gradient
                )
                first_hat = candidate_first_moment / (
                    1.0 - args.beta1**candidate_adam_step
                )
                second_hat = candidate_second_moment / (
                    1.0 - args.beta2**candidate_adam_step
                )
                candidate_update = (
                    args.learning_rate
                    * first_hat
                    / (np.sqrt(second_hat) + args.adam_epsilon)
                )
                proposed_update_rms = rms(candidate_update)
                if args.temporal_scale_guard:
                    temporal_gradient_limit = rolling_robust_limit(
                        accepted_gradient_scales,
                        window=args.temporal_guard_window,
                        min_history=args.temporal_guard_min_history,
                        ratio=args.temporal_gradient_ratio,
                        mad_factor=args.temporal_guard_mad_factor,
                    )
                    temporal_update_limit = rolling_robust_limit(
                        accepted_update_scales,
                        window=args.temporal_guard_window,
                        min_history=args.temporal_guard_min_history,
                        ratio=args.temporal_update_ratio,
                        mad_factor=args.temporal_guard_mad_factor,
                    )
                    temporal_gradient_outlier = (
                        temporal_gradient_limit is not None
                        and gradient_rms > temporal_gradient_limit
                    )
                    temporal_update_outlier = (
                        temporal_update_limit is not None
                        and proposed_update_rms > temporal_update_limit
                    )
                    temporal_step_rejected = (
                        temporal_gradient_outlier or temporal_update_outlier
                    )
                if temporal_step_rejected:
                    gradient_step_applied = False
                    update = np.zeros_like(candidate_update)
                else:
                    adam_step = candidate_adam_step
                    first_moment = candidate_first_moment
                    second_moment = candidate_second_moment
                    update = candidate_update
            else:
                gradient = np.zeros_like(raw_gradient)
                gradient_rms = 0.0
                update = np.zeros_like(raw_gradient)
                proposed_update_rms = 0.0
            full_update = update.copy()
            center_decode_wall_s = 0.0
            center_score_wall_s = 0.0
            center_elapsed = []
            center_backtracking = []
            if temporal_step_rejected:
                current_noise = previous_noise
                current_tokens = previous_tokens
                current_result = previous_result
                proposed_tokens = previous_tokens
                proposed_result = previous_result
                proposed_score = result_score(previous_result)
                center_update_accepted = False
                center_rejection_reason = (
                    "temporal_gradient_and_update_outlier"
                    if temporal_gradient_outlier and temporal_update_outlier
                    else (
                        "temporal_gradient_outlier"
                        if temporal_gradient_outlier
                        else "temporal_update_outlier"
                    )
                )
                center_acceptance_fraction = 0.0
            else:
                current_noise = (
                    current_noise.astype(np.float64) + full_update
                ).astype(np.float32)
                current_batch, center_decode_wall_s = decode_noise_rk4(
                    model,
                    normalizer,
                    current_noise[None],
                    nfp=args.nfp,
                    steps=args.flow_steps,
                    device=device,
                )
                center_results, center_elapsed, center_errors, center_score_wall_s = (
                    score_tokens(
                        pool,
                        current_batch,
                        nfp=args.nfp,
                        target=args.target,
                        timeout_s=args.batch_timeout_s,
                        metadata={"phase": "updated_center", "iteration": iteration},
                    )
                )
                if (
                    any(error is not None for error in center_errors)
                    or center_results[0] is None
                ):
                    raise RuntimeError(
                        "updated-center score failure at iteration "
                        f"{iteration}: {center_errors}"
                    )
                proposed_tokens = current_batch[0]
                proposed_result = center_results[0]
                proposed_score = result_score(proposed_result)
                center_update_accepted = True
                center_rejection_reason = None
                center_acceptance_fraction = 1.0
            if (
                gradient_step_applied
                and args.reject_invalid_center
                and not result_valid(proposed_result)
            ):
                center_update_accepted = False
                center_acceptance_fraction = 0.0
                for fraction in args.invalid_center_backtracking:
                    trial_noise = (
                        previous_noise.astype(np.float64) + fraction * full_update
                    ).astype(np.float32)
                    trial_batch, trial_decode_wall_s = decode_noise_rk4(
                        model,
                        normalizer,
                        trial_noise[None],
                        nfp=args.nfp,
                        steps=args.flow_steps,
                        device=device,
                    )
                    trial_results, trial_elapsed, trial_errors, trial_score_wall_s = (
                        score_tokens(
                            pool,
                            trial_batch,
                            nfp=args.nfp,
                            target=args.target,
                            timeout_s=args.batch_timeout_s,
                            metadata={
                                "phase": "updated_center_backtracking",
                                "iteration": iteration,
                                "fraction": fraction,
                            },
                        )
                    )
                    if any(error is not None for error in trial_errors) or trial_results[0] is None:
                        raise RuntimeError(
                            "backtracked-center score failure at iteration "
                            f"{iteration}, fraction {fraction}: {trial_errors}"
                        )
                    trial_result = trial_results[0]
                    center_backtracking.append(
                        {
                            "fraction": fraction,
                            "status": trial_result.get("status"),
                            "score": result_score(trial_result),
                            "decode_wall_s": trial_decode_wall_s,
                            "score_wall_s": trial_score_wall_s,
                            "score_elapsed_s": trial_elapsed,
                        }
                    )
                    center_decode_wall_s += trial_decode_wall_s
                    center_score_wall_s += trial_score_wall_s
                    center_elapsed.extend(trial_elapsed)
                    if result_valid(trial_result):
                        center_update_accepted = True
                        center_acceptance_fraction = fraction
                        center_rejection_reason = "invalid_full_proposal_backtracked"
                        current_noise = trial_noise
                        current_tokens = trial_batch[0]
                        current_result = trial_result
                        update = fraction * full_update
                        break
                if not center_update_accepted:
                    center_rejection_reason = "invalid_updated_center"
                    current_noise = previous_noise
                    current_tokens = previous_tokens
                    current_result = previous_result
                    first_moment = previous_first_moment
                    second_moment = previous_second_moment
                    adam_step = previous_adam_step
                    update = np.zeros_like(full_update)
            else:
                current_tokens = proposed_tokens
                current_result = proposed_result
            update_rms = rms(update)
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
                "pair_statuses": pair_statuses,
                "raw_direction_deltas": raw_delta.tolist(),
                "used_direction_deltas": used_delta.tolist(),
                "filtered_invalid_directions": invalid_direction.tolist(),
                "filtered_outlier_directions": outlier_direction.tolist(),
                "adaptive_direction_delta_limit": adaptive_delta_limit,
                "temporal_gradient_limit": temporal_gradient_limit,
                "temporal_update_limit": temporal_update_limit,
                "temporal_gradient_outlier": temporal_gradient_outlier,
                "temporal_update_outlier": temporal_update_outlier,
                "temporal_step_rejected": temporal_step_rejected,
                "gradient_step_applied": gradient_step_applied,
                "adam_step": adam_step,
                "proposed_center_score": proposed_score,
                "proposed_center_status": proposed_result.get("status"),
                "center_update_accepted": center_update_accepted,
                "center_rejection_reason": center_rejection_reason,
                "center_acceptance_fraction": center_acceptance_fraction,
                "center_backtracking": center_backtracking,
                "raw_gradient_rms": rms(raw_gradient),
                "gradient_rms": gradient_rms,
                "first_moment_rms": rms(first_moment),
                "second_moment_root_mean": float(np.sqrt(np.mean(second_moment))),
                "gradient_previous_moment_cosine": cosine_similarity(
                    gradient, previous_first_moment
                ),
                "update_gradient_cosine": cosine_similarity(update, gradient),
                "update_previous_moment_cosine": cosine_similarity(
                    update, previous_first_moment
                ),
                "update_rms": update_rms,
                "proposed_update_rms": proposed_update_rms,
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
                "total_wall_s": prior_wall_s + time.perf_counter() - started,
            }
            trajectory_path = write_trajectory_case(
                trajectory_dir,
                current_tokens,
                current_noise,
                current_result,
                nfp=args.nfp,
                target=args.target,
                iteration=iteration,
                optimizer_state={
                    "adam_step": adam_step,
                    "current_score": current_score,
                    "best_score": best_score,
                    "best_iteration": best_iteration,
                    "gradient_step_applied": gradient_step_applied,
                    "temporal_step_rejected": temporal_step_rejected,
                    "center_update_accepted": center_update_accepted,
                    "center_acceptance_fraction": center_acceptance_fraction,
                },
            )
            row["trajectory_case"] = str(
                trajectory_path.relative_to(args.out_dir)
            )
            if gradient_step_applied and center_update_accepted:
                accepted_gradient_scales.append(gradient_rms)
                accepted_update_scales.append(update_rms)
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
                adam_step=adam_step,
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

    total_wall_s = prior_wall_s + time.perf_counter() - started
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
        "completed_adam_steps": adam_step,
        "total_wall_s": total_wall_s,
        "mean_iteration_wall_s": (
            float(np.mean([row["iteration_wall_s"] for row in history]))
            if history
            else float("nan")
        ),
        "initial_decode_wall_s": initial_decode_wall_s,
        "initial_score_wall_s": initial_score_wall_s,
        "initial_score_elapsed_s": initial_elapsed,
        "resumed_from_iteration": start_iteration if args.resume else None,
        "resume_center_score": resumed_center_score if args.resume else None,
        "final_noise_rms": rms(current_noise),
        "final_noise_abs_max": float(np.max(np.abs(current_noise))),
        "best_case": str((args.out_dir / "best.json").resolve()),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
