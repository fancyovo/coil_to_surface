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
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
for path in (REPO_ROOT, GPU_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stellarator_gpu import (  # noqa: E402
    BatchCoilFieldGpu,
    coil_component_gradient_native,
    score_coils_native,
)
from stellarator_eval.psi import build_modes  # noqa: E402
from scripts.optimize_flow_prior_standard_adam import (  # noqa: E402
    rolling_robust_limit,
)
from scripts.optimize_flow_prior_zo_adam import (  # noqa: E402
    TOKEN_DIM,
    cosine_similarity,
    decode_noise_rk4,
    diagnostics_value,
    load_flow_checkpoint,
    load_initial_noise,
    result_score,
    result_valid,
    rms,
)
from scripts.optimize_native_score_cem import (  # noqa: E402
    append_jsonl,
    file_sha256,
    token_case,
    write_json,
)


def score_arguments(tokens: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(tokens, dtype=np.float64)
    return values[..., :33], values[..., 33:66], values[..., 66:99], values[..., 99]


def endpoint_latents(center: np.ndarray, perturbation: float) -> np.ndarray:
    flat = np.asarray(center, dtype=np.float32).reshape(-1)
    count = flat.size
    endpoints = np.repeat(flat[None], 2 * count, axis=0)
    coordinate = np.arange(count)
    endpoints[2 * coordinate, coordinate] -= np.float32(perturbation)
    endpoints[2 * coordinate + 1, coordinate] += np.float32(perturbation)
    return endpoints.reshape((2 * count,) + center.shape)


def coordinate_gradient(scores: np.ndarray, perturbation: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size % 2:
        raise ValueError("coordinate endpoint count must be even")
    return (values[1::2] - values[0::2]) / (2.0 * perturbation)


def score_config(
    *,
    iota_degree: int,
    surface_theta_count: int,
    axis_hint: tuple[float, float] | None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "iota_degree": int(iota_degree),
        "surface_selection_mode": 1,
        "surface_confidence_periods": 1,
        "surface_theta_count": int(surface_theta_count),
        "surface_trace_steps": 400,
        "surface_flux_bisection_iters": 6,
    }
    if axis_hint is not None:
        config.update(
            {
                "axis_hint_enabled": 1,
                "axis_hint_require_continuation": 2,
                "axis_hint_R": float(axis_hint[0]),
                "axis_hint_Z": float(axis_hint[1]),
            }
        )
    return config


def axis_hint(result: dict[str, Any]) -> tuple[float, float]:
    diagnostics = result["diagnostics"]
    value = (float(diagnostics["axis_R"]), float(diagnostics["axis_Z"]))
    if not all(math.isfinite(item) for item in value):
        raise RuntimeError("accepted center does not contain a finite magnetic-axis hint")
    return value


def score_center(
    lib: Path,
    tokens: np.ndarray,
    *,
    nfp: int,
    score_device: int,
    iota_degree: int,
    surface_theta_count: int,
    previous_result: dict[str, Any] | None,
) -> tuple[dict[str, Any], float]:
    x, y, z, current = score_arguments(tokens)
    started = time.perf_counter()
    result = score_coils_native(
        lib,
        x,
        y,
        z,
        current,
        nfp,
        device_id=score_device,
        target_helicity=(1, nfp),
        config_overrides=score_config(
            iota_degree=iota_degree,
            surface_theta_count=surface_theta_count,
            axis_hint=None if previous_result is None else axis_hint(previous_result),
        ),
    )
    return result, time.perf_counter() - started


class LocalFullGradientEstimator:
    def __init__(
        self,
        lib: Path,
        *,
        nfp: int,
        score_device: int,
        segments_per_coil: int,
        psi_iterations: int,
        alpha_iterations: int,
        formal_surface_theta_count: int,
        local_surface_theta_count: int,
        iota_degree: int,
    ) -> None:
        self.lib = lib
        self.nfp = int(nfp)
        self.score_device = int(score_device)
        self.segments_per_coil = int(segments_per_coil)
        self.psi_iterations = int(psi_iterations)
        self.alpha_iterations = int(alpha_iterations)
        self.formal_surface_theta_count = int(formal_surface_theta_count)
        self.local_surface_theta_count = int(local_surface_theta_count)
        self.iota_degree = int(iota_degree)
        modes = build_modes(10, 12)
        self.mode_a = np.asarray([mode.a for mode in modes], dtype=np.int32)
        self.mode_b = np.asarray([mode.b for mode in modes], dtype=np.int32)
        self.mode_m = np.asarray([mode.m for mode in modes], dtype=np.int32)
        self.mode_kind = np.asarray(
            [0 if mode.kind == "cos" else 1 for mode in modes], dtype=np.int32
        )

    def evaluate(
        self,
        center_tokens: np.ndarray,
        endpoint_tokens: np.ndarray,
        center_result: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        center_x, center_y, center_z, center_current = score_arguments(center_tokens)
        x, y, z, current = score_arguments(endpoint_tokens)
        query_count = len(endpoint_tokens)
        center_axis = axis_hint(center_result)
        timings: dict[str, float] = {}

        started = time.perf_counter()
        batch = BatchCoilFieldGpu(
            self.lib,
            x,
            y,
            z,
            current,
            self.nfp,
            segments_per_coil=self.segments_per_coil,
            device_id=self.score_device,
        )
        timings["field_create"] = time.perf_counter() - started
        try:
            started = time.perf_counter()
            capture = batch.capture_psi_center(
                center_x,
                center_y,
                center_z,
                center_current,
                target_helicity=(1, self.nfp),
                config_overrides=score_config(
                    iota_degree=self.iota_degree,
                    surface_theta_count=self.formal_surface_theta_count,
                    axis_hint=center_axis,
                ),
            )
            timings["center_capture"] = time.perf_counter() - started
            captured_result = capture["score_result"]
            if not result_valid(captured_result):
                raise RuntimeError(
                    f"center capture failed with status {captured_result.get('status')}"
                )

            axis_R0 = np.full(query_count, center_axis[0], dtype=np.float64)
            axis_Z0 = np.full(query_count, center_axis[1], dtype=np.float64)
            started = time.perf_counter()
            refined_axis = batch.refine_axis_hint(
                axis_R0,
                axis_Z0,
                trace_steps=960,
                newton_iterations=6,
                finite_difference_step=2.0e-4,
                maximum_newton_step=0.25,
                residual_tolerance=1.0e-7,
                hint_max_distance=0.08,
            )
            timings["axis_refine"] = time.perf_counter() - started

            started = time.perf_counter()
            batch_axis = batch.trace_axis_samples(
                refined_axis["R"],
                refined_axis["Z"],
                integration_steps=960,
                sample_count=240,
            )
            timings["axis_samples"] = time.perf_counter() - started

            started = time.perf_counter()
            psi, psi_rms, psi_stats = batch.fit_psi_pcgls(
                *batch_axis,
                self.mode_a,
                self.mode_b,
                self.mode_m,
                self.mode_kind,
                capture["psi_coefficients"],
                radius_scale=0.05,
                radial_grid=48,
                vertical_grid=48,
                phi_grid=48,
                rho_min=0.002,
                ridge=1.0e-6,
                iterations=self.psi_iterations,
            )
            timings["psi"] = time.perf_counter() - started

            started = time.perf_counter()
            coil_linear = coil_component_gradient_native(
                self.lib,
                center_x,
                center_y,
                center_z,
                center_current,
                self.nfp,
            )
            coil_components = np.full(query_count, coil_linear["component"])
            for values, center_values, name in (
                (x, center_x, "x"),
                (y, center_y, "y"),
                (z, center_z, "z"),
                (current, center_current, "current"),
            ):
                delta = (values - center_values) * coil_linear["gradient"][name]
                coil_components += delta.reshape(query_count, -1).sum(axis=1)
            timings["coil_linear"] = time.perf_counter() - started

            started = time.perf_counter()
            local_results, local_stats = batch.score_local_batch(
                current,
                *batch_axis,
                refined_axis["residual"],
                refined_axis["topology_trace"],
                refined_axis["topology_det"],
                psi,
                psi_rms,
                coil_components,
                capture,
                surface_theta_count=self.local_surface_theta_count,
                alpha_iterations=self.alpha_iterations,
            )
            timings["local_score"] = time.perf_counter() - started
        finally:
            batch.close()
            batch.clear_psi_preconditioner()

        scores = np.asarray([result_score(result) for result in local_results])
        statuses = [str(result.get("status")) for result in local_results]
        timing_total = sum(timings.values())
        details = {
            "timing_s": {**timings, "total": timing_total},
            "psi_stats": psi_stats,
            "local_stats": local_stats,
            "status_counts": {
                status: statuses.count(status) for status in sorted(set(statuses))
            },
            "axis_valid_fraction": float(np.mean(refined_axis["valid"])),
            "center_capture_score": result_score(captured_result),
            "center_capture_score_delta": (
                result_score(captured_result) - result_score(center_result)
            ),
        }
        return scores, captured_result, details


def make_case(
    tokens: np.ndarray,
    noise: np.ndarray,
    result: dict[str, Any],
    *,
    nfp: int,
    iteration: int,
    best_score: float,
    best_iteration: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    case = token_case(tokens, nfp=nfp, target="QH")
    case["flow_prior_local_full_gradient_adam"] = {
        "format": "flow_prior_local_full_gradient_adam_v1",
        "iteration": int(iteration),
        "best_score": float(best_score),
        "best_iteration": int(best_iteration),
        "noise": np.asarray(noise, dtype=np.float32).tolist(),
        "native_score": result,
        "manifest": manifest,
    }
    return case


def write_trajectory(
    directory: Path,
    tokens: np.ndarray,
    noise: np.ndarray,
    result: dict[str, Any],
    *,
    nfp: int,
    iteration: int,
    best_score: float,
    best_iteration: int,
    manifest: dict[str, Any],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{iteration:04d}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            make_case(
                tokens,
                noise,
                result,
                nfp=nfp,
                iteration=iteration,
                best_score=best_score,
                best_iteration=best_iteration,
                manifest=manifest,
            ),
            indent=2,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def plot_progress(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = [row["iteration"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(step, [row["current_score"] for row in rows], label="current")
    axes[0, 0].plot(step, [row["best_score"] for row in rows], label="best")
    axes[0, 0].set(ylabel="formal score", title="Local full-gradient Adam")
    axes[0, 0].legend()
    axes[0, 1].plot(step, [row["current_qh_error"] for row in rows], label="QH")
    axes[0, 1].plot(step, [row["current_qa_error"] for row in rows], label="QA")
    axes[0, 1].plot(step, [row["current_qp_error"] for row in rows], label="QP")
    axes[0, 1].set(yscale="log", ylabel="volume QS residual")
    axes[0, 1].legend()
    axes[1, 0].plot(step, [row["gradient_rms"] for row in rows], label="gradient")
    axes[1, 0].plot(step, [row["update_rms"] for row in rows], label="update")
    axes[1, 0].set(yscale="log", xlabel="iteration", ylabel="latent RMS")
    axes[1, 0].legend()
    axes[1, 1].plot(step, [row["gradient_wall_s"] for row in rows], label="gradient")
    axes[1, 1].plot(step, [row["iteration_wall_s"] for row in rows], label="iteration")
    axes[1, 1].set(xlabel="iteration", ylabel="seconds")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adam with a 300-D coordinate gradient from one local CUDA score batch."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-case", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--nfp", type=int, default=4)
    parser.add_argument("--n-base-coils", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-wall-s", type=float, default=1500.0)
    parser.add_argument("--flow-steps", type=int, default=128)
    parser.add_argument("--perturbation", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.7)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--flow-device", type=int, default=0)
    parser.add_argument("--score-device", type=int, default=1)
    parser.add_argument("--segments-per-coil", type=int, default=256)
    parser.add_argument("--psi-iterations", type=int, default=4)
    parser.add_argument("--alpha-iterations", type=int, default=4)
    parser.add_argument("--formal-surface-theta-count", type=int, default=128)
    parser.add_argument("--local-surface-theta-count", type=int, default=64)
    parser.add_argument("--iota-degree", type=int, default=3)
    parser.add_argument("--temporal-guard-window", type=int, default=20)
    parser.add_argument("--temporal-guard-min-history", type=int, default=20)
    parser.add_argument("--temporal-gradient-ratio", type=float, default=8.0)
    parser.add_argument("--temporal-update-ratio", type=float, default=8.0)
    parser.add_argument("--temporal-guard-mad-factor", type=float, default=8.0)
    parser.add_argument("--backtracking", default="0.5,0.25,0.125")
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.flow_device == args.score_device:
        raise ValueError("flow and score devices must be distinct")
    if torch.cuda.device_count() <= max(args.flow_device, args.score_device):
        raise RuntimeError("two visible CUDA devices are required")
    if args.n_base_coils * TOKEN_DIM != 300:
        raise ValueError("this experiment currently requires three 100-D coil tokens")
    if args.iterations < 1 or args.max_wall_s <= 0.0:
        raise ValueError("iterations and max-wall-s must be positive")
    if not 0.0 < args.beta1 < 1.0 or not 0.0 < args.beta2 < 1.0:
        raise ValueError("Adam betas must be in (0, 1)")
    if args.perturbation <= 0.0 or args.learning_rate <= 0.0:
        raise ValueError("perturbation and learning rate must be positive")
    backtracking = tuple(float(value) for value in args.backtracking.split(","))
    if any(not 0.0 < value < 1.0 for value in backtracking):
        raise ValueError("backtracking fractions must be in (0, 1)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.out_dir / "trajectory"
    paths = {
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
        required = [path for name, path in paths.items() if name != "summary.json"]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"resume artifacts are missing: {missing}")
        if paths["summary.json"].exists():
            raise FileExistsError("refusing to resume a completed run")
        state = np.load(paths["state_latest.npz"], allow_pickle=False)
        current_noise = np.asarray(state["current_noise"], dtype=np.float32)
        best_noise = np.asarray(state["best_noise"], dtype=np.float32)
        first_moment = np.asarray(state["first_moment"], dtype=np.float64)
        second_moment = np.asarray(state["second_moment"], dtype=np.float64)
        start_iteration = int(state["iteration"])
        adam_step = int(state["adam_step"])
        history = [
            json.loads(line)
            for line in paths["history.jsonl"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        progress = json.loads(paths["progress.json"].read_text(encoding="utf-8"))
        initial_score = float(progress["initial_score"])
        best_score = float(progress["best_score"])
        best_iteration = int(progress["best_iteration"])
        prior_wall_s = float(history[-1]["total_wall_s"])
        manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    else:
        if any(path.exists() for path in paths.values()) or trajectory_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing run {args.out_dir}")
        current_noise, initial_payload = load_initial_noise(args.initial_case)
        if current_noise.shape != (args.n_base_coils, TOKEN_DIM):
            raise ValueError("initial latent shape does not match n-base-coils")
        best_noise = current_noise.copy()
        first_moment = np.zeros_like(current_noise, dtype=np.float64)
        second_moment = np.zeros_like(current_noise, dtype=np.float64)
        start_iteration = 0
        adam_step = 0
        history = []
        initial_score = float("nan")
        best_score = float("-inf")
        best_iteration = 0
        prior_wall_s = 0.0
        manifest = {
            "algorithm": "adam_with_query_batched_300d_coordinate_gradient",
            "gradient_objective": (
                "dynamic local full-density score with coordinate and maximum-surface-selection "
                "derivatives omitted"
            ),
            "formal_objective": "complete native QH score for every accepted center",
            "initial_case": str(args.initial_case.resolve()),
            "initial_case_sha256": file_sha256(args.initial_case),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "native_lib": str(args.lib.resolve()),
            "native_lib_sha256": file_sha256(args.lib),
            "nfp": args.nfp,
            "n_base_coils": args.n_base_coils,
            "iterations": args.iterations,
            "max_wall_s": args.max_wall_s,
            "flow": {"method": "rk4", "steps": args.flow_steps, "dtype": "fp32"},
            "coordinate_gradient": {
                "dimension": args.n_base_coils * TOKEN_DIM,
                "endpoint_count": 2 * args.n_base_coils * TOKEN_DIM,
                "perturbation": args.perturbation,
                "difference": "centered",
                "psi_grid": 48,
                "psi_iterations": args.psi_iterations,
                "alpha_iterations": args.alpha_iterations,
                "formal_surface_theta_count": args.formal_surface_theta_count,
                "local_surface_theta_count": args.local_surface_theta_count,
            },
            "adam": {
                "learning_rate": args.learning_rate,
                "beta1": args.beta1,
                "beta2": args.beta2,
                "epsilon": args.adam_epsilon,
                "weight_decay": 0.0,
            },
            "axis_policy": "initial global search, then strict mixed-precision continuation",
            "devices": {"flow": args.flow_device, "score": args.score_device},
            "initial_metadata_keys": sorted(initial_payload.keys()),
        }
        write_json(paths["manifest.json"], manifest)

    torch.cuda.set_device(args.flow_device)
    flow_device = torch.device("cuda", args.flow_device)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, flow_device)
    if int(checkpoint["step"]) != 30000:
        raise RuntimeError("unexpected flow checkpoint step")
    estimator = LocalFullGradientEstimator(
        args.lib,
        nfp=args.nfp,
        score_device=args.score_device,
        segments_per_coil=args.segments_per_coil,
        psi_iterations=args.psi_iterations,
        alpha_iterations=args.alpha_iterations,
        formal_surface_theta_count=args.formal_surface_theta_count,
        local_surface_theta_count=args.local_surface_theta_count,
        iota_degree=args.iota_degree,
    )

    torch.cuda.set_device(args.flow_device)
    current_tokens_batch, initial_decode_wall_s = decode_noise_rk4(
        model,
        normalizer,
        current_noise[None],
        nfp=args.nfp,
        steps=args.flow_steps,
        device=flow_device,
    )
    current_tokens = current_tokens_batch[0]
    current_result, initial_score_wall_s = score_center(
        args.lib,
        current_tokens,
        nfp=args.nfp,
        score_device=args.score_device,
        iota_degree=args.iota_degree,
        surface_theta_count=args.formal_surface_theta_count,
        previous_result=None,
    )
    if not result_valid(current_result):
        raise RuntimeError(f"initial center is invalid: {current_result.get('status')}")
    if not args.resume:
        initial_score = result_score(current_result)
        best_score = initial_score
        best_tokens = current_tokens.copy()
        best_result = current_result
        write_trajectory(
            trajectory_dir,
            current_tokens,
            current_noise,
            current_result,
            nfp=args.nfp,
            iteration=0,
            best_score=best_score,
            best_iteration=0,
            manifest=manifest,
        )
        write_json(
            paths["best.json"],
            make_case(
                best_tokens,
                best_noise,
                best_result,
                nfp=args.nfp,
                iteration=0,
                best_score=best_score,
                best_iteration=0,
                manifest=manifest,
            ),
        )
    else:
        best_payload = json.loads(paths["best.json"].read_text(encoding="utf-8"))
        best_tokens = np.column_stack(
            (
                np.asarray(best_payload["raw"]["x"]),
                np.asarray(best_payload["raw"]["y"]),
                np.asarray(best_payload["raw"]["z"]),
                np.asarray(best_payload["raw"]["current"]),
            )
        )
        best_result = best_payload["flow_prior_local_full_gradient_adam"]["native_score"]

    accepted_gradient_scales = [
        float(row["gradient_rms"])
        for row in history
        if row.get("gradient_step_applied") and row.get("center_update_accepted")
    ]
    accepted_update_scales = [
        float(row["update_rms"])
        for row in history
        if row.get("gradient_step_applied") and row.get("center_update_accepted")
    ]
    recent_walls = [float(row["iteration_wall_s"]) for row in history[-5:]]
    run_started = time.perf_counter()
    stop_reason = "completed_iterations"

    for iteration in range(start_iteration + 1, args.iterations + 1):
        elapsed_total = prior_wall_s + time.perf_counter() - run_started
        reserve = 1.25 * max(recent_walls, default=30.0)
        if elapsed_total + reserve >= args.max_wall_s:
            stop_reason = "max_wall_s"
            break
        iteration_started = time.perf_counter()

        endpoints = endpoint_latents(current_noise, args.perturbation)
        torch.cuda.set_device(args.flow_device)
        endpoint_tokens, endpoint_decode_wall_s = decode_noise_rk4(
            model,
            normalizer,
            endpoints,
            nfp=args.nfp,
            steps=args.flow_steps,
            device=flow_device,
        )
        local_scores, captured_result, gradient_details = estimator.evaluate(
            current_tokens,
            endpoint_tokens,
            current_result,
        )
        current_result = captured_result
        statuses = gradient_details["status_counts"]
        all_endpoints_valid = statuses == {"ok": len(endpoints)}
        raw_gradient = coordinate_gradient(local_scores, args.perturbation).reshape(
            current_noise.shape
        )
        raw_gradient_rms = rms(raw_gradient)
        temporal_gradient_limit = rolling_robust_limit(
            accepted_gradient_scales,
            window=args.temporal_guard_window,
            min_history=args.temporal_guard_min_history,
            ratio=args.temporal_gradient_ratio,
            mad_factor=args.temporal_guard_mad_factor,
        )
        temporal_gradient_outlier = bool(
            temporal_gradient_limit is not None
            and raw_gradient_rms > temporal_gradient_limit
        )
        gradient_step_applied = all_endpoints_valid and not temporal_gradient_outlier

        previous_first = first_moment.copy()
        previous_second = second_moment.copy()
        previous_adam_step = adam_step
        if gradient_step_applied:
            candidate_step = adam_step + 1
            candidate_first = (
                args.beta1 * first_moment + (1.0 - args.beta1) * raw_gradient
            )
            candidate_second = (
                args.beta2 * second_moment + (1.0 - args.beta2) * raw_gradient**2
            )
            first_hat = candidate_first / (1.0 - args.beta1**candidate_step)
            second_hat = candidate_second / (1.0 - args.beta2**candidate_step)
            proposed_update = args.learning_rate * first_hat / (
                np.sqrt(second_hat) + args.adam_epsilon
            )
            temporal_update_limit = rolling_robust_limit(
                accepted_update_scales,
                window=args.temporal_guard_window,
                min_history=args.temporal_guard_min_history,
                ratio=args.temporal_update_ratio,
                mad_factor=args.temporal_guard_mad_factor,
            )
            temporal_update_outlier = bool(
                temporal_update_limit is not None
                and rms(proposed_update) > temporal_update_limit
            )
            if temporal_update_outlier:
                gradient_step_applied = False
        else:
            candidate_step = adam_step
            candidate_first = first_moment
            candidate_second = second_moment
            proposed_update = np.zeros_like(current_noise, dtype=np.float64)
            temporal_update_limit = None
            temporal_update_outlier = False

        previous_noise = current_noise.copy()
        previous_tokens = current_tokens.copy()
        previous_result = current_result
        center_update_accepted = False
        center_acceptance_fraction = 0.0
        proposal_score_wall_s = 0.0
        proposal_decode_wall_s = 0.0
        proposal_attempts: list[dict[str, Any]] = []
        applied_update = np.zeros_like(proposed_update)
        if gradient_step_applied:
            for fraction in (1.0,) + backtracking:
                trial_noise = (
                    previous_noise.astype(np.float64) + fraction * proposed_update
                ).astype(np.float32)
                torch.cuda.set_device(args.flow_device)
                trial_tokens_batch, decode_wall_s = decode_noise_rk4(
                    model,
                    normalizer,
                    trial_noise[None],
                    nfp=args.nfp,
                    steps=args.flow_steps,
                    device=flow_device,
                )
                trial_result, score_wall_s = score_center(
                    args.lib,
                    trial_tokens_batch[0],
                    nfp=args.nfp,
                    score_device=args.score_device,
                    iota_degree=args.iota_degree,
                    surface_theta_count=args.formal_surface_theta_count,
                    previous_result=previous_result,
                )
                proposal_decode_wall_s += decode_wall_s
                proposal_score_wall_s += score_wall_s
                proposal_attempts.append(
                    {
                        "fraction": fraction,
                        "status": trial_result.get("status"),
                        "score": result_score(trial_result),
                    }
                )
                if result_valid(trial_result):
                    current_noise = trial_noise
                    current_tokens = trial_tokens_batch[0]
                    current_result = trial_result
                    first_moment = candidate_first
                    second_moment = candidate_second
                    adam_step = candidate_step
                    applied_update = fraction * proposed_update
                    center_update_accepted = True
                    center_acceptance_fraction = fraction
                    break
        if not center_update_accepted:
            current_noise = previous_noise
            current_tokens = previous_tokens
            current_result = previous_result
            first_moment = previous_first
            second_moment = previous_second
            adam_step = previous_adam_step

        current_score = result_score(current_result)
        if current_score > best_score:
            best_score = current_score
            best_iteration = iteration
            best_noise = current_noise.copy()
            best_tokens = current_tokens.copy()
            best_result = current_result
            write_json(
                paths["best.json"],
                make_case(
                    best_tokens,
                    best_noise,
                    best_result,
                    nfp=args.nfp,
                    iteration=best_iteration,
                    best_score=best_score,
                    best_iteration=best_iteration,
                    manifest=manifest,
                ),
            )

        update_rms = rms(applied_update)
        if gradient_step_applied and center_update_accepted:
            accepted_gradient_scales.append(raw_gradient_rms)
            accepted_update_scales.append(update_rms)
        iteration_wall_s = time.perf_counter() - iteration_started
        recent_walls.append(iteration_wall_s)
        recent_walls = recent_walls[-5:]
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
            "gradient_endpoint_count": len(endpoints),
            "gradient_endpoint_statuses": statuses,
            "gradient_step_applied": gradient_step_applied,
            "center_update_accepted": center_update_accepted,
            "center_acceptance_fraction": center_acceptance_fraction,
            "proposal_attempts": proposal_attempts,
            "adam_step": adam_step,
            "raw_gradient_rms": raw_gradient_rms,
            "gradient_rms": raw_gradient_rms if gradient_step_applied else 0.0,
            "first_moment_rms": rms(first_moment),
            "second_moment_root_mean": float(np.sqrt(np.mean(second_moment))),
            "gradient_previous_moment_cosine": cosine_similarity(
                raw_gradient, previous_first
            ),
            "update_gradient_cosine": cosine_similarity(applied_update, raw_gradient),
            "proposed_update_rms": rms(proposed_update),
            "update_rms": update_rms,
            "temporal_gradient_limit": temporal_gradient_limit,
            "temporal_update_limit": temporal_update_limit,
            "temporal_gradient_outlier": temporal_gradient_outlier,
            "temporal_update_outlier": temporal_update_outlier,
            "noise_rms": rms(current_noise),
            "noise_abs_max": float(np.max(np.abs(current_noise))),
            "endpoint_decode_wall_s": endpoint_decode_wall_s,
            "gradient_pipeline": gradient_details,
            "gradient_wall_s": endpoint_decode_wall_s
            + gradient_details["timing_s"]["total"],
            "proposal_decode_wall_s": proposal_decode_wall_s,
            "proposal_score_wall_s": proposal_score_wall_s,
            "iteration_wall_s": iteration_wall_s,
            "total_wall_s": prior_wall_s + time.perf_counter() - run_started,
        }
        trajectory_path = write_trajectory(
            trajectory_dir,
            current_tokens,
            current_noise,
            current_result,
            nfp=args.nfp,
            iteration=iteration,
            best_score=best_score,
            best_iteration=best_iteration,
            manifest=manifest,
        )
        row["trajectory_case"] = str(trajectory_path.relative_to(args.out_dir))
        history.append(row)
        append_jsonl(paths["history.jsonl"], row)
        write_json(
            paths["progress.json"],
            {
                "manifest": manifest,
                "initial_score": initial_score,
                "best_score": best_score,
                "best_iteration": best_iteration,
                "iterations": history,
            },
        )
        np.savez_compressed(
            paths["state_latest.npz"],
            current_noise=current_noise,
            best_noise=best_noise,
            first_moment=first_moment,
            second_moment=second_moment,
            iteration=np.asarray(iteration, dtype=np.int64),
            adam_step=np.asarray(adam_step, dtype=np.int64),
        )
        if iteration == 1 or iteration % args.plot_every == 0:
            plot_progress(history, args.out_dir / "progress.png")
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "score": current_score,
                    "best": best_score,
                    "gradient_rms": raw_gradient_rms,
                    "update_rms": update_rms,
                    "accepted": center_update_accepted,
                    "wall_s": iteration_wall_s,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    total_wall_s = prior_wall_s + time.perf_counter() - run_started
    plot_progress(history, args.out_dir / "progress.png")
    completed_iterations = history[-1]["iteration"] if history else start_iteration
    summary = {
        "status": "ok",
        "stop_reason": stop_reason,
        "initial_score": initial_score,
        "final_score": result_score(current_result),
        "best_score": best_score,
        "best_iteration": best_iteration,
        "completed_iterations": completed_iterations,
        "completed_adam_steps": adam_step,
        "total_wall_s": total_wall_s,
        "initial_decode_wall_s": initial_decode_wall_s,
        "initial_score_wall_s": initial_score_wall_s,
        "mean_iteration_wall_s": (
            float(np.mean([row["iteration_wall_s"] for row in history]))
            if history
            else None
        ),
        "manifest": manifest,
    }
    write_json(paths["summary.json"], summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
