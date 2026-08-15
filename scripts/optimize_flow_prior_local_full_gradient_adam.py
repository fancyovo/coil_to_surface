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
    orthogonal_directions,
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
from flow_matching.trajectory_dataset import OptimizationTraceRecorder  # noqa: E402


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


def random_direction_endpoints(
    center: np.ndarray,
    perturbation: float,
    directions: np.ndarray,
) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    if directions.ndim != center.ndim + 1 or directions.shape[1:] != center.shape:
        raise ValueError("direction bank shape does not match center")
    endpoints = np.repeat(center[None], 2 * len(directions), axis=0)
    endpoints[0::2] -= np.float32(perturbation) * directions
    endpoints[1::2] += np.float32(perturbation) * directions
    return endpoints


def random_direction_gradient(
    scores: np.ndarray,
    perturbation: float,
    directions: np.ndarray,
) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float64)
    slopes = coordinate_gradient(scores, perturbation)
    if slopes.shape != (len(directions),):
        raise ValueError("directional slope count does not match direction bank")
    return np.mean(
        slopes.reshape((-1,) + (1,) * (directions.ndim - 1)) * directions,
        axis=0,
    )


def gradient_probe(
    center: np.ndarray,
    *,
    mode: str,
    perturbation: float,
    random_direction_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray]:
    if mode == "coordinate":
        return None, endpoint_latents(center, perturbation)
    directions = orthogonal_directions(rng, center.shape, random_direction_count)
    return directions, random_direction_endpoints(center, perturbation, directions)


def damped_inverse_bfgs(
    inverse_hessian: np.ndarray,
    step: np.ndarray,
    objective_gradient_delta: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    inverse = np.asarray(inverse_hessian, dtype=np.float64)
    step = np.asarray(step, dtype=np.float64).reshape(-1)
    delta = np.asarray(objective_gradient_delta, dtype=np.float64).reshape(-1)
    if inverse.shape != (step.size, step.size) or delta.shape != step.shape:
        raise ValueError("incompatible BFGS shapes")
    try:
        hessian_step = np.linalg.solve(inverse, step)
    except np.linalg.LinAlgError:
        return inverse.copy(), {"updated": False, "reason": "singular_inverse"}
    step_hessian_step = float(np.dot(step, hessian_step))
    step_delta = float(np.dot(step, delta))
    if not math.isfinite(step_hessian_step) or step_hessian_step <= 1.0e-14:
        return inverse.copy(), {"updated": False, "reason": "invalid_model"}
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


def recorded_native_result(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in (
        "original_space_local_gradient_adam",
        "flow_prior_screening",
        "flow_prior_local_full_gradient_adam",
        "flow_prior_local_full_gradient_bfgs",
        "flow_prior_standard_adam",
        "flow_prior_zo_adam",
        "flow_prior_cem",
    ):
        metadata = payload.get(key)
        if isinstance(metadata, dict) and isinstance(metadata.get("native_score"), dict):
            return metadata["native_score"]
    return None


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
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
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
        return scores, captured_result, details, local_results


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
    if manifest.get("parameter_space", "latent") == "data":
        metadata_key = "original_space_local_gradient_adam"
    else:
        metadata_key = (
            "flow_prior_local_full_gradient_bfgs"
            if manifest["optimizer"] == "bfgs"
            else "flow_prior_local_full_gradient_adam"
        )
    parameter_name = (
        "normalized_coil_tokens"
        if manifest.get("parameter_space", "latent") == "data"
        else "noise"
    )
    case[metadata_key] = {
        "format": f"{metadata_key}_v1",
        "iteration": int(iteration),
        "best_score": float(best_score),
        "best_iteration": int(best_iteration),
        "parameter_space": manifest.get("parameter_space", "latent"),
        parameter_name: np.asarray(noise, dtype=np.float32).tolist(),
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
    optimizer = str(rows[-1].get("optimizer", "adam")).upper()
    axes[0, 0].set(ylabel="formal score", title=f"Local-gradient {optimizer}")
    axes[0, 0].legend()
    axes[0, 1].plot(step, [row["current_qh_error"] for row in rows], label="QH")
    axes[0, 1].plot(step, [row["current_qa_error"] for row in rows], label="QA")
    axes[0, 1].plot(step, [row["current_qp_error"] for row in rows], label="QP")
    axes[0, 1].set(yscale="log", ylabel="volume QS residual")
    axes[0, 1].legend()
    axes[1, 0].plot(step, [row["gradient_rms"] for row in rows], label="gradient")
    axes[1, 0].plot(step, [row["update_rms"] for row in rows], label="update")
    axes[1, 0].set(yscale="log", xlabel="iteration", ylabel="parameter RMS")
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
        description="Adam with a query-batched local CUDA score gradient."
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
    parser.add_argument(
        "--parameter-space",
        choices=("latent", "data"),
        default="latent",
        help=(
            "Optimize flow noise, or optimize normalized coil tokens directly after "
            "one initial flow decode. The data mode never evaluates the flow again."
        ),
    )
    parser.add_argument("--perturbation", type=float, default=0.005)
    parser.add_argument(
        "--gradient-mode",
        choices=("coordinate", "random-orthogonal"),
        default="coordinate",
    )
    parser.add_argument("--random-directions", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--optimizer", choices=("adam", "bfgs"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.7)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--flow-device", type=int, default=0)
    parser.add_argument("--score-device", type=int, default=0)
    parser.add_argument(
        "--flow-pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Decode an accepted center together with the next iteration's gradient "
            "endpoints. This removes the separate one-sample flow call."
        ),
    )
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
    parser.add_argument("--bfgs-initial-trust-rms", type=float, default=0.01)
    parser.add_argument("--bfgs-min-trust-rms", type=float, default=1.0e-5)
    parser.add_argument("--bfgs-max-trust-rms", type=float, default=0.05)
    parser.add_argument("--bfgs-trust-growth", type=float, default=1.2)
    parser.add_argument("--bfgs-trust-shrink", type=float, default=0.5)
    parser.add_argument("--bfgs-min-improvement", type=float, default=0.0)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--trajectory-every", type=int, default=1)
    parser.add_argument("--state-every", type=int, default=1)
    parser.add_argument("--save-training-trace", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.parameter_space == "data":
        args.flow_pipeline = False
    if torch.cuda.device_count() <= max(args.flow_device, args.score_device):
        raise RuntimeError("requested CUDA device is unavailable")
    if args.n_base_coils < 1:
        raise ValueError("n-base-coils must be positive")
    if args.iterations < 1 or args.max_wall_s <= 0.0:
        raise ValueError("iterations and max-wall-s must be positive")
    if not 0.0 < args.beta1 < 1.0 or not 0.0 < args.beta2 < 1.0:
        raise ValueError("Adam betas must be in (0, 1)")
    if args.perturbation <= 0.0 or args.learning_rate <= 0.0:
        raise ValueError("perturbation and learning rate must be positive")
    if args.state_every < 1 or args.progress_every < 1:
        raise ValueError("state-every and progress-every must be positive")
    if args.plot_every < 0 or args.trajectory_every < 0:
        raise ValueError("plot-every and trajectory-every must be nonnegative")
    dimension = args.n_base_coils * TOKEN_DIM
    if not 1 <= args.random_directions <= dimension:
        raise ValueError("random-directions must be in [1, latent dimension]")
    if args.optimizer == "bfgs" and args.gradient_mode != "coordinate":
        raise ValueError("BFGS requires the full coordinate gradient")
    if not 0.0 < args.bfgs_min_trust_rms <= args.bfgs_initial_trust_rms:
        raise ValueError("BFGS minimum trust RMS must not exceed its initial value")
    if not args.bfgs_initial_trust_rms <= args.bfgs_max_trust_rms:
        raise ValueError("BFGS initial trust RMS must not exceed its maximum")
    if args.bfgs_trust_growth < 1.0 or not 0.0 < args.bfgs_trust_shrink < 1.0:
        raise ValueError("invalid BFGS trust-radius factors")
    backtracking = tuple(float(value) for value in args.backtracking.split(","))
    if any(not 0.0 < value < 1.0 for value in backtracking):
        raise ValueError("backtracking fractions must be in (0, 1)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.out_dir / "trajectory"
    trace_recorder = (
        OptimizationTraceRecorder(args.out_dir) if args.save_training_trace else None
    )
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
        inverse_hessian = (
            np.asarray(state["inverse_hessian"], dtype=np.float64)
            if "inverse_hessian" in state.files
            else np.eye(dimension, dtype=np.float64)
        )
        previous_bfgs_gradient = (
            np.asarray(state["previous_bfgs_gradient"], dtype=np.float64)
            if "previous_bfgs_gradient" in state.files
            else np.empty(0, dtype=np.float64)
        )
        previous_bfgs_step = (
            np.asarray(state["previous_bfgs_step"], dtype=np.float64)
            if "previous_bfgs_step" in state.files
            else np.empty(0, dtype=np.float64)
        )
        bfgs_trust_rms = (
            float(state["bfgs_trust_rms"])
            if "bfgs_trust_rms" in state.files
            else args.bfgs_initial_trust_rms
        )
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
        manifest.setdefault("optimizer", "adam")
        manifest.setdefault("flow_pipeline", False)
        direction_rng = np.random.default_rng(args.seed)
        if "direction_rng_state" in state.files:
            direction_rng.bit_generator.state = json.loads(
                str(state["direction_rng_state"])
            )
        elif args.gradient_mode == "random-orthogonal":
            raise ValueError("random-direction resume state is missing its RNG state")
        prefetched_gradient = None
        if (
            "prefetched_gradient_present" in state.files
            and bool(state["prefetched_gradient_present"].item())
        ):
            saved_directions = np.asarray(
                state["prefetched_gradient_directions"], dtype=np.float32
            )
            prefetched_gradient = {
                "directions": None if saved_directions.size == 0 else saved_directions,
                "tokens": np.asarray(
                    state["prefetched_gradient_tokens"], dtype=np.float64
                ),
                "decode_wall_s": float(state["prefetched_gradient_decode_wall_s"]),
                "source_iteration": int(state["prefetched_gradient_source_iteration"]),
            }
    else:
        if any(path.exists() for path in paths.values()) or trajectory_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing run {args.out_dir}")
        current_noise, initial_payload = load_initial_noise(args.initial_case)
        if current_noise.shape != (args.n_base_coils, TOKEN_DIM):
            raise ValueError("initial latent shape does not match n-base-coils")
        best_noise = current_noise.copy()
        first_moment = np.zeros_like(current_noise, dtype=np.float64)
        second_moment = np.zeros_like(current_noise, dtype=np.float64)
        inverse_hessian = np.eye(dimension, dtype=np.float64)
        previous_bfgs_gradient = np.empty(0, dtype=np.float64)
        previous_bfgs_step = np.empty(0, dtype=np.float64)
        bfgs_trust_rms = args.bfgs_initial_trust_rms
        direction_rng = np.random.default_rng(args.seed)
        start_iteration = 0
        adam_step = 0
        history = []
        initial_score = float("nan")
        best_score = float("-inf")
        best_iteration = 0
        prior_wall_s = 0.0
        prefetched_gradient = None
        manifest = {
            "algorithm": f"{args.optimizer}_with_query_batched_local_gradient",
            "optimizer": args.optimizer,
            "parameter_space": args.parameter_space,
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
            "flow": {
                "method": "rk4",
                "steps": args.flow_steps,
                "dtype": "fp32",
                "use": (
                    "every gradient endpoint and accepted center"
                    if args.parameter_space == "latent"
                    else "initial latent-to-coil decode only"
                ),
            },
            "flow_pipeline": bool(
                args.flow_pipeline and args.parameter_space == "latent"
            ),
            "coordinate_gradient": {
                "dimension": args.n_base_coils * TOKEN_DIM,
                "mode": args.gradient_mode,
                "random_directions": (
                    args.random_directions
                    if args.gradient_mode == "random-orthogonal"
                    else None
                ),
                "endpoint_count": 2 * (
                    args.random_directions
                    if args.gradient_mode == "random-orthogonal"
                    else args.n_base_coils * TOKEN_DIM
                ),
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
            "bfgs": {
                "initial_trust_rms": args.bfgs_initial_trust_rms,
                "min_trust_rms": args.bfgs_min_trust_rms,
                "max_trust_rms": args.bfgs_max_trust_rms,
                "trust_growth": args.bfgs_trust_growth,
                "trust_shrink": args.bfgs_trust_shrink,
                "min_improvement": args.bfgs_min_improvement,
            },
            "seed": args.seed,
            "axis_policy": "initial global search, then strict mixed-precision continuation",
            "devices": {"flow": args.flow_device, "score": args.score_device},
            "initial_metadata_keys": sorted(initial_payload.keys()),
        }
        write_json(paths["manifest.json"], manifest)

    if args.resume:
        initial_payload = json.loads(args.initial_case.read_text(encoding="utf-8"))
        saved_space = manifest.get("parameter_space", "latent")
        if saved_space != args.parameter_space:
            raise ValueError(
                f"resume parameter space {saved_space!r} != requested {args.parameter_space!r}"
            )

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
    if args.resume and args.parameter_space == "data":
        started = time.perf_counter()
        current_tokens_batch = normalizer.inverse(
            current_noise[None], (args.nfp, args.n_base_coils)
        ).astype(np.float64, copy=False)
        initial_decode_wall_s = time.perf_counter() - started
    else:
        current_tokens_batch, initial_decode_wall_s = decode_noise_rk4(
            model,
            normalizer,
            current_noise[None],
            nfp=args.nfp,
            steps=args.flow_steps,
            device=flow_device,
        )
    current_tokens = current_tokens_batch[0]
    if not args.resume and args.parameter_space == "data":
        normalized, clipped_fraction = normalizer.transform(
            current_tokens[None], (args.nfp, args.n_base_coils)
        )
        current_noise = normalized[0]
        best_noise = current_noise.copy()
        reconstructed = normalizer.inverse(
            current_noise[None], (args.nfp, args.n_base_coils)
        )[0]
        reconstruction_relative_rms = float(
            np.linalg.norm(reconstructed - current_tokens)
            / max(np.linalg.norm(current_tokens), 1.0e-30)
        )
        current_tokens = reconstructed.astype(np.float64, copy=False)
        manifest["data_parameterization"] = {
            "definition": "per-coordinate training-set standardization",
            "physical_mapping": "CoilNormalizer.inverse with canonical current L1/sign",
            "initial_clipped_fraction": float(clipped_fraction),
            "initial_roundtrip_relative_rms": reconstruction_relative_rms,
            "flow_calls_after_initialization": 0,
        }
        write_json(paths["manifest.json"], manifest)
    initial_previous_result = recorded_native_result(initial_payload)
    current_result, initial_score_wall_s = score_center(
        args.lib,
        current_tokens,
        nfp=args.nfp,
        score_device=args.score_device,
        iota_degree=args.iota_degree,
        surface_theta_count=args.formal_surface_theta_count,
        previous_result=initial_previous_result,
    )
    if not result_valid(current_result):
        raise RuntimeError(f"initial center is invalid: {current_result.get('status')}")
    if not args.resume:
        initial_score = result_score(current_result)
        best_score = initial_score
        best_tokens = current_tokens.copy()
        best_result = current_result
        if args.trajectory_every > 0:
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
        if trace_recorder is not None:
            trace_recorder.record_initial(current_noise, current_tokens, current_result)
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
        if manifest.get("parameter_space", "latent") == "data":
            best_metadata_key = "original_space_local_gradient_adam"
        else:
            best_metadata_key = (
                "flow_prior_local_full_gradient_bfgs"
                if manifest["optimizer"] == "bfgs"
                else "flow_prior_local_full_gradient_adam"
            )
        best_result = best_payload[best_metadata_key]["native_score"]

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

    def decode_flow(states: np.ndarray) -> tuple[np.ndarray, float]:
        if args.parameter_space == "data":
            started = time.perf_counter()
            decoded = normalizer.inverse(
                states, (args.nfp, args.n_base_coils)
            ).astype(np.float64, copy=False)
            return decoded, time.perf_counter() - started
        torch.cuda.set_device(args.flow_device)
        decoded, wall_s = decode_noise_rk4(
            model,
            normalizer,
            states,
            nfp=args.nfp,
            steps=args.flow_steps,
            device=flow_device,
        )
        if args.flow_device == args.score_device:
            torch.cuda.empty_cache()
        return decoded, wall_s

    for iteration in range(start_iteration + 1, args.iterations + 1):
        elapsed_total = prior_wall_s + time.perf_counter() - run_started
        reserve = 1.25 * max(recent_walls, default=30.0)
        if elapsed_total + reserve >= args.max_wall_s:
            stop_reason = "max_wall_s"
            break
        iteration_started = time.perf_counter()
        probe_noise = current_noise.copy()
        probe_tokens = current_tokens.copy()
        trace_first_moment_before = first_moment.copy()
        trace_second_moment_before = second_moment.copy()

        endpoint_decode_cache_hit = prefetched_gradient is not None
        prefetched_endpoint_decode_wall_s = 0.0
        if prefetched_gradient is not None:
            if int(prefetched_gradient["source_iteration"]) != iteration - 1:
                raise RuntimeError("prefetched gradient endpoints do not match iteration")
            directions = prefetched_gradient["directions"]
            endpoint_tokens = prefetched_gradient["tokens"]
            prefetched_endpoint_decode_wall_s = float(
                prefetched_gradient["decode_wall_s"]
            )
            endpoint_decode_wall_s = 0.0
            endpoint_count = len(endpoint_tokens)
            prefetched_gradient = None
        else:
            directions, endpoints = gradient_probe(
                current_noise,
                mode=args.gradient_mode,
                perturbation=args.perturbation,
                random_direction_count=args.random_directions,
                rng=direction_rng,
            )
            endpoint_tokens, endpoint_decode_wall_s = decode_flow(endpoints)
            endpoint_count = len(endpoints)
        local_scores, captured_result, gradient_details, local_results = estimator.evaluate(
            current_tokens,
            endpoint_tokens,
            current_result,
        )
        current_result = captured_result
        statuses = gradient_details["status_counts"]
        all_endpoints_valid = statuses == {"ok": endpoint_count}
        if directions is None:
            raw_gradient = coordinate_gradient(local_scores, args.perturbation).reshape(
                current_noise.shape
            )
        else:
            raw_gradient = random_direction_gradient(
                local_scores, args.perturbation, directions
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
        previous_inverse_hessian = inverse_hessian.copy()
        previous_bfgs_gradient_state = previous_bfgs_gradient.copy()
        previous_bfgs_step_state = previous_bfgs_step.copy()
        previous_bfgs_trust_rms = bfgs_trust_rms
        previous_adam_step = adam_step
        bfgs_update = {"updated": False, "reason": "not_applicable"}
        if gradient_step_applied:
            if args.optimizer == "adam":
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
            else:
                candidate_step = adam_step + 1
                candidate_first = first_moment
                candidate_second = second_moment
                if previous_bfgs_gradient.size:
                    inverse_hessian, bfgs_update = damped_inverse_bfgs(
                        inverse_hessian,
                        previous_bfgs_step,
                        previous_bfgs_gradient - raw_gradient.reshape(-1),
                    )
                    previous_bfgs_gradient = np.empty(0, dtype=np.float64)
                    previous_bfgs_step = np.empty(0, dtype=np.float64)
                direction = (inverse_hessian @ raw_gradient.reshape(-1)).reshape(
                    current_noise.shape
                )
                direction_rms = rms(direction)
                proposed_update = direction * (
                    bfgs_trust_rms / max(direction_rms, np.finfo(np.float64).eps)
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
        pipeline_decode_wall_s = 0.0
        pipeline_decode_batch_sizes: list[int] = []
        pipeline_wasted_endpoint_count = 0
        proposal_attempts: list[dict[str, Any]] = []
        applied_update = np.zeros_like(proposed_update)
        next_directions: np.ndarray | None = None
        if args.flow_pipeline and iteration < args.iterations:
            next_directions, next_endpoints_template = gradient_probe(
                current_noise,
                mode=args.gradient_mode,
                perturbation=args.perturbation,
                random_direction_count=args.random_directions,
                rng=direction_rng,
            )
        else:
            next_endpoints_template = None
        if gradient_step_applied:
            for fraction in (1.0,) + backtracking:
                trial_noise = (
                    previous_noise.astype(np.float64) + fraction * proposed_update
                ).astype(np.float32)
                if args.flow_pipeline and iteration < args.iterations:
                    if next_directions is None:
                        trial_endpoints = endpoint_latents(
                            trial_noise, args.perturbation
                        )
                    else:
                        trial_endpoints = random_direction_endpoints(
                            trial_noise, args.perturbation, next_directions
                        )
                    trial_states = np.concatenate(
                        (trial_noise[None], trial_endpoints), axis=0
                    )
                else:
                    trial_endpoints = None
                    trial_states = trial_noise[None]
                trial_tokens_batch, decode_wall_s = decode_flow(trial_states)
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
                pipeline_decode_wall_s += decode_wall_s
                pipeline_decode_batch_sizes.append(len(trial_states))
                proposal_score_wall_s += score_wall_s
                proposal_attempts.append(
                    {
                        "fraction": fraction,
                        "status": trial_result.get("status"),
                        "score": result_score(trial_result),
                    }
                )
                score_acceptable = (
                    args.optimizer == "adam"
                    or result_score(trial_result)
                    >= result_score(previous_result) + args.bfgs_min_improvement
                )
                if result_valid(trial_result) and score_acceptable:
                    current_noise = trial_noise
                    current_tokens = trial_tokens_batch[0]
                    current_result = trial_result
                    first_moment = candidate_first
                    second_moment = candidate_second
                    adam_step = candidate_step
                    applied_update = fraction * proposed_update
                    if args.optimizer == "bfgs":
                        previous_bfgs_gradient = raw_gradient.reshape(-1).copy()
                        previous_bfgs_step = applied_update.reshape(-1).copy()
                        if fraction == 1.0:
                            bfgs_trust_rms = min(
                                args.bfgs_max_trust_rms,
                                args.bfgs_trust_growth * bfgs_trust_rms,
                            )
                        else:
                            bfgs_trust_rms = max(
                                args.bfgs_min_trust_rms, fraction * bfgs_trust_rms
                            )
                    center_update_accepted = True
                    center_acceptance_fraction = fraction
                    if trial_endpoints is not None:
                        prefetched_gradient = {
                            "directions": next_directions,
                            "tokens": trial_tokens_batch[1:],
                            "decode_wall_s": decode_wall_s,
                            "source_iteration": iteration,
                        }
                    break
                if trial_endpoints is not None:
                    pipeline_wasted_endpoint_count += len(trial_endpoints)
        if not center_update_accepted:
            current_noise = previous_noise
            current_tokens = previous_tokens
            current_result = previous_result
            first_moment = previous_first
            second_moment = previous_second
            if args.optimizer == "adam":
                inverse_hessian = previous_inverse_hessian
                previous_bfgs_gradient = previous_bfgs_gradient_state
                previous_bfgs_step = previous_bfgs_step_state
                bfgs_trust_rms = previous_bfgs_trust_rms
            elif gradient_step_applied:
                bfgs_trust_rms = max(
                    args.bfgs_min_trust_rms,
                    args.bfgs_trust_shrink * previous_bfgs_trust_rms,
                )
            else:
                inverse_hessian = previous_inverse_hessian
                previous_bfgs_gradient = previous_bfgs_gradient_state
                previous_bfgs_step = previous_bfgs_step_state
                bfgs_trust_rms = previous_bfgs_trust_rms
            adam_step = previous_adam_step

        if args.flow_pipeline and iteration < args.iterations and prefetched_gradient is None:
            if next_directions is None:
                if next_endpoints_template is None:
                    raise RuntimeError("next coordinate endpoints were not prepared")
                next_endpoints = endpoint_latents(current_noise, args.perturbation)
            else:
                next_endpoints = random_direction_endpoints(
                    current_noise, args.perturbation, next_directions
                )
            next_tokens, refill_wall_s = decode_flow(next_endpoints)
            proposal_decode_wall_s += refill_wall_s
            pipeline_decode_wall_s += refill_wall_s
            pipeline_decode_batch_sizes.append(len(next_endpoints))
            prefetched_gradient = {
                "directions": next_directions,
                "tokens": next_tokens,
                "decode_wall_s": refill_wall_s,
                "source_iteration": iteration,
            }

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
            "gradient_endpoint_count": endpoint_count,
            "gradient_mode": args.gradient_mode,
            "random_direction_count": 0 if directions is None else len(directions),
            "gradient_endpoint_statuses": statuses,
            "gradient_step_applied": gradient_step_applied,
            "center_update_accepted": center_update_accepted,
            "center_acceptance_fraction": center_acceptance_fraction,
            "proposal_attempts": proposal_attempts,
            "adam_step": adam_step,
            "optimizer": args.optimizer,
            "bfgs_update": bfgs_update,
            "bfgs_trust_rms": bfgs_trust_rms,
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
            "endpoint_decode_cache_hit": endpoint_decode_cache_hit,
            "prefetched_endpoint_decode_wall_s": prefetched_endpoint_decode_wall_s,
            "gradient_pipeline": gradient_details,
            "gradient_wall_s": endpoint_decode_wall_s
            + gradient_details["timing_s"]["total"],
            "proposal_decode_wall_s": proposal_decode_wall_s,
            "flow_pipeline_decode_wall_s": pipeline_decode_wall_s,
            "flow_pipeline_decode_batch_sizes": pipeline_decode_batch_sizes,
            "flow_pipeline_wasted_endpoint_count": pipeline_wasted_endpoint_count,
            "proposal_score_wall_s": proposal_score_wall_s,
            "iteration_wall_s": iteration_wall_s,
            "total_wall_s": prior_wall_s + time.perf_counter() - run_started,
        }
        if trace_recorder is not None:
            trace_recorder.record_step(
                iteration=iteration,
                probe_noise=probe_noise,
                probe_tokens=probe_tokens,
                directions=(
                    np.empty((0,) + current_noise.shape, dtype=np.float32)
                    if directions is None
                    else np.asarray(directions, dtype=np.float32)
                ),
                endpoint_tokens=np.asarray(endpoint_tokens, dtype=np.float32),
                local_results=local_results,
                probe_result=captured_result,
                raw_gradient=np.asarray(raw_gradient, dtype=np.float32),
                first_moment_before=np.asarray(trace_first_moment_before, dtype=np.float32),
                second_moment_before=np.asarray(trace_second_moment_before, dtype=np.float32),
                first_moment_after=np.asarray(first_moment, dtype=np.float32),
                second_moment_after=np.asarray(second_moment, dtype=np.float32),
                proposed_update=np.asarray(proposed_update, dtype=np.float32),
                applied_update=np.asarray(applied_update, dtype=np.float32),
                center_after_noise=np.asarray(current_noise, dtype=np.float32),
                center_after_tokens=np.asarray(current_tokens, dtype=np.float32),
                center_result=current_result,
                gradient_step_applied=gradient_step_applied,
                center_update_accepted=center_update_accepted,
                center_acceptance_fraction=center_acceptance_fraction,
                adam_step=adam_step,
            )
        if args.trajectory_every > 0 and (
            iteration % args.trajectory_every == 0 or iteration == args.iterations
        ):
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
        if iteration % args.progress_every == 0 or iteration == args.iterations:
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
        prefetched_directions = (
            np.empty((0,), dtype=np.float32)
            if prefetched_gradient is None or prefetched_gradient["directions"] is None
            else np.asarray(prefetched_gradient["directions"], dtype=np.float32)
        )
        prefetched_tokens = (
            np.empty((0, args.n_base_coils, TOKEN_DIM), dtype=np.float64)
            if prefetched_gradient is None
            else np.asarray(prefetched_gradient["tokens"], dtype=np.float64)
        )
        if iteration % args.state_every == 0 or iteration == args.iterations:
            np.savez_compressed(
                paths["state_latest.npz"],
                current_noise=current_noise,
                best_noise=best_noise,
                first_moment=first_moment,
                second_moment=second_moment,
                inverse_hessian=inverse_hessian,
                previous_bfgs_gradient=previous_bfgs_gradient,
                previous_bfgs_step=previous_bfgs_step,
                bfgs_trust_rms=np.asarray(bfgs_trust_rms, dtype=np.float64),
                direction_rng_state=np.asarray(
                    json.dumps(direction_rng.bit_generator.state)
                ),
                prefetched_gradient_present=np.asarray(prefetched_gradient is not None),
                prefetched_gradient_directions=prefetched_directions,
                prefetched_gradient_tokens=prefetched_tokens,
                prefetched_gradient_decode_wall_s=np.asarray(
                    0.0
                    if prefetched_gradient is None
                    else prefetched_gradient["decode_wall_s"],
                    dtype=np.float64,
                ),
                prefetched_gradient_source_iteration=np.asarray(
                    -1
                    if prefetched_gradient is None
                    else prefetched_gradient["source_iteration"],
                    dtype=np.int64,
                ),
                iteration=np.asarray(iteration, dtype=np.int64),
                adam_step=np.asarray(adam_step, dtype=np.int64),
            )
        if args.plot_every > 0 and (iteration == 1 or iteration % args.plot_every == 0):
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
    if args.plot_every > 0:
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
        "completed_adam_steps": adam_step if args.optimizer == "adam" else 0,
        "completed_optimizer_steps": adam_step,
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
    if trace_recorder is not None:
        summary["training_trace"] = trace_recorder.finalize()
    write_json(paths["summary.json"], summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
