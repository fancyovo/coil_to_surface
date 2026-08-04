from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
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
from scripts.optimize_flow_prior_zo_adam import (
    cosine_similarity,
    decode_noise_rk4,
    load_flow_checkpoint,
)
from scripts.qh_blackbox_gradient_reference import (
    branch_fingerprint,
    compact_result,
    file_sha256,
    load_rows,
    write_json,
)
from scripts.qh_score_noise_sensitivity import perturbation_metrics
from stellarator_gpu import score_coils_g2_gradient_native


DEFAULT_STEPS = (32, 64, 128, 256)


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("steps must be positive integers")
    return result


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    left = np.asarray(value, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def token_cotangent(gradient: dict[str, np.ndarray]) -> np.ndarray:
    shape = gradient["x"].shape
    output = np.empty((shape[0], 100), dtype=np.float32)
    output[:, :33] = gradient["x"]
    output[:, 33:66] = gradient["y"]
    output[:, 66:99] = gradient["z"]
    output[:, 99] = gradient["current"]
    return output


def score_arguments(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(tokens, dtype=np.float64)
    return values[:, :33], values[:, 33:66], values[:, 66:99], values[:, 99]


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_vjp(
    model,
    normalizer,
    noise: np.ndarray,
    cotangent: np.ndarray,
    *,
    nfp: int,
    device: torch.device,
    steps: int,
    checkpoint_steps: int,
    use_checkpoint: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    clear_cuda()
    physical, gradient, diagnostics = decode_physical_vjp(
        model,
        normalizer,
        noise,
        cotangent,
        nfp=nfp,
        device=device,
        rk4_steps=steps,
        checkpoint_steps=checkpoint_steps,
        use_checkpoint=use_checkpoint,
    )
    return physical, gradient, asdict(diagnostics)


def profile_vjp(
    model,
    normalizer,
    noise: np.ndarray,
    cotangent: np.ndarray,
    *,
    nfp: int,
    device: torch.device,
    steps: int,
    checkpoint_steps: int,
) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    clear_cuda()
    started = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        _, _, diagnostics = decode_physical_vjp(
            model,
            normalizer,
            noise,
            cotangent,
            nfp=nfp,
            device=device,
            rk4_steps=steps,
            checkpoint_steps=checkpoint_steps,
            use_checkpoint=False,
        )
    wall_s = time.perf_counter() - started

    def device_time_us(row: Any) -> float:
        for name in ("self_device_time_total", "self_cuda_time_total"):
            value = getattr(row, name, None)
            if value is not None:
                return float(value)
        return 0.0

    averages = list(profiler.key_averages())
    device_sorted = sorted(averages, key=device_time_us, reverse=True)
    cpu_sorted = sorted(
        averages,
        key=lambda row: float(getattr(row, "self_cpu_time_total", 0.0)),
        reverse=True,
    )
    events = list(profiler.events())
    cuda_events = [event for event in events if "CUDA" in str(getattr(event, "device_type", ""))]

    def compact_operator(row: Any) -> dict[str, Any]:
        return {
            "name": str(row.key),
            "count": int(row.count),
            "self_device_time_s": device_time_us(row) * 1.0e-6,
            "self_cpu_time_s": float(getattr(row, "self_cpu_time_total", 0.0)) * 1.0e-6,
        }

    return {
        "steps": int(steps),
        "mode": "retained_activations",
        "profiled_wall_s": float(wall_s),
        "flow_diagnostics": asdict(diagnostics),
        "operator_self_device_time_sum_s": float(
            sum(device_time_us(row) for row in averages) * 1.0e-6
        ),
        "operator_self_cpu_time_sum_s": float(
            sum(float(getattr(row, "self_cpu_time_total", 0.0)) for row in averages)
            * 1.0e-6
        ),
        "cuda_event_count": len(cuda_events),
        "profiler_event_count": len(events),
        "top_device_operators": [compact_operator(row) for row in device_sorted[:15]],
        "top_cpu_operators": [compact_operator(row) for row in cpu_sorted[:15]],
        "interpretation": (
            "Profiler operator sums are approximate serialized kernel work; compare them with "
            "the profiled wall time only to diagnose launch/synchronization gaps. Use the "
            "unprofiled FlowVjpDiagnostics timings for performance decisions."
        ),
    }


def reference_position_scale(
    reference_dir: Path,
    cases: list[dict[str, Any]],
    raw: np.ndarray,
    center_index: int,
    center_tokens: np.ndarray,
    perturbation_scale: float,
) -> float:
    endpoints = [
        row
        for row in cases
        if row["kind"] == "endpoint"
        and int(row["center_index"]) == center_index
        and math.isclose(
            float(row["scale"]), perturbation_scale, rel_tol=0.0, abs_tol=1.0e-12
        )
    ]
    if not endpoints:
        raise RuntimeError(
            f"reference {reference_dir} has no endpoints at scale {perturbation_scale}"
        )
    displacements = [
        perturbation_metrics(np.asarray(raw[int(row["case_id"])]), center_tokens)[
            "position_delta_rms_m"
        ]
        for row in endpoints
    ]
    return float(np.median(displacements))


def step_passes(row: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return bool(
        row["status_matches_reference"]
        and row["branch_matches_reference"]
        and row["position_error_to_perturbation"]
        <= thresholds["max_position_error_to_perturbation"]
        and row["score_abs_error"] <= thresholds["max_score_abs_error"]
        and row["component_abs_error_max"] <= thresholds["max_component_abs_error"]
        and row["fixed_cotangent_vjp_cosine"] >= thresholds["min_vjp_cosine"]
        and row["end_to_end_vjp_cosine"] >= thresholds["min_vjp_cosine"]
        and row["fixed_cotangent_vjp_relative_l2"]
        <= thresholds["max_fixed_vjp_relative_l2"]
    )


def evaluate_center(args: argparse.Namespace) -> None:
    manifest = json.loads((args.reference_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "complete":
        raise RuntimeError("reference directory must be complete")
    center_index = next(
        index
        for index, center in enumerate(manifest["centers"])
        if center["center_id"] == args.center_id
    )
    center = manifest["centers"][center_index]
    nfp = int(center["nfp"])
    cases = load_rows(args.reference_dir / "cases.jsonl")
    raw_reference = np.load(args.reference_dir / "raw_tokens.npy", mmap_mode="r")
    banks = np.load(args.reference_dir / "latent_banks.npz")
    noise = np.asarray(banks["centers"][center_index], dtype=np.float32)
    center_case = next(
        row
        for row in cases
        if row["kind"] == "center" and int(row["center_index"]) == center_index
    )
    formal_center_tokens = np.asarray(
        raw_reference[int(center_case["case_id"])], dtype=np.float64
    )
    perturbation_position_rms_m = reference_position_scale(
        args.reference_dir,
        cases,
        raw_reference,
        center_index,
        formal_center_tokens,
        args.perturbation_scale,
    )
    device = torch.device(args.device)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
    if args.compile_model:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        warmup_cotangent = np.ones_like(noise, dtype=np.float32)
        run_vjp(
            model,
            normalizer,
            noise,
            warmup_cotangent,
            nfp=nfp,
            device=device,
            steps=8,
            checkpoint_steps=8,
            use_checkpoint=False,
        )
    decoded_by_steps: dict[int, np.ndarray] = {}
    decode_wall_by_steps: dict[int, float] = {}
    native_by_steps: dict[int, dict[str, Any]] = {}
    cotangent_by_steps: dict[int, np.ndarray] = {}

    for steps in args.steps:
        decoded, decode_wall_s = decode_noise_rk4(
            model,
            normalizer,
            noise[None],
            nfp=nfp,
            steps=steps,
            device=device,
        )
        tokens = decoded[0]
        x, y, z, current = score_arguments(tokens)
        native = score_coils_g2_gradient_native(
            args.gradient_lib,
            x,
            y,
            z,
            current,
            nfp,
            target_helicity=(1, nfp),
        )
        decoded_by_steps[steps] = tokens
        decode_wall_by_steps[steps] = decode_wall_s
        native_by_steps[steps] = native
        cotangent_by_steps[steps] = token_cotangent(native["gradient"])

    reference_steps = max(args.steps)
    reference_tokens = decoded_by_steps[reference_steps]
    reference_score = native_by_steps[reference_steps]["score_result"]
    reference_cotangent = cotangent_by_steps[reference_steps]
    vjp_by_steps: dict[int, dict[str, Any]] = {}

    for steps in args.steps:
        order = (True, False) if center_index % 2 == 0 else (False, True)
        mode_results: dict[str, Any] = {}
        for use_checkpoint in order:
            key = "checkpoint" if use_checkpoint else "retained"
            try:
                physical, gradient, diagnostics = run_vjp(
                    model,
                    normalizer,
                    noise,
                    cotangent_by_steps[steps],
                    nfp=nfp,
                    device=device,
                    steps=steps,
                    checkpoint_steps=args.checkpoint_steps,
                    use_checkpoint=use_checkpoint,
                )
                mode_results[key] = {
                    "physical": physical,
                    "gradient": gradient,
                    "diagnostics": diagnostics,
                    "error": None,
                }
            except torch.cuda.OutOfMemoryError as exc:
                clear_cuda()
                mode_results[key] = {"error": f"{type(exc).__name__}: {exc}"}
        if mode_results["retained"].get("error") is not None:
            raise RuntimeError(
                f"retained-activation VJP failed at RK4-{steps}: "
                f"{mode_results['retained']['error']}"
            )
        if steps == reference_steps:
            fixed = mode_results["retained"]
        else:
            fixed_physical, fixed_gradient, fixed_diagnostics = run_vjp(
                model,
                normalizer,
                noise,
                reference_cotangent,
                nfp=nfp,
                device=device,
                steps=steps,
                checkpoint_steps=args.checkpoint_steps,
                use_checkpoint=False,
            )
            fixed = {
                "physical": fixed_physical,
                "gradient": fixed_gradient,
                "diagnostics": fixed_diagnostics,
                "error": None,
            }
        vjp_by_steps[steps] = {**mode_results, "fixed": fixed}

    reference_end_gradient = vjp_by_steps[reference_steps]["retained"]["gradient"]
    reference_fixed_gradient = vjp_by_steps[reference_steps]["fixed"]["gradient"]
    reference_fingerprint = branch_fingerprint(reference_score)
    rows = []
    thresholds = {
        "max_position_error_to_perturbation": args.max_position_error_to_perturbation,
        "max_score_abs_error": args.max_score_abs_error,
        "max_component_abs_error": args.max_component_abs_error,
        "min_vjp_cosine": args.min_vjp_cosine,
        "max_fixed_vjp_relative_l2": args.max_fixed_vjp_relative_l2,
    }
    for steps in args.steps:
        score = native_by_steps[steps]["score_result"]
        integration = perturbation_metrics(decoded_by_steps[steps], reference_tokens)
        checkpoint_result = vjp_by_steps[steps]["checkpoint"]
        retained_result = vjp_by_steps[steps]["retained"]
        fixed_result = vjp_by_steps[steps]["fixed"]
        components = score["components"]
        component_error = max(
            abs(float(components[name]) - float(reference_score["components"][name]))
            for name in reference_score["components"]
        )
        row = {
            "steps": int(steps),
            "network_evaluations_forward": int(4 * steps),
            "no_grad_decode_wall_s": float(decode_wall_by_steps[steps]),
            "native_forward_wall_s": float(
                native_by_steps[steps]["gradient_diagnostics"]["forward_wall_s"]
            ),
            "native_reverse_wall_s": float(
                native_by_steps[steps]["gradient_diagnostics"]["gradient_wall_s"]
            ),
            "score": compact_result(score),
            "status_matches_reference": score["status"] == reference_score["status"],
            "branch_matches_reference": branch_fingerprint(score) == reference_fingerprint,
            "score_abs_error": abs(float(score["score"]) - float(reference_score["score"])),
            "component_abs_error_max": float(component_error),
            "position_error_rms_m": integration["position_delta_rms_m"],
            "position_error_to_perturbation": float(
                integration["position_delta_rms_m"]
                / max(perturbation_position_rms_m, 1.0e-30)
            ),
            "coefficient_relative_l2": integration["coefficient_relative_l2"],
            "current_relative_l2": integration["current_relative_l2"],
            "fixed_cotangent_vjp_cosine": cosine_similarity(
                fixed_result["gradient"], reference_fixed_gradient
            ),
            "fixed_cotangent_vjp_relative_l2": relative_l2(
                fixed_result["gradient"], reference_fixed_gradient
            ),
            "end_to_end_vjp_cosine": cosine_similarity(
                retained_result["gradient"], reference_end_gradient
            ),
            "end_to_end_vjp_relative_l2": relative_l2(
                retained_result["gradient"], reference_end_gradient
            ),
            "end_to_end_vjp_norm_ratio": rms(retained_result["gradient"])
            / max(rms(reference_end_gradient), 1.0e-30),
            "checkpoint": {
                "error": checkpoint_result.get("error"),
                "diagnostics": checkpoint_result.get("diagnostics"),
            },
            "retained": {"diagnostics": retained_result["diagnostics"]},
            "checkpoint_to_retained_physical_relative_l2": (
                float("nan")
                if checkpoint_result.get("error") is not None
                else relative_l2(checkpoint_result["physical"], retained_result["physical"])
            ),
            "checkpoint_to_retained_vjp_relative_l2": (
                float("nan")
                if checkpoint_result.get("error") is not None
                else relative_l2(checkpoint_result["gradient"], retained_result["gradient"])
            ),
            "checkpoint_to_retained_vjp_cosine": (
                float("nan")
                if checkpoint_result.get("error") is not None
                else cosine_similarity(checkpoint_result["gradient"], retained_result["gradient"])
            ),
        }
        row["passed"] = step_passes(row, thresholds)
        rows.append(row)

    selected_steps = next(
        (int(row["steps"]) for row in rows if row["passed"]), reference_steps
    )
    profiler_summary = None
    if args.profile:
        profiler_summary = profile_vjp(
            model,
            normalizer,
            noise,
            cotangent_by_steps[selected_steps],
            nfp=nfp,
            device=device,
            steps=selected_steps,
            checkpoint_steps=args.checkpoint_steps,
        )
    output = {
        "format": "qh_flow_vjp_benchmark_center_v1",
        "center_id": args.center_id,
        "center_index": center_index,
        "nfp": nfp,
        "n_coils": int(center["n_coils"]),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "gradient_library_sha256": file_sha256(args.gradient_lib),
        "compile_model": bool(args.compile_model),
        "reference_dir": str(args.reference_dir),
        "reference_steps": reference_steps,
        "perturbation_scale": float(args.perturbation_scale),
        "perturbation_position_rms_m": perturbation_position_rms_m,
        "rk4_reference_to_formal_center": perturbation_metrics(
            reference_tokens, formal_center_tokens
        ),
        "thresholds": thresholds,
        "selected_steps": selected_steps,
        "rows": rows,
        "profile": profiler_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"center_{args.center_id}.json", output)
    np.savez_compressed(
        args.output_dir / f"gradients_{args.center_id}.npz",
        steps=np.asarray(args.steps, dtype=np.int64),
        **{
            f"retained_{steps}": np.asarray(
                vjp_by_steps[steps]["retained"]["gradient"], dtype=np.float32
            )
            for steps in args.steps
        },
    )


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def analyze(args: argparse.Namespace) -> None:
    center_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.output_dir.glob("center_*.json"))
    ]
    if len(center_rows) != args.expected_centers:
        raise RuntimeError(
            f"expected {args.expected_centers} center results, found {len(center_rows)}"
        )
    steps = [int(row["steps"]) for row in center_rows[0]["rows"]]
    aggregate_rows = []
    for step in steps:
        selected = [
            next(row for row in center["rows"] if int(row["steps"]) == step)
            for center in center_rows
        ]
        checkpoint_total = [
            row["checkpoint"]["diagnostics"]["decode_wall_s"]
            + row["checkpoint"]["diagnostics"]["backward_wall_s"]
            for row in selected
            if row["checkpoint"]["diagnostics"] is not None
        ]
        retained_total = [
            row["retained"]["diagnostics"]["decode_wall_s"]
            + row["retained"]["diagnostics"]["backward_wall_s"]
            for row in selected
        ]
        retained_incremental_memory = [
            (
                row["retained"]["diagnostics"]["total_peak_memory_allocated_bytes"]
                - row["retained"]["diagnostics"]["baseline_memory_allocated_bytes"]
            )
            / (1024.0**3)
            for row in selected
        ]
        checkpoint_incremental_memory = [
            (
                row["checkpoint"]["diagnostics"]["total_peak_memory_allocated_bytes"]
                - row["checkpoint"]["diagnostics"]["baseline_memory_allocated_bytes"]
            )
            / (1024.0**3)
            for row in selected
            if row["checkpoint"]["diagnostics"] is not None
        ]
        aggregate_rows.append(
            {
                "steps": step,
                "all_centers_passed": all(row["passed"] for row in selected),
                "max_position_error_to_perturbation": max(
                    row["position_error_to_perturbation"] for row in selected
                ),
                "max_score_abs_error": max(row["score_abs_error"] for row in selected),
                "max_component_abs_error": max(
                    row["component_abs_error_max"] for row in selected
                ),
                "min_fixed_cotangent_vjp_cosine": min(
                    row["fixed_cotangent_vjp_cosine"] for row in selected
                ),
                "max_fixed_cotangent_vjp_relative_l2": max(
                    row["fixed_cotangent_vjp_relative_l2"] for row in selected
                ),
                "min_end_to_end_vjp_cosine": min(
                    row["end_to_end_vjp_cosine"] for row in selected
                ),
                "max_end_to_end_vjp_relative_l2": max(
                    row["end_to_end_vjp_relative_l2"] for row in selected
                ),
                "max_checkpoint_to_retained_vjp_relative_l2": max(
                    row["checkpoint_to_retained_vjp_relative_l2"] for row in selected
                ),
                "checkpoint_flow_wall_median_s": percentile(checkpoint_total, 50),
                "checkpoint_flow_wall_p95_s": percentile(checkpoint_total, 95),
                "retained_flow_wall_median_s": percentile(retained_total, 50),
                "retained_flow_wall_p95_s": percentile(retained_total, 95),
                "retained_speedup_over_checkpoint_median": percentile(
                    [left / right for left, right in zip(checkpoint_total, retained_total)], 50
                ),
                "checkpoint_incremental_peak_memory_max_gib": max(
                    checkpoint_incremental_memory
                ),
                "retained_incremental_peak_memory_max_gib": max(
                    retained_incremental_memory
                ),
            }
        )
    selected_steps = next(
        (row["steps"] for row in aggregate_rows if row["all_centers_passed"]), max(steps)
    )
    reference_aggregate = next(row for row in aggregate_rows if row["steps"] == max(steps))
    selected_aggregate = next(row for row in aggregate_rows if row["steps"] == selected_steps)
    combined_speedup = (
        reference_aggregate["checkpoint_flow_wall_median_s"]
        / selected_aggregate["retained_flow_wall_median_s"]
    )
    summary = {
        "format": "qh_flow_vjp_benchmark_summary_v1",
        "center_count": len(center_rows),
        "centers": [center["center_id"] for center in center_rows],
        "reference_steps": max(steps),
        "selected_steps": selected_steps,
        "thresholds": center_rows[0]["thresholds"],
        "aggregate_rows": aggregate_rows,
        "combined_speedup_vs_checkpoint_rk4_256": combined_speedup,
        "profiles": [
            {"center_id": center["center_id"], **center["profile"]}
            for center in center_rows
            if center["profile"] is not None
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    plot_summary(summary, args.output_dir / "flow_vjp_accuracy_speed.png")


def plot_summary(summary: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["aggregate_rows"]
    steps = np.asarray([row["steps"] for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), constrained_layout=True)
    axes[0, 0].plot(
        steps,
        [max(row["max_position_error_to_perturbation"], 1.0e-12) for row in rows],
        "o-",
        label="position / optimizer perturbation",
    )
    axes[0, 0].plot(
        steps,
        [max(row["max_score_abs_error"] / 100.0, 1.0e-12) for row in rows],
        "s-",
        label="score error / 100",
    )
    axes[0, 0].set(yscale="log", ylabel="worst-case error", title="Forward accuracy")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        steps,
        [max(1.0 - row["min_fixed_cotangent_vjp_cosine"], 1.0e-12) for row in rows],
        "o-",
        label="fixed-cotangent 1-cos",
    )
    axes[0, 1].plot(
        steps,
        [max(1.0 - row["min_end_to_end_vjp_cosine"], 1.0e-12) for row in rows],
        "s-",
        label="end-to-end 1-cos",
    )
    axes[0, 1].set(yscale="log", ylabel="worst-case angular error", title="VJP accuracy")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        steps,
        [row["checkpoint_flow_wall_median_s"] for row in rows],
        "o-",
        label="checkpoint every 8 steps",
    )
    axes[1, 0].plot(
        steps,
        [row["retained_flow_wall_median_s"] for row in rows],
        "s-",
        label="retained activations",
    )
    axes[1, 0].set(ylabel="flow forward + VJP [s]", title="Single-sample latency")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        steps,
        [row["checkpoint_incremental_peak_memory_max_gib"] for row in rows],
        "o-",
        label="checkpoint",
    )
    axes[1, 1].plot(
        steps,
        [row["retained_incremental_peak_memory_max_gib"] for row in rows],
        "s-",
        label="retained activations",
    )
    axes[1, 1].set(ylabel="incremental allocated peak [GiB]", title="Peak activation memory")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.ravel():
        axis.set_xlabel("RK4 steps")
        axis.set_xscale("log", base=2)
        axis.set_xticks(steps, labels=[str(value) for value in steps])
        axis.grid(alpha=0.25)
    figure.suptitle("Flow VJP accuracy, latency, and memory on four optimization stages")
    figure.savefig(output, dpi=190)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RK4 accuracy and retained-activation flow VJP latency."
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gradient-lib", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--center-id")
    parser.add_argument("--steps", type=parse_ints, default=DEFAULT_STEPS)
    parser.add_argument("--checkpoint-steps", type=int, default=8)
    parser.add_argument("--perturbation-scale", type=float, default=0.00125)
    parser.add_argument("--max-position-error-to-perturbation", type=float, default=0.01)
    parser.add_argument("--max-score-abs-error", type=float, default=0.02)
    parser.add_argument("--max-component-abs-error", type=float, default=0.02)
    parser.add_argument("--min-vjp-cosine", type=float, default=0.995)
    parser.add_argument("--max-fixed-vjp-relative-l2", type=float, default=0.02)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--expected-centers", type=int, default=4)
    args = parser.parse_args()
    args.reference_dir = args.reference_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.gradient_lib is not None:
        args.gradient_lib = args.gradient_lib.expanduser().resolve()
    if args.analyze_only:
        return args
    if args.checkpoint is None or args.gradient_lib is None or args.center_id is None:
        parser.error("worker mode requires --checkpoint, --gradient-lib, and --center-id")
    if max(args.steps) != 256:
        parser.error("the current acceptance protocol requires RK4-256 as the reference")
    if any(step % args.checkpoint_steps != 0 for step in args.steps):
        parser.error("every RK4 step count must be divisible by --checkpoint-steps")
    return args


def main() -> None:
    args = parse_args()
    if args.analyze_only:
        analyze(args)
    else:
        evaluate_center(args)


if __name__ == "__main__":
    main()
