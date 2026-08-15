from __future__ import annotations

import argparse
import json
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

from scripts.optimize_flow_prior_local_full_gradient_adam import (  # noqa: E402
    LocalFullGradientEstimator,
    recorded_native_result,
    result_score,
    result_valid,
    score_center,
)
from scripts.optimize_flow_prior_zo_adam import (  # noqa: E402
    decode_noise_rk4,
    load_flow_checkpoint,
    load_initial_noise,
    orthogonal_directions,
)
from scripts.optimize_native_score_cem import file_sha256, write_json  # noqa: E402


def parse_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(item) for item in value.split(",") if item.strip())
    if not counts or any(count <= 0 or count % 2 for count in counts):
        raise ValueError("endpoint counts must be positive even integers")
    return counts


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    stages = sorted(trials[0]["timing_s"])
    timing = {
        stage: {
            "median_s": percentile([row["timing_s"][stage] for row in trials], 50),
            "p95_s": percentile([row["timing_s"][stage] for row in trials], 95),
        }
        for stage in stages
    }
    return {
        "repeat_count": len(trials),
        "status_counts": trials[-1]["status_counts"],
        "timing": timing,
        "psi_stats": trials[-1]["psi_stats"],
        "local_stats": trials[-1]["local_stats"],
    }


def add_throughput_metrics(
    row: dict[str, Any],
    *,
    endpoint_count: int,
    nfp: int,
    n_coils: int,
    segments_per_coil: int,
    psi_iterations: int,
    psi_basis_count: int,
) -> None:
    total_s = row["timing"]["total"]["median_s"]
    row["endpoints_per_s"] = endpoint_count / total_s
    row["amortized_ms_per_endpoint"] = 1.0e3 * total_s / endpoint_count

    # These are explicit operation models, not hardware-counter measurements.
    # The flux model counts 28 ordinary FP operations per segment-point
    # Biot-Savart interaction and excludes reciprocal-square-root weighting.
    local = row["local_stats"]
    segment_count = nfp * n_coils * segments_per_coil
    flux_flops = (
        endpoint_count
        * int(local["flux_points_per_query"])
        * segment_count
        * 28
    )
    flux_s = row["timing"]["local_score"]["median_s"] * (
        float(local["flux_s"]) / max(float(local["total_s"]), 1.0e-30)
    )
    row["flux_nominal_tflops"] = flux_flops / max(flux_s, 1.0e-30) / 1.0e12
    row["flux_segment_point_giga_per_s"] = (
        endpoint_count
        * int(local["flux_points_per_query"])
        * segment_count
        / max(flux_s, 1.0e-30)
        / 1.0e9
    )

    psi = row["psi_stats"]
    # One PCGLS iteration applies A and A^T. Counting multiply-add as two FLOPs
    # gives 4*m*n operations per iteration. Basis construction and the shared
    # preconditioner are deliberately not included.
    psi_flops = (
        endpoint_count
        * 4
        * psi_iterations
        * int(psi["point_count"])
        * psi_basis_count
    )
    psi_s = row["timing"]["psi"]["median_s"]
    row["psi_matvec_equivalent_tflops"] = (
        psi_flops / max(psi_s, 1.0e-30) / 1.0e12
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-GPU local-gradient endpoint batch scaling benchmark."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-case", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint-counts", default="2,4,8,16,32,64,128,256,600")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--flow-steps", type=int, default=128)
    parser.add_argument("--perturbation", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=2026081501)
    parser.add_argument("--nfp", type=int, default=4)
    parser.add_argument("--n-base-coils", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--segments-per-coil", type=int, default=256)
    parser.add_argument("--psi-iterations", type=int, default=4)
    parser.add_argument("--alpha-iterations", type=int, default=4)
    parser.add_argument("--formal-surface-theta-count", type=int, default=128)
    parser.add_argument("--local-surface-theta-count", type=int, default=64)
    parser.add_argument("--iota-degree", type=int, default=3)
    args = parser.parse_args()

    counts = parse_counts(args.endpoint_counts)
    dimension = args.n_base_coils * 100
    if max(counts) > 2 * dimension:
        raise ValueError("endpoint count exceeds the centered full-dimensional limit")
    if args.repeats < 1 or args.perturbation <= 0.0:
        raise ValueError("repeats and perturbation must be positive")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
    initial_noise, initial_payload = load_initial_noise(args.initial_case)
    decoded, initial_flow_s = decode_noise_rk4(
        model,
        normalizer,
        initial_noise[None],
        nfp=args.nfp,
        steps=args.flow_steps,
        device=device,
    )
    center_tokens = decoded[0]
    normalized, clipped_fraction = normalizer.transform(
        center_tokens[None], (args.nfp, args.n_base_coils)
    )
    center_parameter = normalized[0]
    center_tokens = normalizer.inverse(
        center_parameter[None], (args.nfp, args.n_base_coils)
    )[0].astype(np.float64, copy=False)
    center_result, center_score_s = score_center(
        args.lib,
        center_tokens,
        nfp=args.nfp,
        score_device=args.device,
        iota_degree=args.iota_degree,
        surface_theta_count=args.formal_surface_theta_count,
        previous_result=recorded_native_result(initial_payload),
    )
    if not result_valid(center_result):
        raise RuntimeError(f"initial center failed: {center_result.get('status')}")

    estimator = LocalFullGradientEstimator(
        args.lib,
        nfp=args.nfp,
        score_device=args.device,
        segments_per_coil=args.segments_per_coil,
        psi_iterations=args.psi_iterations,
        alpha_iterations=args.alpha_iterations,
        formal_surface_theta_count=args.formal_surface_theta_count,
        local_surface_theta_count=args.local_surface_theta_count,
        iota_degree=args.iota_degree,
    )
    rng = np.random.default_rng(args.seed)
    directions = orthogonal_directions(rng, center_parameter.shape, dimension)
    rows = []
    for endpoint_count in counts:
        direction_count = endpoint_count // 2
        selected = directions[:direction_count]
        parameters = np.repeat(center_parameter[None], endpoint_count, axis=0)
        parameters[0::2] -= np.float32(args.perturbation) * selected
        parameters[1::2] += np.float32(args.perturbation) * selected
        endpoint_tokens = normalizer.inverse(
            parameters, (args.nfp, args.n_base_coils)
        ).astype(np.float64, copy=False)

        trials = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            _, _, details, local_results = estimator.evaluate(
                center_tokens, endpoint_tokens, center_result
            )
            outer_s = time.perf_counter() - started
            status_counts: dict[str, int] = {}
            for result in local_results:
                status = str(result.get("status"))
                status_counts[status] = status_counts.get(status, 0) + 1
            trials.append(
                {
                    "outer_wall_s": outer_s,
                    "timing_s": details["timing_s"],
                    "psi_stats": details["psi_stats"],
                    "local_stats": details["local_stats"],
                    "status_counts": status_counts,
                }
            )
        row = {
            "endpoint_count": endpoint_count,
            "direction_count": direction_count,
            **summarize_trials(trials),
        }
        add_throughput_metrics(
            row,
            endpoint_count=endpoint_count,
            nfp=args.nfp,
            n_coils=args.n_base_coils,
            segments_per_coil=args.segments_per_coil,
            psi_iterations=args.psi_iterations,
            psi_basis_count=len(estimator.mode_a),
        )
        rows.append(row)

    output = {
        "format": "local_gradient_batch_scaling_v1",
        "precision": {
            "query_batch_field_and_iterative_fits": "fp32",
            "formal_center_and_axis_control": "mixed fp32/fp64",
        },
        "hardware_counter_tflops_available": False,
        "tflops_note": (
            "Reported flux and psi TFLOPS use explicit operation models. They are "
            "not Nsight hardware counters and must not be compared as exact achieved FLOPS."
        ),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "library_sha256": file_sha256(args.lib),
        "initial_case_sha256": file_sha256(args.initial_case),
        "initial_flow_s": initial_flow_s,
        "initial_formal_score_s": center_score_s,
        "center_score": result_score(center_result),
        "initial_clipped_fraction": float(clipped_fraction),
        "configuration": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "endpoint_counts": list(counts),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, output)


if __name__ == "__main__":
    main()
