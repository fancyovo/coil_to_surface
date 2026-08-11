from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
for path in (REPO_ROOT, GPU_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stellarator_gpu import BatchCoilFieldGpu, CoilFieldGpu, coil_component_gradient_native
from stellarator_eval.axis import interp_periodic_hermite
from stellarator_eval.psi import build_modes


def score_arguments(tokens: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(tokens, dtype=np.float64)
    return values[..., :33], values[..., 33:66], values[..., 66:99], values[..., 99]


def relative_rms(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    denominator = max(float(np.linalg.norm(expected)), 1.0e-30)
    return float(np.linalg.norm(difference) / denominator)


def timed(function):
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def psi_training_points(
    axis: tuple[np.ndarray, ...],
    query: int,
    *,
    nfp: int,
    radius_scale: float,
    grid: int,
    rho_min: float,
) -> tuple[np.ndarray, ...]:
    offsets = np.linspace(-radius_scale, radius_scale, grid)
    phis = np.linspace(0.0, 2.0 * np.pi / nfp, grid, endpoint=False)
    dR, dZ, phi = np.meshgrid(offsets, offsets, phis, indexing="ij")
    keep = (np.hypot(dR, dZ) >= rho_min) & (np.hypot(dR, dZ) <= radius_scale)
    dR = dR[keep]
    dZ = dZ[keep]
    phi = phi[keep]
    phi_axis = np.linspace(0.0, 2.0 * np.pi / nfp, axis[0].shape[1], endpoint=False)
    axis_R, _ = interp_periodic_hermite(
        phi, phi_axis, axis[0][query], axis[2][query], nfp
    )
    axis_Z, _ = interp_periodic_hermite(
        phi, phi_axis, axis[1][query], axis[3][query], nfp
    )
    return axis_R + dR, axis_Z + dZ, phi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=600)
    parser.add_argument("--point-count", type=int, default=256)
    parser.add_argument("--segments-per-coil", type=int, default=256)
    parser.add_argument("--trace-steps", type=int, default=400)
    parser.add_argument("--axis-integration-steps", type=int, default=960)
    parser.add_argument("--axis-samples", type=int, default=240)
    parser.add_argument("--reference-count", type=int, default=4)
    parser.add_argument("--psi-degree", type=int, default=10)
    parser.add_argument("--psi-mtor", type=int, default=12)
    parser.add_argument("--psi-grid", type=int, default=48)
    parser.add_argument("--psi-radius", type=float, default=0.05)
    parser.add_argument("--psi-rho-min", type=float, default=0.002)
    parser.add_argument("--psi-ridge", type=float, default=1.0e-6)
    parser.add_argument("--psi-iterations", type=int, default=4)
    parser.add_argument("--surface-theta-count", type=int, default=64)
    parser.add_argument("--alpha-iterations", type=int, default=4)
    parser.add_argument("--full-reference-dir", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.candidate_dir / "candidates.json").read_text(encoding="utf-8"))
    arrays = np.load(args.candidate_dir / "candidates.npz")
    tokens = np.asarray(arrays["tokens"], dtype=np.float64)
    center = manifest["centers"][0]
    endpoint_rows = [
        row for row in manifest["candidates"]
        if row["kind"] == "endpoint" and np.isclose(row["scale"], 0.005)
    ]
    endpoint_rows.sort(key=lambda row: (int(row["direction_index"]), int(row["sign"])))
    if len(endpoint_rows) < args.query_count:
        raise ValueError(f"only {len(endpoint_rows)} matching endpoints are available")
    endpoint_indices = np.asarray([
        int(row["candidate_index"]) for row in endpoint_rows[: args.query_count]
    ])
    center_x, center_y, center_z, center_current = score_arguments(tokens[0])
    x, y, z, current = score_arguments(tokens[endpoint_indices])
    nfp = int(center["nfp"])

    rng = np.random.default_rng(2026081201)
    radius = rng.uniform(0.005, 0.06, size=(args.query_count, args.point_count))
    angle = rng.uniform(0.0, 2.0 * np.pi, size=(args.query_count, args.point_count))
    phi = rng.uniform(0.0, 2.0 * np.pi / nfp, size=(args.query_count, args.point_count))
    cylindrical_R = float(center["axis_R"]) + radius * np.cos(angle)
    cylindrical_Z = float(center["axis_Z"]) + radius * np.sin(angle)
    points = np.stack(
        (cylindrical_R * np.cos(phi), cylindrical_R * np.sin(phi), cylindrical_Z), axis=-1
    ).astype(np.float32)
    line_offsets = np.asarray([-0.02, -0.01, 0.0, 0.01, 0.02], dtype=np.float64)
    R0 = np.broadcast_to(float(center["axis_R"]) + line_offsets, (args.query_count, 5)).copy()
    Z0 = np.broadcast_to(float(center["axis_Z"]) + line_offsets[::-1], (args.query_count, 5)).copy()
    axis_R0 = np.full(args.query_count, float(center["axis_R"]), dtype=np.float64)
    axis_Z0 = np.full(args.query_count, float(center["axis_Z"]), dtype=np.float64)

    batch, create_wall_s = timed(lambda: BatchCoilFieldGpu(
        args.lib, x, y, z, current, nfp,
        segments_per_coil=args.segments_per_coil, device_id=0,
    ))
    try:
        center_capture, center_capture_wall_s = timed(lambda: batch.capture_psi_center(
            center_x, center_y, center_z, center_current,
            target_helicity=(1, nfp),
            config_overrides={
                "iota_degree": 3,
                "surface_selection_mode": 1,
                "surface_confidence_periods": 1,
                "surface_theta_count": 128,
                "surface_trace_steps": 400,
                "surface_flux_bisection_iters": 6,
                "axis_hint_enabled": 1,
                "axis_hint_require_continuation": 2,
                "axis_hint_R": float(center["axis_R"]),
                "axis_hint_Z": float(center["axis_Z"]),
            },
        ))
        batch_B, eval_B_wall_s = timed(lambda: batch.eval_B(points))
        (batch_B_grad, batch_gradient), eval_B_grad_wall_s = timed(
            lambda: batch.eval_B_grad(points)
        )
        (batch_R1, batch_Z1), trace_wall_s = timed(
            lambda: batch.trace_period(R0, Z0, steps=args.trace_steps)
        )
        refined_axis, refine_axis_wall_s = timed(lambda: batch.refine_axis_hint(
            axis_R0, axis_Z0,
            trace_steps=args.axis_integration_steps,
            newton_iterations=6,
            finite_difference_step=2.0e-4,
            maximum_newton_step=0.25,
            residual_tolerance=1.0e-7,
            hint_max_distance=0.08,
        ))
        batch_axis, axis_wall_s = timed(lambda: batch.trace_axis_samples(
            refined_axis["R"], refined_axis["Z"],
            integration_steps=args.axis_integration_steps,
            sample_count=args.axis_samples,
        ))
        modes = build_modes(args.psi_degree, args.psi_mtor)
        mode_a = np.asarray([mode.a for mode in modes], dtype=np.int32)
        mode_b = np.asarray([mode.b for mode in modes], dtype=np.int32)
        mode_m = np.asarray([mode.m for mode in modes], dtype=np.int32)
        mode_kind = np.asarray(
            [0 if mode.kind == "cos" else 1 for mode in modes], dtype=np.int32
        )
        center_psi = center_capture["psi_coefficients"]
        if center_psi.size != len(modes):
            raise RuntimeError(
                f"center psi coefficient count {center_psi.size} != mode count {len(modes)}"
            )
        (batch_psi, batch_psi_rms, batch_psi_stats), batch_psi_wall_s = timed(
            lambda: batch.fit_psi_pcgls(
                *batch_axis,
                mode_a, mode_b, mode_m, mode_kind,
                center_psi,
                radius_scale=args.psi_radius,
                radial_grid=args.psi_grid,
                vertical_grid=args.psi_grid,
                phi_grid=args.psi_grid,
                rho_min=args.psi_rho_min,
                ridge=args.psi_ridge,
                iterations=args.psi_iterations,
            )
        )
        coil_linear = coil_component_gradient_native(
            args.lib, center_x, center_y, center_z, center_current, nfp
        )
        coil_components = np.full(args.query_count, coil_linear["component"])
        for values, center_values, name in (
            (x, center_x, "x"), (y, center_y, "y"), (z, center_z, "z"),
            (current, center_current, "current"),
        ):
            weighted_delta = (values - center_values) * coil_linear["gradient"][name]
            coil_components += weighted_delta.reshape(args.query_count, -1).sum(axis=1)
        (local_results, local_stats), local_score_wall_s = timed(
            lambda: batch.score_local_batch(
                current, *batch_axis,
                refined_axis["residual"], refined_axis["topology_trace"],
                refined_axis["topology_det"], batch_psi, batch_psi_rms,
                coil_components, center_capture,
                surface_theta_count=args.surface_theta_count,
                alpha_iterations=args.alpha_iterations,
            )
        )

        references = np.linspace(
            0, args.query_count - 1, min(args.reference_count, args.query_count), dtype=int
        )
        errors = []
        reference_wall_s = 0.0
        for query in references:
            started = time.perf_counter()
            field = CoilFieldGpu(
                args.lib, x[query], y[query], z[query], current[query], nfp,
                segments_per_coil=args.segments_per_coil, device_id=0,
            )
            try:
                single_B = field.eval_B(points[query], precision="fp32")
                single_B_grad, single_gradient = field.eval_B_grad(
                    points[query], precision="fp32"
                )
                single_R1, single_Z1 = field.trace_period_blockline_mixed(
                    R0[query], Z0[query], steps=args.trace_steps,
                    threads_per_line=256, mode="bf32_state64",
                )
                single_axis = field.trace_axis_samples(
                    refined_axis["R"][query], refined_axis["Z"][query], nfp=nfp,
                    integration_steps=args.axis_integration_steps,
                    sample_count=args.axis_samples,
                )
                train_R, train_Z, train_phi = psi_training_points(
                    batch_axis, int(query), nfp=nfp,
                    radius_scale=args.psi_radius, grid=args.psi_grid,
                    rho_min=args.psi_rho_min,
                )
                single_psi, single_psi_rms, single_psi_stats = field.fit_psi_fullgpu(
                    train_R, train_Z, train_phi,
                    *single_axis,
                    mode_a, mode_b, mode_m, mode_kind,
                    a=args.psi_radius,
                    poly_degree=args.psi_degree,
                    m_tor=args.psi_mtor,
                    ridge=args.psi_ridge,
                    precision="fp32",
                    solver="qr",
                )
            finally:
                field.close()
            reference_wall_s += time.perf_counter() - started
            errors.append({
                "query": int(query),
                "B_relative_rms": relative_rms(batch_B[query], single_B),
                "B_from_grad_relative_rms": relative_rms(batch_B_grad[query], single_B_grad),
                "grad_B_relative_rms": relative_rms(batch_gradient[query], single_gradient),
                "trace_R_relative_rms": relative_rms(batch_R1[query], single_R1),
                "trace_Z_absolute_rms": float(np.sqrt(np.mean(np.square(
                    batch_Z1[query] - single_Z1
                )))),
                "axis_R_relative_rms": relative_rms(batch_axis[0][query], single_axis[0]),
                "axis_Z_absolute_rms": float(np.sqrt(np.mean(np.square(
                    batch_axis[1][query] - single_axis[1]
                )))),
                "axis_R_phi_relative_rms": relative_rms(batch_axis[2][query], single_axis[2]),
                "axis_Z_phi_relative_rms": relative_rms(batch_axis[3][query], single_axis[3]),
                "psi_coefficient_relative_rms": relative_rms(batch_psi[query], single_psi),
                "psi_train_rms_batch": float(batch_psi_rms[query]),
                "psi_train_rms_exact": float(single_psi_rms),
                "psi_train_rms_ratio": float(batch_psi_rms[query] / single_psi_rms),
                "psi_exact_fit_s": float(single_psi_stats["total_s"]),
            })
    finally:
        batch.clear_psi_preconditioner()
        batch.close()

    local_scores = np.asarray([result["score"] for result in local_results])
    local_statuses = [result["status"] for result in local_results]
    gradient_comparison = None
    if args.full_reference_dir:
        exact_by_candidate = {}
        for path in sorted(args.full_reference_dir.glob("rank_*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                exact_by_candidate[int(row["metadata"]["candidate_index"])] = row["variants"]["exact"]
        weights = np.asarray([10.0, 10.0, 10.0, 10.0, 42.0, 10.0, 8.0])
        component_names = ["axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil"]
        center_coordinate = float(center_capture["score_result"]["components"]["coordinate"])
        exact_scores = []
        exact_scores_fixed_surface = []
        exact_component_rows = []
        for row in endpoint_rows[: args.query_count]:
            exact = exact_by_candidate[int(row["candidate_index"])]
            components = np.asarray([exact["components"][name] for name in component_names])
            exact_component_rows.append(components)
            before = float(np.dot(weights, components) / weights.sum())
            gate = float(exact["score"] / max(before, 1.0e-30))
            before_no_coordinate = before + weights[3] / weights.sum() * (
                center_coordinate - components[3]
            )
            exact_scores.append(before_no_coordinate * gate)
            before_fixed_surface = before_no_coordinate + weights[2] / weights.sum() * (
                float(center_capture["score_result"]["components"]["surface"]) - components[2]
            )
            exact_scores_fixed_surface.append(before_fixed_surface * gate)
        exact_scores = np.asarray(exact_scores)
        exact_scores_fixed_surface = np.asarray(exact_scores_fixed_surface)
        exact_component_rows = np.asarray(exact_component_rows)
        local_component_rows = np.asarray([
            [result["components"][name] for name in component_names]
            for result in local_results
        ])
        local_directional = (local_scores[1::2] - local_scores[0::2]) / 0.01
        exact_directional = (exact_scores[1::2] - exact_scores[0::2]) / 0.01
        exact_fixed_surface_directional = (
            exact_scores_fixed_surface[1::2] - exact_scores_fixed_surface[0::2]
        ) / 0.01
        cosine = float(np.dot(local_directional, exact_directional) / max(
            np.linalg.norm(local_directional) * np.linalg.norm(exact_directional), 1.0e-30
        ))
        fixed_surface_cosine = float(
            np.dot(local_directional, exact_fixed_surface_directional) / max(
                np.linalg.norm(local_directional) *
                np.linalg.norm(exact_fixed_surface_directional), 1.0e-30
            )
        )
        component_comparison = {}
        for index, name in enumerate(component_names):
            local_component_directional = (
                local_component_rows[1::2, index] - local_component_rows[0::2, index]
            ) / 0.01
            exact_component_directional = (
                exact_component_rows[1::2, index] - exact_component_rows[0::2, index]
            ) / 0.01
            denominator = np.linalg.norm(local_component_directional) * np.linalg.norm(
                exact_component_directional
            )
            component_comparison[name] = {
                "cosine": None if denominator == 0.0 else float(
                    np.dot(local_component_directional, exact_component_directional) / denominator
                ),
                "local_rms": float(np.sqrt(np.mean(local_component_directional ** 2))),
                "exact_rms": float(np.sqrt(np.mean(exact_component_directional ** 2))),
                "endpoint_rms_difference": float(np.sqrt(np.mean(
                    (local_component_rows[:, index] - exact_component_rows[:, index]) ** 2
                ))),
            }
        gradient_comparison = {
            "coordinate_omitted_exact_cosine": cosine,
            "coordinate_and_surface_selection_omitted_exact_cosine": fixed_surface_cosine,
            "local_directional_rms": float(np.sqrt(np.mean(local_directional ** 2))),
            "exact_directional_rms": float(np.sqrt(np.mean(exact_directional ** 2))),
            "local_directional": local_directional.tolist(),
            "exact_directional": exact_directional.tolist(),
            "components": component_comparison,
        }

    output = {
        "format": "query_batch_field_benchmark_v1",
        "query_count": args.query_count,
        "point_count": args.point_count,
        "segments_per_coil": args.segments_per_coil,
        "nfp": nfp,
        "trace_steps": args.trace_steps,
        "axis_integration_steps": args.axis_integration_steps,
        "axis_samples": args.axis_samples,
        "psi": {
            "degree": args.psi_degree,
            "m_tor": args.psi_mtor,
            "coefficient_count": int(center_psi.size),
            "grid": args.psi_grid,
            "radius": args.psi_radius,
            "rho_min": args.psi_rho_min,
            "ridge": args.psi_ridge,
            "iterations": args.psi_iterations,
            "batch_stats": batch_psi_stats,
            "train_rms_p50_p95_max": [
                float(value) for value in np.quantile(batch_psi_rms, [0.5, 0.95, 1.0])
            ],
        },
        "center_score": center_capture["score_result"],
        "local_score": {
            "stats": local_stats,
            "wall_s": local_score_wall_s,
            "status_counts": {
                status: local_statuses.count(status) for status in sorted(set(local_statuses))
            },
            "score_p05_p50_p95": [
                float(value) for value in np.quantile(local_scores, [0.05, 0.5, 0.95])
            ],
            "gradient_comparison": gradient_comparison,
        },
        "timing_s": {
            "center_capture": center_capture_wall_s,
            "field_create": create_wall_s,
            "eval_B": eval_B_wall_s,
            "eval_B_grad": eval_B_grad_wall_s,
            "trace_period_5_lines": trace_wall_s,
            "refine_axis_hint": refine_axis_wall_s,
            "trace_axis_samples": axis_wall_s,
            "fit_psi_batch": batch_psi_wall_s,
            "score_local_batch": local_score_wall_s,
            "tested_stage_total": create_wall_s + eval_B_wall_s + eval_B_grad_wall_s +
                trace_wall_s + refine_axis_wall_s + axis_wall_s + batch_psi_wall_s,
            "sequential_reference_subset": reference_wall_s,
        },
        "reference_errors": errors,
        "axis_refinement": {
            "valid_count": int(np.count_nonzero(refined_axis["valid"])),
            "residual_p50_p95_max": [
                float(value) for value in np.quantile(
                    refined_axis["residual"], [0.5, 0.95, 1.0]
                )
            ],
            "hint_distance_p50_p95_max": [
                float(value) for value in np.quantile(np.hypot(
                    refined_axis["R"] - axis_R0,
                    refined_axis["Z"] - axis_Z0,
                ), [0.5, 0.95, 1.0])
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
