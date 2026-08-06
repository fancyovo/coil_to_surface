from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.flow import integrate_flow
from scripts.optimize_flow_prior_zo_adam import load_flow_checkpoint
from scripts.qh_blackbox_gradient_reference import file_sha256, load_rows, write_json
from scripts.qh_score_noise_sensitivity import perturbation_metrics


def parse_steps(value: str) -> tuple[int, ...]:
    steps = tuple(int(item) for item in value.split(",") if item.strip())
    if not steps or any(item < 1 for item in steps):
        raise argparse.ArgumentTypeError("steps must be positive integers")
    if tuple(sorted(set(steps))) != steps:
        raise argparse.ArgumentTypeError("steps must be unique and increasing")
    return steps


def array_error(reference: np.ndarray, value: np.ndarray) -> dict[str, float]:
    reference64 = np.asarray(reference, dtype=np.float64)
    delta = np.asarray(value, dtype=np.float64) - reference64
    return {
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(
            np.linalg.norm(delta) / max(np.linalg.norm(reference64), 1.0e-30)
        ),
    }


@torch.inference_mode()
def flow_roundtrips(
    model,
    latent: torch.Tensor,
    data: torch.Tensor,
    nfp: torch.Tensor,
    *,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    decoded = integrate_flow(
        model, latent, nfp, start_time=0.0, end_time=1.0, steps=steps, method="rk4"
    )
    recovered_latent = integrate_flow(
        model, decoded, nfp, start_time=1.0, end_time=0.0, steps=steps, method="rk4"
    )
    inverted = integrate_flow(
        model, data, nfp, start_time=1.0, end_time=0.0, steps=steps, method="rk4"
    )
    recovered_data = integrate_flow(
        model, inverted, nfp, start_time=0.0, end_time=1.0, steps=steps, method="rk4"
    )
    torch.cuda.synchronize(latent.device)
    return (
        decoded.cpu().numpy(),
        recovered_latent.cpu().numpy(),
        recovered_data.cpu().numpy(),
        float(time.perf_counter() - started),
    )


@torch.inference_mode()
def decode_probes(
    model,
    centers: np.ndarray,
    directions: np.ndarray,
    nfp_values: np.ndarray,
    *,
    probe_rms: float,
    steps: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    endpoint_parts = []
    endpoint_nfp = []
    for center, center_directions, center_nfp in zip(
        centers, directions, nfp_values
    ):
        endpoint_parts.extend(
            (
                center[None] + probe_rms * center_directions,
                center[None] - probe_rms * center_directions,
            )
        )
        endpoint_nfp.extend([int(center_nfp)] * (2 * len(center_directions)))
    endpoints = torch.from_numpy(np.concatenate(endpoint_parts).astype(np.float32)).to(
        device=device
    )
    nfp = torch.tensor(endpoint_nfp, dtype=torch.long, device=device)
    started = time.perf_counter()
    decoded = integrate_flow(
        model, endpoints, nfp, start_time=0.0, end_time=1.0, steps=steps, method="rk4"
    )
    torch.cuda.synchronize(device)
    return decoded.cpu().numpy(), float(time.perf_counter() - started)


def configuration_position_rms(
    tokens: np.ndarray, reference: np.ndarray, *, samples: int = 128
) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.float64)
    center = np.asarray(reference, dtype=np.float64)
    coefficients = values[..., :99].reshape(*values.shape[:-1], 3, 33)
    center_coefficients = center[..., :99].reshape(*center.shape[:-1], 3, 33)
    delta = coefficients - center_coefficients
    time_values = np.arange(samples, dtype=np.float64) / samples
    modes = np.arange(1, 17, dtype=np.float64)
    angles = 2.0 * np.pi * modes[:, None] * time_values[None]
    positions = delta[..., 0, None].copy()
    positions = positions + np.einsum(
        "...cm,mt->...ct", delta[..., 1::2], np.sin(angles)
    )
    positions = positions + np.einsum(
        "...cm,mt->...ct", delta[..., 2::2], np.cos(angles)
    )
    displacement_squared = np.sum(positions * positions, axis=-2)
    return np.sqrt(np.mean(displacement_squared, axis=(-2, -1)))


def plot_summary(summary: dict[str, Any], output: Path) -> None:
    aggregates = summary["aggregate_rows"]
    steps = np.asarray([row["steps"] for row in aggregates])
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    series = (
        ("latent_rms_max", "latent round-trip RMS", "normalized units"),
        ("data_rms_max", "data round-trip RMS", "normalized units"),
        (
            "physical_position_rms_max_m",
            "data round-trip curve RMS",
            "position error [m]",
        ),
    )
    for axis, (key, title, ylabel) in zip(axes, series):
        axis.plot(steps, [row[key] for row in aggregates], "o-", linewidth=1.8)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(steps, labels=[str(item) for item in steps])
        axis.set_title(title)
        axis.set_xlabel("RK4 steps")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
    figure.suptitle("Same-integrator forward/backward closure across four optimization centers")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark same-step flow reconstruction.")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=parse_steps, default=(16, 32, 64, 128, 256))
    parser.add_argument("--probe-rms", type=float, default=0.00125)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.probe_rms <= 0.0:
        raise ValueError("probe RMS must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (args.reference_dir / "manifest.json").read_text(encoding="utf-8")
    )
    cases = load_rows(args.reference_dir / "cases.jsonl")
    raw = np.load(args.reference_dir / "raw_tokens.npy", mmap_mode="r")
    with np.load(args.reference_dir / "latent_banks.npz") as banks:
        latent_np = np.asarray(banks["centers"], dtype=np.float32)
        direction_flat = np.asarray(banks["directions"], dtype=np.float32)
    centers = manifest["centers"]
    if len(centers) != len(latent_np):
        raise RuntimeError("center manifest and latent bank disagree")
    coil_counts = {int(center["n_coils"]) for center in centers}
    if len(coil_counts) != 1 or latent_np.shape[1] != next(iter(coil_counts)):
        raise RuntimeError("this benchmark requires equal coil counts across centers")
    directions_np = direction_flat.reshape(
        len(centers), direction_flat.shape[1], latent_np.shape[1], latent_np.shape[2]
    )

    device = torch.device(args.device)
    model, normalizer, checkpoint = load_flow_checkpoint(args.checkpoint, device)
    formal_tokens = []
    normalized_data = []
    for center_index, center in enumerate(centers):
        center_case = next(
            row
            for row in cases
            if row["kind"] == "center" and int(row["center_index"]) == center_index
        )
        tokens = np.asarray(raw[int(center_case["case_id"])], dtype=np.float32)
        key = (int(center["nfp"]), int(center["n_coils"]))
        normalized, clipped_fraction = normalizer.transform(tokens[None], key)
        if clipped_fraction != 0.0:
            raise RuntimeError(
                f"normalization clipped center {center['center_id']}: {clipped_fraction}"
            )
        formal_tokens.append(normalizer.inverse(normalized, key)[0])
        normalized_data.append(normalized[0])
    data_np = np.stack(normalized_data).astype(np.float32)
    latent = torch.from_numpy(latent_np).to(device=device)
    data = torch.from_numpy(data_np).to(device=device)
    nfp = torch.tensor(
        [int(center["nfp"]) for center in centers], dtype=torch.long, device=device
    )

    # Warm up kernels outside the recorded measurements.
    flow_roundtrips(model, latent, data, nfp, steps=4)
    rows = []
    for steps in args.steps:
        decoded_latent, recovered_latent, recovered_data, wall_s = flow_roundtrips(
            model, latent, data, nfp, steps=steps
        )
        decoded_probes, probe_wall_s = decode_probes(
            model,
            latent_np,
            directions_np,
            nfp.cpu().numpy(),
            probe_rms=args.probe_rms,
            steps=steps,
            device=device,
        )
        probe_offset = 0
        for center_index, center in enumerate(centers):
            key = (int(center["nfp"]), int(center["n_coils"]))
            probe_count = 2 * directions_np.shape[1]
            center_probe_normalized = decoded_probes[
                probe_offset : probe_offset + probe_count
            ]
            probe_offset += probe_count
            center_decoded_tokens = normalizer.inverse(
                decoded_latent[center_index : center_index + 1], key
            )[0]
            probe_tokens = normalizer.inverse(center_probe_normalized, key)
            probe_position_rms = float(
                np.median(
                    configuration_position_rms(
                        probe_tokens, center_decoded_tokens[None]
                    )
                )
            )
            latent_error = array_error(latent_np[center_index], recovered_latent[center_index])
            data_error = array_error(data_np[center_index], recovered_data[center_index])
            recovered_tokens = normalizer.inverse(recovered_data[center_index : center_index + 1], key)[0]
            physical_error = perturbation_metrics(
                recovered_tokens, formal_tokens[center_index]
            )
            rows.append(
                {
                    "center_id": center["center_id"],
                    "nfp": key[0],
                    "n_coils": key[1],
                    "steps": int(steps),
                    "batch_four_roundtrips_wall_s": wall_s,
                    "batch_probe_decode_wall_s": probe_wall_s,
                    "latent_rms": latent_error["rms"],
                    "latent_max_abs": latent_error["max_abs"],
                    "latent_relative_l2": latent_error["relative_l2"],
                    "latent_rms_to_probe_rms": latent_error["rms"] / args.probe_rms,
                    "data_rms": data_error["rms"],
                    "data_max_abs": data_error["max_abs"],
                    "data_relative_l2": data_error["relative_l2"],
                    "physical_position_rms_m": physical_error["position_delta_rms_m"],
                    "physical_position_max_m": physical_error["position_delta_max_m"],
                    "physical_coefficient_relative_l2": physical_error[
                        "coefficient_relative_l2"
                    ],
                    "physical_current_relative_l2": physical_error["current_relative_l2"],
                    "probe_position_rms_m": probe_position_rms,
                    "physical_position_rms_to_probe": physical_error[
                        "position_delta_rms_m"
                    ]
                    / probe_position_rms,
                }
            )

    aggregates = []
    for steps in args.steps:
        selected = [row for row in rows if row["steps"] == steps]
        aggregates.append(
            {
                "steps": int(steps),
                "latent_rms_max": max(row["latent_rms"] for row in selected),
                "latent_rms_to_probe_max": max(
                    row["latent_rms_to_probe_rms"] for row in selected
                ),
                "data_rms_max": max(row["data_rms"] for row in selected),
                "physical_position_rms_max_m": max(
                    row["physical_position_rms_m"] for row in selected
                ),
                "physical_position_rms_to_probe_max": max(
                    row["physical_position_rms_to_probe"] for row in selected
                ),
                "batch_four_roundtrips_wall_s": selected[0][
                    "batch_four_roundtrips_wall_s"
                ],
                "batch_probe_decode_wall_s": selected[0]["batch_probe_decode_wall_s"],
            }
        )
    summary = {
        "format": "qh_flow_same_step_reconstruction_v1",
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "reference_dir": str(args.reference_dir.resolve()),
        "probe_rms": args.probe_rms,
        "centers": [center["center_id"] for center in centers],
        "rows": rows,
        "aggregate_rows": aggregates,
    }
    write_json(args.output_dir / "summary.json", summary)
    plot_summary(summary, args.output_dir / "same_step_reconstruction.png")
    print(json.dumps(summary["aggregate_rows"], indent=2), flush=True)


if __name__ == "__main__":
    main()
