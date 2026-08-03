from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibrated_probability(
    raw_logit: np.ndarray, *, scale: float, bias: float
) -> np.ndarray:
    calibrated = np.clip(scale * np.asarray(raw_logit, dtype=np.float64) + bias, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-calibrated))


@torch.no_grad()
def project_to_rms_(latent: torch.Tensor, target_rms: torch.Tensor) -> None:
    if target_rms.shape != (latent.shape[0], 1, 1):
        raise ValueError("target_rms must have shape (batch, 1, 1)")
    current = latent.float().square().mean(dim=(1, 2), keepdim=True).sqrt()
    latent.mul_((target_rms / current.clamp_min(1.0e-12)).to(latent.dtype))


def optimize_latents(
    model: torch.nn.Module,
    initial: torch.Tensor,
    nfp: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    projected: bool,
    log_interval: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float | int]]]:
    if initial.ndim != 3 or nfp.shape != (initial.shape[0],):
        raise ValueError("initial and nfp batch dimensions must match")
    if steps < 1 or learning_rate <= 0.0 or log_interval < 1:
        raise ValueError("steps, learning_rate, and log_interval must be positive")

    latent = torch.nn.Parameter(initial.detach().clone())
    target_rms = initial.float().square().mean(dim=(1, 2), keepdim=True).sqrt()
    optimizer = torch.optim.Adam(
        [latent], lr=learning_rate, betas=(0.9, 0.999), eps=1.0e-8, fused=latent.is_cuda
    )
    with torch.no_grad():
        best_logit = model(latent, nfp).float()
        best_latent = latent.detach().clone()
    history: list[dict[str, float | int]] = []

    def record(step: int, logits: torch.Tensor) -> None:
        values = logits.detach().float()
        rms = latent.detach().float().square().mean(dim=(1, 2)).sqrt()
        quantiles = torch.quantile(values, torch.tensor((0.0, 0.5, 0.9, 0.99), device=values.device))
        history.append(
            {
                "step": step,
                "logit_min": float(quantiles[0].cpu()),
                "logit_median": float(quantiles[1].cpu()),
                "logit_p90": float(quantiles[2].cpu()),
                "logit_p99": float(quantiles[3].cpu()),
                "logit_max": float(values.max().cpu()),
                "latent_rms_median": float(rms.median().cpu()),
                "latent_rms_max": float(rms.max().cpu()),
            }
        )

    record(0, best_logit)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(latent, nfp).float()
        with torch.no_grad():
            improved = logits > best_logit
            best_logit[improved] = logits[improved]
            best_latent[improved] = latent.detach()[improved]
        (-logits.sum()).backward()
        if projected:
            with torch.no_grad():
                flat_latent = latent.detach().flatten(1)
                flat_gradient = latent.grad.flatten(1)
                radial = (flat_gradient * flat_latent).sum(dim=1, keepdim=True)
                radial /= flat_latent.square().sum(dim=1, keepdim=True).clamp_min(1.0e-12)
                flat_gradient.sub_(radial * flat_latent)
        optimizer.step()
        if projected:
            project_to_rms_(latent, target_rms)
        if step % log_interval == 0:
            with torch.no_grad():
                monitored_logit = model(latent, nfp).float()
                improved = monitored_logit > best_logit
                best_logit[improved] = monitored_logit[improved]
                best_latent[improved] = latent.detach()[improved]
            record(step, monitored_logit)

    with torch.no_grad():
        final_logit = model(latent, nfp).float()
        improved = final_logit > best_logit
        best_logit[improved] = final_logit[improved]
        best_latent[improved] = latent.detach()[improved]
    if not history or history[-1]["step"] != steps:
        record(steps, final_logit)
    if not torch.isfinite(best_logit).all() or not torch.isfinite(best_latent).all():
        raise FloatingPointError("proxy optimization produced non-finite values")
    return best_latent, best_logit, history


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("latent proxy optimization requires CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def load_calibration(path: Path, proxy_checkpoint: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("checkpoint_sha256") != file_sha256(proxy_checkpoint):
        raise ValueError("calibration summary does not match proxy checkpoint")
    calibration = summary["calibration"]
    if float(calibration["scale"]) <= 0.0:
        raise ValueError("calibration scale must be positive")
    return calibration


def plot_optimization(histories: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"free": "#9a4d42", "projected": "#237a57"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for variant in ("free", "projected"):
        matching = [row for row in histories if row["variant"] == variant]
        for rank_history in matching:
            values = rank_history["history"]
            axes[0].plot(
                [row["step"] for row in values],
                [row["logit_median"] for row in values],
                color=colors[variant],
                alpha=0.25,
            )
            axes[1].plot(
                [row["step"] for row in values],
                [row["latent_rms_median"] for row in values],
                color=colors[variant],
                alpha=0.25,
            )
        axes[0].plot([], [], color=colors[variant], label=variant)
        axes[1].plot([], [], color=colors[variant], label=variant)
    axes[0].set(xlabel="Adam step", ylabel="raw proxy logit", title="Median proxy objective by GPU shard")
    axes[1].set(xlabel="Adam step", ylabel="latent RMS", title="Median latent radius by GPU shard")
    for axis in axes:
        axis.legend()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def aggregate_shards(
    output_dir: Path,
    *,
    world_size: int,
    scale: float,
    bias: float,
) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {}
    for rank in range(world_size):
        with np.load(output_dir / f"optimizer_rank_{rank}.npz", allow_pickle=False) as payload:
            for name in payload.files:
                parts.setdefault(name, []).append(np.asarray(payload[name]))
    merged = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
    merged["initial_probability"] = calibrated_probability(
        merged["initial_raw_logit"], scale=scale, bias=bias
    )
    merged["best_probability"] = calibrated_probability(
        merged["best_raw_logit"], scale=scale, bias=bias
    )
    np.savez_compressed(output_dir / "optimized_latents.npz", **merged)
    return merged


def prepare_candidates(
    args: argparse.Namespace,
    optimized: dict[str, np.ndarray],
    *,
    device: torch.device,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    from flow_matching.data import CoilNormalizer
    from flow_matching.flow import integrate_flow
    from flow_matching.model import CoilFlowTransformer
    from scripts.optimize_native_score_cem import token_case

    checkpoint = torch.load(args.flow_checkpoint, map_location="cpu", weights_only=False)
    flow = CoilFlowTransformer(**checkpoint["model_config"]).to(device=device, dtype=torch.float32)
    flow.load_state_dict(checkpoint["ema"])
    flow.eval()
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])

    chosen_parts = []
    for variant in ("free", "projected"):
        available = np.flatnonzero(optimized["variant"] == variant)
        if len(available) < args.candidates_per_variant:
            raise ValueError(f"not enough {variant} candidates")
        order = available[np.argsort(-optimized["best_raw_logit"][available], kind="stable")]
        chosen_parts.append(order[: args.candidates_per_variant])
    chosen = np.concatenate(chosen_parts)
    latent = optimized["best_latent"][chosen].astype(np.float32, copy=False)
    decoded_parts = []
    decode_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(latent), args.flow_batch):
            stop = min(start + args.flow_batch, len(latent))
            state = torch.from_numpy(latent[start:stop]).to(device=device)
            nfp = torch.full((stop - start,), args.nfp, dtype=torch.long, device=device)
            decoded_parts.append(
                integrate_flow(
                    flow,
                    state,
                    nfp,
                    start_time=0.0,
                    end_time=1.0,
                    steps=args.flow_steps,
                    method="rk4",
                ).cpu().numpy()
            )
    torch.cuda.synchronize(device)
    decode_s = time.perf_counter() - decode_started
    decoded = np.concatenate(decoded_parts)
    raw_tokens = normalizer.inverse(decoded, (args.nfp, args.n_base_coils))

    prepared_path = args.output_dir / "prepared_cases.jsonl"
    with prepared_path.open("w", encoding="utf-8") as stream:
        for case_id, source_index in enumerate(chosen):
            variant = str(optimized["variant"][source_index])
            initial = optimized["initial_latent"][source_index].astype(np.float64)
            best = optimized["best_latent"][source_index].astype(np.float64)
            metadata = {
                "optimization_variant": variant,
                "optimizer_start_index": int(optimized["start_index"][source_index]),
                "initial_proxy_probability": float(optimized["initial_probability"][source_index]),
                "initial_proxy_raw_logit": float(optimized["initial_raw_logit"][source_index]),
                "latent_l2_from_initial": float(np.linalg.norm(best - initial)),
            }
            case = token_case(
                raw_tokens[case_id].astype(np.float64),
                nfp=args.nfp,
                target="QH",
                metadata=metadata,
            )
            row = {
                "case_id": case_id,
                "pool_index": int(source_index),
                "sampling_modes": [f"optimized_{variant}"],
                "proxy_probability": float(optimized["best_probability"][source_index]),
                "proxy_logit": float(
                    float(calibration["scale"]) * optimized["best_raw_logit"][source_index]
                    + float(calibration["bias"])
                ),
                "proxy_raw_logit": float(optimized["best_raw_logit"][source_index]),
                "latent_rms": float(np.sqrt(np.mean(best * best))),
                "metadata": metadata,
                "case": case,
            }
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    np.savez_compressed(
        args.output_dir / "selected_optimized_latents.npz",
        source_index=chosen,
        latent=latent,
        variant=optimized["variant"][chosen],
        proxy_probability=optimized["best_probability"][chosen],
        proxy_raw_logit=optimized["best_raw_logit"][chosen],
    )
    return {
        "selected_count": int(len(chosen)),
        "selected_per_variant": int(args.candidates_per_variant),
        "decode_s": decode_s,
        "flow_checkpoint_step": int(checkpoint["step"]),
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize the QH latent support proxy from many Gaussian starts.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proxy-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--starts-per-variant", type=int, default=4096)
    parser.add_argument("--candidates-per-variant", type=int, default=384)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--nfp", type=int, default=4)
    parser.add_argument("--n-base-coils", type=int, default=3)
    parser.add_argument("--flow-steps", type=int, default=256)
    parser.add_argument("--flow-batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def main() -> None:
    from flow_matching.proxy import LatentProxyTransformer

    args = parse_args()
    if args.starts_per_variant < 1 or args.candidates_per_variant < 1:
        raise ValueError("start and candidate counts must be positive")
    rank, local_rank, world_size, device = setup_distributed()
    if args.starts_per_variant % world_size:
        raise ValueError("starts-per-variant must be divisible by world size")
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    barrier(world_size)
    process_started = time.perf_counter()
    torch.set_float32_matmul_precision("high")

    calibration = load_calibration(args.calibration_summary, args.proxy_checkpoint)
    checkpoint = torch.load(args.proxy_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "qh_latent_proxy_v1":
        raise ValueError("unsupported proxy checkpoint")
    model = LatentProxyTransformer(**checkpoint["model_config"]).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    local_count = args.starts_per_variant // world_size
    generator = torch.Generator(device=device).manual_seed(args.seed + rank * 1000003)
    initial = torch.randn(
        (local_count, args.n_base_coils, 100),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    nfp = torch.full((local_count,), args.nfp, dtype=torch.long, device=device)
    with torch.no_grad():
        initial_logit = model(initial, nfp).float()

    shard_values: dict[str, list[np.ndarray]] = {
        "variant": [],
        "start_index": [],
        "initial_latent": [],
        "initial_raw_logit": [],
        "best_latent": [],
        "best_raw_logit": [],
    }
    histories = []
    for variant, projected in (("free", False), ("projected", True)):
        best_latent, best_logit, history = optimize_latents(
            model,
            initial,
            nfp,
            steps=args.steps,
            learning_rate=args.learning_rate,
            projected=projected,
            log_interval=args.log_interval,
        )
        shard_values["variant"].append(np.full(local_count, variant, dtype="U16"))
        shard_values["start_index"].append(
            rank + world_size * np.arange(local_count, dtype=np.int64)
        )
        shard_values["initial_latent"].append(initial.detach().cpu().numpy())
        shard_values["initial_raw_logit"].append(initial_logit.cpu().numpy())
        shard_values["best_latent"].append(best_latent.cpu().numpy())
        shard_values["best_raw_logit"].append(best_logit.cpu().numpy())
        histories.append({"rank": rank, "variant": variant, "history": history})
        if rank == 0:
            print(json.dumps({"event": "optimized", "variant": variant, **history[-1]}), flush=True)
    np.savez_compressed(
        args.output_dir / f"optimizer_rank_{rank}.npz",
        **{name: np.concatenate(values, axis=0) for name, values in shard_values.items()},
    )
    (args.output_dir / f"optimizer_history_rank_{rank}.json").write_text(
        json.dumps(histories, indent=2) + "\n", encoding="utf-8"
    )
    barrier(world_size)

    if rank != 0:
        if world_size > 1:
            dist.destroy_process_group()
        return

    all_histories = []
    for source_rank in range(world_size):
        all_histories.extend(
            json.loads((args.output_dir / f"optimizer_history_rank_{source_rank}.json").read_text(encoding="utf-8"))
        )
    optimized = aggregate_shards(
        args.output_dir,
        world_size=world_size,
        scale=float(calibration["scale"]),
        bias=float(calibration["bias"]),
    )
    plot_optimization(all_histories, args.output_dir / "proxy_optimization_monitor.png")
    if world_size > 1:
        dist.destroy_process_group()
    prepare_summary = prepare_candidates(
        args, optimized, device=device, calibration=calibration
    )
    manifest = {
        "format": "qh_latent_proxy_optimization_v1",
        "stage": "prepared",
        "args": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "world_size": world_size,
        "proxy_checkpoint_sha256": file_sha256(args.proxy_checkpoint),
        "proxy_checkpoint_step": int(checkpoint["step"]),
        "flow_checkpoint_sha256": file_sha256(args.flow_checkpoint),
        "calibration": calibration,
        "optimized_count": int(len(optimized["variant"])),
        "optimization_summary": {
            variant: {
                "count": int(np.sum(optimized["variant"] == variant)),
                "initial_probability_median": float(np.median(optimized["initial_probability"][optimized["variant"] == variant])),
                "best_probability_median": float(np.median(optimized["best_probability"][optimized["variant"] == variant])),
                "best_raw_logit_median": float(np.median(optimized["best_raw_logit"][optimized["variant"] == variant])),
                "best_raw_logit_max": float(np.max(optimized["best_raw_logit"][optimized["variant"] == variant])),
                "latent_rms_median": float(
                    np.median(np.sqrt(np.mean(optimized["best_latent"][optimized["variant"] == variant].astype(np.float64) ** 2, axis=(1, 2))))
                ),
            }
            for variant in ("free", "projected")
        },
        "prepare": prepare_summary,
        "runtime_s": time.perf_counter() - process_started,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "prepared", **manifest}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
