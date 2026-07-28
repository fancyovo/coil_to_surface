from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import distributed as dist
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.data import (
    CoilNormalizer,
    GroupKey,
    GroupStore,
    RawGroup,
    group_counts,
    load_raw_groups,
)
from flow_matching.flow import flow_matching_batch, flow_matching_loss, sample_heun
from flow_matching.geometry import (
    curve_metrics,
    geometry_eligible,
    reference_bounds,
    summarize_metrics,
)
from flow_matching.model import CoilFlowTransformer
from flow_matching.monitoring import append_metrics, plot_metrics


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float):
        self.model = CoilFlowTransformer(**model.config).to(next(model.parameters()).device)
        self.model.load_state_dict(model.state_dict())
        self.model.eval()
        self.decay = float(decay)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema, current in zip(self.model.parameters(), model.parameters(), strict=True):
            ema.lerp_(current.detach(), 1.0 - self.decay)
        for ema, current in zip(self.model.buffers(), model.buffers(), strict=True):
            ema.copy_(current)


def distributed_setup() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if world_size > 1:
            dist.init_process_group("nccl", device_id=device)
    else:
        if world_size > 1:
            dist.init_process_group("gloo")
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        device_ids = [torch.cuda.current_device()] if torch.cuda.is_available() else None
        dist.barrier(device_ids=device_ids)


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value


def learning_rate_scale(step: int, *, warmup: int, total: int) -> float:
    if step < warmup:
        return max(step, 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def calculate_reference_bounds(
    groups: dict[GroupKey, RawGroup], device: torch.device, max_samples: int = 8192
) -> dict[str, tuple[float, float]]:
    counts = np.asarray([len(group.tokens) for group in groups.values()], dtype=float)
    probabilities = counts / counts.sum()
    rng = np.random.default_rng(20260728)
    collected: dict[str, list[torch.Tensor]] = {}
    for key, probability in zip(groups, probabilities, strict=True):
        group = groups[key]
        count = min(len(group.tokens), max(16, int(round(max_samples * probability))))
        indices = rng.choice(len(group.tokens), size=count, replace=False)
        values = torch.from_numpy(group.tokens[indices]).to(device)
        metrics = curve_metrics(values)
        for name, metric in metrics.items():
            collected.setdefault(name, []).append(metric.detach())
    merged = {name: torch.cat(parts) for name, parts in collected.items()}
    return reference_bounds(merged)


@torch.no_grad()
def validation_loss(
    model: nn.Module,
    store: GroupStore,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    batch_size: int,
    use_bf16: bool,
) -> float:
    total = torch.zeros(2, device=device, dtype=torch.float64)
    for key_index, key in enumerate(store.keys):
        if key_index % world_size != rank:
            continue
        source = store.groups[key]
        count = min(len(source), int(batch_size))
        generator = torch.Generator().manual_seed(100000 + key[0] * 100 + key[1])
        indices = torch.randperm(len(source), generator=generator)[:count]
        data = source[indices].to(device)
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(200000 + key[0] * 100 + key[1])
            mixed, time_value, target = flow_matching_batch(data)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else nullcontext()
        )
        with autocast:
            prediction = model(
                mixed,
                time_value,
                torch.full((count,), key[0], dtype=torch.long, device=device),
            )
        loss = flow_matching_loss(prediction, target)
        total[0] += loss.double() * count
        total[1] += count
    if world_size > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return float((total[0] / total[1].clamp_min(1.0)).cpu())


@torch.no_grad()
def generate_monitor_samples(
    model: nn.Module,
    store: GroupStore,
    normalizer: CoilNormalizer,
    bounds: dict[str, tuple[float, float]],
    *,
    count: int,
    steps: int,
    seed: int,
    device: torch.device,
    use_bf16: bool,
) -> tuple[dict[str, float], list[tuple[GroupKey, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(store.keys), size=count, p=store.probabilities)
    group_counts_selected = np.bincount(selected, minlength=len(store.keys))
    collected: dict[str, list[torch.Tensor]] = {}
    candidates: list[tuple[GroupKey, np.ndarray]] = []
    torch_generator = torch.Generator(device=device).manual_seed(seed)
    for key_index, group_count in enumerate(group_counts_selected):
        if not group_count:
            continue
        key = store.keys[key_index]
        noise = torch.randn(
            (int(group_count), key[1], 100), device=device, generator=torch_generator
        )
        nfp = torch.full((int(group_count),), key[0], device=device, dtype=torch.long)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else nullcontext()
        )
        with autocast:
            normalized = sample_heun(model, noise, nfp, steps=steps)
        raw = normalizer.inverse(normalized.float().cpu().numpy(), key)
        metrics = curve_metrics(torch.from_numpy(raw).to(device))
        eligible = geometry_eligible(metrics, bounds).cpu().numpy()
        for name, metric in metrics.items():
            collected.setdefault(name, []).append(metric.detach())
        candidates.extend((key, raw[index]) for index in np.flatnonzero(eligible))
    merged = {name: torch.cat(parts) for name, parts in collected.items()}
    return summarize_metrics(merged, bounds), candidates


def score_candidates(
    candidates: list[tuple[GroupKey, np.ndarray]],
    *,
    count: int,
    lib_path: Path,
    gpu_ids: list[int],
    timeout_s: float,
    output_path: Path,
) -> dict[str, float]:
    from scripts.optimize_native_score_cem import NativeScorePool, token_case

    selected = candidates[:count]
    cases = [
        token_case(tokens, nfp=key[0], target="QH", metadata={"flow_monitor": True})
        for key, tokens in selected
    ]
    if not cases:
        return {"score_count": 0, "score_ok_rate": 0.0, "score_mean_all": 0.0}
    with NativeScorePool(lib_path, gpu_ids) as pool:
        evaluated = pool.map(cases, target="QH", timeout_s=timeout_s)
    rows = []
    scores = []
    ok_scores = []
    for case, (result, elapsed, error) in zip(cases, evaluated, strict=True):
        score = 0.0 if result is None else float(result["score"])
        status = "error" if result is None else str(result["status"])
        scores.append(score)
        if status == "ok":
            ok_scores.append(score)
        rows.append(
            {"case": case, "elapsed_s": elapsed, "error": error, "native_score": result}
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, allow_nan=True) + "\n", encoding="utf-8")
    return {
        "score_count": len(scores),
        "score_ok_rate": len(ok_scores) / len(scores),
        "score_mean_all": float(np.mean(scores)),
        "score_mean_ok": float(np.mean(ok_scores)) if ok_scores else 0.0,
        "score_p90": float(np.percentile(scores, 90)),
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    normalizer: CoilNormalizer,
    step: int,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    geometry_bounds: dict[str, tuple[float, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model_config": model.config,
            "model": model.state_dict(),
            "ema": ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "normalizer": normalizer.to_dict(),
            "step": int(step),
            "args": vars(args),
            "data_manifest": manifest,
            "geometry_reference_bounds": geometry_bounds,
        },
        temporary,
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a QH coil rectified-flow model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-per-gpu", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=1408)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--validation-interval", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=250)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--score-lib", type=Path)
    parser.add_argument("--score-start-step", type=int, default=2000)
    parser.add_argument("--score-interval", type=int, default=2000)
    parser.add_argument("--score-count", type=int, default=32)
    parser.add_argument("--score-min-eligible-rate", type=float, default=0.25)
    parser.add_argument("--score-timeout-s", type=float, default=900.0)
    parser.add_argument("--verify-data", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    process_started = time.perf_counter()
    args = parse_args()
    rank, local_rank, world_size, device = distributed_setup()
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    use_bf16 = device.type == "cuda" and not args.no_bf16

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    if rank == 0 and metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite existing run {metrics_path}")
    barrier(world_size)

    train_raw, manifest = load_raw_groups(
        args.data_dir, "train", verify_hashes=args.verify_data and rank == 0
    )
    validation_raw, _ = load_raw_groups(args.data_dir, "validation")
    normalizer = CoilNormalizer.fit(train_raw)
    train_store = GroupStore(train_raw, normalizer)
    validation_store = GroupStore(validation_raw, normalizer)
    bounds = calculate_reference_bounds(train_raw, device) if rank == 0 else None
    if world_size > 1:
        values = [bounds]
        dist.broadcast_object_list(values, src=0)
        bounds = values[0]
    assert bounds is not None

    base_model = CoilFlowTransformer(
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        hidden=args.hidden,
    ).to(device)
    ema = ExponentialMovingAverage(base_model, args.ema_decay)
    train_model: nn.Module = base_model
    if args.compile:
        train_model = torch.compile(train_model)
    if world_size > 1:
        train_model = torch.nn.parallel.DistributedDataParallel(
            train_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_scale(step, warmup=args.warmup_steps, total=args.steps),
    )
    if rank == 0:
        run_manifest = {
            "args": vars(args),
            "world_size": world_size,
            "device": str(device),
            "torch_version": torch.__version__,
            "parameter_count": base_model.parameter_count,
            "train_counts": group_counts(train_raw),
            "validation_counts": group_counts(validation_raw),
            "normalizer": normalizer.to_dict(),
            "geometry_reference_bounds": bounds,
            "data_format": manifest.get("format"),
        }
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "start", **run_manifest}, default=str), flush=True)

    key_rng = np.random.default_rng(args.seed)
    batch_generator = torch.Generator().manual_seed(args.seed + 1000 * rank)
    log_started = time.perf_counter()
    training_started = log_started
    accumulated_loss = 0.0
    accumulated_grad = 0.0
    accumulated_steps = 0
    last_sample_summary: dict[str, float] = {}
    last_candidates: list[tuple[GroupKey, np.ndarray]] = []

    try:
        for step in range(1, args.steps + 1):
            key = train_store.choose_key(key_rng)
            data = train_store.batch(
                key,
                args.batch_per_gpu,
                device=device,
                generator=batch_generator,
                permute=True,
            )
            mixed, time_value, target = flow_matching_batch(data)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with autocast:
                prediction = train_model(
                    mixed,
                    time_value,
                    torch.full((data.shape[0],), key[0], dtype=torch.long, device=device),
                )
                loss = flow_matching_loss(prediction, target)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema.update(base_model)
            accumulated_loss += float(loss.detach())
            accumulated_grad += float(grad_norm)
            accumulated_steps += 1

            if step % args.log_interval == 0:
                elapsed = time.perf_counter() - log_started
                loss_value = torch.tensor(accumulated_loss / accumulated_steps, device=device)
                grad_value = torch.tensor(accumulated_grad / accumulated_steps, device=device)
                loss_value = reduce_mean(loss_value, world_size)
                grad_value = reduce_mean(grad_value, world_size)
                if rank == 0:
                    row = {
                        "event": "train",
                        "step": step,
                        "train_loss": float(loss_value.cpu()),
                        "grad_norm": float(grad_value.cpu()),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "samples_per_s": args.batch_per_gpu * world_size * accumulated_steps / elapsed,
                        "elapsed_s": time.perf_counter() - training_started,
                    }
                    append_metrics(metrics_path, row)
                    print(json.dumps(row, separators=(",", ":")), flush=True)
                accumulated_loss = accumulated_grad = 0.0
                accumulated_steps = 0
                log_started = time.perf_counter()

            if step % args.validation_interval == 0:
                event_started = time.perf_counter()
                value = validation_loss(
                    ema.model,
                    validation_store,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    batch_size=128,
                    use_bf16=use_bf16,
                )
                if rank == 0:
                    row = {
                        "event": "validation",
                        "step": step,
                        "validation_loss": value,
                        "duration_s": time.perf_counter() - event_started,
                    }
                    append_metrics(metrics_path, row)
                    print(json.dumps(row, separators=(",", ":")), flush=True)

            if step % args.sample_interval == 0:
                event_started = time.perf_counter()
                if rank == 0:
                    last_sample_summary, last_candidates = generate_monitor_samples(
                        ema.model,
                        train_store,
                        normalizer,
                        bounds,
                        count=args.sample_count,
                        steps=args.sample_steps,
                        seed=args.seed + step,
                        device=device,
                        use_bf16=use_bf16,
                    )
                    row = {
                        "event": "sample",
                        "step": step,
                        "duration_s": time.perf_counter() - event_started,
                        **last_sample_summary,
                    }
                    append_metrics(metrics_path, row)
                    print(json.dumps(row, separators=(",", ":")), flush=True)
                    plot_metrics(metrics_path, args.output_dir / "monitor.png")
                barrier(world_size)

            should_score = (
                args.score_lib is not None
                and step >= args.score_start_step
                and step % args.score_interval == 0
            )
            if should_score:
                event_started = time.perf_counter()
                if rank == 0:
                    if last_sample_summary.get("geometry_eligible_rate", 0.0) >= args.score_min_eligible_rate:
                        score_summary = score_candidates(
                            last_candidates,
                            count=args.score_count,
                            lib_path=args.score_lib,
                            gpu_ids=list(range(world_size)),
                            timeout_s=args.score_timeout_s,
                            output_path=args.output_dir / "score_monitor" / f"step_{step:08d}.json",
                        )
                        row = {"event": "score", "step": step, **score_summary}
                    else:
                        row = {
                            "event": "score_skipped",
                            "step": step,
                            "reason": "geometry_eligible_rate_below_threshold",
                            **last_sample_summary,
                        }
                    row["duration_s"] = time.perf_counter() - event_started
                    append_metrics(metrics_path, row)
                    print(json.dumps(row, separators=(",", ":")), flush=True)
                    plot_metrics(metrics_path, args.output_dir / "monitor.png")
                barrier(world_size)

            if step % args.checkpoint_interval == 0 or step == args.steps:
                event_started = time.perf_counter()
                if rank == 0:
                    save_checkpoint(
                        args.output_dir / "checkpoints" / f"step_{step:08d}.pt",
                        model=base_model,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        normalizer=normalizer,
                        step=step,
                        args=args,
                        manifest=manifest,
                        geometry_bounds=bounds,
                    )
                    save_checkpoint(
                        args.output_dir / "checkpoint_latest.pt",
                        model=base_model,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        normalizer=normalizer,
                        step=step,
                        args=args,
                        manifest=manifest,
                        geometry_bounds=bounds,
                    )
                barrier(world_size)
                if rank == 0:
                    row = {
                        "event": "checkpoint",
                        "step": step,
                        "duration_s": time.perf_counter() - event_started,
                    }
                    append_metrics(metrics_path, row)
                    print(json.dumps(row, separators=(",", ":")), flush=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(device),
                    torch.cuda.max_memory_reserved(device),
                ],
                dtype=torch.float64,
                device=device,
            )
            if world_size > 1:
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        else:
            peak = torch.zeros(2, dtype=torch.float64, device=device)
        if rank == 0:
            row = {
                "event": "complete",
                "step": args.steps,
                "training_elapsed_s": time.perf_counter() - training_started,
                "process_elapsed_s": time.perf_counter() - process_started,
                "peak_memory_allocated_gib": float(peak[0].cpu()) / (1024**3),
                "peak_memory_reserved_gib": float(peak[1].cpu()) / (1024**3),
            }
            append_metrics(metrics_path, row)
            print(json.dumps(row, separators=(",", ":")), flush=True)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
