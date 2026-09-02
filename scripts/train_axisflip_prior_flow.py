from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import subprocess
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

from flow_matching.axis_prior_rl import (
    FORMAT,
    NFP,
    N_BASE_COILS,
    diversity_summary,
    feature_weights,
    fit_prior_normalizer,
    initialize_model,
    inverse_tokens,
    load_teacher_dataset,
    model_config,
    random_permute_coils,
    split_masks,
    transform_tokens,
)
from flow_matching.flow import integrate_flow


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float):
        self.model = type(model)(**model.config).to(next(model.parameters()).device)
        self.model.load_state_dict(model.state_dict())
        self.model.eval()
        self.decay = float(decay)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for target, source in zip(self.model.parameters(), model.parameters(), strict=True):
            target.lerp_(source.detach(), 1.0 - self.decay)
        for target, source in zip(self.model.buffers(), model.buffers(), strict=True):
            target.copy_(source)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values))
    os.replace(temporary, path)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    normalizer,
    epoch: int,
    step: int,
    teacher_manifest: dict[str, Any],
    code_commit: str,
    converged: bool,
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "format": FORMAT,
            "stage": "distilled_q0",
            "model_config": model.config,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "ema": {key: value.detach().cpu() for key, value in ema.model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "normalizer": normalizer.to_dict(),
            "epoch": int(epoch),
            "step": int(step),
            "teacher_repository_commit": teacher_manifest["repository_commit"],
            "code_commit": code_commit,
            "converged": bool(converged),
        },
        temporary,
    )
    os.replace(temporary, path)


def distributed_setup() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("prior Flow distillation requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier(device_ids=[torch.cuda.current_device()])


def fixed_validation_loss(
    model: nn.Module,
    normalized: np.ndarray,
    indices: np.ndarray,
    *,
    rank: int,
    world_size: int,
    batch_size: int,
    weights: torch.Tensor,
    device: torch.device,
    seed: int,
) -> float:
    local = indices[rank::world_size]
    total = torch.zeros(2, dtype=torch.float64, device=device)
    model.eval()
    with torch.inference_mode(), torch.random.fork_rng(devices=[device]):
        torch.manual_seed(int(seed) + rank)
        for offset in range(0, len(local), batch_size):
            selected = local[offset : offset + batch_size]
            data = torch.from_numpy(normalized[selected]).to(device=device)
            noise = torch.randn_like(data)
            time_value = torch.rand(len(data), device=device, dtype=torch.float32)
            mixed = (1.0 - time_value[:, None, None]) * noise + time_value[:, None, None] * data
            target = data - noise
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(
                    mixed,
                    time_value,
                    torch.full((len(data),), NFP, dtype=torch.long, device=device),
                )
            square = (prediction.float() - target.float()).square() * weights
            loss_sum = square.sum()
            denominator = len(data) * N_BASE_COILS * weights.sum()
            total[0] += loss_sum.double()
            total[1] += denominator.double()
    if world_size > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    model.train()
    return float((total[0] / total[1].clamp_min(1.0)).cpu())


@torch.inference_mode()
def generated_monitor(
    model: nn.Module,
    normalizer,
    teacher_normalized: np.ndarray,
    *,
    count: int,
    flow_steps: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn((count, N_BASE_COILS, 100), generator=generator, device=device)
    nfp = torch.full((count,), NFP, dtype=torch.long, device=device)
    model.eval()
    generated = integrate_flow(
        model,
        noise,
        nfp,
        start_time=0.0,
        end_time=1.0,
        steps=flow_steps,
        method="rk4",
    ).float().cpu().numpy()
    teacher = np.asarray(teacher_normalized[:count], dtype=np.float32)
    mean_error = float(np.mean(np.abs(generated.mean(axis=(0, 1)) - teacher.mean(axis=(0, 1)))))
    std_error = float(np.mean(np.abs(generated.std(axis=(0, 1)) - teacher.std(axis=(0, 1)))))
    physical = inverse_tokens(generated, normalizer)
    current_l1 = np.sum(np.abs(physical[..., -1]), axis=1)
    return {
        "count": int(count),
        "normalized_mean_abs_error": mean_error,
        "normalized_std_abs_error": std_error,
        "moment_error": mean_error + std_error,
        "current_l1_min_a": float(np.min(current_l1)),
        "current_l1_max_a": float(np.max(current_l1)),
        "diversity": diversity_summary(generated, seed=seed),
    }


def distribution_stable(values: list[float], *, absolute_tolerance: float = 0.01) -> bool:
    if len(values) < 3:
        return False
    recent = np.asarray(values[-3:], dtype=np.float64)
    return bool(np.max(recent) - np.min(recent) <= absolute_tolerance)


def main() -> None:
    args = parser().parse_args()
    rank, local_rank, world_size, device = distributed_setup()
    if world_size != 4:
        raise ValueError("the registered distillation uses exactly four GPUs")
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if code_commit != args.expected_commit:
        raise RuntimeError(f"repository commit {code_commit} != expected {args.expected_commit}")
    dirty = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"], cwd=REPO_ROOT, text=True
    )
    if dirty.strip():
        raise RuntimeError("tracked experiment worktree is dirty")

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    barrier(world_size)
    tokens, seeds, teacher_manifest = load_teacher_dataset(
        args.dataset_dir, verify_hashes=rank == 0
    )
    masks = split_masks(seeds, int(teacher_manifest["seed_start"]))
    train_indices = np.flatnonzero(masks["train"])
    validation_indices = np.flatnonzero(masks["validation"])
    test_indices = np.flatnonzero(masks["test"])
    normalizer = fit_prior_normalizer(tokens[train_indices])
    normalized = transform_tokens(tokens, normalizer)
    if float(np.max(np.abs(normalized[train_indices, :, -1]))) > 1.0e-6:
        raise RuntimeError("teacher q0 current channel is not the expected equal-current target")
    if rank == 0:
        atomic_save_npy(
            args.output_dir / "q0_train_normalized.npy",
            normalized[train_indices].astype(np.float32),
        )
    barrier(world_size)

    config = model_config(
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        hidden=args.hidden,
    )
    base_model = initialize_model(config, seed=args.seed).to(device)
    ema = ExponentialMovingAverage(base_model, args.ema_decay)
    train_model: nn.Module = base_model
    if world_size > 1:
        train_model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1.0e-6
    )
    weights = feature_weights(normalizer, device)
    if rank == 0:
        manifest = {
            "format": FORMAT,
            "stage": "q0_distillation",
            "status": "running",
            "code_commit": code_commit,
            "teacher_dataset": str(args.dataset_dir.resolve()),
            "teacher_repository_commit": teacher_manifest["repository_commit"],
            "condition": {"nfp": NFP, "n_base_coils": N_BASE_COILS},
            "split_counts": {
                "train": int(len(train_indices)),
                "validation": int(len(validation_indices)),
                "test": int(len(test_indices)),
            },
            "model_config": config,
            "parameter_count": base_model.parameter_count,
            "normalizer": {
                "source": "axis-flip analytic teacher only",
                "current_mean_a": float(normalizer.mean[-1]),
                "current_scale_a": float(normalizer.std[-1]),
                "current_l1_a": normalizer.current_l1_a,
            },
            "q0_normalized_training_cache": "q0_train_normalized.npy",
            "convergence": {
                "minimum_epochs": args.minimum_epochs,
                "validation_relative_improvement": args.minimum_relative_improvement,
                "validation_patience": args.patience,
                "generated_moment_stability_checks": 3,
                "generated_moment_stability_absolute_tolerance": 0.01,
                "maximum_epochs_is_safety_only": args.maximum_epochs,
            },
            "created_unix_s": time.time(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        atomic_write_json(args.output_dir / "manifest.json", manifest)

    best_loss = math.inf
    stale = 0
    global_step = 0
    moment_history: list[float] = []
    converged = False
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.perf_counter()
    for epoch in range(1, args.maximum_epochs + 1):
        epoch_started = time.perf_counter()
        rng = np.random.default_rng(args.seed + epoch)
        order = train_indices.copy()
        rng.shuffle(order)
        local_indices = order[rank::world_size]
        base_model.train()
        train_total = torch.zeros(2, dtype=torch.float64, device=device)
        permutation_generator = torch.Generator().manual_seed(args.seed + 100000 * epoch + rank)
        for offset in range(0, len(local_indices), args.batch_per_gpu):
            selected = local_indices[offset : offset + args.batch_per_gpu]
            data_cpu = torch.from_numpy(normalized[selected])
            data_cpu = random_permute_coils(data_cpu, generator=permutation_generator)
            data = data_cpu.to(device=device, non_blocking=True)
            noise = torch.randn_like(data)
            time_value = torch.rand(len(data), dtype=torch.float32, device=device)
            mixed = (1.0 - time_value[:, None, None]) * noise + time_value[:, None, None] * data
            target = data - noise
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = train_model(
                    mixed,
                    time_value,
                    torch.full((len(data),), NFP, dtype=torch.long, device=device),
                )
                square = (prediction.float() - target.float()).square() * weights
                loss = square.sum() / (len(data) * N_BASE_COILS * weights.sum())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
            optimizer.step()
            ema.update(base_model)
            train_total[0] += loss.detach().double() * len(data)
            train_total[1] += len(data)
            global_step += 1
        if world_size > 1:
            dist.all_reduce(train_total, op=dist.ReduceOp.SUM)
        train_loss = float((train_total[0] / train_total[1]).cpu())
        validation = fixed_validation_loss(
            ema.model,
            normalized,
            validation_indices,
            rank=rank,
            world_size=world_size,
            batch_size=args.validation_batch_per_gpu,
            weights=weights,
            device=device,
            seed=args.seed + 900000,
        )
        scheduler.step(validation)
        improved = (
            not math.isfinite(best_loss)
            or validation < best_loss * (1.0 - args.minimum_relative_improvement)
        )
        if improved:
            best_loss = validation
            stale = 0
        else:
            stale += 1
        monitor = None
        if rank == 0:
            monitor = generated_monitor(
                ema.model,
                normalizer,
                normalized[test_indices],
                count=min(args.monitor_count, len(test_indices)),
                flow_steps=args.monitor_flow_steps,
                device=device,
                seed=args.seed + 800000,
            )
            moment_history.append(float(monitor["moment_error"]))
            stable = distribution_stable(moment_history)
            row = {
                "event": "distillation_epoch",
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "validation_loss": validation,
                "best_validation_loss": best_loss,
                "meaningful_improvement": improved,
                "stale_checks": stale,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "generated": monitor,
                "generated_stable": stable,
                "epoch_wall_s": time.perf_counter() - epoch_started,
                "total_wall_s": time.perf_counter() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            print(json.dumps(row, separators=(",", ":")), flush=True)
            save_checkpoint(
                args.output_dir / "checkpoint_latest.pt",
                model=base_model,
                ema=ema,
                optimizer=optimizer,
                normalizer=normalizer,
                epoch=epoch,
                step=global_step,
                teacher_manifest=teacher_manifest,
                code_commit=code_commit,
                converged=False,
            )
            if improved:
                save_checkpoint(
                    args.output_dir / "checkpoint_q0.pt",
                    model=base_model,
                    ema=ema,
                    optimizer=optimizer,
                    normalizer=normalizer,
                    epoch=epoch,
                    step=global_step,
                    teacher_manifest=teacher_manifest,
                    code_commit=code_commit,
                    converged=False,
                )
            converged = bool(
                epoch >= args.minimum_epochs and stale >= args.patience and stable
            )
        decision = torch.tensor([1 if converged else 0], dtype=torch.int32, device=device)
        if world_size > 1:
            dist.broadcast(decision, src=0)
        converged = bool(decision.item())
        barrier(world_size)
        if converged:
            break

    if rank == 0:
        if not converged:
            atomic_write_json(
                args.output_dir / "convergence.json",
                {
                    "format": FORMAT,
                    "status": "not_converged_at_safety_limit",
                    "epochs": args.maximum_epochs,
                    "best_validation_loss": best_loss,
                    "finished_unix_s": time.time(),
                },
            )
        else:
            best_path = args.output_dir / "checkpoint_q0.pt"
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            best["converged"] = True
            temporary = best_path.with_name(f".{best_path.name}.{os.getpid()}.tmp")
            torch.save(best, temporary)
            os.replace(temporary, best_path)
            atomic_write_json(
                args.output_dir / "convergence.json",
                {
                    "format": FORMAT,
                    "status": "converged",
                    "stopped_epoch": epoch,
                    "global_step": global_step,
                    "best_validation_loss": best_loss,
                    "validation_stale_checks": stale,
                    "recent_generated_moment_errors": moment_history[-3:],
                    "checkpoint": str(best_path.resolve()),
                    "finished_unix_s": time.time(),
                },
            )
            manifest = json.loads(
                (args.output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["status"] = "converged"
            manifest["convergence_artifact"] = "convergence.json"
            atomic_write_json(args.output_dir / "manifest.json", manifest)
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()
    if not converged:
        raise RuntimeError("q0 distillation did not meet convergence criteria")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Distill the fixed axis-flip analytic prior into Flow matching.")
    value.add_argument("--dataset-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--expected-commit", required=True)
    value.add_argument("--batch-per-gpu", type=int, default=256)
    value.add_argument("--validation-batch-per-gpu", type=int, default=256)
    value.add_argument("--learning-rate", type=float, default=1.0e-4)
    value.add_argument("--ema-decay", type=float, default=0.999)
    value.add_argument("--minimum-epochs", type=int, default=10)
    value.add_argument("--maximum-epochs", type=int, default=200)
    value.add_argument("--patience", type=int, default=10)
    value.add_argument("--minimum-relative-improvement", type=float, default=0.003)
    value.add_argument("--monitor-count", type=int, default=512)
    value.add_argument("--monitor-flow-steps", type=int, default=32)
    value.add_argument("--width", type=int, default=256)
    value.add_argument("--layers", type=int, default=6)
    value.add_argument("--heads", type=int, default=8)
    value.add_argument("--hidden", type=int, default=704)
    value.add_argument("--seed", type=int, default=2026090301)
    return value


if __name__ == "__main__":
    main()
