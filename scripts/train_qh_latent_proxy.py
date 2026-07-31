from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
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

from flow_matching.proxy import (
    LatentProxyTransformer,
    binary_metrics,
    enrichment_at_prior_rates,
    roc_auc,
    validation_threshold,
)


GroupKey = tuple[int, int]


def autocast_context(use_bf16: bool):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )


def setup() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if world_size > 1:
            dist.init_process_group("nccl", device_id=device)
    else:
        device = torch.device("cpu")
        if world_size > 1:
            dist.init_process_group("gloo")
    return rank, local_rank, world_size, device


def barrier(world_size: int, local_rank: int) -> None:
    if world_size > 1:
        device_ids = [local_rank] if torch.cuda.is_available() else None
        dist.barrier(device_ids=device_ids)


def load_latent_groups(
    latent_dir: Path,
    split: str,
    *,
    device: torch.device,
) -> tuple[dict[GroupKey, torch.Tensor], dict[GroupKey, torch.Tensor], dict[str, Any]]:
    manifest = json.loads((latent_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "qh_flow_latents_v1":
        raise ValueError("unsupported latent dataset format")
    latent_parts: dict[GroupKey, list[torch.Tensor]] = {}
    id_parts: dict[GroupKey, list[torch.Tensor]] = {}
    for shard in manifest["shards"]:
        if shard["split"] != split:
            continue
        payload = torch.load(latent_dir / shard["file"], map_location="cpu", weights_only=False)
        if payload.get("format") != "qh_flow_latents_v1" or payload["split"] != split:
            raise ValueError(f"invalid latent shard {shard['file']}")
        key = tuple(int(value) for value in payload["key"])
        latent_parts.setdefault(key, []).append(payload["latents"].float())
        id_parts.setdefault(key, []).append(payload["ids"].int())
    groups = {
        key: torch.cat(parts).contiguous().to(device=device)
        for key, parts in latent_parts.items()
    }
    ids = {
        key: torch.cat(id_parts[key]).contiguous().to(device=device)
        for key in groups
    }
    if not groups:
        raise ValueError(f"latent dataset has no {split!r} samples")
    return groups, ids, manifest


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


@torch.inference_mode()
def evaluate_groups(
    model: nn.Module,
    groups: dict[GroupKey, torch.Tensor],
    ids: dict[GroupKey, torch.Tensor],
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, np.ndarray]:
    was_training = model.training
    model.eval()
    probability_parts = []
    label_parts = []
    group_parts = []
    id_output_parts = []
    rms_parts = []
    for key in sorted(groups):
        positive = groups[key]
        generator = torch.Generator(device=device).manual_seed(
            int(seed) + key[0] * 1000003 + key[1] * 10007
        )
        for start in range(0, len(positive), batch_size):
            stop = min(start + batch_size, len(positive))
            positive_batch = positive[start:stop]
            negative_batch = torch.randn(
                positive_batch.shape,
                dtype=positive_batch.dtype,
                device=device,
                generator=generator,
            )
            tokens = torch.cat([positive_batch, negative_batch], dim=0)
            nfp = torch.full((len(tokens),), key[0], dtype=torch.long, device=device)
            with autocast_context(use_bf16):
                logits = model(tokens, nfp)
            probabilities = torch.sigmoid(logits.float())
            probability_parts.append(probabilities.cpu().numpy())
            label_parts.append(
                np.r_[np.ones(len(positive_batch), dtype=np.int8), np.zeros(len(negative_batch), dtype=np.int8)]
            )
            group_parts.append(
                np.full(len(tokens), f"nfp{key[0]}_nc{key[1]}", dtype="U16")
            )
            id_output_parts.append(
                np.r_[ids[key][start:stop].cpu().numpy(), np.full(len(negative_batch), -1, dtype=np.int32)]
            )
            rms_parts.append(
                torch.sqrt(torch.mean(tokens.float() ** 2, dim=(1, 2))).cpu().numpy()
            )
    if was_training:
        model.train()
    return {
        "probability": np.concatenate(probability_parts),
        "label": np.concatenate(label_parts),
        "group": np.concatenate(group_parts),
        "id": np.concatenate(id_output_parts),
        "latent_rms": np.concatenate(rms_parts),
    }


def summarize_evaluation(
    values: dict[str, np.ndarray],
    *,
    threshold: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = binary_metrics(
        values["probability"], values["label"], threshold=threshold
    )
    summary["enrichment"] = enrichment_at_prior_rates(
        values["probability"], values["label"]
    )
    summary["groups"] = {}
    for group in sorted(np.unique(values["group"])):
        selected = values["group"] == group
        summary["groups"][str(group)] = binary_metrics(
            values["probability"][selected],
            values["label"][selected],
            threshold=threshold,
        )
    return summary


def radial_baseline(
    validation: dict[str, np.ndarray], test: dict[str, np.ndarray]
) -> dict[str, float | str]:
    positive_auc = roc_auc(validation["latent_rms"], validation["label"])
    direction = 1.0 if positive_auc >= 0.5 else -1.0
    return {
        "orientation": "larger_rms_is_positive" if direction > 0 else "smaller_rms_is_positive",
        "validation_roc_auc": roc_auc(direction * validation["latent_rms"], validation["label"]),
        "test_roc_auc": roc_auc(direction * test["latent_rms"], test["label"]),
    }


def roc_pr_curves(
    probability: np.ndarray, label: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-probability, kind="stable")
    sorted_label = label[order].astype(np.int64)
    tp = np.cumsum(sorted_label)
    fp = np.cumsum(1 - sorted_label)
    recall = tp / max(int(np.sum(label)), 1)
    false_positive_rate = fp / max(int(np.sum(1 - label)), 1)
    precision = tp / np.arange(1, len(label) + 1)
    return (
        np.r_[0.0, false_positive_rate],
        np.r_[0.0, recall],
        np.r_[0.0, recall],
        np.r_[1.0, precision],
    )


def plot_monitor(metrics_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    train = [row for row in rows if row["event"] == "train"]
    validation = [row for row in rows if row["event"] == "validation"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    if train:
        axes[0, 0].plot([row["step"] for row in train], [row["loss"] for row in train], color="#145a78")
        axes[0, 0].set(title="Training BCE", xlabel="step", ylabel="BCE")
        axes[1, 0].plot([row["step"] for row in train], [row["accuracy"] for row in train], color="#9b4f1d")
        axes[1, 0].set(title="Training accuracy", xlabel="step", ylabel="accuracy", ylim=(0.45, 1.01))
    if validation:
        steps = [row["step"] for row in validation]
        axes[0, 1].plot(steps, [row["log_loss"] for row in validation], color="#7a3e9d", label="BCE")
        axes[0, 1].set(title="Held-out validation BCE", xlabel="step", ylabel="BCE")
        axes[1, 1].plot(steps, [row["roc_auc"] for row in validation], color="#1f7a4d", label="ROC-AUC")
        axes[1, 1].plot(steps, [row["average_precision"] for row in validation], color="#bb3e55", label="AP")
        axes[1, 1].set(title="Held-out validation ranking", xlabel="step", ylabel="metric", ylim=(0.45, 1.01))
        axes[1, 1].legend()
        learning_rate = axes[0, 1].twinx()
        learning_rate.plot(steps, [row["learning_rate"] for row in validation], color="#777777", alpha=0.5)
        learning_rate.set_ylabel("learning rate")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_test_evaluation(
    values: dict[str, np.ndarray],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confusion = summary["confusion"]
    matrix = np.asarray(
        [[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]],
        dtype=float,
    )
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, image, title, fmt in (
        (axes[0], matrix, "Test confusion matrix: counts", ".0f"),
        (axes[1], normalized, "Test confusion matrix: row-normalized", ".3f"),
    ):
        axis.imshow(image, cmap="Blues", vmin=0.0)
        axis.set_xticks((0, 1), ("predicted random", "predicted QUASR"))
        axis.set_yticks((0, 1), ("true random", "true QUASR"))
        axis.tick_params(axis="x", rotation=15)
        axis.set_title(title)
        for row in range(2):
            for column in range(2):
                axis.text(column, row, format(image[row, column], fmt), ha="center", va="center")
    figure.savefig(output_dir / "test_confusion_matrix.png", dpi=190)
    plt.close(figure)

    fpr, tpr, recall, precision = roc_pr_curves(values["probability"], values["label"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    axes[0].plot(fpr, tpr, color="#176b87", lw=2, label=f"AUC = {summary['roc_auc']:.4f}")
    axes[0].plot((0, 1), (0, 1), color="#888888", ls="--")
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title="Held-out test ROC")
    axes[0].legend()
    axes[1].plot(recall, precision, color="#a34137", lw=2, label=f"AP = {summary['average_precision']:.4f}")
    axes[1].axhline(0.5, color="#888888", ls="--")
    axes[1].set(xlabel="recall", ylabel="precision", title="Held-out test precision-recall")
    axes[1].legend()
    figure.savefig(output_dir / "test_roc_pr.png", dpi=190)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.hist(values["probability"][values["label"] == 0], bins=60, density=True, alpha=0.6, label="random prior", color="#486a88")
    axis.hist(values["probability"][values["label"] == 1], bins=60, density=True, alpha=0.6, label="inverse QUASR", color="#b5523b")
    axis.axvline(summary["threshold"], color="#222222", ls="--", label="validation threshold")
    axis.set(xlabel="proxy probability", ylabel="density", title="Held-out test predictions")
    axis.legend()
    figure.savefig(output_dir / "test_probability_distribution.png", dpi=190)
    plt.close(figure)

    enrichment = summary["enrichment"]
    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.plot(
        [row["actual_prior_pass_rate"] for row in enrichment],
        [row["positive_retention_rate"] for row in enrichment],
        "o-",
        color="#2e7d5b",
    )
    axis.plot((1.0e-3, 1.0), (1.0e-3, 1.0), color="#888888", ls="--", label="random")
    axis.set(xscale="log", yscale="log", xlabel="random-prior pass rate", ylabel="held-out QUASR retention", title="Proxy screening enrichment")
    axis.legend()
    figure.savefig(output_dir / "test_enrichment.png", dpi=190)
    plt.close(figure)


def save_checkpoint(
    path: Path,
    *,
    model: LatentProxyTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_auc: float,
    validation: dict[str, Any],
    args: argparse.Namespace,
    latent_manifest: dict[str, Any],
) -> None:
    torch.save(
        {
            "format": "qh_latent_proxy_v1",
            "model": model.state_dict(),
            "model_config": model.config,
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "best_validation_auc": float(best_auc),
            "validation": validation,
            "args": vars(args),
            "latent_checkpoint_sha256": latent_manifest["checkpoint_sha256"],
            "latent_source_manifest_sha256": latent_manifest["source_manifest_sha256"],
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test the QH flow latent support proxy.")
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--batch-per-gpu", type=int, default=2048)
    parser.add_argument("--eval-batch", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--plateau-validations", type=int, default=10)
    parser.add_argument("--final-plateau-validations", type=int, default=20)
    parser.add_argument("--max-lr-reductions", type=int, default=3)
    parser.add_argument("--lr-reduction-factor", type=float, default=0.3)
    parser.add_argument("--minimum-improvement", type=float, default=1.0e-5)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=704)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--no-bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps < 1 or args.batch_per_gpu < 1 or args.validation_interval < 1:
        raise ValueError("step, batch, and validation intervals must be positive")
    process_started = time.perf_counter()
    rank, local_rank, world_size, device = setup()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "checkpoints").mkdir()
    barrier(world_size, local_rank)
    use_bf16 = device.type == "cuda" and not args.no_bf16

    train_groups, train_ids, latent_manifest = load_latent_groups(args.latent_dir, "train", device=device)
    validation_groups: dict[GroupKey, torch.Tensor] = {}
    validation_ids: dict[GroupKey, torch.Tensor] = {}
    test_groups: dict[GroupKey, torch.Tensor] = {}
    test_ids: dict[GroupKey, torch.Tensor] = {}
    if rank == 0:
        validation_groups, validation_ids, _ = load_latent_groups(args.latent_dir, "validation", device=device)
        test_groups, test_ids, _ = load_latent_groups(args.latent_dir, "test", device=device)

    model_config = {
        "token_dim": 100,
        "width": args.width,
        "layers": args.layers,
        "heads": args.heads,
        "hidden": args.hidden,
        "max_nfp": 16,
    }
    base_model = LatentProxyTransformer(**model_config).to(device)
    train_model: nn.Module = base_model
    if world_size > 1:
        train_model = torch.nn.parallel.DistributedDataParallel(
            base_model,
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
    criterion = nn.BCEWithLogitsLoss()
    keys = sorted(train_groups)
    counts = np.asarray([len(train_groups[key]) for key in keys], dtype=np.float64)
    probabilities = counts / counts.sum()
    key_rng = np.random.default_rng(args.seed)
    sample_generator = torch.Generator(device=device).manual_seed(args.seed + rank * 1000003)
    metrics_path = args.output_dir / "metrics.jsonl"
    if rank == 0:
        run_manifest = {
            "args": vars(args),
            "world_size": world_size,
            "device": str(device),
            "use_bf16": use_bf16,
            "parameter_count": base_model.parameter_count,
            "model_config": model_config,
            "train_counts": {f"nfp{key[0]}_nc{key[1]}": len(train_groups[key]) for key in keys},
            "latent_manifest": str((args.latent_dir / "manifest.json").resolve()),
            "latent_checkpoint_sha256": latent_manifest["checkpoint_sha256"],
        }
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "start", **run_manifest}, default=str), flush=True)

    best_loss = math.inf
    best_auc = -math.inf
    best_auc_loss = math.inf
    stale_validations = 0
    lr_reductions = 0
    stopped_reason = "max_steps"
    step = 0
    interval_loss = torch.zeros(3, device=device, dtype=torch.float64)
    interval_started = time.perf_counter()
    training_started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        if args.warmup_steps > 0 and step <= args.warmup_steps:
            warmup_lr = args.learning_rate * step / args.warmup_steps
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr
        key = keys[int(key_rng.choice(len(keys), p=probabilities))]
        positive_source = train_groups[key]
        indices = torch.randint(
            len(positive_source),
            (args.batch_per_gpu,),
            generator=sample_generator,
            device=device,
        )
        positive = positive_source[indices]
        negative = torch.randn(
            positive.shape,
            dtype=positive.dtype,
            device=device,
            generator=sample_generator,
        )
        tokens = torch.cat([positive, negative], dim=0)
        labels = torch.cat(
            [
                torch.ones(len(positive), device=device),
                torch.zeros(len(negative), device=device),
            ]
        )
        nfp = torch.full((len(tokens),), key[0], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_bf16):
            logits = train_model(tokens, nfp)
            loss = criterion(logits.float(), labels)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            accuracy = torch.mean(((logits >= 0) == labels.bool()).float())
            interval_loss += torch.stack((loss.detach().double(), accuracy.double(), gradient_norm.detach().double()))

        if step % args.log_interval == 0:
            values = interval_loss.clone()
            if world_size > 1:
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= args.log_interval * world_size
            if rank == 0:
                row = {
                    "event": "train",
                    "step": step,
                    "loss": float(values[0].cpu()),
                    "accuracy": float(values[1].cpu()),
                    "gradient_norm": float(values[2].cpu()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "steps_per_s": args.log_interval / max(time.perf_counter() - interval_started, 1.0e-9),
                }
                append_jsonl(metrics_path, row)
                print(json.dumps(row), flush=True)
            interval_loss.zero_()
            interval_started = time.perf_counter()

        if step % args.validation_interval != 0:
            continue
        barrier(world_size, local_rank)
        validation_payload: list[dict[str, Any] | None] = [None]
        if rank == 0:
            evaluated = evaluate_groups(
                base_model,
                validation_groups,
                validation_ids,
                batch_size=args.eval_batch,
                seed=args.seed + 100000000,
                device=device,
                use_bf16=use_bf16,
            )
            threshold = validation_threshold(evaluated["probability"], evaluated["label"])
            validation_summary = summarize_evaluation(evaluated, threshold=threshold)
            validation_summary.update(
                {
                    "event": "validation",
                    "step": step,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            validation_payload[0] = validation_summary
        if world_size > 1:
            dist.broadcast_object_list(validation_payload, src=0)
        validation_summary = validation_payload[0]
        assert validation_summary is not None

        current_loss = float(validation_summary["log_loss"])
        current_auc = float(validation_summary["roc_auc"])
        improved_loss = current_loss < best_loss - args.minimum_improvement
        if improved_loss:
            best_loss = current_loss
            stale_validations = 0
        else:
            stale_validations += 1
        improved_auc = current_auc > best_auc + 1.0e-5 or (
            abs(current_auc - best_auc) <= 1.0e-5 and current_loss < best_auc_loss
        )
        if improved_auc:
            best_auc = current_auc
            best_auc_loss = current_loss
        if rank == 0:
            validation_summary["best_loss"] = best_loss
            validation_summary["best_auc"] = best_auc
            validation_summary["stale_validations"] = stale_validations
            validation_summary["lr_reductions"] = lr_reductions
            append_jsonl(metrics_path, validation_summary)
            print(json.dumps(validation_summary, separators=(",", ":")), flush=True)
            if improved_auc:
                save_checkpoint(
                    args.output_dir / "checkpoint_best_auc.pt",
                    model=base_model,
                    optimizer=optimizer,
                    step=step,
                    best_auc=best_auc,
                    validation=validation_summary,
                    args=args,
                    latent_manifest=latent_manifest,
                )
            plot_monitor(metrics_path, args.output_dir / "training_monitor.png")

        should_stop = False
        patience = (
            args.plateau_validations
            if lr_reductions < args.max_lr_reductions
            else args.final_plateau_validations
        )
        if stale_validations >= patience:
            if lr_reductions < args.max_lr_reductions:
                for group in optimizer.param_groups:
                    group["lr"] *= args.lr_reduction_factor
                lr_reductions += 1
                stale_validations = 0
                if rank == 0:
                    append_jsonl(
                        metrics_path,
                        {
                            "event": "learning_rate_reduction",
                            "step": step,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "reduction": lr_reductions,
                        },
                    )
            else:
                should_stop = True
                stopped_reason = "validation_plateau_after_final_lr_reduction"
        barrier(world_size, local_rank)
        if should_stop:
            break

    barrier(world_size, local_rank)
    if rank == 0:
        checkpoint_path = args.output_dir / "checkpoint_best_auc.pt"
        if not checkpoint_path.is_file():
            raise RuntimeError("training completed without a best-AUC checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint["model"])
        validation_values = evaluate_groups(
            base_model,
            validation_groups,
            validation_ids,
            batch_size=args.eval_batch,
            seed=args.seed + 100000000,
            device=device,
            use_bf16=use_bf16,
        )
        threshold = validation_threshold(validation_values["probability"], validation_values["label"])
        validation_summary = summarize_evaluation(validation_values, threshold=threshold)
        test_values = evaluate_groups(
            base_model,
            test_groups,
            test_ids,
            batch_size=args.eval_batch,
            seed=args.seed + 200000000,
            device=device,
            use_bf16=use_bf16,
        )
        test_summary = summarize_evaluation(test_values, threshold=threshold)
        summary = {
            "format": "qh_latent_proxy_evaluation_v1",
            "selected_checkpoint": str(checkpoint_path),
            "selected_step": int(checkpoint["step"]),
            "training_final_step": step,
            "training_stop_reason": stopped_reason,
            "validation": validation_summary,
            "test": test_summary,
            "radial_baseline": radial_baseline(validation_values, test_values),
            "runtime": {
                "training_s": time.perf_counter() - training_started,
                "process_s": time.perf_counter() - process_started,
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            },
        }
        (args.output_dir / "evaluation_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            args.output_dir / "test_predictions.npz",
            **test_values,
            threshold=np.asarray(threshold),
        )
        np.savez_compressed(
            args.output_dir / "validation_predictions.npz",
            **validation_values,
            threshold=np.asarray(threshold),
        )
        plot_test_evaluation(test_values, test_summary, args.output_dir)
        plot_monitor(metrics_path, args.output_dir / "training_monitor.png")
        print(json.dumps({"event": "complete", **summary}, separators=(",", ":")), flush=True)
    barrier(world_size, local_rank)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
