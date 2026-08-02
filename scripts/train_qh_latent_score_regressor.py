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
from scipy.stats import pearsonr, spearmanr
import torch
from torch import distributed as dist
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.proxy import LatentProxyTransformer
from scripts.prepare_qh_score_regression_dataset import (
    DATASET_FORMAT,
    SCORE_BINS,
    file_sha256,
    score_bin_label,
)


CHECKPOINT_FORMAT = "qh_latent_score_regressor_v1"


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
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)


def autocast_context(use_bf16: bool):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def load_split(dataset_dir: Path, split: str, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(dataset_dir / f"{split}.pt", map_location="cpu", weights_only=False)
    if payload.get("format") != DATASET_FORMAT or payload.get("split") != split:
        raise ValueError(f"invalid {split} score-regression dataset")
    required = ("tokens", "mask", "nfp", "n_coils", "target", "status")
    if any(name not in payload for name in required):
        raise ValueError(f"{split} dataset is missing required tensors")
    if not torch.isfinite(payload["tokens"]).all() or not torch.isfinite(payload["target"]).all():
        raise ValueError(f"{split} dataset contains non-finite values")
    if torch.any((payload["target"] < 0.0) | (payload["target"] > 1.0)):
        raise ValueError(f"{split} target lies outside [0, 1]")
    return {name: payload[name].to(device=device) for name in required}


def safe_correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float | None:
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    value = pearsonr(x, y).statistic if kind == "pearson" else spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def basic_regression_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    if predicted.shape != actual.shape or predicted.size == 0:
        raise ValueError("predicted and actual scores must be nonempty with equal shape")
    error = predicted - actual
    mse = float(np.mean(error**2))
    variance = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "count": int(len(actual)),
        "mse": mse,
        "mse_normalized": mse / 10000.0,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error**2) / variance) if variance > 0.0 else None,
        "pearson": safe_correlation(predicted, actual, "pearson"),
        "spearman": safe_correlation(predicted, actual, "spearman"),
        "predicted_mean": float(np.mean(predicted)),
        "actual_mean": float(np.mean(actual)),
        "actual_median": float(np.median(actual)),
        "actual_p10": float(np.quantile(actual, 0.10)),
        "actual_p90": float(np.quantile(actual, 0.90)),
        "actual_max": float(np.max(actual)),
    }


def subset_metrics(predicted: np.ndarray, actual: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=bool)
    if not np.any(selected):
        return {"count": 0}
    summary = basic_regression_metrics(predicted[selected], actual[selected])
    summary.update(
        {
            "fraction": float(np.mean(selected)),
            "actual_fraction_gt_20": float(np.mean(actual[selected] > 20.0)),
            "actual_fraction_gt_30": float(np.mean(actual[selected] > 30.0)),
            "actual_min": float(np.min(actual[selected])),
        }
    )
    return summary


def complete_regression_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    *,
    nfp: np.ndarray,
    n_coils: np.ndarray,
    status: np.ndarray,
    status_names: dict[int, str],
) -> dict[str, Any]:
    summary = basic_regression_metrics(predicted, actual)
    summary["predicted_thresholds"] = {
        f"gt_{threshold:g}": subset_metrics(predicted, actual, predicted > threshold)
        for threshold in (10.0, 20.0, 30.0, 40.0)
    }
    summary["actual_score_strata"] = {}
    for index in range(len(SCORE_BINS) - 1):
        lower = SCORE_BINS[index]
        upper = SCORE_BINS[index + 1]
        selected = (actual >= lower) & (actual < upper)
        summary["actual_score_strata"][score_bin_label(index)] = subset_metrics(
            predicted, actual, selected
        )
    summary["top_prediction_fractions"] = {}
    order = np.argsort(-predicted, kind="stable")
    for fraction in (0.10, 0.05, 0.01, 0.005):
        count = max(1, int(math.ceil(len(predicted) * fraction)))
        selected = np.zeros(len(predicted), dtype=bool)
        selected[order[:count]] = True
        summary["top_prediction_fractions"][f"top_{fraction:g}"] = subset_metrics(
            predicted, actual, selected
        )
    summary["status"] = {}
    for code in sorted(np.unique(status)):
        selected = status == code
        summary["status"][status_names[int(code)]] = subset_metrics(predicted, actual, selected)
    summary["conditions"] = {}
    for nfp_value, coil_value in sorted(set(zip(nfp.tolist(), n_coils.tolist()))):
        selected = (nfp == nfp_value) & (n_coils == coil_value)
        summary["conditions"][f"nfp{nfp_value}_nc{coil_value}"] = subset_metrics(
            predicted, actual, selected
        )
    return summary


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    data: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    was_training = model.training
    model.eval()
    predictions = []
    for start in range(0, len(data["target"]), batch_size):
        stop = min(start + batch_size, len(data["target"]))
        logits = model(
            data["tokens"][start:stop],
            data["nfp"][start:stop],
            data["mask"][start:stop],
        )
        predictions.append((100.0 * torch.sigmoid(logits.float())).cpu().numpy())
    if was_training:
        model.train()
    return {
        "predicted": np.concatenate(predictions),
        "actual": (100.0 * data["target"]).cpu().numpy(),
        "nfp": data["nfp"].cpu().numpy(),
        "n_coils": data["n_coils"].cpu().numpy(),
        "status": data["status"].cpu().numpy(),
    }


def epoch_indices(
    count: int,
    *,
    epoch: int,
    seed: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + epoch * 1000003)
    order = torch.randperm(count, generator=generator)
    per_rank = int(math.ceil(count / world_size))
    total = per_rank * world_size
    if total > count:
        order = torch.cat([order, order[: total - count]])
    return order[rank:total:world_size]


def save_checkpoint(
    path: Path,
    *,
    model: LatentProxyTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    best_validation_mse: float,
    validation: dict[str, Any],
    args: argparse.Namespace,
    dataset_manifest: dict[str, Any],
) -> None:
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model": model.state_dict(),
            "model_config": model.config,
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "epoch": int(epoch),
            "best_validation_mse_normalized": float(best_validation_mse),
            "validation": validation,
            "args": vars(args),
            "dataset_snapshot_digest": dataset_manifest["snapshot"]["included_shard_digest"],
            "score_library_sha256": dataset_manifest["score_library_sha256"],
        },
        path,
    )


def plot_training(metrics_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    train = [row for row in rows if row["event"] == "train"]
    validation = [row for row in rows if row["event"] == "validation"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot([row["step"] for row in train], [row["rmse_score"] for row in train], color="#176b87")
    axes[0, 0].set(title="Training RMSE", xlabel="step", ylabel="score")
    axes[1, 0].plot([row["step"] for row in train], [row["mae_score"] for row in train], color="#a04a35")
    axes[1, 0].set(title="Training MAE", xlabel="step", ylabel="score")
    steps = [row["step"] for row in validation]
    axes[0, 1].plot(steps, [row["rmse"] for row in validation], color="#2b7a4b")
    axes[0, 1].set(title="Held-out validation RMSE", xlabel="step", ylabel="score")
    axes[1, 1].plot(steps, [row["pearson"] for row in validation], color="#8a4d9f", label="Pearson")
    axes[1, 1].plot(steps, [row["spearman"] for row in validation], color="#d28b25", label="Spearman")
    axes[1, 1].set(title="Held-out validation correlation", xlabel="step", ylabel="correlation")
    axes[1, 1].legend()
    if validation:
        learning_rate = axes[0, 1].twinx()
        learning_rate.plot(steps, [row["learning_rate"] for row in validation], color="#777777", alpha=0.45)
        learning_rate.set_ylabel("learning rate")
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def plot_test(
    values: dict[str, np.ndarray],
    summary: dict[str, Any],
    output_dir: Path,
    status_names: dict[int, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predicted = values["predicted"]
    actual = values["actual"]
    figure, axis = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    palette = {
        "ok": "#237a57",
        "no_axis": "#c74b3b",
        "no_surface": "#d48632",
        "drift_rejected": "#7d5aa6",
        "flux_rejected": "#4385b5",
    }
    for code in sorted(np.unique(values["status"])):
        selected = values["status"] == code
        label = status_names[int(code)]
        axis.scatter(
            actual[selected],
            predicted[selected],
            s=10,
            alpha=0.38,
            linewidths=0,
            color=palette.get(label, "#777777"),
            label=f"{label} ({np.sum(selected)})",
        )
    limit = max(45.0, float(np.max(actual)) + 2.0, float(np.max(predicted)) + 2.0)
    axis.plot((0.0, limit), (0.0, limit), color="#222222", ls="--", lw=1.3, label="ideal")
    axis.set(xlim=(0, limit), ylim=(0, limit), xlabel="actual native score", ylabel="predicted score", title="Held-out test: predicted vs actual score")
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(output_dir / "test_prediction_scatter.png", dpi=210)
    plt.close(figure)

    edges = np.quantile(predicted, np.linspace(0.0, 1.0, 11))
    bins = np.digitize(predicted, edges[1:-1], right=True)
    bin_predicted = []
    bin_actual = []
    bin_p10 = []
    bin_p90 = []
    for index in range(10):
        selected = bins == index
        bin_predicted.append(float(np.mean(predicted[selected])))
        bin_actual.append(float(np.mean(actual[selected])))
        bin_p10.append(float(np.quantile(actual[selected], 0.10)))
        bin_p90.append(float(np.quantile(actual[selected], 0.90)))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].errorbar(
        bin_predicted,
        bin_actual,
        yerr=[np.asarray(bin_actual) - np.asarray(bin_p10), np.asarray(bin_p90) - np.asarray(bin_actual)],
        fmt="o-",
        color="#176b87",
        capsize=3,
        label="actual mean and P10-P90",
    )
    calibration_limit = max(max(bin_predicted), max(bin_actual)) + 1.0
    axes[0].plot((0, calibration_limit), (0, calibration_limit), color="#777777", ls="--")
    axes[0].set(xlabel="mean predicted score", ylabel="actual score", title="Prediction-decile calibration")
    axes[0].legend()
    axes[1].hist(actual, bins=60, alpha=0.58, density=True, color="#2f7f5f", label="actual")
    axes[1].hist(predicted, bins=60, alpha=0.58, density=True, color="#b85b3f", label="predicted")
    axes[1].set(xlabel="score", ylabel="density", title="Held-out score distributions")
    axes[1].legend()
    figure.savefig(output_dir / "test_calibration_distribution.png", dpi=200)
    plt.close(figure)

    thresholds = (10.0, 20.0, 30.0, 40.0)
    counts = [summary["predicted_thresholds"][f"gt_{value:g}"]["count"] for value in thresholds]
    actual_means = [
        summary["predicted_thresholds"][f"gt_{value:g}"].get("actual_mean", 0.0)
        for value in thresholds
    ]
    actual_p10 = [
        summary["predicted_thresholds"][f"gt_{value:g}"].get("actual_p10", 0.0)
        for value in thresholds
    ]
    actual_p90 = [
        summary["predicted_thresholds"][f"gt_{value:g}"].get("actual_p90", 0.0)
        for value in thresholds
    ]
    figure, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    axis.errorbar(
        thresholds,
        actual_means,
        yerr=[np.asarray(actual_means) - np.asarray(actual_p10), np.asarray(actual_p90) - np.asarray(actual_means)],
        fmt="o-",
        color="#7b4c9d",
        capsize=4,
    )
    for x_value, y_value, count in zip(thresholds, actual_means, counts, strict=True):
        axis.annotate(f"n={count}", (x_value, y_value), xytext=(0, 8), textcoords="offset points", ha="center")
    axis.plot((0, 45), (0, 45), color="#888888", ls="--", lw=1)
    axis.set(xlabel="prediction threshold", ylabel="actual score", title="Actual quality above prediction thresholds")
    figure.savefig(output_dir / "test_high_prediction_tail.png", dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a four-GPU latent native-score regressor.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--batch-per-gpu", type=int, default=2048)
    parser.add_argument("--eval-batch", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--plateau-validations", type=int, default=16)
    parser.add_argument("--final-plateau-validations", type=int, default=32)
    parser.add_argument("--max-lr-reductions", type=int, default=4)
    parser.add_argument("--lr-reduction-factor", type=float, default=0.3)
    parser.add_argument("--minimum-improvement", type=float, default=1.0e-7)
    parser.add_argument("--rise-relative-margin", type=float, default=0.002)
    parser.add_argument("--rise-window", type=int, default=5)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=704)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--no-bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.max_steps, args.batch_per_gpu, args.validation_interval) < 1:
        raise ValueError("training counts and intervals must be positive")
    process_started = time.perf_counter()
    rank, local_rank, world_size, device = setup()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "checkpoints").mkdir()
    barrier(world_size, local_rank)
    dataset_manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if dataset_manifest.get("format") != DATASET_FORMAT:
        raise ValueError("unsupported frozen regression dataset")
    train_data = load_split(args.dataset_dir, "train", device)
    validation_data = load_split(args.dataset_dir, "validation", device) if rank == 0 else None
    test_data = load_split(args.dataset_dir, "test", device) if rank == 0 else None
    status_names = {
        int(code): name
        for name, code in dataset_manifest["representation"]["status_codes"].items()
    }

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
    use_bf16 = device.type == "cuda" and not args.no_bf16
    metrics_path = args.output_dir / "metrics.jsonl"
    if rank == 0:
        run_manifest = {
            "args": vars(args),
            "world_size": world_size,
            "device": str(device),
            "use_bf16_training_forward": use_bf16,
            "fp32_loss_validation_test": True,
            "parameter_count": base_model.parameter_count,
            "model_config": model_config,
            "dataset_manifest": str((args.dataset_dir / "manifest.json").resolve()),
            "dataset_snapshot_digest": dataset_manifest["snapshot"]["included_shard_digest"],
            "score_library_sha256": dataset_manifest["score_library_sha256"],
            "split_counts": {name: dataset_manifest["splits"][name]["count"] for name in ("train", "validation", "test")},
        }
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "start", **run_manifest}, default=str), flush=True)

    step = 0
    epoch = 0
    best_validation_mse = math.inf
    best_validation_step = 0
    stale_validations = 0
    lr_reductions = 0
    rise_observed = False
    recent_validation: list[float] = []
    stop_requested = False
    stopped_reason = "max_steps"
    interval_squared_error = torch.zeros(3, dtype=torch.float64, device=device)
    interval_started = time.perf_counter()
    training_started = time.perf_counter()
    while step < args.max_steps and not stop_requested:
        indices = epoch_indices(
            len(train_data["target"]),
            epoch=epoch,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )
        for start in range(0, len(indices), args.batch_per_gpu):
            if step >= args.max_steps or stop_requested:
                break
            batch_indices = indices[start : start + args.batch_per_gpu].to(device=device)
            step += 1
            if args.warmup_steps > 0 and step <= args.warmup_steps:
                learning_rate = args.learning_rate * step / args.warmup_steps
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(use_bf16):
                logits = train_model(
                    train_data["tokens"][batch_indices],
                    train_data["nfp"][batch_indices],
                    train_data["mask"][batch_indices],
                )
            prediction = torch.sigmoid(logits.float())
            target = train_data["target"][batch_indices]
            error = prediction - target
            loss = torch.mean(error**2)
            loss.backward()
            optimizer.step()
            interval_squared_error += torch.stack(
                [torch.sum(error.detach() ** 2), torch.sum(torch.abs(error.detach())), torch.tensor(float(len(error)), device=device)]
            ).to(torch.float64)

            if step % args.log_interval == 0:
                totals = interval_squared_error.clone()
                if world_size > 1:
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                if rank == 0:
                    count = max(float(totals[2].item()), 1.0)
                    row = {
                        "event": "train",
                        "step": step,
                        "epoch": epoch,
                        "mse_normalized": float(totals[0].item() / count),
                        "rmse_score": float(100.0 * math.sqrt(totals[0].item() / count)),
                        "mae_score": float(100.0 * totals[1].item() / count),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "interval_wall_s": time.perf_counter() - interval_started,
                    }
                    append_jsonl(metrics_path, row)
                    print(json.dumps(row), flush=True)
                interval_squared_error.zero_()
                interval_started = time.perf_counter()

            if step % args.validation_interval == 0:
                barrier(world_size, local_rank)
                control = torch.zeros(5, dtype=torch.float64, device=device)
                if rank == 0:
                    assert validation_data is not None
                    validation_values = evaluate_model(base_model, validation_data, batch_size=args.eval_batch)
                    validation = basic_regression_metrics(
                        validation_values["predicted"], validation_values["actual"]
                    )
                    validation_mse = float(validation["mse_normalized"])
                    improved = validation_mse < best_validation_mse - args.minimum_improvement
                    if improved:
                        best_validation_mse = validation_mse
                        best_validation_step = step
                        stale_validations = 0
                        save_checkpoint(
                            args.output_dir / "checkpoint_best_validation.pt",
                            model=base_model,
                            optimizer=optimizer,
                            step=step,
                            epoch=epoch,
                            best_validation_mse=best_validation_mse,
                            validation=validation,
                            args=args,
                            dataset_manifest=dataset_manifest,
                        )
                    else:
                        stale_validations += 1
                    recent_validation.append(validation_mse)
                    recent_validation = recent_validation[-args.rise_window :]
                    if (
                        len(recent_validation) == args.rise_window
                        and np.median(recent_validation)
                        > best_validation_mse * (1.0 + args.rise_relative_margin)
                    ):
                        rise_observed = True
                    if stale_validations >= args.plateau_validations and lr_reductions < args.max_lr_reductions:
                        for group in optimizer.param_groups:
                            group["lr"] *= args.lr_reduction_factor
                        lr_reductions += 1
                        stale_validations = 0
                    elif (
                        lr_reductions >= args.max_lr_reductions
                        and stale_validations >= args.final_plateau_validations
                        and rise_observed
                    ):
                        stop_requested = True
                        stopped_reason = "validation_rise_after_final_lr_reduction"
                    row = {
                        "event": "validation",
                        "step": step,
                        "epoch": epoch,
                        **validation,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "best_validation_mse_normalized": best_validation_mse,
                        "best_validation_step": best_validation_step,
                        "stale_validations": stale_validations,
                        "lr_reductions": lr_reductions,
                        "validation_rise_observed": rise_observed,
                    }
                    append_jsonl(metrics_path, row)
                    print(json.dumps(row), flush=True)
                    control[:] = torch.tensor(
                        [optimizer.param_groups[0]["lr"], float(stop_requested), best_validation_mse, stale_validations, lr_reductions],
                        dtype=torch.float64,
                        device=device,
                    )
                if world_size > 1:
                    dist.broadcast(control, src=0)
                for group in optimizer.param_groups:
                    group["lr"] = float(control[0].item())
                stop_requested = bool(round(float(control[1].item())))
                best_validation_mse = float(control[2].item())
                stale_validations = int(round(float(control[3].item())))
                lr_reductions = int(round(float(control[4].item())))
                barrier(world_size, local_rank)
        epoch += 1

    barrier(world_size, local_rank)
    if rank == 0:
        if not (args.output_dir / "checkpoint_best_validation.pt").is_file():
            raise RuntimeError("training produced no validation checkpoint")
        save_checkpoint(
            args.output_dir / "checkpoint_latest.pt",
            model=base_model,
            optimizer=optimizer,
            step=step,
            epoch=epoch,
            best_validation_mse=best_validation_mse,
            validation={"best_step": best_validation_step},
            args=args,
            dataset_manifest=dataset_manifest,
        )
        checkpoint = torch.load(
            args.output_dir / "checkpoint_best_validation.pt",
            map_location=device,
            weights_only=False,
        )
        base_model.load_state_dict(checkpoint["model"])
        assert validation_data is not None and test_data is not None
        validation_values = evaluate_model(base_model, validation_data, batch_size=args.eval_batch)
        test_values = evaluate_model(base_model, test_data, batch_size=args.eval_batch)
        validation_summary = complete_regression_metrics(
            **validation_values,
            status_names=status_names,
        )
        test_summary = complete_regression_metrics(
            **test_values,
            status_names=status_names,
        )

        train_actual = (100.0 * train_data["target"]).cpu().numpy()
        global_mean = float(np.mean(train_actual))
        global_prediction = np.full_like(test_values["actual"], global_mean)
        condition_means: dict[tuple[int, int], float] = {}
        train_nfp = train_data["nfp"].cpu().numpy()
        train_n_coils = train_data["n_coils"].cpu().numpy()
        for key in set(zip(train_nfp.tolist(), train_n_coils.tolist())):
            selected = (train_nfp == key[0]) & (train_n_coils == key[1])
            condition_means[key] = float(np.mean(train_actual[selected]))
        condition_prediction = np.asarray(
            [condition_means.get((int(nfp), int(nc)), global_mean) for nfp, nc in zip(test_values["nfp"], test_values["n_coils"], strict=True)],
            dtype=np.float64,
        )
        baselines = {
            "global_train_mean": basic_regression_metrics(global_prediction, test_values["actual"]),
            "condition_train_mean": basic_regression_metrics(condition_prediction, test_values["actual"]),
        }
        sample_ids = (args.dataset_dir / "test_sample_ids.txt").read_text(encoding="utf-8").splitlines()
        if len(sample_ids) != len(test_values["actual"]):
            raise ValueError("test sample-id count does not match tensors")
        np.savez_compressed(
            args.output_dir / "test_predictions.npz",
            sample_id=np.asarray(sample_ids),
            predicted_score=test_values["predicted"],
            actual_score=test_values["actual"],
            nfp=test_values["nfp"],
            n_coils=test_values["n_coils"],
            status=test_values["status"],
        )
        plot_training(metrics_path, args.output_dir / "training_monitor.png")
        plot_test(test_values, test_summary, args.output_dir, status_names)
        summary = {
            "format": "qh_latent_score_regression_evaluation_v1",
            "checkpoint": {
                "file": "checkpoint_best_validation.pt",
                "sha256": file_sha256(args.output_dir / "checkpoint_best_validation.pt"),
                "step": int(checkpoint["step"]),
                "epoch": int(checkpoint["epoch"]),
            },
            "stopping": {
                "reason": stopped_reason,
                "final_step": step,
                "final_epoch": epoch,
                "best_validation_step": best_validation_step,
                "validation_rise_observed": rise_observed,
                "lr_reductions": lr_reductions,
            },
            "validation": validation_summary,
            "test": test_summary,
            "test_baselines": baselines,
            "timing": {
                "training_s": time.perf_counter() - training_started,
                "total_process_s": time.perf_counter() - process_started,
            },
        }
        (args.output_dir / "evaluation_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "complete", **summary}, allow_nan=False), flush=True)
    barrier(world_size, local_rank)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
