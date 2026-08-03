from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.proxy import (
    LatentProxyTransformer,
    apply_logit_calibration,
    fit_logit_calibration,
    validation_threshold,
)
from scripts.train_qh_latent_proxy import (
    evaluate_groups,
    load_latent_groups,
    plot_test_evaluation,
    radial_baseline,
    summarize_evaluation,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibrated_values(
    values: dict[str, np.ndarray], calibration: dict[str, Any]
) -> dict[str, np.ndarray]:
    output = dict(values)
    output["raw_probability"] = values["probability"]
    output["raw_logit"] = values["logit"]
    output["logit"] = (
        float(calibration["scale"]) * values["logit"] + float(calibration["bias"])
    )
    output["probability"] = apply_logit_calibration(
        values["logit"],
        scale=float(calibration["scale"]),
        bias=float(calibration["bias"]),
    )
    return output


def compact_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in (
            "count",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "roc_auc",
            "average_precision",
            "log_loss",
            "brier",
            "ece",
            "confusion",
        )
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authoritative FP32 evaluation of a trained QH latent proxy."
    )
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-batch", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_batch < 1:
        raise ValueError("eval batch must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("authoritative proxy evaluation requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    started = time.perf_counter()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "qh_latent_proxy_v1":
        raise ValueError("unsupported proxy checkpoint format")
    model = LatentProxyTransformer(**checkpoint["model_config"]).to(
        device=device, dtype=torch.float32
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    validation_groups, validation_ids, latent_manifest = load_latent_groups(
        args.latent_dir, "validation", device=device
    )
    test_groups, test_ids, _ = load_latent_groups(args.latent_dir, "test", device=device)

    inference_started = time.perf_counter()
    validation_raw = evaluate_groups(
        model,
        validation_groups,
        validation_ids,
        batch_size=args.eval_batch,
        seed=args.seed + 100000000,
        device=device,
        use_bf16=False,
    )
    test_raw = evaluate_groups(
        model,
        test_groups,
        test_ids,
        batch_size=args.eval_batch,
        seed=args.seed + 200000000,
        device=device,
        use_bf16=False,
    )
    torch.cuda.synchronize(device)
    inference_s = time.perf_counter() - inference_started

    calibration = fit_logit_calibration(
        validation_raw["logit"], validation_raw["label"]
    )
    if not calibration["success"]:
        raise RuntimeError(f"validation calibration failed: {calibration['message']}")
    validation = calibrated_values(validation_raw, calibration)
    test = calibrated_values(test_raw, calibration)
    threshold = validation_threshold(validation["probability"], validation["label"])
    validation_summary = summarize_evaluation(validation, threshold=threshold)
    test_summary = summarize_evaluation(test, threshold=threshold)

    raw_validation_threshold = validation_threshold(
        validation_raw["probability"], validation_raw["label"]
    )
    raw_test_summary = summarize_evaluation(test_raw, threshold=raw_validation_threshold)
    summary = {
        "format": "qh_latent_proxy_evaluation_v2",
        "authoritative": True,
        "inference_precision": "FP32",
        "metric_precision": "FP64",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "selected_step": int(checkpoint["step"]),
        "recorded_best_validation_auc": float(checkpoint["best_validation_auc"]),
        "latent_checkpoint_sha256": latent_manifest["checkpoint_sha256"],
        "calibration": calibration,
        "validation": validation_summary,
        "test": test_summary,
        "raw_test_before_calibration": compact_metrics(raw_test_summary),
        "radial_baseline": radial_baseline(validation_raw, test_raw),
        "runtime": {
            "inference_s": inference_s,
            "process_s": time.perf_counter() - started,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }
    (args.output_dir / "evaluation_summary_fp32.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "validation_predictions_fp32.npz",
        **validation,
        threshold=np.asarray(threshold),
    )
    np.savez_compressed(
        args.output_dir / "test_predictions_fp32.npz",
        **test,
        threshold=np.asarray(threshold),
    )
    plot_test_evaluation(test, test_summary, args.output_dir)
    print(json.dumps({"event": "complete", **summary}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
