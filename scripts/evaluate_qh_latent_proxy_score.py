from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "gpu_backend" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ints(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise ValueError("GPU IDs must be distinct nonnegative integers")
    return values


def select_cases(
    probabilities: np.ndarray,
    *,
    stratified_count: int,
    iid_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[int, list[str]]]:
    if stratified_count < 1 or iid_count < 1 or stratified_count + iid_count > len(probabilities):
        raise ValueError("invalid stratified/iid sample counts")
    order = np.argsort(probabilities, kind="stable")
    positions = np.linspace(0, len(order) - 1, stratified_count, dtype=np.int64)
    stratified = np.unique(order[positions])
    if len(stratified) != stratified_count:
        remaining_positions = np.setdiff1d(np.arange(len(order)), positions, assume_unique=False)
        needed = stratified_count - len(stratified)
        stratified = np.r_[stratified, order[remaining_positions[:needed]]]
    available = np.setdiff1d(np.arange(len(probabilities)), stratified, assume_unique=False)
    iid = rng.choice(available, size=iid_count, replace=False)
    modes = {int(index): ["prediction_rank_stratified"] for index in stratified}
    for index in iid:
        modes.setdefault(int(index), []).append("iid_prior")
    selected = np.asarray(sorted(modes), dtype=np.int64)
    return selected, modes


def prepare(args: argparse.Namespace) -> None:
    import torch

    from flow_matching.data import CoilNormalizer
    from flow_matching.flow import integrate_flow
    from flow_matching.model import CoilFlowTransformer
    from flow_matching.proxy import LatentProxyTransformer, apply_logit_calibration
    from scripts.optimize_native_score_cem import token_case

    if not torch.cuda.is_available():
        raise RuntimeError("proxy prediction and flow decoding require CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    started = time.perf_counter()

    proxy_checkpoint = torch.load(args.proxy_checkpoint, map_location="cpu", weights_only=False)
    if proxy_checkpoint.get("format") != "qh_latent_proxy_v1":
        raise ValueError("unsupported proxy checkpoint")
    proxy = LatentProxyTransformer(**proxy_checkpoint["model_config"]).to(device=device)
    proxy.load_state_dict(proxy_checkpoint["model"])
    proxy.eval()
    calibration = {"method": "identity", "scale": 1.0, "bias": 0.0}
    if args.calibration_summary is not None:
        calibration_summary = json.loads(args.calibration_summary.read_text(encoding="utf-8"))
        if calibration_summary.get("checkpoint_sha256") != file_sha256(args.proxy_checkpoint):
            raise ValueError("calibration summary does not match the proxy checkpoint")
        calibration = calibration_summary["calibration"]
    flow_checkpoint = torch.load(args.flow_checkpoint, map_location="cpu", weights_only=False)
    flow = CoilFlowTransformer(**flow_checkpoint["model_config"]).to(device=device, dtype=torch.float32)
    flow.load_state_dict(flow_checkpoint["ema"])
    flow.eval()
    normalizer = CoilNormalizer.from_dict(flow_checkpoint["normalizer"])

    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal(
        (args.pool_count, args.n_base_coils, 100), dtype=np.float32
    )
    raw_logits = []
    prediction_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(noise), args.proxy_batch):
            stop = min(start + args.proxy_batch, len(noise))
            tokens = torch.from_numpy(noise[start:stop]).to(device=device)
            nfp = torch.full((stop - start,), args.nfp, dtype=torch.long, device=device)
            logits = proxy(tokens, nfp)
            raw_logits.append(logits.float().cpu().numpy())
    raw_logits_np = np.concatenate(raw_logits).astype(np.float64)
    calibrated_logits_np = (
        float(calibration["scale"]) * raw_logits_np + float(calibration["bias"])
    )
    probabilities_np = apply_logit_calibration(
        raw_logits_np,
        scale=float(calibration["scale"]),
        bias=float(calibration["bias"]),
    )
    torch.cuda.synchronize(device)
    prediction_s = time.perf_counter() - prediction_started
    selected, modes = select_cases(
        probabilities_np,
        stratified_count=args.stratified_count,
        iid_count=args.iid_count,
        rng=rng,
    )

    selected_noise = noise[selected]
    decoded_parts = []
    decode_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(selected_noise), args.flow_batch):
            stop = min(start + args.flow_batch, len(selected_noise))
            latent = torch.from_numpy(selected_noise[start:stop]).to(device=device)
            nfp = torch.full((stop - start,), args.nfp, dtype=torch.long, device=device)
            decoded = integrate_flow(
                flow,
                latent,
                nfp,
                start_time=0.0,
                end_time=1.0,
                steps=args.flow_steps,
                method="rk4",
            )
            decoded_parts.append(decoded.cpu().numpy())
    torch.cuda.synchronize(device)
    decode_s = time.perf_counter() - decode_started
    decoded_normalized = np.concatenate(decoded_parts)
    raw_tokens = normalizer.inverse(decoded_normalized, (args.nfp, args.n_base_coils))

    prepared_path = args.output_dir / "prepared_cases.jsonl"
    with prepared_path.open("w", encoding="utf-8") as stream:
        for local_index, pool_index in enumerate(selected):
            case = token_case(
                raw_tokens[local_index].astype(np.float64),
                nfp=args.nfp,
                target="QH",
                metadata={"proxy_pool_index": int(pool_index)},
            )
            row = {
                "case_id": local_index,
                "pool_index": int(pool_index),
                "sampling_modes": modes[int(pool_index)],
                "proxy_probability": float(probabilities_np[pool_index]),
                "proxy_logit": float(calibrated_logits_np[pool_index]),
                "proxy_raw_logit": float(raw_logits_np[pool_index]),
                "latent_rms": float(np.sqrt(np.mean(selected_noise[local_index].astype(np.float64) ** 2))),
                "case": case,
            }
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    np.savez_compressed(
        args.output_dir / "proxy_pool_predictions.npz",
        probability=probabilities_np,
        calibrated_logit=calibrated_logits_np,
        raw_logit=raw_logits_np,
        selected_pool_index=selected,
        selected_probability=probabilities_np[selected],
        selected_noise=selected_noise,
    )
    manifest = {
        "format": "qh_latent_proxy_score_correlation_v1",
        "stage": "prepared",
        "args": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "proxy_checkpoint_sha256": file_sha256(args.proxy_checkpoint),
        "flow_checkpoint_sha256": file_sha256(args.flow_checkpoint),
        "proxy_checkpoint_step": int(proxy_checkpoint["step"]),
        "calibration": calibration,
        "flow_checkpoint_step": int(flow_checkpoint["step"]),
        "pool_probability": {
            "min": float(np.min(probabilities_np)),
            "median": float(np.median(probabilities_np)),
            "max": float(np.max(probabilities_np)),
        },
        "selected_count": len(selected),
        "runtime": {
            "prediction_s": prediction_s,
            "decode_s": decode_s,
            "process_s": time.perf_counter() - started,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "prepared", **manifest}, separators=(",", ":")), flush=True)


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | None]:
    from scipy.stats import pearsonr, spearmanr

    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return {"count": len(x), "pearson": None, "spearman": None}
    return {
        "count": len(x),
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def analyze(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    probability = np.asarray([row["proxy_probability"] for row in rows], dtype=float)
    score = np.asarray([row["score"] for row in rows], dtype=float)
    status = np.asarray([row["status"] for row in rows], dtype="U32")
    logit = np.asarray(
        [
            row.get(
                "proxy_logit",
                np.log(np.clip(row["proxy_probability"], 1.0e-7, 1.0 - 1.0e-7))
                - np.log1p(-np.clip(row["proxy_probability"], 1.0e-7, 1.0 - 1.0e-7)),
            )
            for row in rows
        ],
        dtype=float,
    )
    subsets = {
        "all": np.ones(len(rows), dtype=bool),
        "iid_prior": np.asarray(["iid_prior" in row["sampling_modes"] for row in rows]),
        "prediction_rank_stratified": np.asarray(
            ["prediction_rank_stratified" in row["sampling_modes"] for row in rows]
        ),
        "status_ok": status == "ok",
    }
    correlations = {}
    for name, selected in subsets.items():
        correlations[name] = {
            "probability_vs_score": correlation(probability[selected], score[selected]),
            "logit_vs_score": correlation(logit[selected], score[selected]),
        }

    order = np.argsort(probability, kind="stable")
    bins = []
    for index, indices in enumerate(np.array_split(order, 10)):
        values = score[indices]
        bins.append(
            {
                "bin": index,
                "count": len(indices),
                "probability_min": float(np.min(probability[indices])),
                "probability_max": float(np.max(probability[indices])),
                "probability_mean": float(np.mean(probability[indices])),
                "score_mean": float(np.mean(values)),
                "score_median": float(np.median(values)),
                "score_p90": float(np.percentile(values, 90)),
                "status_ok_rate": float(np.mean(status[indices] == "ok")),
            }
        )
    summary = {
        "count": len(rows),
        "score": {
            "mean": float(np.mean(score)),
            "median": float(np.median(score)),
            "p90": float(np.percentile(score, 90)),
            "max": float(np.max(score)),
        },
        "status_counts": {
            str(value): int(np.sum(status == value)) for value in np.unique(status)
        },
        "correlation": correlations,
        "prediction_bins": bins,
    }
    (output_dir / "score_correlation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_score_correlation(probability, logit, score, status, bins, output_dir)
    return summary


def plot_score_correlation(
    probability: np.ndarray,
    logit: np.ndarray,
    score: np.ndarray,
    status: np.ndarray,
    bins: list[dict[str, float | int]],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = np.where(status == "ok", "#237a57", "#9a4d42")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].scatter(probability, score, c=colors, s=12, alpha=0.48, edgecolors="none")
    axes[0].plot(
        [row["probability_mean"] for row in bins],
        [row["score_mean"] for row in bins],
        "o-",
        color="#111111",
        lw=2,
        label="decile mean",
    )
    axes[0].set(xlabel="proxy probability", ylabel="native score", title="Native score vs proxy prediction")
    axes[0].legend()
    axes[1].scatter(logit, score, c=colors, s=12, alpha=0.48, edgecolors="none")
    axes[1].set(xlabel="proxy logit", ylabel="native score", title="Native score vs proxy logit")
    figure.savefig(output_dir / "proxy_prediction_vs_native_score.png", dpi=190)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.7), constrained_layout=True)
    x = np.arange(len(bins))
    axis.plot(x, [row["score_mean"] for row in bins], "o-", label="mean score", color="#176b87")
    axis.plot(x, [row["score_median"] for row in bins], "s-", label="median score", color="#a34137")
    axis.plot(x, [row["score_p90"] for row in bins], "^-", label="P90 score", color="#6b4c9a")
    axis.set(xlabel="proxy prediction decile", ylabel="native score", title="Score trend across proxy prediction deciles")
    axis.legend()
    figure.savefig(output_dir / "proxy_score_decile_trend.png", dpi=190)
    plt.close(figure)


def score(args: argparse.Namespace) -> None:
    from scripts.optimize_native_score_cem import NativeScorePool, compact_score_diagnostics

    prepared_path = args.output_dir / "prepared_cases.jsonl"
    if not prepared_path.is_file():
        raise FileNotFoundError(prepared_path)
    score_path = args.output_dir / "scored_cases.jsonl"
    if score_path.exists():
        raise FileExistsError(score_path)
    prepared = [json.loads(line) for line in prepared_path.read_text(encoding="utf-8").splitlines()]
    gpu_ids = parse_ints(args.gpu_ids)
    started = time.perf_counter()
    with NativeScorePool(args.lib, gpu_ids) as pool:
        results = pool.map(
            [row["case"] for row in prepared],
            target="QH",
            timeout_s=args.timeout_s,
        )
    rows = []
    with score_path.open("w", encoding="utf-8") as stream:
        for row, (result, elapsed, error) in zip(prepared, results, strict=True):
            compact = compact_score_diagnostics(result) if result is not None else None
            output = {
                "case_id": row["case_id"],
                "pool_index": row["pool_index"],
                "sampling_modes": row["sampling_modes"],
                "proxy_probability": row["proxy_probability"],
                "proxy_logit": row.get("proxy_logit"),
                "proxy_raw_logit": row.get("proxy_raw_logit"),
                "latent_rms": row["latent_rms"],
                "score": float(compact["score"]) if compact is not None else 0.0,
                "status": compact["status"] if compact is not None else "error",
                "native": compact,
                "score_wall_s": elapsed,
                "error": error,
            }
            rows.append(output)
            stream.write(json.dumps(output, separators=(",", ":"), allow_nan=True) + "\n")
    summary = analyze(rows, args.output_dir)
    manifest_path = args.output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "stage": "complete",
            "score_library_sha256": file_sha256(args.lib),
            "gpu_ids": gpu_ids,
            "score_runtime_s": time.perf_counter() - started,
            "score_summary": summary,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "scored", **summary}, separators=(",", ":")), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure native score against latent proxy prediction.")
    parser.add_argument("--mode", choices=("prepare", "score"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proxy-checkpoint", type=Path)
    parser.add_argument("--flow-checkpoint", type=Path)
    parser.add_argument("--calibration-summary", type=Path)
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--nfp", type=int, default=4)
    parser.add_argument("--n-base-coils", type=int, default=3)
    parser.add_argument("--pool-count", type=int, default=131072)
    parser.add_argument("--stratified-count", type=int, default=768)
    parser.add_argument("--iid-count", type=int, default=256)
    parser.add_argument("--proxy-batch", type=int, default=16384)
    parser.add_argument("--flow-batch", type=int, default=256)
    parser.add_argument("--flow-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--timeout-s", type=float, default=10800.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        if args.proxy_checkpoint is None or args.flow_checkpoint is None:
            raise ValueError("prepare mode requires proxy and flow checkpoints")
        prepare(args)
    else:
        if args.lib is None:
            raise ValueError("score mode requires --lib")
        score(args)


if __name__ == "__main__":
    main()
