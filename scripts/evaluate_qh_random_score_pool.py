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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    score = np.asarray([row["score"] for row in rows], dtype=float)
    status = np.asarray([row["status"] for row in rows], dtype="U32")
    ok = status == "ok"
    return {
        "count": len(rows),
        "score": {
            "mean": float(np.mean(score)),
            "median": float(np.median(score)),
            "p90": float(np.percentile(score, 90)),
            "p95": float(np.percentile(score, 95)),
            "p99": float(np.percentile(score, 99)),
            "p99_5": float(np.percentile(score, 99.5)),
            "max": float(np.max(score)),
        },
        "score_exceedance_counts": {
            str(threshold): int(np.sum(score >= threshold)) for threshold in (10, 20, 30, 40, 50)
        },
        "score_exceedance_rates": {
            str(threshold): float(np.mean(score >= threshold)) for threshold in (10, 20, 30, 40, 50)
        },
        "status_counts": {str(value): int(np.sum(status == value)) for value in np.unique(status)},
        "status_ok_rate": float(np.mean(ok)),
        "status_ok_score": {
            "count": int(np.sum(ok)),
            "mean": float(np.mean(score[ok])) if np.any(ok) else None,
            "median": float(np.median(score[ok])) if np.any(ok) else None,
            "max": float(np.max(score[ok])) if np.any(ok) else None,
        },
    }


def plot_distribution(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    score = np.asarray([row["score"] for row in rows], dtype=float)
    status = np.asarray([row["status"] for row in rows], dtype="U32")
    ok = status == "ok"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    bins = np.linspace(0.0, max(55.0, float(score.max()) + 1.0), 70)
    axes[0].hist(score[~ok], bins=bins, alpha=0.65, color="#9a4d42", label="rejected")
    axes[0].hist(score[ok], bins=bins, alpha=0.65, color="#237a57", label="status=ok")
    axes[0].axvline(40.0, color="#444444", ls="--", label="40 / 50 thresholds")
    axes[0].axvline(50.0, color="#111111", ls=":")
    axes[0].set(xlabel="native score", ylabel="count", title="IID random latent scores")
    axes[0].legend()

    ordered = np.sort(score)
    survival = 1.0 - np.arange(len(ordered)) / len(ordered)
    axes[1].step(ordered, survival, where="post", color="#486a88")
    axes[1].axvline(40.0, color="#444444", ls="--")
    axes[1].axvline(50.0, color="#111111", ls=":")
    axes[1].set(yscale="log", xlabel="native score", ylabel="empirical survival", title="IID score upper tail")
    figure.savefig(output_dir / "iid_random_score_distribution.png", dpi=190)
    plt.close(figure)


def prepare(args: argparse.Namespace) -> None:
    import torch

    from flow_matching.data import CoilNormalizer
    from flow_matching.flow import integrate_flow
    from flow_matching.model import CoilFlowTransformer
    from scripts.optimize_native_score_cem import token_case

    if not torch.cuda.is_available():
        raise RuntimeError("random-pool flow decoding requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    checkpoint = torch.load(args.flow_checkpoint, map_location="cpu", weights_only=False)
    flow = CoilFlowTransformer(**checkpoint["model_config"]).to(device=device, dtype=torch.float32)
    flow.load_state_dict(checkpoint["ema"])
    flow.eval()
    normalizer = CoilNormalizer.from_dict(checkpoint["normalizer"])
    rng = np.random.default_rng(args.seed)
    latent = rng.standard_normal((args.count, args.n_base_coils, 100), dtype=np.float32)

    decoded_parts = []
    started = time.perf_counter()
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
    decode_s = time.perf_counter() - started
    decoded = np.concatenate(decoded_parts)
    raw_tokens = normalizer.inverse(decoded, (args.nfp, args.n_base_coils))
    with (args.output_dir / "prepared_cases.jsonl").open("w", encoding="utf-8") as stream:
        for case_id in range(len(latent)):
            case = token_case(
                raw_tokens[case_id].astype(np.float64),
                nfp=args.nfp,
                target="QH",
                metadata={"random_pool_case_id": case_id, "seed": args.seed},
            )
            row = {
                "case_id": case_id,
                "latent_rms": float(np.sqrt(np.mean(latent[case_id].astype(np.float64) ** 2))),
                "case": case,
            }
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    np.savez_compressed(args.output_dir / "random_latents.npz", latent=latent)
    manifest = {
        "format": "qh_iid_random_score_pool_v1",
        "stage": "prepared",
        "args": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "flow_checkpoint_sha256": file_sha256(args.flow_checkpoint),
        "flow_checkpoint_step": int(checkpoint["step"]),
        "runtime": {
            "decode_s": decode_s,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "prepared", **manifest}, separators=(",", ":")), flush=True)


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
        results = pool.map([row["case"] for row in prepared], target="QH", timeout_s=args.timeout_s)
    rows = []
    with score_path.open("w", encoding="utf-8") as stream:
        for row, (result, elapsed, error) in zip(prepared, results, strict=True):
            compact = compact_score_diagnostics(result) if result is not None else None
            output = {
                "case_id": row["case_id"],
                "latent_rms": row["latent_rms"],
                "score": float(compact["score"]) if compact is not None else 0.0,
                "status": compact["status"] if compact is not None else "error",
                "native": compact,
                "score_wall_s": elapsed,
                "error": error,
            }
            rows.append(output)
            stream.write(json.dumps(output, separators=(",", ":"), allow_nan=True) + "\n")
    summary = summarize(rows)
    (args.output_dir / "score_distribution_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_distribution(rows, args.output_dir)
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
    parser = argparse.ArgumentParser(description="Decode and native-score a pure IID QH flow-latent pool.")
    parser.add_argument("--mode", choices=("prepare", "score"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path)
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--nfp", type=int, default=4)
    parser.add_argument("--n-base-coils", type=int, default=3)
    parser.add_argument("--flow-steps", type=int, default=256)
    parser.add_argument("--flow-batch", type=int, default=256)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--timeout-s", type=float, default=10800.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        if args.flow_checkpoint is None:
            raise ValueError("prepare mode requires --flow-checkpoint")
        prepare(args)
    else:
        if args.lib is None:
            raise ValueError("score mode requires --lib")
        score(args)


if __name__ == "__main__":
    main()
