from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
COEFF_COUNT = 33
TOKEN_DIM = 3 * COEFF_COUNT + 1


def load_pca_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    components = np.asarray(payload["components"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    if mean.shape != (TOKEN_DIM,):
        raise ValueError(f"PCA mean must have shape ({TOKEN_DIM},), got {mean.shape}")
    if components.shape[0] != TOKEN_DIM or components.shape[1] != scale.size:
        raise ValueError("PCA components/scale shapes are inconsistent")
    return {
        "mean": mean,
        "components": components,
        "scale": scale,
        "current_scale": float(payload.get("current_scale", 1.0e6)),
    }


def decode_latents(latents: np.ndarray, pca: dict[str, Any]) -> np.ndarray:
    latents = np.asarray(latents, dtype=np.float32)
    decoded = (
        (latents * pca["scale"]) @ pca["components"].T + pca["mean"]
    ).astype(np.float64)
    decoded[..., -1] *= float(pca["current_scale"])
    return decoded


def normalize_current_l1(tokens: np.ndarray, target_l1_a: float) -> np.ndarray:
    out = np.asarray(tokens, dtype=np.float64).copy()
    currents_a = out[..., -1]
    norms = np.sum(np.abs(currents_a), axis=-1, keepdims=True)
    fallback = np.full_like(currents_a, -float(target_l1_a) / currents_a.shape[-1])
    scaled = np.where(
        norms > 1.0e-12,
        currents_a * (float(target_l1_a) / np.maximum(norms, 1.0e-12)),
        fallback,
    )
    out[..., -1] = scaled
    return out


def token_case(tokens: np.ndarray, *, nfp: int, target: str, metadata: dict[str, Any] | None = None) -> dict:
    tokens = np.atleast_2d(np.asarray(tokens, dtype=np.float64))
    if tokens.shape[1] != TOKEN_DIM:
        raise ValueError(f"tokens must have {TOKEN_DIM} columns")
    helicity = 0 if target.upper() == "QA" else 1
    raw_metadata = {"helicity": helicity, "native_score_target": target.upper()}
    raw_metadata.update(metadata or {})
    return {
        "nfp": int(nfp),
        "raw": {
            "x": tokens[:, :COEFF_COUNT].tolist(),
            "y": tokens[:, COEFF_COUNT : 2 * COEFF_COUNT].tolist(),
            "z": tokens[:, 2 * COEFF_COUNT : 3 * COEFF_COUNT].tolist(),
            "current": tokens[:, -1].tolist(),
            "current_unit": "A",
            "nfp": int(nfp),
            "metadata": raw_metadata,
        },
    }


def update_distribution(
    mean: np.ndarray,
    sigma: np.ndarray,
    elite: np.ndarray,
    *,
    smoothing: float,
    min_sigma: float,
    max_sigma: float,
    latent_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    elite_mean = np.mean(elite, axis=0)
    elite_sigma = np.std(elite, axis=0)
    next_mean = smoothing * mean + (1.0 - smoothing) * elite_mean
    next_sigma = smoothing * sigma + (1.0 - smoothing) * elite_sigma
    return (
        np.clip(next_mean, -latent_limit, latent_limit).astype(np.float32),
        np.clip(next_sigma, min_sigma, max_sigma).astype(np.float32),
    )


def _score_worker(task_q: mp.Queue, result_q: mp.Queue, lib_path: str, gpu_id: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(GPU_PYTHON))
    from stellarator_gpu import score_coils_native

    while True:
        task = task_q.get()
        if task is None:
            return
        task_id, case, target = task
        started = time.perf_counter()
        try:
            raw = case["raw"]
            nfp = int(case["nfp"])
            target_helicity = (1, 0 if target == "QA" else nfp)
            result = score_coils_native(
                lib_path,
                raw["x"],
                raw["y"],
                raw["z"],
                raw["current"],
                nfp,
                device_id=0,
                target_helicity=target_helicity,
            )
            result_q.put((task_id, result, time.perf_counter() - started, None))
        except Exception as exc:
            result_q.put(
                (task_id, None, time.perf_counter() - started, f"{exc!r}\n{traceback.format_exc()}")
            )


class NativeScorePool:
    def __init__(self, lib_path: Path, gpu_ids: list[int]):
        self.ctx = mp.get_context("spawn")
        self.task_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.workers = [
            self.ctx.Process(
                target=_score_worker,
                args=(self.task_q, self.result_q, str(lib_path.resolve()), gpu_id),
                name=f"native-score-gpu-{gpu_id}",
            )
            for gpu_id in gpu_ids
        ]
        for worker in self.workers:
            worker.start()

    def map(
        self,
        cases: list[dict],
        *,
        target: str,
        timeout_s: float,
    ) -> list[tuple[dict[str, Any] | None, float, str | None]]:
        for index, case in enumerate(cases):
            self.task_q.put((index, case, target))
        deadline = time.monotonic() + timeout_s
        results: dict[int, tuple[dict[str, Any] | None, float, str | None]] = {}
        while len(results) < len(cases):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(set(range(len(cases))) - set(results))
                raise TimeoutError(f"native score batch timed out; missing candidate IDs {missing}")
            try:
                task_id, result, elapsed, error = self.result_q.get(timeout=min(5.0, remaining))
            except queue.Empty:
                dead = [worker.name for worker in self.workers if not worker.is_alive()]
                if dead:
                    raise RuntimeError(f"native score workers exited unexpectedly: {dead}")
                continue
            results[int(task_id)] = (result, float(elapsed), error)
        return [results[index] for index in range(len(cases))]

    def close(self) -> None:
        for _ in self.workers:
            self.task_q.put(None)
        for worker in self.workers:
            worker.join(timeout=20)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)

    def __enter__(self) -> "NativeScorePool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, separators=(",", ":"), allow_nan=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the native C++/CUDA score using diagonal CEM.")
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument(
        "--lib",
        type=Path,
        default=REPO_ROOT / "gpu_backend" / "build_native_score" / "libstellarator_gpu.so",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("QA", "QH"), required=True)
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--n-base-coils", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=32)
    parser.add_argument("--elite", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--min-sigma", type=float, default=0.01)
    parser.add_argument("--max-sigma", type=float, default=0.8)
    parser.add_argument("--smoothing", type=float, default=0.55)
    parser.add_argument("--latent-limit", type=float, default=3.0)
    parser.add_argument("--random-start-std", type=float, default=1.0)
    parser.add_argument("--current-l1-a", type=float)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-timeout-s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    if args.nfp <= 0 or args.n_base_coils <= 0:
        raise ValueError("nfp and n-base-coils must be positive")
    if not 1 <= args.elite <= args.popsize:
        raise ValueError("elite must be in [1, popsize]")
    gpu_ids = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("at least one GPU is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.out_dir / "candidates.jsonl"
    if candidates_path.exists():
        raise FileExistsError(f"refusing to overwrite existing run {candidates_path}")

    pca = load_pca_checkpoint(args.pca)
    latent_dim = int(pca["scale"].size)
    rng = np.random.default_rng(args.seed)
    mean = rng.normal(
        0.0,
        args.random_start_std,
        size=(args.n_base_coils, latent_dim),
    ).astype(np.float32)
    mean = np.clip(mean, -args.latent_limit, args.latent_limit)
    sigma = np.full_like(mean, args.sigma, dtype=np.float32)

    center_tokens = decode_latents(mean[None], pca)[0]
    center_l1_a = float(np.sum(np.abs(center_tokens[:, -1])))
    current_l1_a = float(args.current_l1_a or max(center_l1_a, 1.0e5))
    manifest = {
        "target": args.target,
        "nfp": args.nfp,
        "n_base_coils": args.n_base_coils,
        "latent_dim_per_coil": latent_dim,
        "seed": args.seed,
        "iterations": args.iterations,
        "popsize": args.popsize,
        "elite": args.elite,
        "sigma": args.sigma,
        "smoothing": args.smoothing,
        "latent_limit": args.latent_limit,
        "current_l1_a": current_l1_a,
        "pca_path": str(args.pca.resolve()),
        "pca_sha256": file_sha256(args.pca),
        "native_lib_path": str(args.lib.resolve()),
        "native_lib_sha256": file_sha256(args.lib),
        "gpu_ids": gpu_ids,
    }
    write_json(args.out_dir / "manifest.json", manifest)

    best_score = float("-inf")
    best_latent: np.ndarray | None = None
    best_case: dict[str, Any] | None = None
    best_result: dict[str, Any] | None = None
    best_generation = 0
    best_candidate = -1
    start_score: float | None = None
    generation_rows = []
    started = time.perf_counter()

    with NativeScorePool(args.lib, gpu_ids) as pool:
        for generation in range(1, args.iterations + 1):
            generation_started = time.perf_counter()
            eps = rng.normal(size=(args.popsize, *mean.shape)).astype(np.float32)
            population = mean[None] + sigma[None] * eps
            population[0] = mean
            if best_latent is not None and args.popsize > 1:
                population[1] = best_latent
            population = np.clip(population, -args.latent_limit, args.latent_limit)
            tokens = decode_latents(population, pca)
            tokens = normalize_current_l1(tokens, current_l1_a)
            cases = [
                token_case(
                    tokens[index],
                    nfp=args.nfp,
                    target=args.target,
                    metadata={
                        "cem_generation": generation,
                        "cem_candidate": index,
                        "cem_seed": args.seed,
                    },
                )
                for index in range(args.popsize)
            ]
            evaluated = pool.map(cases, target=args.target, timeout_s=args.batch_timeout_s)
            scores = np.full(args.popsize, -np.inf, dtype=np.float64)
            statuses: dict[str, int] = {}
            errors = []
            for index, (result, elapsed, error) in enumerate(evaluated):
                if error is not None or result is None:
                    errors.append({"candidate": index, "error": error})
                    append_jsonl(
                        candidates_path,
                        {
                            "generation": generation,
                            "candidate": index,
                            "elapsed_s": elapsed,
                            "error": error,
                        },
                    )
                    continue
                score = float(result["score"])
                status = str(result["status"])
                scores[index] = score
                statuses[status] = statuses.get(status, 0) + 1
                append_jsonl(
                    candidates_path,
                    {
                        "generation": generation,
                        "candidate": index,
                        "score": score,
                        "status": status,
                        "elapsed_s": elapsed,
                        "latent": population[index].tolist(),
                        "native_score": result,
                    },
                )
                if generation == 1 and index == 0:
                    start_score = score
                if score > best_score:
                    best_score = score
                    best_latent = population[index].copy()
                    best_case = cases[index]
                    best_result = result
                    best_generation = generation
                    best_candidate = index

            finite = np.isfinite(scores)
            if np.count_nonzero(finite) < args.elite:
                raise RuntimeError(
                    f"generation {generation} has only {np.count_nonzero(finite)} finite scores"
                )
            elite_indices = np.argsort(scores)[::-1][: args.elite]
            mean, sigma = update_distribution(
                mean,
                sigma,
                population[elite_indices],
                smoothing=args.smoothing,
                min_sigma=args.min_sigma,
                max_sigma=args.max_sigma,
                latent_limit=args.latent_limit,
            )
            row = {
                "generation": generation,
                "score_mean": float(np.mean(scores[finite])),
                "score_median": float(np.median(scores[finite])),
                "score_max": float(np.max(scores[finite])),
                "best_score": best_score,
                "best_generation": best_generation,
                "best_candidate": best_candidate,
                "sigma_mean": float(np.mean(sigma)),
                "sigma_max": float(np.max(sigma)),
                "statuses": statuses,
                "errors": errors,
                "wall_s": time.perf_counter() - generation_started,
            }
            generation_rows.append(row)
            write_json(args.out_dir / "progress.json", {"manifest": manifest, "generations": generation_rows})
            print(json.dumps(row, separators=(",", ":")), flush=True)

    if best_case is None or best_result is None or best_latent is None:
        raise RuntimeError("CEM did not produce a valid candidate")
    best_case["cem"] = {
        "target": args.target,
        "seed": args.seed,
        "generation": best_generation,
        "candidate": best_candidate,
        "start_score": start_score,
        "best_score": best_score,
        "latent": best_latent.tolist(),
        "native_score": best_result,
        "manifest": manifest,
    }
    write_json(args.out_dir / "best.json", best_case)
    summary = {
        "manifest": manifest,
        "start_score": start_score,
        "best_score": best_score,
        "best_generation": best_generation,
        "best_candidate": best_candidate,
        "best_status": best_result["status"],
        "best_components": best_result["components"],
        "best_diagnostics": best_result["diagnostics"],
        "generations": generation_rows,
        "total_wall_s": time.perf_counter() - started,
        "best_case": str((args.out_dir / "best.json").resolve()),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps({"best_score": best_score, "best_case": summary["best_case"]}), flush=True)


if __name__ == "__main__":
    main()
