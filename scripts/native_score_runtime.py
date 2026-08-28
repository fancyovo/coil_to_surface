from __future__ import annotations

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


def token_case(
    tokens: np.ndarray,
    *,
    nfp: int,
    target: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
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


def _score_worker(
    task_q: mp.Queue,
    result_q: mp.Queue,
    lib_path: str,
    gpu_id: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    sys.path.insert(0, str(GPU_PYTHON))
    from stellarator_gpu import score_coils_native

    while True:
        task = task_q.get()
        if task is None:
            return
        task_id, case, target, config_overrides = task
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
                config_overrides=config_overrides,
            )
            result_q.put((task_id, result, time.perf_counter() - started, None))
        except Exception as exc:
            result_q.put(
                (
                    task_id,
                    None,
                    time.perf_counter() - started,
                    f"{exc!r}\n{traceback.format_exc()}",
                )
            )


class NativeScorePool:
    """One persistent native-score process per selected GPU."""

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
        cases: list[dict[str, Any]],
        *,
        target: str,
        timeout_s: float,
        config_overrides: dict[str, Any] | None = None,
    ) -> list[tuple[dict[str, Any] | None, float, str | None]]:
        for index, case in enumerate(cases):
            self.task_q.put((index, case, target, config_overrides))
        deadline = time.monotonic() + timeout_s
        results: dict[int, tuple[dict[str, Any] | None, float, str | None]] = {}
        while len(results) < len(cases):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(set(range(len(cases))) - set(results))
                raise TimeoutError(
                    f"native score batch timed out; missing candidate IDs {missing}"
                )
            try:
                task_id, result, elapsed, error = self.result_q.get(
                    timeout=min(5.0, remaining)
                )
            except queue.Empty:
                dead = [worker.name for worker in self.workers if not worker.is_alive()]
                if dead:
                    raise RuntimeError(
                        f"native score workers exited unexpectedly: {dead}"
                    )
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
        self.task_q.close()
        self.result_q.close()
        self.task_q.join_thread()
        self.result_q.join_thread()

    def __enter__(self) -> "NativeScorePool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, separators=(",", ":"), allow_nan=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
