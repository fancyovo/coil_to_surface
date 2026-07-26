from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_WORKER: dict[str, Any] = {}
_CUDA_FAILURE_MARKERS = (
    "out of memory",
    "cudaerrormemoryallocation",
    "cuda-capable device is busy",
    "all cuda-capable devices are busy",
    "illegal memory access",
)


def _write_json(path: Path, payload: Any) -> None:
    from stellarator_eval.serialization import write_json

    write_json(path, payload)


def _gpu_snapshot(gpu_tokens: list[str] | None = None) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_rows = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
        process_rows = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        parsed_gpus = []
        selected_uuids = set()
        tokens = set(gpu_tokens or [])
        for row in gpu_rows.splitlines():
            fields = [field.strip() for field in row.split(",")]
            if len(fields) < 2:
                continue
            if not tokens or fields[0] in tokens or fields[1] in tokens:
                parsed_gpus.append(row)
                selected_uuids.add(fields[1])
        selected_processes = []
        for row in process_rows.splitlines():
            fields = [field.strip() for field in row.split(",")]
            if fields and (not selected_uuids or fields[0] in selected_uuids):
                selected_processes.append(row)
        return {"gpus": parsed_gpus, "compute_processes": selected_processes}
    except Exception as exc:
        return {"error": repr(exc)}


def _worker_init(
    gpu_tokens: list[str], workers_per_gpu: int, worker_count: int, settings: dict[str, Any]
) -> None:
    identity = mp.current_process()._identity
    worker_index = (int(identity[0]) - 1) % worker_count if identity else 0
    gpu_index = worker_index // workers_per_gpu
    gpu_token = gpu_tokens[gpu_index]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_token)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"

    started = time.perf_counter()
    import torch

    a = torch.eye(32, device="cuda:0", dtype=torch.float32)
    b = torch.ones((32, 1), device="cuda:0", dtype=torch.float32)
    torch.linalg.lstsq(a, b)
    torch.cuda.synchronize()
    _WORKER.update(
        settings=settings,
        worker_index=worker_index,
        gpu_index=gpu_index,
        gpu_token=str(gpu_token),
        warmup_s=float(time.perf_counter() - started),
    )


def _worker_task(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    metadata = task["metadata"]
    device_id = int(task["id"])
    settings = _WORKER["settings"]
    try:
        from stellarator_eval.config import EvalConfig
        from stellarator_eval.field import load_case_file
        from stellarator_eval.volume_pipeline import evaluate_coils_to_volume_qs
        from stellarator_eval.volume_score import evaluate_volume_quality_score

        field_input = load_case_file(task["case_path"], "raw")
        psi_a = float(settings["psi_a"])
        if settings["psi_a_mode"] == "metadata":
            minor_radius = metadata.get("minor_radius")
            if minor_radius is not None:
                psi_a = min(psi_a, 0.9 * float(minor_radius))
        base = EvalConfig(current_unit="A", omp_threads=1)
        volume = replace(
            base.volume_qs,
            point_count=int(settings["points"]),
            alpha_fit_point_count=int(settings["alpha_fit_points"]),
            alpha_radial_order=int(settings["alpha_order"]),
            alpha_poloidal_order=int(settings["alpha_order"]),
            alpha_toroidal_order=int(settings["alpha_order"]),
            precision=str(settings["precision"]),
            gpu_device=0,
        )
        scan = replace(
            base.scan,
            levels=tuple(float(value) for value in settings["levels"]),
            gpu_device=0,
        )
        config = replace(
            base,
            axis=replace(base.axis, gpu_device=0),
            psi=replace(base.psi, a=psi_a, gpu_device=0),
            scan=scan,
            volume_qs=volume,
        )
        helicity = int(metadata["helicity"])
        target = (1, 0 if helicity == 0 else int(field_input.nfp))
        result = evaluate_coils_to_volume_qs(
            field_input, config, target_helicity=target, output_dir=None
        )
        score = evaluate_volume_quality_score(
            result, field_input=field_input, current_unit="A"
        )
        elapsed = float(time.perf_counter() - started)
        record = {
            "id": device_id,
            "split": task["split"],
            "helicity": helicity,
            "nfp": int(field_input.nfp),
            "metadata_qs_error": float(metadata["qs_error"]),
            "metadata_mean_iota": float(metadata["mean_iota"]),
            "psi_a": psi_a,
            "status": score["status"],
            "pipeline_status": result.get("status"),
            "score": score["score"],
            "components": score["components"],
            "details": score["details"],
            "pipeline_timing": result.get("timing") or {},
            "worker_elapsed_s": elapsed,
            "reason": result.get("reason"),
            "worker_pid": os.getpid(),
            "worker_index": _WORKER["worker_index"],
            "gpu_index": _WORKER["gpu_index"],
            "gpu_token": _WORKER["gpu_token"],
            "worker_warmup_s": _WORKER["warmup_s"],
        }
        output_path = Path(task["output_path"])
        _write_json(
            output_path,
            {
                "metadata": metadata,
                "result": result,
                "volume_quality_score": score,
                "execution": {
                    key: record[key]
                    for key in (
                        "worker_elapsed_s",
                        "worker_pid",
                        "worker_index",
                        "gpu_index",
                        "gpu_token",
                        "worker_warmup_s",
                    )
                },
            },
        )
        return record
    except Exception as exc:
        return {
            "id": device_id,
            "split": task.get("split"),
            "helicity": int(metadata.get("helicity", -1)),
            "metadata_qs_error": float(metadata.get("qs_error", "nan")),
            "status": "error",
            "pipeline_status": "error",
            "score": None,
            "worker_elapsed_s": float(time.perf_counter() - started),
            "reason": repr(exc),
            "traceback": traceback.format_exc(),
            "worker_pid": os.getpid(),
            "worker_index": _WORKER.get("worker_index"),
            "gpu_index": _WORKER.get("gpu_index"),
            "gpu_token": _WORKER.get("gpu_token"),
            "worker_warmup_s": _WORKER.get("warmup_s"),
        }


def _load_completed(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed[int(row["id"])] = row
    return completed


def _tokens(gpu_count: int) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if not tokens:
        tokens = [str(index) for index in range(gpu_count)]
    if len(tokens) < gpu_count:
        raise ValueError(
            f"requested {gpu_count} GPUs but CUDA_VISIBLE_DEVICES exposes {len(tokens)}: {visible!r}"
        )
    return tokens[:gpu_count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "validation", "all"), default="all")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--alpha-fit-points", type=int, default=30000)
    parser.add_argument("--alpha-order", type=int, default=12)
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--psi-a", type=float, default=0.05)
    parser.add_argument("--psi-a-mode", choices=("fixed", "metadata"), default="fixed")
    parser.add_argument(
        "--levels",
        default="0.001,0.002,0.004,0.008,0.012,0.02,0.04,0.08,0.12,0.16,0.25,0.36,0.49,0.64,0.81",
    )
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if args.gpu_count <= 0 or args.workers_per_gpu <= 0:
        raise ValueError("gpu-count and workers-per-gpu must be positive")
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    metadata_rows = json.loads(
        (args.dataset_dir / "metadata_selected.json").read_text(encoding="utf-8")
    )
    metadata = {int(row["ID"]): row for row in metadata_rows}
    samples = [
        sample
        for sample in manifest["samples"]
        if args.split == "all" or sample.get("split", "calibration") == args.split
    ]
    if args.limit is not None:
        samples = samples[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "results.jsonl"
    if args.fresh and jsonl_path.exists():
        jsonl_path.unlink()
    completed = _load_completed(jsonl_path)
    tasks = []
    for sample in samples:
        device_id = int(sample["id"])
        if device_id in completed:
            continue
        tasks.append(
            {
                "id": device_id,
                "split": sample.get("split", "calibration"),
                "metadata": metadata[device_id],
                "case_path": str(args.dataset_dir / sample["file"]),
                "output_path": str(
                    args.output_dir / "cases" / f"id_{device_id:07d}" / "summary.json"
                ),
            }
        )

    gpu_tokens = _tokens(args.gpu_count)
    worker_count = args.gpu_count * args.workers_per_gpu
    settings = {
        "points": args.points,
        "alpha_fit_points": args.alpha_fit_points,
        "alpha_order": args.alpha_order,
        "precision": args.precision,
        "psi_a": args.psi_a,
        "psi_a_mode": args.psi_a_mode,
        "levels": [float(value) for value in args.levels.split(",") if value.strip()],
    }
    preflight = _gpu_snapshot(gpu_tokens)
    run_config = {
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "selected_count": len(samples),
        "pending_count": len(tasks),
        "gpu_count": args.gpu_count,
        "gpu_tokens": gpu_tokens,
        "workers_per_gpu": args.workers_per_gpu,
        "worker_count": worker_count,
        "settings": settings,
        "preflight_gpu": preflight,
    }
    _write_json(args.output_dir / "run_config.json", run_config)
    if preflight.get("compute_processes"):
        raise RuntimeError(
            "timing run requires idle allocated GPUs; preflight found compute processes: "
            + repr(preflight["compute_processes"])
        )

    print(json.dumps(run_config, indent=2), flush=True)
    batch_started = time.perf_counter()
    new_rows = []
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=worker_count,
        initializer=_worker_init,
        initargs=(gpu_tokens, args.workers_per_gpu, worker_count, settings),
    )
    try:
        with jsonl_path.open("a", encoding="utf-8") as stream:
            for index, row in enumerate(pool.imap_unordered(_worker_task, tasks), start=1):
                new_rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(
                    f"[{index:04d}/{len(tasks):04d}] id={row['id']} status={row['status']} "
                    f"score={row.get('score')} time={row['worker_elapsed_s']:.3f}s",
                    flush=True,
                )
                reason = str(row.get("reason") or "").lower()
                if row["status"] == "error" and any(
                    marker in reason for marker in _CUDA_FAILURE_MARKERS
                ):
                    raise RuntimeError(f"CUDA resource failure for id={row['id']}: {row['reason']}")
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    batch_wall_s = float(time.perf_counter() - batch_started)

    all_rows = list(_load_completed(jsonl_path).values())
    selected_ids = {int(sample["id"]) for sample in samples}
    all_rows = [row for row in all_rows if int(row["id"]) in selected_ids]
    statuses = Counter(str(row["status"]) for row in all_rows)
    finite_scores = [float(row["score"]) for row in all_rows if row.get("score") is not None]
    import numpy as np

    summary = {
        "config": run_config,
        "completed_count": len(all_rows),
        "new_count": len(new_rows),
        "status_counts": dict(statuses),
        "score": {
            "mean": float(np.mean(finite_scores)) if finite_scores else None,
            "median": float(np.median(finite_scores)) if finite_scores else None,
            "p10": float(np.percentile(finite_scores, 10)) if finite_scores else None,
            "p90": float(np.percentile(finite_scores, 90)) if finite_scores else None,
            "min": float(np.min(finite_scores)) if finite_scores else None,
            "max": float(np.max(finite_scores)) if finite_scores else None,
        },
        "timing": {
            "new_batch_wall_s": batch_wall_s,
            "amortized_s_per_new_sample": batch_wall_s / len(new_rows) if new_rows else None,
            "throughput_new_samples_per_s": len(new_rows) / batch_wall_s if batch_wall_s else None,
        },
        "worker_warmup_s_by_pid": {
            str(row["worker_pid"]): row.get("worker_warmup_s") for row in new_rows
        },
        "postflight_gpu": _gpu_snapshot(gpu_tokens),
        "rows": all_rows,
    }
    _write_json(args.output_dir / "batch_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("completed_count", "status_counts", "score", "timing")}, indent=2))


if __name__ == "__main__":
    main()
