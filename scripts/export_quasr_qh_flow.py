from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.quasr_export import load_simson_coils, stable_split


def serial_path(root: str | Path, device_id: int) -> Path:
    device_id = int(device_id)
    return Path(root) / "simsopt_serials" / f"{device_id // 1000:04d}" / f"serial{device_id:07d}.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_qh_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if int(float(row["helicity"])) == 1]
    rows.sort(key=lambda row: int(float(row["ID"])))
    return rows


def _optional_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _extract(task: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    root, row = task
    device_id = int(float(row["ID"]))
    try:
        parsed = load_simson_coils(serial_path(root, device_id))
        expected_nfp = int(float(row["nfp"]))
        expected_coils = int(float(row["nc_per_hp"]))
        if parsed.nfp != expected_nfp:
            raise ValueError(f"nfp mismatch: metadata={expected_nfp}, serial={parsed.nfp}")
        if parsed.tokens.shape[0] != expected_coils:
            raise ValueError(
                f"base-coil mismatch: metadata={expected_coils}, serial={parsed.tokens.shape[0]}"
            )
        return {
            "ok": True,
            "id": device_id,
            "nfp": parsed.nfp,
            "n_coils": parsed.tokens.shape[0],
            "curve_order": parsed.curve_order,
            "tokens": parsed.tokens,
            "split": stable_split(device_id),
            "qs_error": _optional_float(row, "qs_error"),
            "mean_iota": _optional_float(row, "mean_iota"),
            "minor_radius": _optional_float(row, "minor_radius"),
        }
    except Exception as exc:
        return {"ok": False, "id": device_id, "error": f"{type(exc).__name__}: {exc}"}


class ShardWriter:
    def __init__(self, output_dir: Path, shard_size: int):
        self.output_dir = output_dir
        self.shard_size = int(shard_size)
        self.buffers: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        self.parts: Counter[tuple[int, int]] = Counter()
        self.shards: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        key = (int(row["nfp"]), int(row["n_coils"]))
        self.buffers[key].append(row)
        if len(self.buffers[key]) >= self.shard_size:
            self.flush(key)

    def flush(self, key: tuple[int, int]) -> None:
        rows = self.buffers[key]
        if not rows:
            return
        nfp, n_coils = key
        part = self.parts[key]
        self.parts[key] += 1
        filename = f"qh_nfp{nfp:02d}_nc{n_coils:02d}_part{part:04d}.npz"
        path = self.output_dir / filename
        tokens = np.stack([row["tokens"] for row in rows]).astype(np.float32, copy=False)
        np.savez(
            path,
            tokens=tokens,
            ids=np.asarray([row["id"] for row in rows], dtype=np.int32),
            split=np.asarray([row["split"] for row in rows], dtype=np.uint8),
            qs_error=np.asarray([row["qs_error"] for row in rows], dtype=np.float32),
            mean_iota=np.asarray([row["mean_iota"] for row in rows], dtype=np.float32),
            minor_radius=np.asarray([row["minor_radius"] for row in rows], dtype=np.float32),
            curve_order=np.asarray([row["curve_order"] for row in rows], dtype=np.uint8),
        )
        self.shards.append(
            {
                "file": filename,
                "count": len(rows),
                "nfp": nfp,
                "n_coils": n_coils,
                "shape": list(tokens.shape),
                "sha256": sha256(path),
            }
        )
        rows.clear()

    def finish(self) -> None:
        for key in sorted(self.buffers):
            self.flush(key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract all QUASR QH base coils without Simsopt.")
    parser.add_argument("--quasr-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.workers < 1 or args.workers > 16:
        raise ValueError("--workers must be between 1 and 16")
    if args.shard_size < 1:
        raise ValueError("--shard-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {args.output_dir}")

    rows = read_qh_rows(args.metadata)
    if args.limit is not None:
        rows = rows[: args.limit]
    started = time.perf_counter()
    writer = ShardWriter(args.output_dir, args.shard_size)
    failures = []
    group_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    order_counts: Counter[str] = Counter()
    tasks = ((str(args.quasr_root), row) for row in rows)
    context = mp.get_context("fork")
    with context.Pool(processes=args.workers) as pool:
        for index, result in enumerate(pool.imap(_extract, tasks, chunksize=8), start=1):
            if result["ok"]:
                writer.add(result)
                group_counts[f"nfp{result['nfp']}_nc{result['n_coils']}"] += 1
                split_counts[("train", "validation", "test")[result["split"]]] += 1
                order_counts[str(result["curve_order"])] += 1
            else:
                failures.append(result)
            if index % 1000 == 0 or index == len(rows):
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "processed": index,
                            "total": len(rows),
                            "failures": len(failures),
                            "samples_per_s": index / max(elapsed, 1.0e-9),
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    writer.finish()
    failures_path = args.output_dir / "failures.jsonl"
    with failures_path.open("w", encoding="utf-8") as stream:
        for row in failures:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {
        "format": "quasr_qh_flow_v1",
        "helicity": 1,
        "token_layout": "x[33],y[33],z[33],current_A",
        "curve_order": 16,
        "source_root": str(args.quasr_root),
        "source_metadata": str(args.metadata),
        "metadata_sha256": sha256(args.metadata),
        "requested_count": len(rows),
        "success_count": sum(group_counts.values()),
        "failure_count": len(failures),
        "group_counts": dict(sorted(group_counts.items())),
        "curve_order_counts": dict(sorted(order_counts.items(), key=lambda item: int(item[0]))),
        "split_counts": dict(split_counts),
        "shards": writer.shards,
        "failures_file": failures_path.name,
        "failures_sha256": sha256(failures_path),
        "wall_s": time.perf_counter() - started,
        "workers": args.workers,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
