from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.quasr_export import load_simson_coils


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serial_path(root: Path, device_id: int) -> Path:
    return root / "simsopt_serials" / f"{device_id // 1000:04d}" / f"serial{device_id:07d}.json"


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a compact QUASR QH flow dataset.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--quasr-root", type=Path)
    parser.add_argument("--source-check-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "quasr_qh_flow_v1":
        fail(f"unexpected format: {manifest.get('format')!r}")

    failures_path = args.data_dir / manifest["failures_file"]
    if sha256(failures_path) != manifest["failures_sha256"]:
        fail("failure-list SHA-256 mismatch")

    all_ids: list[np.ndarray] = []
    locations: dict[int, tuple[Path, int]] = {}
    group_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    order_counts: Counter[str] = Counter()
    total_bytes = manifest_path.stat().st_size + failures_path.stat().st_size

    for shard in manifest["shards"]:
        path = args.data_dir / shard["file"]
        if sha256(path) != shard["sha256"]:
            fail(f"shard SHA-256 mismatch: {path.name}")
        total_bytes += path.stat().st_size
        with np.load(path, allow_pickle=False) as data:
            tokens = data["tokens"]
            ids = data["ids"]
            split = data["split"]
            curve_order = data["curve_order"]
            expected_shape = tuple(shard["shape"])
            if tokens.shape != expected_shape or tokens.shape != (
                int(shard["count"]),
                int(shard["n_coils"]),
                100,
            ):
                fail(f"shape mismatch: {path.name}: {tokens.shape} != {expected_shape}")
            if tokens.dtype != np.float32 or not np.all(np.isfinite(tokens)):
                fail(f"tokens are not finite float32: {path.name}")
            if ids.shape != (tokens.shape[0],) or split.shape != ids.shape:
                fail(f"metadata array shape mismatch: {path.name}")
            if np.any(split > 2):
                fail(f"invalid split value: {path.name}")
            if np.any(np.sum(np.abs(tokens[:, :, 99]), axis=1) == 0):
                fail(f"zero-current configuration: {path.name}")
            all_ids.append(ids.astype(np.int64, copy=False))
            group_counts[f"nfp{shard['nfp']}_nc{shard['n_coils']}"] += tokens.shape[0]
            for value, count in zip(*np.unique(split, return_counts=True)):
                split_counts[("train", "validation", "test")[int(value)]] += int(count)
            for value, count in zip(*np.unique(curve_order, return_counts=True)):
                order_counts[str(int(value))] += int(count)
            if args.source_check_count:
                for index, device_id in enumerate(ids):
                    locations[int(device_id)] = (path, index)

    ids = np.concatenate(all_ids)
    if np.unique(ids).size != ids.size:
        fail("duplicate device IDs found")
    if ids.size != int(manifest["success_count"]):
        fail(f"success count mismatch: arrays={ids.size}, manifest={manifest['success_count']}")
    if dict(sorted(group_counts.items())) != manifest["group_counts"]:
        fail("group counts do not match manifest")
    if dict(split_counts) != manifest["split_counts"]:
        fail("split counts do not match manifest")
    if dict(sorted(order_counts.items(), key=lambda item: int(item[0]))) != manifest[
        "curve_order_counts"
    ]:
        fail("curve-order counts do not match manifest")

    source_checked = 0
    if args.source_check_count:
        if args.quasr_root is None:
            fail("--quasr-root is required when --source-check-count is nonzero")
        if args.source_check_count > ids.size:
            fail("--source-check-count exceeds the dataset size")
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(ids, size=args.source_check_count, replace=False)
        shard_cache: dict[Path, np.ndarray] = {}
        for device_id in selected:
            path, index = locations[int(device_id)]
            if path not in shard_cache:
                with np.load(path, allow_pickle=False) as data:
                    shard_cache[path] = data["tokens"]
            source = load_simson_coils(serial_path(args.quasr_root, int(device_id)))
            compact = shard_cache[path][index]
            if source.tokens.shape != compact.shape or not np.array_equal(source.tokens, compact):
                fail(f"source token mismatch for device {int(device_id)}")
            source_checked += 1

    result = {
        "data_dir": str(args.data_dir.resolve()),
        "success_count": int(ids.size),
        "failure_count": int(manifest["failure_count"]),
        "shard_count": len(manifest["shards"]),
        "group_counts": dict(sorted(group_counts.items())),
        "split_counts": dict(split_counts),
        "curve_order_counts": dict(sorted(order_counts.items(), key=lambda item: int(item[0]))),
        "source_checked": source_checked,
        "bytes": total_bytes,
        "status": "ok",
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
