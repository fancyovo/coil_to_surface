from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(root: Path) -> dict:
    shard_count = 0
    sample_count = 0
    status_counts: Counter[str] = Counter()
    stream_ids = set()
    for path in sorted((root / "shards").glob("*.meta.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        shard_count += 1
        sample_count += int(payload["row_count"])
        stream_ids.add(str(payload["stream_id"]))
        status_counts.update(
            {str(key): int(value) for key, value in payload["status_counts"].items()}
        )
    return {
        "dataset_root": str(root.resolve()),
        "completed_samples": sample_count,
        "completed_shards": shard_count,
        "streams_with_completed_shards": len(stream_ids),
        "status_counts": dict(sorted(status_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed QH IID score shards.")
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.dataset_root), indent=2))


if __name__ == "__main__":
    main()
