#!/usr/bin/env python3
"""Validate the fixed full-evaluation code bundle before Slurm submission."""

from __future__ import annotations

import json
from pathlib import Path
import py_compile
import subprocess


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("code_manifest.json")


def listed_paths(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from listed_paths(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {"entrypoint", "implementation", "implementations"}:
                yield from listed_paths(item)
            elif isinstance(item, (dict, list)):
                yield from listed_paths(item)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = sorted({ROOT / path for path in listed_paths(manifest)})
    missing = [path for path in paths if not path.is_file()]
    if missing:
        for path in missing:
            print(f"ERROR: missing evaluation code: {path}")
        return 1

    for path in paths:
        if path.suffix == ".py":
            py_compile.compile(str(path), doraise=True)
        elif path.suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True)
    print(f"PASS: validated {len(paths)} fixed evaluation files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
