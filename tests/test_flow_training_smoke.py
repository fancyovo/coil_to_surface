from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from scripts import train_qh_flow


def write_dataset(root: Path) -> None:
    rng = np.random.default_rng(11)
    tokens = rng.normal(scale=0.1, size=(32, 2, 100)).astype(np.float32)
    tokens[..., 0] += 1.0
    tokens[..., 2] += 0.2
    tokens[..., 33] += 0.2
    tokens[..., 35] += 0.1
    tokens[..., -1] = rng.normal(5.0e5, 1.0e5, size=(32, 2))
    split = np.zeros(32, dtype=np.uint8)
    split[24:28] = 1
    split[28:] = 2
    shard = root / "qh_nfp03_nc02_part0000.npz"
    np.savez(
        shard,
        tokens=tokens,
        ids=np.arange(32, dtype=np.int32),
        split=split,
        qs_error=np.ones(32, dtype=np.float32),
        mean_iota=np.ones(32, dtype=np.float32),
        minor_radius=np.ones(32, dtype=np.float32),
        curve_order=np.full(32, 16, dtype=np.uint8),
    )
    manifest = {
        "format": "quasr_qh_flow_v1",
        "shards": [
            {
                "file": shard.name,
                "count": 32,
                "nfp": 3,
                "n_coils": 2,
                "sha256": "not-checked-in-smoke",
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_two_step_cpu_training_writes_monitor_and_checkpoint(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "run"
    data_dir.mkdir()
    write_dataset(data_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_qh_flow.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--batch-per-gpu",
            "4",
            "--width",
            "32",
            "--layers",
            "1",
            "--heads",
            "4",
            "--hidden",
            "64",
            "--log-interval",
            "1",
            "--validation-interval",
            "1",
            "--sample-interval",
            "1",
            "--sample-count",
            "4",
            "--sample-steps",
            "1",
            "--checkpoint-interval",
            "1",
        ],
    )
    train_qh_flow.main()
    assert (output_dir / "metrics.jsonl").is_file()
    assert (output_dir / "monitor.png").is_file()
    assert (output_dir / "checkpoint_latest.pt").is_file()
