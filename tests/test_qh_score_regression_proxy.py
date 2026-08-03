from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from flow_matching.proxy import LatentScoreRegressorTransformer
from scripts.prepare_qh_score_regression_dataset import (
    CORPUS_FORMAT,
    DATASET_FORMAT,
    assign_stratified_splits,
    prepare_dataset,
)
from scripts.train_qh_latent_score_regressor import (
    basic_regression_metrics,
    complete_regression_metrics,
    evaluate_model,
)
from scripts import train_qh_latent_score_regressor


CURRENT_SHA = "a" * 64
OLD_SHA = "b" * 64


def write_stream(root: Path, stream_id: str, library_sha: str, count: int) -> None:
    stream_dir = root / "streams" / stream_id
    stream_dir.mkdir(parents=True)
    (stream_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": CORPUS_FORMAT,
                "stream_id": stream_id,
                "native_score": {"library_sha256": library_sha},
                "flow_checkpoint": {"sha256": "c" * 64},
            }
        ),
        encoding="utf-8",
    )
    rows = []
    score_values = (1.0, 3.0, 7.0, 15.0, 25.0, 35.0, 45.0)
    statuses = ("ok", "no_axis", "no_surface", "drift_rejected", "flux_rejected")
    for index in range(count):
        n_coils = 2 + index % 2
        rows.append(
            {
                "format": CORPUS_FORMAT,
                "sample_id": f"{stream_id}_{index:06d}",
                "condition": {"nfp": 3 + index % 3, "n_base_coils": n_coils},
                "latent": np.full((n_coils, 100), index / 100.0, dtype=np.float32).tolist(),
                "native_score": {
                    "score": score_values[index % len(score_values)],
                    "status": statuses[index % len(statuses)],
                },
                "provenance": {"score_library_sha256": library_sha},
            }
        )
    shard_dir = root / "shards"
    shard_dir.mkdir(exist_ok=True)
    shard_path = shard_dir / f"{stream_id}_shard_000000.jsonl.gz"
    with gzip.open(shard_path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    shard_sha = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    (shard_dir / f"{stream_id}_shard_000000.meta.json").write_text(
        json.dumps(
            {
                "format": CORPUS_FORMAT,
                "stream_id": stream_id,
                "shard_index": 0,
                "file": shard_path.name,
                "sha256": shard_sha,
                "row_count": count,
                "status_counts": {},
            }
        ),
        encoding="utf-8",
    )


def test_frozen_dataset_filters_library_and_has_disjoint_stratified_splits(tmp_path: Path):
    corpus = tmp_path / "corpus"
    output = tmp_path / "frozen"
    write_stream(corpus, "current", CURRENT_SHA, 140)
    write_stream(corpus, "old", OLD_SHA, 21)
    manifest = prepare_dataset(
        corpus,
        output,
        score_library_sha256=CURRENT_SHA,
        seed=19,
    )
    assert manifest["snapshot"]["included_rows"] == 140
    assert manifest["snapshot"]["excluded_library_rows"] == {OLD_SHA: 21}
    assert sum(manifest["splits"][name]["count"] for name in ("train", "validation", "test")) == 140
    split_ids = {
        name: set((output / f"{name}_sample_ids.txt").read_text(encoding="utf-8").splitlines())
        for name in ("train", "validation", "test")
    }
    assert not split_ids["train"] & split_ids["validation"]
    assert not split_ids["train"] & split_ids["test"]
    assert not split_ids["validation"] & split_ids["test"]
    payload = torch.load(output / "train.pt", weights_only=False)
    assert payload["format"] == DATASET_FORMAT
    assert payload["tokens"].shape[1:] == (3, 100)
    assert torch.all(payload["target"] >= 0.0) and torch.all(payload["target"] <= 1.0)


def test_stratified_assignment_is_deterministic():
    records = [
        {"sample_id": f"sample-{index}", "status": "ok", "score": float(index % 50)}
        for index in range(200)
    ]
    first = assign_stratified_splits(records, seed=7, validation_fraction=0.1, test_fraction=0.1)
    second = assign_stratified_splits(records, seed=7, validation_fraction=0.1, test_fraction=0.1)
    assert first == second
    assert {"train", "validation", "test"} <= set(first)


def test_regression_metrics_and_bounded_model_evaluation():
    actual = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0])
    perfect = basic_regression_metrics(actual, actual)
    assert perfect["mse"] == 0.0
    assert perfect["r2"] == 1.0
    assert perfect["pearson"] == pytest.approx(1.0)
    summary = complete_regression_metrics(
        actual,
        actual,
        nfp=np.asarray([3, 3, 4, 4, 5]),
        n_coils=np.asarray([2, 2, 3, 3, 4]),
        status=np.asarray([0, 0, 0, 1, 1]),
        status_names={0: "ok", 1: "no_axis"},
    )
    assert summary["predicted_thresholds"]["gt_20"]["count"] == 2
    assert summary["predicted_thresholds"]["gt_30"]["actual_min"] == 40.0

    model = LatentScoreRegressorTransformer(
        width=32, layers=1, heads=4, hidden=64, max_coils=3
    ).eval()
    data = {
        "tokens": torch.randn(6, 3, 100),
        "mask": torch.tensor([[True, True, False]]).expand(6, -1),
        "nfp": torch.full((6,), 4, dtype=torch.long),
        "n_coils": torch.full((6,), 2, dtype=torch.long),
        "target": torch.linspace(0.0, 1.0, 6),
        "status": torch.zeros(6, dtype=torch.int16),
    }
    values = evaluate_model(model, data, batch_size=4)
    assert np.all(values["predicted"] > 0.0)
    assert np.all(values["predicted"] < 100.0)


def test_cpu_regression_training_writes_heldout_outputs(tmp_path: Path, monkeypatch):
    corpus = tmp_path / "corpus"
    dataset = tmp_path / "dataset"
    output = tmp_path / "training"
    write_stream(corpus, "current", CURRENT_SHA, 140)
    manifest = prepare_dataset(corpus, dataset, score_library_sha256=CURRENT_SHA, seed=23)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_qh_latent_score_regressor.py",
            "--dataset-dir",
            str(dataset),
            "--output-dir",
            str(output),
            "--max-steps",
            "2",
            "--batch-per-gpu",
            "8",
            "--eval-batch",
            "32",
            "--width",
            "32",
            "--layers",
            "1",
            "--heads",
            "4",
            "--hidden",
            "64",
            "--warmup-steps",
            "0",
            "--log-interval",
            "1",
            "--validation-interval",
            "1",
        ],
    )
    train_qh_latent_score_regressor.main()
    summary = json.loads((output / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["test"]["count"] == manifest["splits"]["test"]["count"]
    assert summary["checkpoint"]["step"] >= 0
    assert (output / "test_prediction_scatter.png").is_file()
    assert (output / "test_calibration_distribution.png").is_file()
    assert (output / "test_high_prediction_tail.png").is_file()
