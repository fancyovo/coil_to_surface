from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import sys
import torch

from flow_matching.proxy import (
    LatentProxyTransformer,
    binary_metrics,
    enrichment_at_prior_rates,
    validation_threshold,
)
from scripts.evaluate_qh_latent_proxy_score import select_cases
from scripts.invert_qh_flow_latents import curve_position_rms
from scripts import train_qh_latent_proxy


def small_proxy() -> LatentProxyTransformer:
    return LatentProxyTransformer(width=32, layers=2, heads=4, hidden=64)


def test_proxy_is_permutation_invariant_and_respects_mask():
    torch.manual_seed(7)
    model = small_proxy().eval()
    tokens = torch.randn(3, 4, 100)
    nfp = torch.tensor([3, 4, 5])
    permutation = torch.tensor([2, 0, 3, 1])
    expected = model(tokens, nfp)
    actual = model(tokens[:, permutation], nfp)
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=2.0e-6)

    padded = torch.cat([tokens[:, :2], torch.randn(3, 2, 100)], dim=1)
    mask = torch.tensor([[True, True, False, False]]).expand(3, -1)
    torch.testing.assert_close(
        model(padded, nfp, mask), model(tokens[:, :2], nfp), rtol=1.0e-5, atol=2.0e-6
    )


def test_binary_metrics_and_validation_threshold_are_exact_for_separated_data():
    probability = np.asarray([0.05, 0.2, 0.8, 0.95])
    label = np.asarray([0, 0, 1, 1])
    threshold = validation_threshold(probability, label)
    metrics = binary_metrics(probability, label, threshold=threshold)
    assert threshold == 0.8
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["confusion"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_enrichment_uses_negative_pass_rate_as_denominator():
    probability = np.asarray([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0])
    label = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    row = enrichment_at_prior_rates(probability, label, rates=(0.25,))[0]
    assert row["actual_prior_pass_rate"] == 0.25
    assert row["positive_retention_rate"] == 1.0
    assert row["enrichment"] == 4.0


def test_score_sample_selection_is_disjoint_and_spans_probability_range():
    probability = np.linspace(0.0, 1.0, 1000)
    selected, modes = select_cases(
        probability,
        stratified_count=80,
        iid_count=20,
        rng=np.random.default_rng(11),
    )
    assert len(selected) == 100
    assert len(modes) == 100
    stratified = [index for index, value in modes.items() if "prediction_rank_stratified" in value]
    iid = [index for index, value in modes.items() if "iid_prior" in value]
    assert len(stratified) == 80
    assert len(iid) == 20
    assert not set(stratified) & set(iid)
    assert min(stratified) == 0
    assert max(stratified) == 999


def test_curve_position_rms_matches_parseval_constant_and_harmonic_weights():
    delta = np.zeros((2, 1, 100), dtype=np.float32)
    delta[0, 0, 0] = 2.0
    delta[1, 0, 1] = 2.0
    np.testing.assert_allclose(curve_position_rms(delta), [2.0, np.sqrt(2.0)])


def write_latent_dataset(root: Path) -> None:
    generator = torch.Generator().manual_seed(31)
    shards = []
    for split, count in (("train", 32), ("validation", 12), ("test", 12)):
        filename = f"{split}_nfp04_nc03_rank00.pt"
        torch.save(
            {
                "format": "qh_flow_latents_v1",
                "split": split,
                "key": (4, 3),
                "rank": 0,
                "ids": torch.arange(count, dtype=torch.int32),
                "latents": 2.5 + 0.2 * torch.randn((count, 3, 100), generator=generator),
            },
            root / filename,
        )
        shards.append(
            {
                "file": filename,
                "split": split,
                "nfp": 4,
                "n_coils": 3,
                "rank": 0,
                "count": count,
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "qh_flow_latents_v1",
                "checkpoint_sha256": "flow-checkpoint",
                "source_manifest_sha256": "source-manifest",
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )


def test_cpu_proxy_training_writes_heldout_confusion_matrix(tmp_path, monkeypatch):
    latent_dir = tmp_path / "latents"
    output_dir = tmp_path / "training"
    latent_dir.mkdir()
    write_latent_dataset(latent_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_qh_latent_proxy.py",
            "--latent-dir",
            str(latent_dir),
            "--output-dir",
            str(output_dir),
            "--max-steps",
            "2",
            "--batch-per-gpu",
            "4",
            "--eval-batch",
            "8",
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
    train_qh_latent_proxy.main()
    summary = json.loads((output_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["test"]["confusion"]["tp"] + summary["test"]["confusion"]["fn"] == 12
    assert summary["test"]["confusion"]["tn"] + summary["test"]["confusion"]["fp"] == 12
    assert (output_dir / "test_confusion_matrix.png").is_file()
    assert (output_dir / "checkpoint_best_auc.pt").is_file()
