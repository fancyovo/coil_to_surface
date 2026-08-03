from __future__ import annotations

import numpy as np
import torch

from scripts.evaluate_qh_latent_proxy_score import analyze
from scripts.analyze_qh_latent_proxy_optimization import latent_diversity
from scripts.optimize_qh_latent_proxy import calibrated_probability, project_to_rms_


def test_project_to_rms_preserves_requested_batch_radii() -> None:
    latent = torch.randn(5, 3, 100)
    target = torch.linspace(0.8, 1.2, 5).reshape(5, 1, 1)
    project_to_rms_(latent, target)
    actual = latent.square().mean(dim=(1, 2), keepdim=True).sqrt()
    torch.testing.assert_close(actual, target, rtol=1.0e-6, atol=1.0e-6)


def test_calibrated_probability_is_monotone_and_finite() -> None:
    logits = np.asarray([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    probability = calibrated_probability(logits, scale=0.7, bias=2.0)
    assert np.isfinite(probability).all()
    assert np.all(np.diff(probability) >= 0.0)
    assert probability[0] > 0.0
    assert probability[-1] == 1.0


def test_score_analysis_accepts_fewer_than_ten_cases(tmp_path) -> None:
    rows = [
        {
            "proxy_probability": 0.2 + 0.1 * index,
            "proxy_logit": float(index),
            "score": float(index + 1),
            "status": "ok",
            "sampling_modes": ["optimized_projected"],
        }
        for index in range(4)
    ]
    summary = analyze(rows, tmp_path)
    assert summary["count"] == 4
    assert len(summary["prediction_bins"]) == 4


def test_latent_diversity_detects_duplicate_pair() -> None:
    latent = np.zeros((3, 1, 2), dtype=np.float32)
    latent[2] = 1.0
    summary = latent_diversity(latent)
    assert summary["rounded_1e4_unique_count"] == 2
    assert summary["nearest_neighbor_rms_distance"]["min"] == 0.0
