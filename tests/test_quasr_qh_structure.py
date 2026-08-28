from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from flow_matching.data import file_sha256
from scripts.analyze_quasr_qh_structure import (
    adjusted_rand_index,
    fit_kmeans,
    set_features,
)


def random_tokens(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tokens = rng.normal(scale=0.03, size=(3, 3, 100)).astype(np.float64)
    tokens[..., 0] += np.asarray([0.8, 1.1, 1.4])[None]
    tokens[..., 33] += np.asarray([0.2, -0.1, 0.4])[None]
    tokens[..., 66] += np.asarray([-0.2, 0.3, 0.1])[None]
    tokens[..., -1] = np.asarray([2.0, -1.0, 3.0])[None]
    return tokens


def rotate_toroidally(tokens: np.ndarray, angle: float) -> np.ndarray:
    output = tokens.copy()
    x = tokens[..., :33]
    y = tokens[..., 33:66]
    output[..., :33] = np.cos(angle) * x - np.sin(angle) * y
    output[..., 33:66] = np.sin(angle) * x + np.cos(angle) * y
    return output


def shift_curve_parameter(tokens: np.ndarray, phase: float) -> np.ndarray:
    output = tokens.copy()
    for coordinate in range(3):
        start = 33 * coordinate
        for mode in range(1, 17):
            sine_index = start + 2 * mode - 1
            cosine_index = start + 2 * mode
            sine = tokens[..., sine_index]
            cosine = tokens[..., cosine_index]
            angle = mode * phase
            output[..., sine_index] = sine * np.cos(angle) - cosine * np.sin(angle)
            output[..., cosine_index] = sine * np.sin(angle) + cosine * np.cos(angle)
    return output


def reverse_curve_parameter(tokens: np.ndarray) -> np.ndarray:
    output = tokens.copy()
    for coordinate in range(3):
        start = 33 * coordinate
        output[..., start + 1 : start + 33 : 2] *= -1.0
    return output


def test_set_descriptors_ignore_coil_permutation_and_global_rotation() -> None:
    tokens = random_tokens()
    baseline = set_features(tokens, samples=96)
    permuted = set_features(tokens[:, [2, 0, 1]], samples=96)
    rotated = set_features(rotate_toroidally(tokens, 0.73), samples=96)
    np.testing.assert_allclose(permuted, baseline, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(rotated, baseline, rtol=2.0e-5, atol=2.0e-6)


def test_set_descriptors_ignore_grid_aligned_phase_shift_and_reversal() -> None:
    tokens = random_tokens()
    baseline = set_features(tokens, samples=96)
    shifted = set_features(shift_curve_parameter(tokens, 2.0 * np.pi * 7 / 96), samples=96)
    reversed_tokens = set_features(reverse_curve_parameter(tokens), samples=96)
    np.testing.assert_allclose(shifted, baseline, rtol=3.0e-5, atol=3.0e-5)
    np.testing.assert_allclose(reversed_tokens, baseline, rtol=3.0e-5, atol=3.0e-5)


def test_kmeans_and_adjusted_rand_recover_separated_blobs() -> None:
    rng = np.random.default_rng(11)
    values = np.concatenate(
        (
            rng.normal(loc=-3.0, scale=0.1, size=(50, 2)),
            rng.normal(loc=3.0, scale=0.1, size=(50, 2)),
        )
    ).astype(np.float32)
    centers, labels, _, _ = fit_kmeans(
        values, 2, seed=5, n_init=2, max_iterations=30
    )
    truth = np.repeat([0, 1], 50)
    assert centers.shape == (2, 2)
    assert adjusted_rand_index(labels, truth) == 1.0


def test_small_atlas_runs_end_to_end(tmp_path: Path) -> None:
    rng = np.random.default_rng(23)
    tokens = rng.normal(scale=0.015, size=(40, 2, 100)).astype(np.float32)
    tokens[:20, :, 0] += 0.7
    tokens[20:, :, 0] += 1.4
    tokens[..., 33] += 0.2
    tokens[..., -1] = np.asarray([2.0, 1.0], dtype=np.float32)
    ids = np.arange(1000, 1040, dtype=np.int32)
    shard = tmp_path / "qh_nfp04_nc02_part0000.npz"
    np.savez_compressed(
        shard,
        tokens=tokens,
        ids=ids,
        split=np.zeros(len(ids), dtype=np.uint8),
        curve_order=np.full(len(ids), 16, dtype=np.uint8),
    )
    manifest = {
        "format": "quasr_qh_flow_v1",
        "requested_count": len(ids),
        "shards": [
            {
                "count": len(ids),
                "file": shard.name,
                "n_coils": 2,
                "nfp": 4,
                "sha256": file_sha256(shard),
                "shape": [len(ids), 2, 100],
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "atlas"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "analyze_quasr_qh_structure.py"),
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--verify-hashes",
            "--curve-samples",
            "32",
            "--feature-batch-size",
            "16",
            "--fit-count",
            "30",
            "--silhouette-count",
            "20",
            "--max-pca-components",
            "8",
            "--k-values",
            "2,3",
            "--gallery-group-count",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((output / "atlas_summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate"]["sample_count"] == 40
    assert summary["groups"][0]["selected_k"] in {2, 3}
    assert (output / "novelty_reference.json").is_file()
