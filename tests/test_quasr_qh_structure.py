from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from flow_matching.data import file_sha256
from scripts.analyze_quasr_qh_structure import (
    adjusted_rand_index,
    balanced_fit_indices,
    fit_kmeans,
    normalized_mutual_information,
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


def test_balanced_fit_and_nfp_mixing_metrics() -> None:
    strata = np.repeat([2, 3, 4], 20)
    indices = balanced_fit_indices(strata, 15, rng=np.random.default_rng(17))
    assert len(indices) == 15
    np.testing.assert_array_equal(np.unique(strata[indices], return_counts=True)[1], [5, 5, 5])

    cluster = np.repeat([0, 1], 4)
    assert normalized_mutual_information(cluster, cluster) == 1.0
    assert normalized_mutual_information(cluster, np.tile([4, 5], 4)) < 1.0e-12


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
    assert summary["aggregate"]["hard_group_count"] == 1
    assert summary["aggregate"]["n_base_coils_values"] == [2]
    assert summary["aggregate"]["leaf_cluster_count"] == summary["groups"][0]["selected_k"]
    assert summary["aggregate"]["leaf_cluster_count_by_n_base_coils"] == {
        "2": summary["groups"][0]["selected_k"]
    }
    assert summary["aggregate"]["selected_k_per_hard_group_range"] == [
        summary["groups"][0]["selected_k"],
        summary["groups"][0]["selected_k"],
    ]
    assert (output / "novelty_reference.json").is_file()


def test_cross_nfp_atlas_keeps_ncoils_hard_and_mixes_nfp(tmp_path: Path) -> None:
    rng = np.random.default_rng(29)
    shards = []
    next_id = 2000
    for nfp in (4, 5):
        tokens = rng.normal(scale=0.01, size=(20, 2, 100)).astype(np.float32)
        tokens[:10, :, 0] += 0.7
        tokens[10:, :, 0] += 1.4
        tokens[..., 33] += 0.2
        tokens[..., -1] = np.asarray([2.0, 1.0], dtype=np.float32)
        ids = np.arange(next_id, next_id + len(tokens), dtype=np.int32)
        next_id += len(tokens)
        shard = tmp_path / f"qh_nfp{nfp:02d}_nc02_part0000.npz"
        np.savez_compressed(
            shard,
            tokens=tokens,
            ids=ids,
            split=np.zeros(len(ids), dtype=np.uint8),
            curve_order=np.full(len(ids), 16, dtype=np.uint8),
        )
        shards.append(
            {
                "count": len(ids),
                "file": shard.name,
                "n_coils": 2,
                "nfp": nfp,
                "sha256": file_sha256(shard),
                "shape": [len(ids), 2, 100],
            }
        )
    manifest = {
        "format": "quasr_qh_flow_v1",
        "requested_count": 40,
        "shards": shards,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "cross_nfp_atlas"
    subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "analyze_quasr_qh_structure.py"
            ),
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--verify-hashes",
            "--stratification",
            "ncoils",
            "--curve-samples",
            "32",
            "--feature-batch-size",
            "16",
            "--fit-count",
            "40",
            "--silhouette-count",
            "20",
            "--max-pca-components",
            "8",
            "--k-values",
            "2",
            "--gallery-group-count",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((output / "atlas_summary.json").read_text(encoding="utf-8"))
    group = summary["groups"][0]
    assert summary["method"]["stratification"] == "ncoils"
    assert summary["aggregate"]["hard_group_count"] == 1
    assert summary["aggregate"]["n_base_coils_values"] == [2]
    assert summary["aggregate"]["leaf_cluster_count"] == 2
    assert summary["aggregate"]["cross_nfp_leaf_cluster_count"] == 2
    assert group["group"] == "nc2"
    assert group["nfp"] is None
    assert group["nfp_values"] == [4, 5]
    assert group["nfp_counts"] == {"4": 20, "5": 20}
    assert group["cluster_nfp_normalized_mutual_information"] < 0.1
    assert all(row["nfp_value_count"] == 2 for row in group["cluster_nfp_composition"])
    assert (output / "cross_nfp_composition.png").is_file()
    assert (output / "galleries" / "nc2_representatives.png").is_file()
    with np.load(output / group["assignment_file"], allow_pickle=False) as payload:
        np.testing.assert_array_equal(np.unique(payload["nfp"], return_counts=True)[1], [20, 20])
