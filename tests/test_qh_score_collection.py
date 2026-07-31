from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from flow_matching.collection import (
    derive_stream_seed,
    load_train_condition_prior,
    write_jsonl_gzip_atomic,
)
from scripts.summarize_qh_iid_score_corpus import summarize


def test_training_manifest_prior_uses_joint_empirical_counts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text('{"shards": []}\n', encoding="utf-8")
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "train_counts": {
                    "nfp4_nc3": 7,
                    "nfp2_nc1": 2,
                    "nfp8_nc5": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    prior = load_train_condition_prior(
        data_dir,
        training_run_manifest=run_manifest,
        supported_keys={(2, 1), (4, 3), (8, 5)},
    )

    assert prior.keys == ((2, 1), (4, 3), (8, 5))
    assert prior.counts == (2, 7, 1)
    np.testing.assert_allclose(prior.probabilities, [0.2, 0.7, 0.1])
    assert prior.to_dict()["definition"].startswith("empirical_joint_distribution")


def test_dataset_fallback_counts_only_training_split(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.savez_compressed(
        data_dir / "a.npz",
        split=np.asarray([0, 1, 0, 2], dtype=np.uint8),
    )
    np.savez_compressed(
        data_dir / "b.npz",
        split=np.asarray([0, 0, 1], dtype=np.uint8),
    )
    manifest = {
        "shards": [
            {"file": "a.npz", "nfp": 3, "n_coils": 2},
            {"file": "b.npz", "nfp": 5, "n_coils": 4},
        ]
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    prior = load_train_condition_prior(data_dir)

    assert prior.keys == ((3, 2), (5, 4))
    assert prior.counts == (2, 2)
    np.testing.assert_allclose(prior.probabilities, [0.5, 0.5])


def test_prior_rejects_checkpoint_condition_mismatch(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text('{"shards": []}\n', encoding="utf-8")
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(
        json.dumps({"train_counts": {"nfp4_nc3": 8}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="checkpoint normalizer differ"):
        load_train_condition_prior(
            data_dir,
            training_run_manifest=run_manifest,
            supported_keys={(4, 3), (5, 3)},
        )


def test_stream_seeds_are_disjoint_across_jobs_and_ranks() -> None:
    seeds = {
        derive_stream_seed(job_id, rank)
        for job_id in (30001, 30002)
        for rank in range(6)
    }
    assert len(seeds) == 12


def test_atomic_shards_preserve_latent_tokens_and_full_native_result(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    shard = root / "shards" / "slurm_1_rank_0_shard_000000.jsonl.gz"
    rows = [
        {
            "sample_id": "slurm_1_rank_0_000000000000",
            "latent": [[0.1] * 100],
            "decoded_tokens": [[0.2] * 99 + [1.0e6]],
            "native_score": {
                "score": 12.5,
                "status": "ok",
                "components": {"axis": 0.9},
                "diagnostics": {"iota_min": 1.2},
            },
        }
    ]

    count, digest = write_jsonl_gzip_atomic(shard, rows)
    with gzip.open(shard, "rt", encoding="utf-8") as stream:
        restored = [json.loads(line) for line in stream]

    assert count == 1
    assert len(digest) == 64
    assert restored == rows
    assert not shard.with_name(shard.name + ".partial").exists()
    with pytest.raises(FileExistsError):
        write_jsonl_gzip_atomic(shard, rows)

    metadata = {
        "stream_id": "slurm_1_rank_0",
        "row_count": 1,
        "status_counts": {"ok": 1},
    }
    (root / "shards" / "slurm_1_rank_0_shard_000000.meta.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    summary = summarize(root)
    assert summary["completed_samples"] == 1
    assert summary["status_counts"] == {"ok": 1}
