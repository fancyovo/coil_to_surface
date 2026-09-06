import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import prepare_score_gradient_continuation as continuation


@pytest.mark.parametrize("corrupt", [None, "replay_round", "optimizer", "hash"])
def test_continuation_preserves_state_and_rejects_mixed_snapshots(tmp_path, monkeypatch, corrupt):
    rl = continuation.rl
    source = tmp_path / "source"
    (source / "checkpoints").mkdir(parents=True)
    strategy = {"flow_optimizer_steps_per_round": rl.FLOW_OPTIMIZER_STEPS_PER_ROUND,
        "ema_lerp": rl.EMA_LERP, "beta": 0.006, "gradient_directions": 64,
        "monte_carlo_samples": 4, "invalid_alpha": 0.05, "samples_per_round": 64,
        "world_size": 2, "loss": "original"}
    score_metadata = tmp_path / "score.json"
    score_metadata.write_text("{}")
    coordinate_checkpoint = tmp_path / "coordinates.pt"
    coordinate_checkpoint.write_bytes(b"unchanged")
    manifest = {"protocol": {"id": "source"}, "repository": {"commit": "source-code"},
        "strategy": strategy, "evaluator": {"library": "unused", "manifest": str(score_metadata),
        "manifest_sha256": rl.file_sha256(score_metadata)}, "paths": {
        "optimizer_checkpoint": str(coordinate_checkpoint),
        "optimizer_checkpoint_sha256": rl.file_sha256(coordinate_checkpoint)}}
    for filename, payload in [("manifest.json", manifest),
        ("progress.json", {"stage": "round_trained", "completed_round": 189, "next_round": 190}),
        ("replay_pool.json", {"source_round": 188 if corrupt == "replay_round" else 189})]:
        (source / filename).write_text(json.dumps(payload))
    checkpoint = {"stage": "score_gradient_online_policy", "outer_round": 190,
        "code_commit": "source-code", "model": {"x": torch.ones(3)},
        "ema": {"x": torch.full((3,), 2.0)}, "normalizer": {"frozen": True}, "model_config": {},
        "online_optimizer": {"state": {0: {"step": 1900, "exp_avg": torch.ones(3)}},
            "param_groups": [{"lr": 1e-3 if corrupt == "optimizer" else 5e-5,
                              "betas": (0.9, 0.95), "weight_decay": 0.01}]}}
    torch.save(checkpoint, source / "checkpoints/round_0190.pt")
    np.savez(source / "replay_pool.npz", **{key: np.zeros(512) for key in rl.REPLAY_ARRAY_KEYS})
    hashes = {name: rl.file_sha256(source / name)
              for name in ("manifest.json", "checkpoints/round_0190.pt", "replay_pool.npz")}
    protocol = {"source_run": str(source), "start_round": 190, "source_sha256": hashes,
        "source_protocol_id": "source", "source_commit": "source-code", "beta": 0.006,
        "format": "weighted", "protocol_id": "weighted", "inherit": ["all_state"],
        "valid_score_weighting": {"tau": 7.5, "epsilon": 0.01}, "loss": "weighted"}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    if corrupt == "hash":
        (source / "replay_pool.npz").write_bytes(b"changed")
    monkeypatch.setattr(rl, "require_clean_repository", lambda commit: {"commit": commit})
    monkeypatch.setattr(rl, "validate_r04_score", lambda *args: {})
    args = argparse.Namespace(protocol=protocol_path, run_root=tmp_path / "new", expected_commit="new-code")
    if corrupt:
        with pytest.raises(ValueError):
            continuation.prepare(args)
        assert not args.run_root.exists()
        return
    continuation.prepare(args)
    for relative in ("checkpoints/round_0190.pt", "replay_pool.npz"):
        assert (args.run_root / relative).read_bytes() == (source / relative).read_bytes()
    result = rl.load_json(args.run_root / "manifest.json")
    expected = copy.deepcopy(strategy)
    expected.update(loss="weighted", valid_score_weighting=protocol["valid_score_weighting"])
    assert result["strategy"] == expected
    assert result["continuation"]["start_round"] == 190
