import json
from pathlib import Path

import numpy as np
import torch

from flow_matching.data import CoilNormalizer
from scripts.select_highest_score_gradient_case import select


def test_selects_highest_valid_completed_round_and_records_provenance(tmp_path):
    run = tmp_path / "run"
    (run / "rounds/round_0000").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    (run / "manifest.json").write_text(json.dumps({"format": "test"}))
    (run / "progress.json").write_text(json.dumps({"next_round": 1}))
    (run / "rounds/round_0000/training_summary.json").write_text(json.dumps({"stage": "round_trained"}))
    normalizer = CoilNormalizer(mean=np.zeros(100, dtype=np.float32), std=np.ones(100, dtype=np.float32),
                                current_l1_a={"8:3": 3.0}, clip=float("inf"))
    checkpoint = {"normalizer": normalizer.to_dict()}
    torch.save(checkpoint, run / "checkpoints/round_0000.pt")
    records = [{"sample_id": "low", "valid": True, "initial": {"score": 20.0, "components": {"volume_qs": 1.0}}, "gradient_ok": True},
               {"sample_id": "high", "valid": True, "initial": {"score": 80.0, "components": {"volume_qs": 2.0}}, "gradient_ok": True},
               {"sample_id": "invalid-high", "valid": False, "initial": {"score": 100.0}, "gradient_ok": False}]
    (run / "rounds/round_0000/rank_00.json").write_text(json.dumps({"records": records}))
    np.savez(run / "rounds/round_0000/rank_00.npz", current=np.zeros((3, 3, 100), dtype=np.float32))
    output = tmp_path / "selection"
    metadata = select(run, output)
    assert metadata["source_sample_id"] == "high"
    assert metadata["source_score"] == 80.0
    case = json.loads((output / "selected_case.json").read_text())
    assert case["nfp"] == 8
    assert case["raw"]["metadata"]["rl_selection"]["source_round"] == 0
