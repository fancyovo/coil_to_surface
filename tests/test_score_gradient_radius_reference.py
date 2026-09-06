import copy

import pytest

from scripts.score_gradient_flow_rl import inherit_reference_strategy


def _reference():
    return {"protocol": {"id": "qh-axisflip-r012-score-gradient-replay10-ema10-rl-r04-abi11-v1"},
        "condition": {"nfp": 8, "n_base_coils": 3}, "evaluator": {"hash": "frozen"},
        "paths": {"optimizer_checkpoint_sha256": "frozen"},
        "baseline_r04": {"q0_checkpoint_sha256": "r012", "flow_steps": 32},
        "q0": {"model_config": {"width": 256}},
        "strategy": {"beta": 0.006021959241479635, "beta_calibration_result": {"beta": 0.006},
                     "flow_optimizer_steps_per_round": 10, "ema_lerp": 0.1}}


def test_radius_control_copies_frozen_beta_and_retains_radius_q0():
    reference = _reference()
    manifest = copy.deepcopy(reference)
    manifest["baseline_r04"]["q0_checkpoint_sha256"] = "r015"
    manifest["strategy"]["beta"] = None
    manifest["strategy"].pop("beta_calibration_result")
    inherit_reference_strategy(manifest, reference)
    assert manifest["strategy"] == reference["strategy"]
    assert manifest["strategy"] is not reference["strategy"]
    assert manifest["baseline_r04"]["q0_checkpoint_sha256"] == "r015"


@pytest.mark.parametrize("change", ["schedule", "scorer", "normalizer", "flow_steps", "weighted", "uncalibrated"])
def test_radius_control_rejects_extra_experimental_changes(change):
    reference = _reference()
    manifest = copy.deepcopy(reference)
    if change == "schedule":
        manifest["strategy"]["flow_optimizer_steps_per_round"] = 50
    elif change == "scorer":
        manifest["evaluator"]["hash"] = "other"
    elif change == "normalizer":
        manifest["paths"]["optimizer_checkpoint_sha256"] = "other"
    elif change == "flow_steps":
        manifest["baseline_r04"]["flow_steps"] = 128
    elif change == "weighted":
        reference["strategy"]["valid_score_weighting"] = {"tau": 7.5}
    elif change == "uncalibrated":
        reference["strategy"]["beta"] = None
    with pytest.raises(ValueError):
        inherit_reference_strategy(manifest, reference)
