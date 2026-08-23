from __future__ import annotations

import json

import numpy as np
import pytest

from stellarator_eval.native_evaluator import (
    CoilSet,
    EvaluationMode,
    Evaluator,
    NativeScorePolicy,
    PRODUCTION_SCORE_CONFIG,
    WeightedComponentPolicy,
)
from stellarator_eval.experimental.neighborhood import NeighborhoodEvaluator


def coils(nfp: int = 4) -> CoilSet:
    values = np.arange(2 * 5, dtype=np.float64).reshape(2, 5) / 10.0
    return CoilSet(values, values + 1.0, values - 1.0, [1.0e6, -1.0e6], nfp)


def native_result() -> dict:
    return {
        "score": 81.5,
        "status": "ok",
        "components": {
            "axis": 100.0,
            "psi": 90.0,
            "surface": 80.0,
            "coordinate": 70.0,
            "volume_qs": 60.0,
            "iota": 50.0,
            "coil": 40.0,
        },
        "timing": {"total_s": 0.75, "axis_search_s": 0.2},
        "diagnostics": {"axis_R": 1.2, "axis_Z": -0.1, "axis_used_hint": 0},
    }


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, library, x, y, z, currents, nfp, **kwargs):
        self.calls.append(
            {
                "library": library,
                "shape": x.shape,
                "currents": currents.copy(),
                "nfp": nfp,
                **kwargs,
            }
        )
        return native_result()


def test_independent_evaluation_returns_structured_metadata() -> None:
    backend = RecordingBackend()
    result = Evaluator("unused.so", backend=backend).evaluate(coils())

    assert result.ok
    assert result.score == result.native_score == 81.5
    assert result.mode is EvaluationMode.INDEPENDENT
    assert result.target_helicity == (1, 4)
    assert backend.calls[0]["config_overrides"] == dict(PRODUCTION_SCORE_CONFIG)
    assert result.continuation_state().R == 1.2
    assert result.to_dict()["input"]["current_unit"] == "A"


def test_strict_continuation_forces_mixed_same_branch_mode() -> None:
    backend = RecordingBackend()
    evaluator = Evaluator("unused.so", backend=backend)
    center = evaluator.evaluate(coils())
    evaluator.evaluate(
        coils(),
        mode=EvaluationMode.STRICT_CONTINUATION,
        continuation=center.continuation_state(),
    )

    config = backend.calls[-1]["config_overrides"]
    assert config["axis_hint_enabled"] == 1
    assert config["axis_hint_require_continuation"] == 2
    assert config["axis_hint_R"] == 1.2
    assert config["axis_hint_Z"] == -0.1


def test_raw_abi_defaults_require_explicit_opt_out() -> None:
    backend = RecordingBackend()
    Evaluator("unused.so", backend=backend, use_production_defaults=False).evaluate(coils())

    assert backend.calls[0]["config_overrides"] == {}


def test_axis_fields_cannot_be_smuggled_through_config() -> None:
    evaluator = Evaluator("unused.so", backend=RecordingBackend())
    with pytest.raises(ValueError, match="controlled by EvaluationMode"):
        evaluator.evaluate(coils(), config_overrides={"axis_hint_enabled": 1})


def test_strict_continuation_rejects_incompatible_state() -> None:
    evaluator = Evaluator("unused.so", backend=RecordingBackend())
    state = evaluator.evaluate(coils()).continuation_state()
    with pytest.raises(ValueError, match="nfp"):
        evaluator.evaluate(
            coils(nfp=5),
            mode=EvaluationMode.STRICT_CONTINUATION,
            continuation=state,
        )


def test_custom_component_policy_preserves_native_score() -> None:
    policy = WeightedComponentPolicy({"volume_qs": 3.0, "coil": 1.0})
    result = Evaluator("unused.so", backend=RecordingBackend(), score_policy=policy).evaluate(
        coils()
    )

    assert result.score == 55.0
    assert result.native_score == 81.5
    assert result.score_policy == "weighted_components"


def test_result_json_is_finite_and_self_describing(tmp_path) -> None:
    result = Evaluator(
        "unused.so", backend=RecordingBackend(), score_policy=NativeScorePolicy()
    ).evaluate(coils())
    path = tmp_path / "result.json"
    result.write_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == "stellarator-native-evaluation-v1"
    assert payload["mode"] == "independent"
    assert payload["diagnostics"]["axis_R"] == 1.2


def test_coil_set_rejects_nonfinite_values() -> None:
    values = np.zeros((1, 3))
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        CoilSet(values, np.zeros_like(values), np.zeros_like(values), [1.0], 4)


def test_neighborhood_rejects_layout_mismatch_before_gpu_setup() -> None:
    backend = RecordingBackend()
    center_coils = coils()
    center = Evaluator("unused.so", backend=backend).evaluate(center_coils)
    neighborhood = NeighborhoodEvaluator(
        "unused.so",
        batch_factory=lambda *args, **kwargs: pytest.fail("GPU setup should not run"),
        coil_gradient_backend=lambda *args, **kwargs: {},
    )
    with pytest.raises(ValueError, match="does not match center"):
        neighborhood.evaluate(center_coils, [coils(nfp=5)], center)
