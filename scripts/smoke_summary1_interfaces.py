#!/usr/bin/env python3
"""Dependency-light smoke checks for the public evaluator interfaces."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stellarator_eval.experimental.neighborhood import NeighborhoodEvaluator
from stellarator_eval.native_evaluator import (
    CoilSet,
    EvaluationMode,
    Evaluator,
    PRODUCTION_SCORE_CONFIG,
    WeightedComponentPolicy,
)


def _coils(nfp: int = 4) -> CoilSet:
    values = np.arange(10, dtype=np.float64).reshape(2, 5) / 10.0
    return CoilSet(values, values + 1.0, values - 1.0, [1.0e6, -1.0e6], nfp)


def _native_result() -> dict:
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
        "timing": {"total_s": 0.75},
        "diagnostics": {"axis_R": 1.2, "axis_Z": -0.1},
    }


class _Backend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, library, x, y, z, currents, nfp, **kwargs):
        self.calls.append({"nfp": nfp, **kwargs})
        return _native_result()


def _expect_value_error(call, message: str) -> None:
    try:
        call()
    except ValueError as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def main() -> None:
    backend = _Backend()
    evaluator = Evaluator("unused.so", backend=backend)
    center = evaluator.evaluate(_coils())
    assert center.ok and center.score == center.native_score == 81.5
    assert backend.calls[-1]["config_overrides"] == dict(PRODUCTION_SCORE_CONFIG)

    evaluator.evaluate(
        _coils(),
        mode=EvaluationMode.STRICT_CONTINUATION,
        continuation=center.continuation_state(),
    )
    config = backend.calls[-1]["config_overrides"]
    assert config["axis_hint_require_continuation"] == 2
    assert (config["axis_hint_R"], config["axis_hint_Z"]) == (1.2, -0.1)

    raw_backend = _Backend()
    Evaluator(
        "unused.so", backend=raw_backend, use_production_defaults=False
    ).evaluate(_coils())
    assert raw_backend.calls[-1]["config_overrides"] == {}

    _expect_value_error(
        lambda: evaluator.evaluate(
            _coils(), config_overrides={"axis_hint_enabled": 1}
        ),
        "controlled by EvaluationMode",
    )
    _expect_value_error(
        lambda: evaluator.evaluate(
            _coils(nfp=5),
            mode=EvaluationMode.STRICT_CONTINUATION,
            continuation=center.continuation_state(),
        ),
        "nfp",
    )

    weighted = Evaluator(
        "unused.so",
        backend=_Backend(),
        score_policy=WeightedComponentPolicy({"volume_qs": 3.0, "coil": 1.0}),
    ).evaluate(_coils())
    assert weighted.score == 55.0 and weighted.native_score == 81.5

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "result.json"
        center.write_json(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == "stellarator-native-evaluation-v1"

    bad = np.zeros((1, 3))
    bad[0, 0] = np.nan
    _expect_value_error(
        lambda: CoilSet(bad, np.zeros_like(bad), np.zeros_like(bad), [1.0], 4),
        "finite",
    )

    neighborhood = NeighborhoodEvaluator(
        "unused.so",
        batch_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("GPU setup should not run")
        ),
        coil_gradient_backend=lambda *args, **kwargs: {},
    )
    _expect_value_error(
        lambda: neighborhood.evaluate(_coils(), [_coils(nfp=5)], center),
        "does not match center",
    )
    print("summary1 interface smoke checks: 9 passed")


if __name__ == "__main__":
    main()
