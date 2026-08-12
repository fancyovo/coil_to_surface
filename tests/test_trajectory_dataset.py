from __future__ import annotations

import gzip
import json

import numpy as np

from flow_matching.trajectory_dataset import COMPONENT_KEYS, OptimizationTraceRecorder


def native_result(score: float) -> dict:
    return {
        "score": score,
        "status": "ok",
        "components": {key: score + index for index, key in enumerate(COMPONENT_KEYS)},
        "timing": {"axis": 0.1},
        "diagnostics": {"iota_min": 1.2, "error_message": ""},
    }


def test_optimization_trace_preserves_training_supervision(tmp_path) -> None:
    recorder = OptimizationTraceRecorder(tmp_path)
    center = np.zeros((2, 100), dtype=np.float32)
    recorder.record_initial(center, center + 10.0, native_result(20.0))
    for iteration in (1, 2):
        directions = np.ones((3, 2, 100), dtype=np.float32) * iteration
        endpoints = np.ones((6, 2, 100), dtype=np.float32) * (iteration + 1)
        recorder.record_step(
            iteration=iteration,
            probe_noise=center + iteration,
            probe_tokens=center + iteration + 10.0,
            directions=directions,
            endpoint_tokens=endpoints,
            local_results=[native_result(float(index)) for index in range(6)],
            probe_result=native_result(20.0 + iteration),
            raw_gradient=center + 1.0,
            first_moment_before=center,
            second_moment_before=center,
            first_moment_after=center + 0.1,
            second_moment_after=center + 0.2,
            proposed_update=center + 0.3,
            applied_update=center + 0.3,
            center_after_noise=center + iteration + 0.3,
            center_after_tokens=center + iteration + 10.3,
            center_result=native_result(21.0 + iteration),
            gradient_step_applied=True,
            center_update_accepted=True,
            center_acceptance_fraction=1.0,
            adam_step=iteration,
        )

    schema = recorder.finalize()
    with np.load(tmp_path / "training_trace.npz", allow_pickle=False) as payload:
        assert payload["directions"].shape == (2, 3, 2, 100)
        assert payload["endpoint_tokens"].shape == (2, 6, 2, 100)
        assert payload["endpoint_components"].shape == (2, 6, len(COMPONENT_KEYS))
        np.testing.assert_allclose(payload["endpoint_score"][0], np.arange(6))
    with gzip.open(tmp_path / "center_native_results.jsonl.gz", "rt") as stream:
        center_rows = [json.loads(line) for line in stream]
    assert len(center_rows) == 3
    assert center_rows[1]["probe_captured_native_score"]["score"] == 21.0
    assert center_rows[1]["center_after_native_score"]["score"] == 22.0
    assert schema["completed_steps"] == 2
