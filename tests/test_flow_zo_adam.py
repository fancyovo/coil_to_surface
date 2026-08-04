from __future__ import annotations

import json

import numpy as np

from scripts.optimize_flow_prior_zo_adam import (
    gradient_from_pairs,
    load_initial_noise,
    orthogonal_directions,
    prior_penalty_and_gradient,
)
from scripts.optimize_flow_prior_standard_adam import (
    parse_backtracking_fractions,
    robust_direction_deltas,
    rolling_robust_limit,
    write_trajectory_case,
)


def test_load_initial_noise_accepts_generic_start(tmp_path):
    path = tmp_path / "start.json"
    noise = np.arange(300, dtype=np.float32).reshape(3, 100)
    path.write_text(
        json.dumps({"flow_prior_start": {"noise": noise.tolist()}}),
        encoding="utf-8",
    )

    loaded, payload = load_initial_noise(path)

    np.testing.assert_array_equal(loaded, noise)
    assert "flow_prior_start" in payload


def test_load_initial_noise_accepts_optimizer_outputs_and_raw_trajectory(tmp_path):
    noise = np.arange(300, dtype=np.float32).reshape(3, 100)
    for key in (
        "flow_prior_standard_adam",
        "flow_prior_subspace_bfgs",
        "flow_prior_g3_informed_subspace_adam",
    ):
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps({key: {"noise": noise.tolist()}}), encoding="utf-8")
        loaded, payload = load_initial_noise(path)
        np.testing.assert_array_equal(loaded, noise)
        assert key in payload

    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps({"noise": noise.tolist()}), encoding="utf-8")
    loaded, payload = load_initial_noise(trajectory_path)
    np.testing.assert_array_equal(loaded, noise)
    assert "noise" in payload


def test_orthogonal_directions_have_unit_rms():
    directions = orthogonal_directions(np.random.default_rng(73), (2, 3), 4)
    flat = directions.reshape(4, -1).astype(np.float64)
    np.testing.assert_allclose(
        np.sqrt(np.mean(flat * flat, axis=1)), np.ones(4), atol=2.0e-7
    )
    gram = flat @ flat.T
    np.testing.assert_allclose(
        gram, np.eye(4) * flat.shape[1], atol=2.0e-6
    )


def test_full_orthogonal_gradient_recovers_linear_objective():
    rng = np.random.default_rng(91)
    shape = (2, 3)
    center = rng.normal(size=shape)
    coefficients = rng.normal(size=shape)
    directions = orthogonal_directions(rng, shape, int(np.prod(shape)))
    perturbation = 0.03

    def objective(value):
        return float(np.sum(coefficients * value))

    plus = np.asarray(
        [objective(center + perturbation * direction) for direction in directions]
    )
    minus = np.asarray(
        [objective(center - perturbation * direction) for direction in directions]
    )
    gradient, raw_delta = gradient_from_pairs(
        plus,
        minus,
        directions,
        perturbation,
        delta_clip=None,
    )
    assert np.all(np.abs(raw_delta) > 0.0)
    np.testing.assert_allclose(gradient, coefficients, rtol=2.0e-6, atol=2.0e-6)


def test_direction_delta_clipping_limits_gradient_contribution():
    directions = np.asarray([[[1.0, -1.0]]], dtype=np.float32)
    gradient, raw_delta = gradient_from_pairs(
        np.asarray([100.0]),
        np.asarray([0.0]),
        directions,
        0.5,
        delta_clip=4.0,
    )
    np.testing.assert_allclose(raw_delta, [100.0])
    np.testing.assert_allclose(gradient, [[4.0, -4.0]])


def test_robust_direction_filter_drops_invalid_and_rescales_valid_directions():
    used, invalid, outlier, limit = robust_direction_deltas(
        np.asarray([-58.0, -0.5, -57.0, -0.2]),
        ["no_axis", "ok", "no_axis", "ok", "ok", "ok", "ok", "ok"],
        outlier_ratio=8.0,
        mad_factor=8.0,
    )

    np.testing.assert_array_equal(invalid, [True, False, True, False])
    np.testing.assert_array_equal(outlier, [False, False, False, False])
    np.testing.assert_allclose(used, [0.0, -0.5, 0.0, -0.2])
    assert limit is None


def test_robust_direction_filter_is_scale_invariant_for_valid_outlier():
    statuses = ["ok"] * 8
    delta = np.asarray([0.4, 0.5, 0.6, 100.0])
    used, invalid, outlier, limit = robust_direction_deltas(
        delta, statuses, outlier_ratio=8.0, mad_factor=8.0
    )
    scaled, scaled_invalid, scaled_outlier, scaled_limit = robust_direction_deltas(
        100.0 * delta, statuses, outlier_ratio=8.0, mad_factor=8.0
    )

    np.testing.assert_array_equal(invalid, np.zeros(4, dtype=bool))
    np.testing.assert_array_equal(outlier, [False, False, False, True])
    np.testing.assert_array_equal(scaled_invalid, invalid)
    np.testing.assert_array_equal(scaled_outlier, outlier)
    np.testing.assert_allclose(scaled, 100.0 * used)
    assert np.isclose(scaled_limit, 100.0 * limit)


def test_rolling_robust_limit_rejects_step_185_scale_spike():
    prior_gradient_rms = [4.7 + 0.03 * index for index in range(20)]
    prior_update_rms = [0.0021 + 0.00002 * index for index in range(20)]

    gradient_limit = rolling_robust_limit(
        prior_gradient_rms,
        window=20,
        min_history=20,
        ratio=8.0,
        mad_factor=8.0,
    )
    update_limit = rolling_robust_limit(
        prior_update_rms,
        window=20,
        min_history=20,
        ratio=8.0,
        mad_factor=8.0,
    )

    assert gradient_limit is not None and 337.109 > gradient_limit
    assert update_limit is not None and 0.042175 > update_limit


def test_rolling_robust_limit_has_warmup_and_is_scale_invariant():
    values = [1.0 + 0.01 * index for index in range(20)]
    assert (
        rolling_robust_limit(
            values[:19],
            window=20,
            min_history=20,
            ratio=8.0,
            mad_factor=8.0,
        )
        is None
    )
    limit = rolling_robust_limit(
        values,
        window=20,
        min_history=20,
        ratio=8.0,
        mad_factor=8.0,
    )
    scaled_limit = rolling_robust_limit(
        [100.0 * value for value in values],
        window=20,
        min_history=20,
        ratio=8.0,
        mad_factor=8.0,
    )
    assert np.isclose(scaled_limit, 100.0 * limit)


def test_parse_backtracking_fractions_requires_decreasing_unit_interval():
    assert parse_backtracking_fractions("0.5,0.25,0.125") == (0.5, 0.25, 0.125)
    assert parse_backtracking_fractions("") == ()

    for invalid in ("1.0", "0.0", "0.25,0.5", "0.5,0.5"):
        try:
            parse_backtracking_fractions(invalid)
        except Exception as exc:
            assert "backtracking fractions" in str(exc)
        else:
            raise AssertionError(f"expected invalid backtracking schedule: {invalid}")


def test_write_trajectory_case_preserves_latent_coils_and_score(tmp_path):
    tokens = np.arange(200, dtype=np.float64).reshape(2, 100)
    noise = np.arange(200, dtype=np.float32).reshape(2, 100) / 10.0
    result = {
        "score": 81.25,
        "status": "ok",
        "components": {"coil": 72.5, "volume_qs": 91.0},
        "diagnostics": {"qs_target_global_error_per_helicity": 0.003},
    }

    path = write_trajectory_case(
        tmp_path,
        tokens,
        noise,
        result,
        nfp=4,
        target="QH",
        iteration=7,
        optimizer_state={"adam_step": 6, "current_score": 81.25},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "step_0007.json"
    assert payload["nfp"] == 4
    assert payload["raw"]["current"] == tokens[:, -1].tolist()
    metadata = payload["flow_prior_standard_adam_trajectory"]
    assert metadata["iteration"] == 7
    assert metadata["noise"] == noise.tolist()
    assert metadata["native_score"] == result
    assert metadata["optimizer_state"]["adam_step"] == 6


def test_prior_penalty_is_inactive_inside_soft_region():
    noise = np.asarray([[0.5, -1.0, 1.5]], dtype=float)
    penalty, gradient = prior_penalty_and_gradient(
        noise,
        rms_soft=2.0,
        coordinate_soft=4.0,
        rms_weight=5.0,
        coordinate_weight=5.0,
    )
    assert penalty == 0.0
    np.testing.assert_array_equal(gradient, np.zeros_like(noise))


def test_prior_gradient_points_outward_beyond_soft_region():
    noise = np.asarray([[5.0, -5.0, 3.0]], dtype=float)
    penalty, gradient = prior_penalty_and_gradient(
        noise,
        rms_soft=2.0,
        coordinate_soft=4.0,
        rms_weight=5.0,
        coordinate_weight=5.0,
    )
    assert penalty > 0.0
    assert np.all(gradient * noise > 0.0)
