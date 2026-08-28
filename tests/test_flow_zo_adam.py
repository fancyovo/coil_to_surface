from __future__ import annotations

import json

import numpy as np

from flow_matching.optimization import QH_OPTIMIZATION_DEFAULTS
from scripts.optimize_flow_prior_zo_adam import (
    gradient_from_pairs,
    load_initial_noise,
    orthogonal_directions,
    prior_penalty_and_gradient,
)
from scripts.optimize_flow_prior_standard_adam import (
    directional_score_deltas,
    gradient_from_direction_deltas,
    parse_backtracking_fractions,
    parse_arguments,
    robust_direction_deltas,
    rolling_robust_limit,
    sample_direction_probe,
    write_trajectory_case,
)


def test_standard_adam_production_defaults():
    args = parse_arguments(
        ["--checkpoint", "checkpoint.pt", "--out-dir", "run"]
    )

    assert args.iterations == QH_OPTIMIZATION_DEFAULTS.iterations
    assert args.learning_rate == QH_OPTIMIZATION_DEFAULTS.learning_rate
    assert args.perturbation == QH_OPTIMIZATION_DEFAULTS.perturbation
    assert args.beta1 == QH_OPTIMIZATION_DEFAULTS.beta1
    assert args.beta2 == QH_OPTIMIZATION_DEFAULTS.beta2
    assert args.flow_steps == QH_OPTIMIZATION_DEFAULTS.flow_steps
    assert args.flow_pipeline
    assert args.directions == QH_OPTIMIZATION_DEFAULTS.directions
    assert args.gradient_estimator == "central"
    assert args.score_surface_mode == "continuous"
    assert args.axis_continuation
    assert args.robust_direction_filter
    assert args.reject_invalid_center
    assert args.invalid_center_backtracking == (0.5, 0.25, 0.125)


def test_standard_adam_production_defaults_can_be_disabled():
    args = parse_arguments(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--out-dir",
            "run",
            "--no-flow-pipeline",
            "--no-axis-continuation",
            "--no-robust-direction-filter",
            "--no-reject-invalid-center",
            "--score-surface-mode",
            "legacy",
        ]
    )

    assert not args.flow_pipeline
    assert not args.axis_continuation
    assert not args.robust_direction_filter
    assert not args.reject_invalid_center
    assert args.invalid_center_backtracking == ()
    assert args.score_surface_mode == "legacy"


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


def test_load_initial_noise_accepts_standardized_data_prior(tmp_path):
    parameters = np.arange(200, dtype=np.float32).reshape(2, 100)
    path = tmp_path / "data_start.json"
    path.write_text(
        json.dumps(
            {
                "data_prior_screening": {
                    "normalized_coil_tokens": parameters.tolist()
                }
            }
        ),
        encoding="utf-8",
    )

    loaded, payload = load_initial_noise(path)

    np.testing.assert_array_equal(loaded, parameters)
    assert "data_prior_screening" in payload


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


def test_one_sided_probe_recovers_linear_objective_with_full_direction_bank():
    direction_rng = np.random.default_rng(123)
    sign_rng = np.random.default_rng(456)
    center = np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0
    coefficients = np.asarray(
        [[0.5, -1.0, 2.0], [1.5, -0.25, 0.75]], dtype=np.float64
    )
    perturbation = 0.005
    directions, signs, states = sample_direction_probe(
        direction_rng,
        sign_rng,
        center,
        directions=6,
        direction_bank_size=6,
        gradient_estimator="one-sided",
        perturbation=perturbation,
    )
    center_score = float(np.sum(center * coefficients))
    endpoint_scores = np.asarray(
        [float(np.sum(state * coefficients)) for state in states]
    )
    deltas = directional_score_deltas(
        endpoint_scores,
        center_score=center_score,
        signs=signs,
        gradient_estimator="one-sided",
    )
    gradient = gradient_from_direction_deltas(deltas, directions, perturbation)

    assert len(states) == 6
    assert set(signs.tolist()) <= {-1, 1}
    np.testing.assert_allclose(gradient, coefficients, rtol=2.0e-5, atol=2.0e-5)


def test_central_and_one_sided_share_leading_direction_bank():
    center = np.zeros((2, 3), dtype=np.float32)
    central, _, central_states = sample_direction_probe(
        np.random.default_rng(22),
        np.random.default_rng(33),
        center,
        directions=4,
        direction_bank_size=4,
        gradient_estimator="central",
        perturbation=0.01,
    )
    one_sided, signs, one_sided_states = sample_direction_probe(
        np.random.default_rng(22),
        np.random.default_rng(44),
        center,
        directions=4,
        direction_bank_size=4,
        gradient_estimator="one-sided",
        perturbation=0.01,
    )
    two_direction, _, _ = sample_direction_probe(
        np.random.default_rng(22),
        np.random.default_rng(55),
        center,
        directions=2,
        direction_bank_size=4,
        gradient_estimator="central",
        perturbation=0.01,
    )

    np.testing.assert_array_equal(one_sided, central)
    np.testing.assert_array_equal(two_direction, central[:2])
    assert len(central_states) == 8
    assert len(one_sided_states) == 4
    np.testing.assert_allclose(
        one_sided_states,
        center[None] + 0.01 * signs[:, None, None] * central,
    )


def test_update_anchored_probe_uses_anchor_and_orthogonal_random_direction():
    center = np.zeros((2, 3), dtype=np.float32)
    anchor = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]], dtype=np.float32)
    directions, signs, states = sample_direction_probe(
        np.random.default_rng(71),
        np.random.default_rng(72),
        center,
        directions=2,
        direction_bank_size=2,
        gradient_estimator="central",
        perturbation=0.005,
        anchor_direction=anchor,
    )

    expected_anchor = anchor / np.sqrt(np.mean(anchor.astype(np.float64) ** 2))
    np.testing.assert_allclose(directions[0], expected_anchor, rtol=2.0e-7, atol=2.0e-7)
    gram = np.mean(
        directions[:, None].astype(np.float64)
        * directions[None, :].astype(np.float64),
        axis=(2, 3),
    )
    np.testing.assert_allclose(gram, np.eye(2), rtol=2.0e-7, atol=2.0e-7)
    np.testing.assert_array_equal(signs, np.ones(2, dtype=np.int8))
    np.testing.assert_allclose(states[:2], 0.005 * directions)
    np.testing.assert_allclose(states[2:], -0.005 * directions)


def test_robust_direction_filter_accepts_one_sided_endpoint_statuses():
    used, invalid, outlier, limit = robust_direction_deltas(
        np.asarray([0.4, -0.5, 0.6, -0.2]),
        ["ok", "no_axis", "ok", "ok"],
        gradient_estimator="one-sided",
        outlier_ratio=8.0,
        mad_factor=8.0,
    )

    np.testing.assert_array_equal(invalid, [False, True, False, False])
    np.testing.assert_array_equal(outlier, np.zeros(4, dtype=bool))
    np.testing.assert_allclose(used, [0.4, 0.0, 0.6, -0.2])
    assert np.isclose(limit, 3.2)


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
