import numpy as np

from scripts.optimize_native_score_cem import (
    COEFF_COUNT,
    TOKEN_DIM,
    compact_score_diagnostics,
    decode_latents,
    normalize_current_l1,
    token_case,
    update_distribution,
)


def test_decode_latents_uses_whitened_pca_directions():
    components = np.zeros((TOKEN_DIM, 2), dtype=np.float32)
    components[0, 0] = 1.0
    components[1, 1] = 1.0
    pca = {
        "mean": np.ones(TOKEN_DIM, dtype=np.float32),
        "components": components,
        "scale": np.array([2.0, 3.0], dtype=np.float32),
        "current_scale": 1.0e6,
    }
    decoded = decode_latents(np.array([[[0.5, -2.0]]], dtype=np.float32), pca)
    assert decoded.shape == (1, 1, TOKEN_DIM)
    np.testing.assert_allclose(decoded[0, 0, :3], [2.0, -5.0, 1.0])


def test_normalize_current_l1_preserves_ratios():
    tokens = np.zeros((1, 2, TOKEN_DIM), dtype=float)
    tokens[0, :, -1] = [-0.25, 0.75]
    normalized = normalize_current_l1(tokens, 2.0)
    np.testing.assert_allclose(normalized[0, :, -1], [-0.5, 1.5])


def test_token_case_keeps_xyz_layout_and_target():
    tokens = np.arange(2 * TOKEN_DIM, dtype=float).reshape(2, TOKEN_DIM)
    case = token_case(tokens, nfp=4, target="QH")
    assert case["nfp"] == 4
    assert case["raw"]["x"][0] == tokens[0, :COEFF_COUNT].tolist()
    assert case["raw"]["y"][0] == tokens[0, COEFF_COUNT : 2 * COEFF_COUNT].tolist()
    assert case["raw"]["z"][0] == tokens[0, 2 * COEFF_COUNT : 3 * COEFF_COUNT].tolist()
    assert case["raw"]["metadata"]["helicity"] == 1


def test_distribution_update_is_smoothed_and_bounded():
    mean = np.zeros((1, 2), dtype=np.float32)
    sigma = np.ones((1, 2), dtype=np.float32)
    elite = np.array([[[2.0, -2.0]], [[4.0, -4.0]]], dtype=np.float32)
    next_mean, next_sigma = update_distribution(
        mean,
        sigma,
        elite,
        smoothing=0.5,
        min_sigma=0.1,
        max_sigma=0.8,
        latent_limit=1.0,
    )
    np.testing.assert_allclose(next_mean, [[1.0, -1.0]])
    np.testing.assert_allclose(next_sigma, [[0.8, 0.8]])


def test_compact_diagnostics_tolerates_early_failure_fields():
    result = {
        "score": 7.0,
        "status": "no_axis",
        "components": {"axis": 0.0},
        "diagnostics": {"surface_level": float("nan")},
    }
    compact = compact_score_diagnostics(result)
    assert compact["status"] == "no_axis"
    assert np.isnan(compact["diagnostics"]["iota_min"])
