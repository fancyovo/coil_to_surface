from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.qh_flow_landscape import (
    CaseRegistry,
    curve_roughness,
    first_drop_radius,
    first_drop_width,
    normalize_direction,
)


def test_case_registry_deduplicates_identical_physical_tokens():
    registry = CaseRegistry()
    tokens = np.arange(300, dtype=np.float32).reshape(3, 100)
    assert registry.register(tokens, 4) == 0
    assert registry.register(tokens.copy(), 4) == 0
    assert registry.register(tokens, 5) == 1
    changed = tokens.copy()
    changed[0, 0] += 1.0
    assert registry.register(changed, 4) == 2


def test_normalize_direction_sets_unit_rms():
    direction = normalize_direction(np.arange(1, 13).reshape(3, 4))
    np.testing.assert_allclose(np.sqrt(np.mean(direction**2)), 1.0, rtol=1.0e-6)


def test_first_drop_width_interpolates_both_sides():
    alphas = np.linspace(-0.2, 0.2, 5)
    scores = 70.0 - 100.0 * np.abs(alphas)
    width = first_drop_width(alphas, scores, drop=5.0)
    np.testing.assert_allclose(width["negative"], 0.05)
    np.testing.assert_allclose(width["positive"], 0.05)
    np.testing.assert_allclose(width["total"], 0.1)
    assert not width["negative_censored"]
    assert not width["positive_censored"]


def test_first_drop_width_reports_censored_scan_boundary():
    alphas = np.linspace(-0.2, 0.2, 5)
    width = first_drop_width(alphas, np.full(5, 70.0), drop=5.0)
    assert width["total"] == 0.4
    assert width["negative_censored"]
    assert width["positive_censored"]


def test_first_drop_radius_uses_physical_displacement_at_crossing():
    alphas = np.linspace(-0.2, 0.2, 5)
    scores = 70.0 - 100.0 * np.abs(alphas)
    radii = 0.25 * np.abs(alphas)
    radius = first_drop_radius(alphas, radii, scores, drop=5.0)
    np.testing.assert_allclose(radius["negative"], 0.0125)
    np.testing.assert_allclose(radius["positive"], 0.0125)
    np.testing.assert_allclose(radius["mean"], 0.0125)


def test_curve_roughness_is_zero_for_linear_curve_on_nonuniform_grid():
    alphas = np.asarray([-0.2, -0.03, -0.001, 0.0, 0.007, 0.08, 0.2])
    roughness = curve_roughness(alphas, 3.0 * alphas + 2.0)
    assert roughness["second_derivative_rms"] < 1.0e-10


def test_landscape_analysis_import_does_not_load_torch():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.qh_flow_landscape; "
            "assert 'torch' not in sys.modules",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
