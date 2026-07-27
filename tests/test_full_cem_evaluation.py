import numpy as np

from scripts.evaluate_cem_candidate_full import (
    parse_float_list,
    rotate_z,
    select_largest_surface,
)


def test_parse_float_list_requires_positive_values():
    assert parse_float_list("0.05,0.1") == [0.05, 0.1]


def test_select_largest_surface_ignores_failures():
    rows = [
        {"a": 0.05, "best_surface": None},
        {"a": 0.08, "best_surface": {"volume": 0.2}},
        {"a": 0.12, "best_surface": {"volume": 0.5}},
    ]
    assert select_largest_surface(rows)["a"] == 0.12


def test_rotate_z_rotates_all_points_without_changing_z():
    points = np.array([[[1.0, 0.0, 2.0], [0.0, 1.0, -3.0]]])
    rotated = rotate_z(points, np.pi / 2)
    np.testing.assert_allclose(rotated[0, :, :2], [[0.0, 1.0], [-1.0, 0.0]], atol=1e-14)
    np.testing.assert_allclose(rotated[..., 2], points[..., 2])
