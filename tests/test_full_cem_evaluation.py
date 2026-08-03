import numpy as np

from scripts.evaluate_cem_candidate_full import (
    build_full_period_surface_mesh,
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


def test_full_period_mesh_connects_period_boundaries_instead_of_capping_each_period():
    xyz = np.zeros((3, 3, 3), dtype=float)
    xyz[..., 0] = 1.0
    xyz[1, :, 1] = 0.25
    xyz[2, :, 1] = 0.5
    colors = np.ones_like(xyz)

    full_xyz, full_colors, indices = build_full_period_surface_mesh(xyz, colors, nfp=3)

    assert full_xyz.shape == (9, 3, 3)
    assert full_colors.shape == full_xyz.shape
    assert indices.size == 9 * 3 * 6
    triangles = indices.reshape(-1, 3)
    edges = {
        tuple(sorted((int(triangle[start]), int(triangle[(start + 1) % 3]))))
        for triangle in triangles
        for start in range(3)
    }
    assert (6, 9) in edges
    assert (0, 6) not in edges
    assert (0, 24) in edges
