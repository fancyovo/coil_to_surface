from __future__ import annotations

import numpy as np

from scripts.generate_axisflip_prior_dataset import (
    GENERATOR_FORMAT,
    N_BASE_COILS,
    TOKEN_DIM,
    block_tasks,
    generate_block,
)


def test_block_tasks_cover_requested_seed_range() -> None:
    assert block_tasks(100, 10, 4) == [
        (0, 100, 4),
        (4, 104, 4),
        (8, 108, 2),
    ]


def test_teacher_generation_is_deterministic_and_fixed_condition() -> None:
    offset_a, tokens_a = generate_block((7, 123456, 1))
    offset_b, tokens_b = generate_block((7, 123456, 1))
    assert GENERATOR_FORMAT.endswith("axis_flip_r012_v1")
    assert offset_a == offset_b == 7
    assert tokens_a.shape == (1, N_BASE_COILS, TOKEN_DIM)
    assert tokens_a.dtype == np.float32
    np.testing.assert_array_equal(tokens_a, tokens_b)
    np.testing.assert_allclose(tokens_a[0, :, -1], tokens_a[0, 0, -1])


def test_experimental_radius_formats_are_registered() -> None:
    for radius, tag in ((0.15, "r015"), (0.20, "r020")):
        offset, tokens = generate_block((3, 987654, 1, radius))
        assert offset == 3
        assert tokens.shape == (1, N_BASE_COILS, TOKEN_DIM)
        assert np.all(np.isfinite(tokens))
        from flow_matching.axis_surface_prior_v2 import axis_flip_registered_format_for_radius

        assert axis_flip_registered_format_for_radius(radius).endswith(f"{tag}_v1")
