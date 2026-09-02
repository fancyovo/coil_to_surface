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
    assert GENERATOR_FORMAT.endswith("axis_flip_v4")
    assert offset_a == offset_b == 7
    assert tokens_a.shape == (1, N_BASE_COILS, TOKEN_DIM)
    assert tokens_a.dtype == np.float32
    np.testing.assert_array_equal(tokens_a, tokens_b)
    np.testing.assert_allclose(tokens_a[0, :, -1], tokens_a[0, 0, -1])
