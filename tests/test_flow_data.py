from __future__ import annotations

import numpy as np

from flow_matching.data import CoilNormalizer, RawGroup, canonicalize_currents


def test_canonicalize_currents_fixes_l1_and_global_sign():
    tokens = np.zeros((2, 2, 100), dtype=np.float32)
    tokens[0, :, -1] = [-1.0, 3.0]
    tokens[1, :, -1] = [-4.0, 1.0]
    result = canonicalize_currents(tokens, 2.0)
    np.testing.assert_allclose(np.abs(result[..., -1]).sum(axis=1), 2.0)
    assert result[0, 1, -1] > 0.0
    assert result[1, 0, -1] > 0.0


def test_normalizer_roundtrip_preserves_canonical_tokens():
    rng = np.random.default_rng(7)
    tokens = rng.normal(size=(16, 3, 100)).astype(np.float32)
    tokens[..., -1] *= 1.0e6
    groups = {(4, 3): RawGroup(tokens=tokens, ids=np.arange(16))}
    normalizer = CoilNormalizer.fit(groups)
    normalized, clipped = normalizer.transform(tokens, (4, 3))
    restored = normalizer.inverse(normalized, (4, 3))
    expected = canonicalize_currents(tokens, normalizer.current_l1_a["4:3"])
    assert clipped == 0.0
    np.testing.assert_allclose(restored, expected, rtol=2.0e-5, atol=1.0e-4)
