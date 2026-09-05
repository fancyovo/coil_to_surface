"""Provenance guards for the formal full-physical-evaluation pipeline."""

from __future__ import annotations

from typing import Mapping

import numpy as np


ALPHA_NU_INITIALIZER_KIND = "alpha_nu"
STANDARD_ALPHA_NU_SURFACE_KIND = "alpha_nu_standard_ls_newton"


def scalar_text(values: Mapping[str, object], key: str) -> str:
    """Read one text scalar from a loaded NPZ-like mapping."""

    if key not in values:
        raise ValueError(f"surface artifact is missing required provenance field {key!r}")
    value = np.asarray(values[key])
    if value.size != 1:
        raise ValueError(f"surface provenance field {key!r} must be scalar")
    item = value.reshape(()).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def require_surface_kind(
    values: Mapping[str, object], expected: str, *, stage: str
) -> str:
    """Reject artifacts produced by a different surface-construction route."""

    actual = scalar_text(values, "kind")
    if actual != expected:
        raise ValueError(
            f"{stage} requires surface kind={expected!r}; got {actual!r}. "
            "Formal full evaluation must use psi -> alpha -> nu -> Simsopt LS/Newton."
        )
    return actual
