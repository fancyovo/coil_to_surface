from __future__ import annotations

from collections.abc import Mapping


SCORE_COMPONENTS = (
    "axis",
    "psi",
    "surface",
    "coordinate",
    "volume_qs",
    "iota",
    "coil",
)
SCORE_WEIGHTS = {
    "axis": 10.0,
    "psi": 10.0,
    "surface": 10.0,
    "coordinate": 10.0,
    "volume_qs": 42.0,
    "iota": 10.0,
    "coil": 8.0,
}
GRADIENT_OMITTED_COMPONENT = "coordinate"


def coordinate_omitted_gradient_score(
    formal_score: float,
    components: Mapping[str, float],
) -> float:
    """Return the score-like scalar used only for local gradient estimation.

    The formal score's multiplicative gate is retained, while the coordinate
    component is omitted and the remaining component weights are normalized to
    100. Formal center/proposal acceptance must continue to use formal_score.
    """
    total_weight = sum(SCORE_WEIGHTS.values())
    weighted_total = sum(
        SCORE_WEIGHTS[name] * float(components[name]) for name in SCORE_COMPONENTS
    )
    full_average = weighted_total / total_weight
    gate = float(formal_score) / full_average if full_average > 0.0 else 0.0

    retained_weight = total_weight - SCORE_WEIGHTS[GRADIENT_OMITTED_COMPONENT]
    retained_total = sum(
        SCORE_WEIGHTS[name] * float(components[name])
        for name in SCORE_COMPONENTS
        if name != GRADIENT_OMITTED_COMPONENT
    )
    return gate * retained_total / retained_weight
