"""Quality grading for standard Boozer surface candidates."""

from __future__ import annotations


def assess_surface_acceptance(
    *,
    solver_converged: bool,
    dense_relative_l2: float,
    dense_normal_field_p95: float,
    toroidal_winding: float,
    normal_min: float,
    max_final_relative_l2: float,
    max_final_normal_p95: float,
) -> dict[str, object]:
    """Separate hard downstream safety checks from strict quality warnings.

    Dense residual limits are a reproducible quality grade, not a theorem that
    a Simsopt surface ceases to exist immediately above the limit. A
    converged, non-degenerate, correctly wound surface may continue through
    the diagnostic pipeline with an explicit warning.
    """

    strict_checks = {
        "dense_relative_l2": dense_relative_l2 <= max_final_relative_l2,
        "dense_normal_field_p95": dense_normal_field_p95 <= max_final_normal_p95,
    }
    hard_checks = {
        "newton_converged": solver_converged,
        "toroidal_winding": toroidal_winding > 0.0,
        "normal_nonzero": normal_min > 1e-12,
    }
    warnings: list[dict[str, object]] = []
    for metric, value, limit, code in (
        (
            "dense_relative_l2",
            dense_relative_l2,
            max_final_relative_l2,
            "dense_relative_l2_above_strict_limit",
        ),
        (
            "dense_normal_field_p95",
            dense_normal_field_p95,
            max_final_normal_p95,
            "dense_normal_field_p95_above_strict_limit",
        ),
    ):
        if value > limit:
            warnings.append(
                {
                    "code": code,
                    "metric": metric,
                    "value": float(value),
                    "strict_limit": float(limit),
                    "ratio_to_limit": float(value / max(limit, 1e-30)),
                    "severity": "warning",
                }
            )

    strict_pass = all(hard_checks.values()) and all(strict_checks.values())
    accepted_for_downstream = all(hard_checks.values())
    return {
        "strict_checks": strict_checks,
        "hard_checks": hard_checks,
        "warnings": warnings,
        "strict_pass": strict_pass,
        "accepted_for_downstream": accepted_for_downstream,
        "evaluation_quality": (
            "strict_pass"
            if strict_pass
            else "accepted_with_quality_warning"
            if accepted_for_downstream
            else "rejected"
        ),
    }
