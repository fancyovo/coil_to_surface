from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CURRENT_QH_PROTOCOL_ID = "qh-flow-screen32-adam200-64d-v1"
DEPRECATED_QH_DIRECTION_COUNT = 2
CURRENT_NATIVE_SCORE_ABI = 10
CURRENT_NATIVE_SCORE_LIBRARY_SHA256 = (
    "565c32073b145d97a1f2244705fb06e4b3458ce798cd74d0c97ee4e0129dc729"
)


@dataclass(frozen=True)
class QHOptimizationDefaults:
    """Validated defaults used by the 309-trajectory QH experiment."""

    candidate_count: int = 32
    iterations: int = 200
    directions: int = 64
    perturbation: float = 0.005
    learning_rate: float = 0.02
    beta1: float = 0.7
    beta2: float = 0.999
    flow_steps: int = 128
    gradient_mode: str = "random-orthogonal"


QH_OPTIMIZATION_DEFAULTS = QHOptimizationDefaults()


def validate_qh_direction_count(directions: int) -> None:
    """Reject the retired two-direction recipe on all current QH entry points."""

    if directions == DEPRECATED_QH_DIRECTION_COUNT:
        raise ValueError(
            "the 2-direction QH recipe is deprecated historical evidence and "
            "cannot be launched from current main; define and review a new "
            "experimental protocol instead"
        )
    if directions < 1:
        raise ValueError("direction count must be positive")


def validate_qh_resume_protocol(
    saved_protocol: Any, requested_protocol: dict[str, Any]
) -> None:
    """Require an exact, classified protocol match before restoring run state."""

    if not isinstance(saved_protocol, dict):
        raise ValueError(
            "legacy or unclassified optimizer manifests cannot resume on current "
            "main; preserve the run as history and start a new named experiment"
        )
    saved_actual = saved_protocol.get("actual")
    if not isinstance(saved_actual, dict):
        raise ValueError("saved optimizer protocol is missing its actual settings")
    saved_directions = saved_actual.get("directions")
    if not isinstance(saved_directions, int):
        raise ValueError("saved optimizer protocol has no integer direction count")
    validate_qh_direction_count(saved_directions)
    if saved_protocol != requested_protocol:
        raise ValueError(
            "resume protocol does not exactly match the saved manifest; an "
            "override is a new experiment, not a continuation"
        )


def _protocol_description(
    *, stage: str, actual: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    differences = {
        key: {"expected": expected[key], "actual": actual.get(key)}
        for key in expected
        if actual.get(key) != expected[key]
    }
    return {
        "id": CURRENT_QH_PROTOCOL_ID if not differences else "unregistered-experimental",
        "status": (
            "current-default" if not differences else "unregistered-experimental"
        ),
        "stage": stage,
        "requirements": {
            "native_score_abi": CURRENT_NATIVE_SCORE_ABI,
            "native_score_library_sha256": CURRENT_NATIVE_SCORE_LIBRARY_SHA256,
        },
        "actual": actual,
        "differences_from_current": differences,
    }


def describe_qh_screening_protocol(
    *, candidate_count: int, flow_steps: int
) -> dict[str, Any]:
    expected = {
        "candidate_count": QH_OPTIMIZATION_DEFAULTS.candidate_count,
        "flow_steps": QH_OPTIMIZATION_DEFAULTS.flow_steps,
    }
    actual = {
        "candidate_count": int(candidate_count),
        "flow_steps": int(flow_steps),
    }
    return _protocol_description(stage="screening", actual=actual, expected=expected)


def describe_qh_optimization_protocol(
    *,
    parameter_space: str,
    optimizer: str,
    iterations: int,
    directions: int,
    perturbation: float,
    learning_rate: float,
    beta1: float,
    beta2: float,
    flow_steps: int,
    gradient_mode: str,
    difference: str,
) -> dict[str, Any]:
    validate_qh_direction_count(directions)
    expected = asdict(QH_OPTIMIZATION_DEFAULTS)
    expected.pop("candidate_count")
    expected.update(
        {
            "parameter_space": "latent",
            "optimizer": "adam",
            "difference": "centered",
        }
    )
    actual = {
        "parameter_space": str(parameter_space),
        "optimizer": str(optimizer),
        "iterations": int(iterations),
        "directions": int(directions),
        "perturbation": float(perturbation),
        "learning_rate": float(learning_rate),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "flow_steps": int(flow_steps),
        "gradient_mode": str(gradient_mode),
        "difference": str(difference),
    }
    return _protocol_description(stage="optimization", actual=actual, expected=expected)


__all__ = [
    "CURRENT_QH_PROTOCOL_ID",
    "CURRENT_NATIVE_SCORE_ABI",
    "CURRENT_NATIVE_SCORE_LIBRARY_SHA256",
    "DEPRECATED_QH_DIRECTION_COUNT",
    "QHOptimizationDefaults",
    "QH_OPTIMIZATION_DEFAULTS",
    "describe_qh_optimization_protocol",
    "describe_qh_screening_protocol",
    "validate_qh_direction_count",
    "validate_qh_resume_protocol",
]
