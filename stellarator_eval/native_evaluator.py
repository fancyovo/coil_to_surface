from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np


RESULT_SCHEMA = "stellarator-native-evaluation-v1"
PRODUCTION_SCORE_CONFIG = MappingProxyType(
    {
        "iota_degree": 3,
        "surface_selection_mode": 1,
        "surface_confidence_periods": 1,
        "surface_theta_count": 128,
        "surface_trace_steps": 400,
        "surface_flux_bisection_iters": 6,
    }
)
_AXIS_CONFIG_FIELDS = {
    "axis_hint_enabled",
    "axis_hint_require_continuation",
    "axis_hint_R",
    "axis_hint_Z",
    "axis_hint_max_distance",
}


class EvaluationMode(str, Enum):
    """How the formal native evaluator locates the magnetic axis."""

    INDEPENDENT = "independent"
    STRICT_CONTINUATION = "strict_continuation"
    NEIGHBORHOOD_PROXY = "neighborhood_proxy"


class ScorePolicy(Protocol):
    """Map native metadata to a user-facing scalar score."""

    def __call__(self, metadata: Mapping[str, Any]) -> float: ...


@dataclass(frozen=True)
class NativeScorePolicy:
    """Use the score assembled inside the validated C++/CUDA evaluator."""

    name: str = "native_default"

    def __call__(self, metadata: Mapping[str, Any]) -> float:
        return float(metadata["score"])


@dataclass(frozen=True)
class WeightedComponentPolicy:
    """Build a transparent weighted mean from the seven 0--100 components.

    This intentionally does not reproduce the native QH multiplicative gates.
    Users that need those gates should use :class:`NativeScorePolicy` or supply
    a custom callable that reads the diagnostic fields explicitly.
    """

    weights: Mapping[str, float]
    name: str = "weighted_components"

    def __post_init__(self) -> None:
        copied = {str(key): float(value) for key, value in self.weights.items()}
        if not copied or any(not math.isfinite(value) or value < 0.0 for value in copied.values()):
            raise ValueError("component weights must be finite, nonnegative, and nonempty")
        if sum(copied.values()) <= 0.0:
            raise ValueError("component weights must have a positive sum")
        object.__setattr__(self, "weights", copied)

    def __call__(self, metadata: Mapping[str, Any]) -> float:
        components = metadata.get("components")
        if not isinstance(components, Mapping):
            raise ValueError("native result does not contain score components")
        missing = set(self.weights) - set(components)
        if missing:
            raise ValueError(f"native result is missing components: {sorted(missing)}")
        denominator = sum(self.weights.values())
        return float(
            sum(self.weights[name] * float(components[name]) for name in self.weights)
            / denominator
        )


@dataclass(frozen=True)
class CoilSet:
    """Fourier coefficients and currents for one stellarator configuration.

    Coefficient arrays have shape ``(n_base_coils, n_coefficients)`` and
    currents are expressed in amperes.
    """

    coeffs_x: np.ndarray
    coeffs_y: np.ndarray
    coeffs_z: np.ndarray
    currents_a: np.ndarray
    nfp: int

    def __post_init__(self) -> None:
        arrays = [
            np.ascontiguousarray(value, dtype=np.float64)
            for value in (self.coeffs_x, self.coeffs_y, self.coeffs_z)
        ]
        currents = np.ascontiguousarray(self.currents_a, dtype=np.float64).reshape(-1)
        if any(value.ndim != 2 for value in arrays):
            raise ValueError("coefficient arrays must be two-dimensional")
        if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
            raise ValueError("coeffs_x, coeffs_y, and coeffs_z must have the same shape")
        if arrays[0].shape[0] == 0 or arrays[0].shape[1] == 0:
            raise ValueError("at least one coil and one coefficient are required")
        if currents.shape != (arrays[0].shape[0],):
            raise ValueError("currents_a must contain one current per base coil")
        if int(self.nfp) <= 0:
            raise ValueError("nfp must be positive")
        if not all(np.all(np.isfinite(value)) for value in (*arrays, currents)):
            raise ValueError("coil coefficients and currents must be finite")
        for name, value in zip(("coeffs_x", "coeffs_y", "coeffs_z"), arrays, strict=True):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        currents.setflags(write=False)
        object.__setattr__(self, "currents_a", currents)
        object.__setattr__(self, "nfp", int(self.nfp))

    @property
    def n_base_coils(self) -> int:
        return int(self.coeffs_x.shape[0])

    @property
    def n_coefficients(self) -> int:
        return int(self.coeffs_x.shape[1])


@dataclass(frozen=True)
class AxisContinuationState:
    """Magnetic-axis seed tied to a particular coil layout and field period."""

    R: float
    Z: float
    nfp: int
    n_base_coils: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.R) or not math.isfinite(self.Z):
            raise ValueError("axis continuation coordinates must be finite")
        if self.nfp <= 0 or self.n_base_coils <= 0:
            raise ValueError("axis continuation dimensions must be positive")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class EvaluationResult:
    """Structured output from a formal evaluation or an explicit proxy."""

    status: str
    score: float
    native_score: float
    components: Mapping[str, float]
    diagnostics: Mapping[str, Any]
    timing: Mapping[str, float]
    mode: EvaluationMode
    target_helicity: tuple[int, int]
    nfp: int
    n_base_coils: int
    n_coefficients: int
    requested_config: Mapping[str, Any] = field(default_factory=dict)
    score_policy: str = "native_default"

    @classmethod
    def from_native(
        cls,
        raw: Mapping[str, Any],
        *,
        coils: CoilSet,
        mode: EvaluationMode,
        target_helicity: tuple[int, int],
        requested_config: Mapping[str, Any],
        policy: ScorePolicy,
    ) -> "EvaluationResult":
        components = {
            str(name): float(value)
            for name, value in dict(raw.get("components") or {}).items()
        }
        timing = {
            str(name): float(value)
            for name, value in dict(raw.get("timing") or {}).items()
        }
        diagnostics = dict(raw.get("diagnostics") or {})
        policy_score = float(policy(raw))
        native_score = float(raw.get("score", math.nan))
        if not math.isfinite(policy_score):
            raise ValueError("score policy returned a non-finite value")
        return cls(
            status=str(raw.get("status", "internal_error")),
            score=policy_score,
            native_score=native_score,
            components=components,
            diagnostics=diagnostics,
            timing=timing,
            mode=mode,
            target_helicity=(int(target_helicity[0]), int(target_helicity[1])),
            nfp=coils.nfp,
            n_base_coils=coils.n_base_coils,
            n_coefficients=coils.n_coefficients,
            requested_config=dict(requested_config),
            score_policy=str(getattr(policy, "name", getattr(policy, "__name__", type(policy).__name__))),
        )

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def continuation_state(self, *, require_ok: bool = True) -> AxisContinuationState:
        if require_ok and not self.ok:
            raise ValueError(f"cannot continue from result with status {self.status!r}")
        try:
            R = float(self.diagnostics["axis_R"])
            Z = float(self.diagnostics["axis_Z"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("result does not contain a finite magnetic axis") from exc
        return AxisContinuationState(R, Z, self.nfp, self.n_base_coils)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(
            {
                "schema": RESULT_SCHEMA,
                "status": self.status,
                "score": self.score,
                "native_score": self.native_score,
                "score_policy": self.score_policy,
                "mode": self.mode.value,
                "target_helicity": self.target_helicity,
                "input": {
                    "nfp": self.nfp,
                    "n_base_coils": self.n_base_coils,
                    "n_coefficients": self.n_coefficients,
                    "current_unit": "A",
                },
                "requested_config": self.requested_config,
                "components": self.components,
                "diagnostics": self.diagnostics,
                "timing": self.timing,
            }
        )

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


NativeBackend = Callable[..., Mapping[str, Any]]


def _load_default_backend() -> NativeBackend:
    from gpu_backend.python.stellarator_gpu import score_coils_native

    return score_coils_native


class Evaluator:
    """Stable Python entry point for the formal C++/CUDA evaluator."""

    def __init__(
        self,
        library: str | Path,
        *,
        device_id: int = 0,
        score_policy: ScorePolicy | None = None,
        config_overrides: Mapping[str, Any] | None = None,
        use_production_defaults: bool = True,
        backend: NativeBackend | None = None,
    ) -> None:
        self.library = Path(library)
        self.device_id = int(device_id)
        self.score_policy = score_policy or NativeScorePolicy()
        defaults = PRODUCTION_SCORE_CONFIG if use_production_defaults else {}
        self.config_overrides = {**defaults, **dict(config_overrides or {})}
        self.use_production_defaults = bool(use_production_defaults)
        self._backend = backend

    def evaluate(
        self,
        coils: CoilSet,
        *,
        mode: EvaluationMode | str = EvaluationMode.INDEPENDENT,
        continuation: AxisContinuationState | None = None,
        target_helicity: tuple[int, int] | None = None,
        config_overrides: Mapping[str, Any] | None = None,
        score_policy: ScorePolicy | None = None,
    ) -> EvaluationResult:
        selected_mode = EvaluationMode(mode)
        if selected_mode is EvaluationMode.NEIGHBORHOOD_PROXY:
            raise ValueError("neighborhood_proxy is not a formal Evaluator mode")
        target = target_helicity or (1, coils.nfp)
        if len(target) != 2:
            raise ValueError("target_helicity must contain (M, N)")
        overrides = {**self.config_overrides, **dict(config_overrides or {})}
        protected = _AXIS_CONFIG_FIELDS.intersection(overrides)
        if protected:
            raise ValueError(
                "axis continuation fields are controlled by EvaluationMode: "
                f"{sorted(protected)}"
            )
        if selected_mode is EvaluationMode.INDEPENDENT:
            if continuation is not None:
                raise ValueError("independent evaluation cannot receive an axis continuation state")
        else:
            if continuation is None:
                raise ValueError("strict continuation requires an axis continuation state")
            if continuation.nfp != coils.nfp:
                raise ValueError("continuation state nfp does not match the coil set")
            if continuation.n_base_coils != coils.n_base_coils:
                raise ValueError("continuation state coil count does not match the coil set")
            overrides.update(
                {
                    "axis_hint_enabled": 1,
                    "axis_hint_require_continuation": 2,
                    "axis_hint_R": continuation.R,
                    "axis_hint_Z": continuation.Z,
                }
            )
        backend = self._backend or _load_default_backend()
        if self._backend is None and not self.library.is_file():
            raise FileNotFoundError(f"native score library not found: {self.library}")
        raw = backend(
            self.library,
            coils.coeffs_x,
            coils.coeffs_y,
            coils.coeffs_z,
            coils.currents_a,
            coils.nfp,
            device_id=self.device_id,
            target_helicity=(int(target[0]), int(target[1])),
            config_overrides=overrides,
        )
        return EvaluationResult.from_native(
            raw,
            coils=coils,
            mode=selected_mode,
            target_helicity=(int(target[0]), int(target[1])),
            requested_config=overrides,
            policy=score_policy or self.score_policy,
        )

    def evaluate_many(
        self,
        coil_sets: Sequence[CoilSet],
        *,
        mode: EvaluationMode | str = EvaluationMode.INDEPENDENT,
        continuations: Sequence[AxisContinuationState | None] | None = None,
        target_helicity: tuple[int, int] | None = None,
        config_overrides: Mapping[str, Any] | None = None,
        score_policy: ScorePolicy | None = None,
    ) -> list[EvaluationResult]:
        states = list(continuations) if continuations is not None else [None] * len(coil_sets)
        if len(states) != len(coil_sets):
            raise ValueError("continuations must contain one entry per coil set")
        return [
            self.evaluate(
                coils,
                mode=mode,
                continuation=state,
                target_helicity=target_helicity,
                config_overrides=config_overrides,
                score_policy=score_policy,
            )
            for coils, state in zip(coil_sets, states, strict=True)
        ]


__all__ = [
    "AxisContinuationState",
    "CoilSet",
    "EvaluationMode",
    "EvaluationResult",
    "Evaluator",
    "NativeScorePolicy",
    "PRODUCTION_SCORE_CONFIG",
    "RESULT_SCHEMA",
    "ScorePolicy",
    "WeightedComponentPolicy",
]
