from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import time
from pathlib import Path
from typing import Any

import numpy as np

from stellarator_eval.native_evaluator import (
    CoilSet,
    EvaluationMode,
    EvaluationResult,
    NativeScorePolicy,
    PRODUCTION_SCORE_CONFIG,
    ScorePolicy,
)
from stellarator_eval.psi import build_modes


@dataclass(frozen=True)
class NeighborhoodSettings:
    """Numerical settings for the query-major local proxy."""

    segments_per_coil: int = 256
    axis_trace_steps: int = 960
    axis_samples: int = 240
    axis_newton_iterations: int = 6
    axis_finite_difference_step: float = 2.0e-4
    axis_maximum_newton_step: float = 0.25
    axis_residual_tolerance: float = 1.0e-7
    axis_hint_max_distance: float = 0.08
    psi_degree: int = 10
    psi_toroidal_order: int = 12
    psi_radius: float = 0.05
    psi_grid: int = 48
    psi_rho_min: float = 0.002
    psi_ridge: float = 1.0e-6
    psi_iterations: int = 4
    alpha_iterations: int = 4
    formal_surface_theta_count: int = 128
    local_surface_theta_count: int = 64
    iota_degree: int = 3

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "segments_per_coil",
            "axis_trace_steps",
            "axis_samples",
            "axis_newton_iterations",
            "psi_degree",
            "psi_toroidal_order",
            "psi_grid",
            "psi_iterations",
            "alpha_iterations",
            "formal_surface_theta_count",
            "local_surface_theta_count",
        )
        if any(int(getattr(self, name)) <= 0 for name in positive_integer_fields):
            raise ValueError("neighborhood integer settings must be positive")
        if self.iota_degree < 0:
            raise ValueError("iota_degree must be nonnegative")


@dataclass(frozen=True)
class NeighborhoodBatchResult:
    """Center replay and candidate proxy results from one query-major batch."""

    center: EvaluationResult
    candidates: tuple[EvaluationResult, ...]
    timing: Mapping[str, float]
    diagnostics: Mapping[str, Any]
    assumptions: tuple[str, ...] = field(
        default=(
            "all candidates remain near the supplied center",
            "each candidate refines the supplied magnetic-axis branch without global fallback",
            "psi uses the center factorization as a preconditioned iterative starting point",
            "the coordinate component and selected surface branch are inherited locally",
            "accepted optimizer centers require a separate formal evaluation",
        )
    )

    @property
    def scores(self) -> np.ndarray:
        return np.asarray([result.score for result in self.candidates], dtype=np.float64)

    @property
    def ok_fraction(self) -> float:
        if not self.candidates:
            return 0.0
        return float(np.mean([result.ok for result in self.candidates]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "stellarator-neighborhood-proxy-v1",
            "center": self.center.to_dict(),
            "candidates": [result.to_dict() for result in self.candidates],
            "timing": dict(self.timing),
            "diagnostics": dict(self.diagnostics),
            "assumptions": list(self.assumptions),
        }


BatchFactory = Callable[..., Any]
CoilGradientBackend = Callable[..., Mapping[str, Any]]


def _load_backends() -> tuple[BatchFactory, CoilGradientBackend]:
    from gpu_backend.python.stellarator_gpu import (
        BatchCoilFieldGpu,
        coil_component_gradient_native,
    )

    return BatchCoilFieldGpu, coil_component_gradient_native


def _elapsed(started: float) -> float:
    return time.perf_counter() - started


class NeighborhoodEvaluator:
    """Experimental batched proxy for finite differences near one center.

    The output is useful for local ordering and directional derivatives. It is
    not a replacement for :class:`stellarator_eval.native_evaluator.Evaluator`.
    """

    def __init__(
        self,
        library: str | Path,
        *,
        device_id: int = 0,
        settings: NeighborhoodSettings | None = None,
        score_policy: ScorePolicy | None = None,
        batch_factory: BatchFactory | None = None,
        coil_gradient_backend: CoilGradientBackend | None = None,
    ) -> None:
        self.library = Path(library)
        self.device_id = int(device_id)
        self.settings = settings or NeighborhoodSettings()
        self.score_policy = score_policy or NativeScorePolicy()
        self._batch_factory = batch_factory
        self._coil_gradient_backend = coil_gradient_backend

    @staticmethod
    def _validate_inputs(
        center_coils: CoilSet,
        candidates: Sequence[CoilSet],
        center_result: EvaluationResult,
    ) -> None:
        if not candidates:
            raise ValueError("at least one neighborhood candidate is required")
        if center_result.mode is EvaluationMode.NEIGHBORHOOD_PROXY:
            raise ValueError("a proxy result cannot anchor another proxy batch")
        if not center_result.ok:
            raise ValueError("the neighborhood center must have a formal ok result")
        if (
            center_result.nfp != center_coils.nfp
            or center_result.n_base_coils != center_coils.n_base_coils
            or center_result.n_coefficients != center_coils.n_coefficients
        ):
            raise ValueError("center result does not describe center_coils")
        expected = (
            center_coils.nfp,
            center_coils.n_base_coils,
            center_coils.n_coefficients,
        )
        for index, candidate in enumerate(candidates):
            actual = (candidate.nfp, candidate.n_base_coils, candidate.n_coefficients)
            if actual != expected:
                raise ValueError(
                    f"candidate {index} layout {actual} does not match center {expected}"
                )

    def evaluate(
        self,
        center_coils: CoilSet,
        candidates: Sequence[CoilSet],
        center_result: EvaluationResult,
        *,
        target_helicity: tuple[int, int] | None = None,
        score_policy: ScorePolicy | None = None,
    ) -> NeighborhoodBatchResult:
        candidates = tuple(candidates)
        self._validate_inputs(center_coils, candidates, center_result)
        if self._batch_factory is None and not self.library.is_file():
            raise FileNotFoundError(f"native score library not found: {self.library}")
        default_factory, default_gradient = _load_backends()
        batch_factory = self._batch_factory or default_factory
        coil_gradient = self._coil_gradient_backend or default_gradient
        settings = self.settings
        target = target_helicity or center_result.target_helicity
        policy = score_policy or self.score_policy
        continuation = center_result.continuation_state()

        x = np.stack([candidate.coeffs_x for candidate in candidates])
        y = np.stack([candidate.coeffs_y for candidate in candidates])
        z = np.stack([candidate.coeffs_z for candidate in candidates])
        current = np.stack([candidate.currents_a for candidate in candidates])
        query_count = len(candidates)
        timing: dict[str, float] = {}
        started = time.perf_counter()
        batch = batch_factory(
            self.library,
            x,
            y,
            z,
            current,
            center_coils.nfp,
            segments_per_coil=settings.segments_per_coil,
            device_id=self.device_id,
        )
        timing["field_create_s"] = _elapsed(started)
        try:
            formal_config = {
                **PRODUCTION_SCORE_CONFIG,
                "iota_degree": settings.iota_degree,
                "surface_theta_count": settings.formal_surface_theta_count,
                "axis_hint_enabled": 1,
                "axis_hint_require_continuation": 2,
                "axis_hint_R": continuation.R,
                "axis_hint_Z": continuation.Z,
            }
            started = time.perf_counter()
            capture = batch.capture_psi_center(
                center_coils.coeffs_x,
                center_coils.coeffs_y,
                center_coils.coeffs_z,
                center_coils.currents_a,
                target_helicity=target,
                config_overrides=formal_config,
            )
            timing["center_capture_s"] = _elapsed(started)
            captured_raw = capture["score_result"]
            if captured_raw.get("status") != "ok":
                raise RuntimeError(
                    f"formal center replay failed with status {captured_raw.get('status')}"
                )

            axis_R0 = np.full(query_count, continuation.R, dtype=np.float64)
            axis_Z0 = np.full(query_count, continuation.Z, dtype=np.float64)
            started = time.perf_counter()
            refined = batch.refine_axis_hint(
                axis_R0,
                axis_Z0,
                trace_steps=settings.axis_trace_steps,
                newton_iterations=settings.axis_newton_iterations,
                finite_difference_step=settings.axis_finite_difference_step,
                maximum_newton_step=settings.axis_maximum_newton_step,
                residual_tolerance=settings.axis_residual_tolerance,
                hint_max_distance=settings.axis_hint_max_distance,
            )
            timing["axis_refine_s"] = _elapsed(started)

            started = time.perf_counter()
            axes = batch.trace_axis_samples(
                refined["R"],
                refined["Z"],
                integration_steps=settings.axis_trace_steps,
                sample_count=settings.axis_samples,
            )
            timing["axis_trace_s"] = _elapsed(started)

            modes = build_modes(settings.psi_degree, settings.psi_toroidal_order)
            mode_a = np.asarray([mode.a for mode in modes], dtype=np.int32)
            mode_b = np.asarray([mode.b for mode in modes], dtype=np.int32)
            mode_m = np.asarray([mode.m for mode in modes], dtype=np.int32)
            mode_kind = np.asarray(
                [0 if mode.kind == "cos" else 1 for mode in modes], dtype=np.int32
            )
            started = time.perf_counter()
            psi, psi_rms, psi_stats = batch.fit_psi_pcgls(
                *axes,
                mode_a,
                mode_b,
                mode_m,
                mode_kind,
                capture["psi_coefficients"],
                radius_scale=settings.psi_radius,
                radial_grid=settings.psi_grid,
                vertical_grid=settings.psi_grid,
                phi_grid=settings.psi_grid,
                rho_min=settings.psi_rho_min,
                ridge=settings.psi_ridge,
                iterations=settings.psi_iterations,
            )
            timing["psi_update_s"] = _elapsed(started)

            started = time.perf_counter()
            coil_linear = coil_gradient(
                self.library,
                center_coils.coeffs_x,
                center_coils.coeffs_y,
                center_coils.coeffs_z,
                center_coils.currents_a,
                center_coils.nfp,
            )
            coil_components = np.full(query_count, float(coil_linear["component"]))
            for values, center_values, name in (
                (x, center_coils.coeffs_x, "x"),
                (y, center_coils.coeffs_y, "y"),
                (z, center_coils.coeffs_z, "z"),
                (current, center_coils.currents_a, "current"),
            ):
                delta = (values - center_values) * coil_linear["gradient"][name]
                coil_components += delta.reshape(query_count, -1).sum(axis=1)
            timing["coil_linearization_s"] = _elapsed(started)

            started = time.perf_counter()
            local_raw, local_stats = batch.score_local_batch(
                current,
                *axes,
                refined["residual"],
                refined["topology_trace"],
                refined["topology_det"],
                psi,
                psi_rms,
                coil_components,
                capture,
                surface_theta_count=settings.local_surface_theta_count,
                alpha_iterations=settings.alpha_iterations,
            )
            timing["local_physics_s"] = _elapsed(started)
        finally:
            try:
                batch.clear_psi_preconditioner()
            finally:
                batch.close()
        timing["total_s"] = sum(timing.values())

        center_replay = EvaluationResult.from_native(
            captured_raw,
            coils=center_coils,
            mode=EvaluationMode.STRICT_CONTINUATION,
            target_helicity=target,
            requested_config=formal_config,
            policy=policy,
        )
        proxy_config = {
            "settings": asdict(settings),
            "center_score": center_replay.native_score,
            "center_axis_R": continuation.R,
            "center_axis_Z": continuation.Z,
        }
        results = tuple(
            EvaluationResult.from_native(
                raw,
                coils=candidate,
                mode=EvaluationMode.NEIGHBORHOOD_PROXY,
                target_helicity=target,
                requested_config=proxy_config,
                policy=policy,
            )
            for raw, candidate in zip(local_raw, candidates, strict=True)
        )
        diagnostics = {
            "query_count": query_count,
            "ok_count": sum(result.ok for result in results),
            "axis_valid_fraction": float(np.mean(refined["valid"])),
            "center_capture_score_delta": center_replay.native_score - center_result.native_score,
            "psi": psi_stats,
            "local": local_stats,
        }
        return NeighborhoodBatchResult(center_replay, results, timing, diagnostics)


__all__ = [
    "NeighborhoodBatchResult",
    "NeighborhoodEvaluator",
    "NeighborhoodSettings",
]
