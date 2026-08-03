from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .score import (
    ScoreConfig,
    _axis_score,
    _blend,
    _clip01,
    _coil_score,
    _finite,
    _psi_score,
    _q_down,
    _q_up,
    coil_geometry_metrics,
)


@dataclass
class VolumeScoreConfig:
    """Scales and weights for the stable coil-to-volume-QS score."""

    base: ScoreConfig = field(default_factory=ScoreConfig)
    surface_inverse_aspect_saturation: float = 0.03
    surface_drift_scale: float = 0.02
    flux_section_std_scale: float = 0.01
    flux_boundary_residual_scale: float = 1e-9
    alpha_normal_B_scale: float = 1e-4
    alpha_relative_l2_scale: float = 0.25
    qs_global_scale: float = 0.05
    qs_edge_scale: float = 0.07
    qh_iota_threshold: float = 1.0
    qh_iota_power: float = 2.0
    volume_qs_size_floor: float = 0.65
    volume_qs_iota_floor: float = 0.50
    qh_total_iota_floor: float = 0.10
    qh_total_helicity_floor: float = 0.10
    qh_helicity_bad: float = 0.10
    qh_helicity_good: float = 0.30
    qh_helicity_exploration_fraction: float = 0.20
    missing_coordinate_score: float = 0.08
    missing_volume_qs_score: float = 0.04
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "axis": 10.0,
            "psi": 10.0,
            "surface": 10.0,
            "coordinate": 10.0,
            "volume_qs": 42.0,
            "iota": 10.0,
            "coil": 8.0,
        }
    )


def _q_saturating_up(value: Any, saturation: float, default: float = 0.0) -> float:
    value = _finite(value, np.nan)
    if not np.isfinite(value) or value <= 0.0 or saturation <= 0.0:
        return default
    x = _clip01(value / saturation)
    return float(x * x * (3.0 - 2.0 * x))


def _minimum_absolute_iota(iota_min: Any, iota_max: Any) -> float:
    low = _finite(iota_min, np.nan)
    high = _finite(iota_max, np.nan)
    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0
    if low <= 0.0 <= high:
        return 0.0
    return float(min(abs(low), abs(high)))


def _surface_stats(result: dict) -> dict[str, float]:
    screen = result.get("surface_screen") or {}
    levels = screen.get("levels") or []
    accepted = [float(x) for x in screen.get("volume_qs_accepted_levels") or []]
    stable = [float(x) for x in screen.get("stable_accepted_levels") or []]
    drifts = [
        _finite(item.get("rel_end_distance_p95"), np.nan)
        for item in levels
        if np.isfinite(_finite(item.get("rel_end_distance_p95"), np.nan))
    ]
    best_item = None
    if accepted:
        best_level = max(accepted)
        best_item = min(
            levels,
            key=lambda item: abs(_finite(item.get("psi_level"), np.inf) - best_level),
            default=None,
        )
    axis_R = abs(_finite((result.get("axis") or {}).get("best_R"), np.nan))
    fallback_radius = _finite((best_item or {}).get("radius_mean"), np.nan)
    flux = _flux_diagnostics(result)
    effective_radius = _finite(
        flux.get("boundary_effective_minor_radius_edge"), fallback_radius
    )
    major_radius = _finite(flux.get("boundary_axis_major_radius_mean"), axis_R)
    inverse_aspect = effective_radius / major_radius if major_radius > 0.0 else float("nan")
    return {
        "screen_level_count": float(len(levels)),
        "screen_stable_count": float(len(stable)),
        "screen_strict_count": float(len(accepted)),
        "screen_best_strict_level": float(max(accepted)) if accepted else float("nan"),
        "screen_min_rel_distance_p95": float(min(drifts)) if drifts else float("nan"),
        "surface_effective_minor_radius": effective_radius,
        "surface_major_radius": major_radius,
        "surface_inverse_aspect_ratio": inverse_aspect,
        "surface_volume": _finite(flux.get("boundary_volume_edge"), np.nan),
    }


def _surface_score(result: dict, cfg: VolumeScoreConfig) -> tuple[float, dict[str, float]]:
    stats = _surface_stats(result)
    size = _q_saturating_up(
        stats["surface_inverse_aspect_ratio"],
        cfg.surface_inverse_aspect_saturation,
    )
    drift = _q_down(
        stats["screen_min_rel_distance_p95"], cfg.surface_drift_scale, 1.0, default=0.15
    )
    count = _q_up(stats["screen_strict_count"], 2.0, 1.0)
    return _blend([(0.65, size), (0.25, drift), (0.10, count)]), {
        "surface_size_score": size,
        "surface_drift_score": drift,
        "surface_count_score": count,
        **stats,
    }


def _flux_diagnostics(result: dict) -> dict:
    volume = result.get("volume_qs") or {}
    diagnostics = ((volume.get("flux") or {}).get("diagnostics") or {})
    if diagnostics:
        return diagnostics
    attempts = volume.get("flux_attempts") or []
    if not attempts:
        return {}

    def rank(item: dict) -> tuple[float, float]:
        section = _finite(item.get("section_relative_std_edge"), np.inf)
        boundary = _finite(item.get("boundary_residual_max"), np.inf)
        return section, boundary

    return min(attempts, key=rank)


def _coordinate_score(result: dict, cfg: VolumeScoreConfig) -> tuple[float, dict[str, Any]]:
    volume = result.get("volume_qs") or {}
    flux = _flux_diagnostics(result)
    alpha = ((volume.get("alpha") or {}).get("diagnostics") or {})
    if not flux and not alpha:
        return cfg.missing_coordinate_score, {
            "coordinate_flux_score": 0.0,
            "coordinate_normal_B_score": 0.0,
            "coordinate_alpha_score": 0.0,
            "coordinate_consistency_score": 0.0,
        }

    section_score = _q_down(
        flux.get("section_relative_std_edge"), cfg.flux_section_std_scale, 1.0
    )
    boundary_score = _q_down(
        flux.get("boundary_residual_max"), cfg.flux_boundary_residual_scale, 1.0
    )
    flux_score = _blend([(0.75, section_score), (0.25, boundary_score)])
    normal_score = _q_down(
        alpha.get("normal_B_relative_l2"), cfg.alpha_normal_B_scale, 1.0
    )
    alpha_score = _q_down(
        alpha.get("relative_l2"), cfg.alpha_relative_l2_scale, 1.0
    )
    iota_min = _finite(alpha.get("iota_min"), np.nan)
    iota_max = _finite(alpha.get("iota_max"), np.nan)
    monotone = bool(flux.get("monotone", False))
    consistent = monotone and np.isfinite(iota_min) and np.isfinite(iota_max)
    consistency_score = 1.0 if consistent else 0.0
    score = _blend(
        [
            (0.35, flux_score),
            (0.35, normal_score),
            (0.20, alpha_score),
            (0.10, consistency_score),
        ]
    )
    return score, {
        "coordinate_flux_score": flux_score,
        "coordinate_flux_section_score": section_score,
        "coordinate_flux_boundary_score": boundary_score,
        "coordinate_normal_B_score": normal_score,
        "coordinate_alpha_score": alpha_score,
        "coordinate_consistency_score": consistency_score,
        "flux_section_relative_std_edge": flux.get("section_relative_std_edge"),
        "flux_boundary_residual_max": flux.get("boundary_residual_max"),
        "flux_monotone": monotone,
        "alpha_relative_l2": alpha.get("relative_l2"),
        "alpha_normal_B_relative_l2": alpha.get("normal_B_relative_l2"),
        "alpha_iota_min": alpha.get("iota_min"),
        "alpha_iota_max": alpha.get("iota_max"),
    }


def _edge_qs_error(metric: dict) -> float:
    bins = metric.get("radial_bins") or []
    for item in reversed(bins):
        value = _finite(item.get("f_C_over_B3_rms"), np.nan)
        if np.isfinite(value):
            return value
    return float("nan")


def _iota_score(result: dict, cfg: VolumeScoreConfig) -> tuple[float, dict[str, Any]]:
    volume = result.get("volume_qs") or {}
    target = volume.get("target_helicity") or [1, 0]
    alpha = ((volume.get("alpha") or {}).get("diagnostics") or {})
    minimum = _minimum_absolute_iota(alpha.get("iota_min"), alpha.get("iota_max"))
    is_qh = len(target) >= 2 and int(target[0]) != 0 and int(target[1]) != 0
    score = (
        _clip01(minimum / cfg.qh_iota_threshold) ** cfg.qh_iota_power
        if is_qh
        else 1.0
    )
    return float(score), {
        "iota_score": float(score),
        "iota_minimum_absolute": minimum,
        "iota_qh_gate_enabled": is_qh,
    }


def _qh_helicity_advantage(result: dict) -> tuple[float, dict[str, Any]]:
    volume = result.get("volume_qs") or {}
    target = volume.get("target_helicity") or [1, 0]
    is_qh = len(target) >= 2 and int(target[0]) != 0 and int(target[1]) != 0
    if not is_qh:
        return 1.0, {
            "qh_helicity_advantage": 1.0,
            "qh_competitor_error": float("nan"),
            "qh_target_error_per_helicity": float("nan"),
        }

    metrics = volume.get("metrics") or {}
    target_error = _finite((metrics.get("target") or {}).get("f_C_over_B3_rms"), np.nan)
    qa_error = _finite((metrics.get("QA") or {}).get("f_C_over_B3_rms"), np.nan)
    qp_error = _finite((metrics.get("QP") or {}).get("f_C_over_B3_rms"), np.nan)
    target_norm = max(float(np.hypot(int(target[0]), int(target[1]))), 1.0)
    qp_norm = max(float(abs(int(target[1]))), 1.0)
    target_per_helicity = target_error / target_norm
    competitor_error = min(qa_error, qp_error / qp_norm)
    if not np.isfinite(target_per_helicity) or not np.isfinite(competitor_error):
        advantage = 1.0
    else:
        advantage = competitor_error / max(target_per_helicity + competitor_error, 1e-300)
    return _clip01(advantage), {
        "qh_helicity_advantage": float(_clip01(advantage)),
        "qh_competitor_error": float(competitor_error),
        "qh_target_error_per_helicity": float(target_per_helicity),
        "qh_qa_error": float(qa_error),
        "qh_qp_error_per_helicity": float(qp_error / qp_norm),
    }


def _qh_helicity_quality(advantage: float, cfg: VolumeScoreConfig) -> float:
    linear = _clip01(advantage / cfg.qh_helicity_good)
    window = _q_saturating_up(
        advantage - cfg.qh_helicity_bad,
        cfg.qh_helicity_good - cfg.qh_helicity_bad,
    )
    fraction = _clip01(cfg.qh_helicity_exploration_fraction)
    return float(fraction * linear + (1.0 - fraction) * window)


def _volume_qs_score(
    result: dict, cfg: VolumeScoreConfig, iota_score: float
) -> tuple[float, dict[str, Any]]:
    volume = result.get("volume_qs") or {}
    target = ((volume.get("metrics") or {}).get("target") or {})
    global_error = _finite(target.get("f_C_over_B3_rms"), np.nan)
    edge_error = _edge_qs_error(target)
    if not np.isfinite(global_error):
        return cfg.missing_volume_qs_score, {
            "volume_qs_global_score": 0.0,
            "volume_qs_edge_score": 0.0,
            "volume_qs_global_error": float("nan"),
            "volume_qs_edge_error": float("nan"),
        }
    global_score = _q_down(global_error, cfg.qs_global_scale, 0.9)
    edge_score = _q_down(edge_error, cfg.qs_edge_scale, 0.9, default=global_score)
    residual_score = _blend([(0.80, global_score), (0.20, edge_score)])
    size_stats = _surface_stats(result)
    size_score = _q_saturating_up(
        size_stats["surface_inverse_aspect_ratio"],
        cfg.surface_inverse_aspect_saturation,
    )
    size_factor = cfg.volume_qs_size_floor + (1.0 - cfg.volume_qs_size_floor) * size_score
    target = volume.get("target_helicity") or [1, 0]
    is_qh = len(target) >= 2 and int(target[0]) != 0 and int(target[1]) != 0
    iota_factor = (
        cfg.volume_qs_iota_floor + (1.0 - cfg.volume_qs_iota_floor) * iota_score
        if is_qh
        else 1.0
    )
    useful_score = residual_score * size_factor * iota_factor
    return useful_score, {
        "volume_qs_global_score": global_score,
        "volume_qs_edge_score": edge_score,
        "volume_qs_residual_score": residual_score,
        "volume_qs_size_score": size_score,
        "volume_qs_size_factor": size_factor,
        "volume_qs_iota_factor": iota_factor,
        "volume_qs_global_error": global_error,
        "volume_qs_edge_error": edge_error,
    }


def _status(result: dict) -> str:
    if not (result.get("axis") or {}).get("has_axis"):
        return "no_axis"
    screen = result.get("surface_screen") or {}
    if not screen.get("stable_accepted_levels"):
        return "no_surface"
    if not screen.get("volume_qs_accepted_levels"):
        return "drift_rejected"
    volume = result.get("volume_qs") or {}
    if volume.get("status") == "ok":
        return "volume_qs"
    if volume.get("flux_attempts"):
        return "flux_rejected"
    return "error"


def evaluate_volume_quality_score(
    result: dict,
    *,
    field_input: Any | None = None,
    current_unit: str = "MA",
    config: VolumeScoreConfig | None = None,
) -> dict[str, Any]:
    cfg = config or VolumeScoreConfig()
    axis_score, axis_parts = _axis_score(result, cfg.base)
    psi_score, psi_parts = _psi_score(result, cfg.base)
    surface_score, surface_parts = _surface_score(result, cfg)
    coordinate_score, coordinate_parts = _coordinate_score(result, cfg)
    iota_score, iota_parts = _iota_score(result, cfg)
    helicity_advantage, helicity_parts = _qh_helicity_advantage(result)
    qs_score, qs_parts = _volume_qs_score(result, cfg, iota_score)
    coil_metrics = (
        coil_geometry_metrics(field_input, current_unit=current_unit)
        if field_input is not None
        else None
    )
    coil_score, coil_parts = _coil_score(coil_metrics, cfg.base)
    components = {
        "axis": axis_score,
        "psi": psi_score,
        "surface": surface_score,
        "coordinate": coordinate_score,
        "volume_qs": qs_score,
        "iota": iota_score,
        "coil": coil_score,
    }
    unknown = set(cfg.weights) ^ set(components)
    if unknown:
        raise ValueError(f"score weights and components differ: {sorted(unknown)}")
    total_weight = sum(float(value) for value in cfg.weights.values())
    if total_weight <= 0.0:
        raise ValueError("score weights must have a positive sum")
    unit_score_before_gate = (
        sum(cfg.weights[name] * value for name, value in components.items()) / total_weight
    )
    qh_total_iota_factor = (
        cfg.qh_total_iota_floor + (1.0 - cfg.qh_total_iota_floor) * iota_score
        if iota_parts["iota_qh_gate_enabled"]
        else 1.0
    )
    qh_total_helicity_factor = (
        cfg.qh_total_helicity_floor
        + (1.0 - cfg.qh_total_helicity_floor)
        * _qh_helicity_quality(helicity_advantage, cfg)
        if iota_parts["iota_qh_gate_enabled"]
        else 1.0
    )
    unit_score = (
        unit_score_before_gate * qh_total_iota_factor * qh_total_helicity_factor
    )
    details: dict[str, Any] = {}
    for part in (
        axis_parts,
        psi_parts,
        surface_parts,
        coordinate_parts,
        qs_parts,
        iota_parts,
        helicity_parts,
        coil_parts,
    ):
        details.update(part)
    details["score_before_qh_iota_gate"] = float(100.0 * _clip01(unit_score_before_gate))
    details["score_qh_total_iota_factor"] = float(qh_total_iota_factor)
    details["score_qh_total_helicity_factor"] = float(qh_total_helicity_factor)
    details["score_qh_helicity_quality"] = float(
        _qh_helicity_quality(helicity_advantage, cfg)
        if iota_parts["iota_qh_gate_enabled"]
        else 1.0
    )
    return {
        "score": float(100.0 * _clip01(unit_score)),
        "score_unit_interval": float(_clip01(unit_score)),
        "status": _status(result),
        "components": {name: float(100.0 * _clip01(value)) for name, value in components.items()},
        "raw_components": components,
        "details": details,
        "config": asdict(cfg),
    }
