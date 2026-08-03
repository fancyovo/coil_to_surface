from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

TWOPI = 2.0 * np.pi


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _q_down(value: Any, scale: float, power: float = 1.0, *, default: float = 0.0) -> float:
    x = _finite(value, np.nan)
    if not np.isfinite(x):
        return float(default)
    x = max(x, 0.0)
    return float(1.0 / (1.0 + (x / scale) ** power))


def _q_up(value: Any, scale: float, power: float = 1.0, *, default: float = 0.0) -> float:
    x = _finite(value, np.nan)
    if not np.isfinite(x) or x <= 0.0:
        return float(default)
    return float(1.0 / (1.0 + (scale / x) ** power))


def _q_saturating_up(value: Any, saturation: float, *, default: float = 0.0) -> float:
    x = _finite(value, np.nan)
    if not np.isfinite(x) or x <= 0.0 or saturation <= 0.0:
        return float(default)
    x = float(np.clip(x / saturation, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _clip01(value: Any) -> float:
    return float(np.clip(_finite(value, 0.0), 0.0, 1.0))


def _blend(items: list[tuple[float, float]]) -> float:
    weight = sum(w for w, _ in items if w > 0.0)
    if weight <= 0.0:
        return 0.0
    return float(sum(w * _clip01(v) for w, v in items if w > 0.0) / weight)


@dataclass
class ScoreConfig:
    axis_residual_scale: float = 1e-5
    axis_residual_power: float = 0.8
    axis_topology_margin: float = 2e-2
    psi_angle_p95_scale: float = 3e-3
    psi_angle_l2_scale: float = 1e-3
    max_psi_level_reference: float = 0.16
    drift_rel_scale: float = 0.30
    boozer_newton_residual_scale: float = 1e-6
    boozer_initial_residual_scale: float = 30.0
    qs_error_scale: float = 1e-3
    volume_scale: float = 0.003
    iota_target_scale: float = 0.35
    coil_length_scale: float = 7.0
    coil_curvature_p95_scale: float = 10.0
    coil_curvature_max_scale: float = 35.0
    coil_min_distance_scale: float = 0.08
    coil_axis_distance_scale: float = 0.20
    coil_high_mode_scale: float = 0.05
    current_scale_a: float = 2.0e6
    missing_coil_score: float = 0.86
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "axis": 18.0,
            "psi": 18.0,
            "surface": 18.0,
            "boozer": 14.0,
            "physics": 20.0,
            "coil": 12.0,
        }
    )


def _curve_values_and_derivatives(coeff: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coeff = np.asarray(coeff, dtype=float)
    order = (coeff.size - 1) // 2
    val = np.full_like(t, coeff[0], dtype=float)
    d1 = np.zeros_like(t, dtype=float)
    d2 = np.zeros_like(t, dtype=float)
    for m in range(1, order + 1):
        arg = TWOPI * m * t
        s = np.sin(arg)
        c = np.cos(arg)
        s_coeff = coeff[2 * m - 1]
        c_coeff = coeff[2 * m]
        omega = TWOPI * m
        val += s_coeff * s + c_coeff * c
        d1 += omega * (s_coeff * c - c_coeff * s)
        d2 += -(omega * omega) * (s_coeff * s + c_coeff * c)
    return val, d1, d2


def _normalize_currents(currents: np.ndarray, current_unit: str) -> np.ndarray:
    unit = current_unit.lower()
    if unit in {"ma", "megaamp", "megaamps"}:
        return np.asarray(currents, dtype=float) * 1e6
    if unit in {"a", "amp", "amps"}:
        return np.asarray(currents, dtype=float)
    raise ValueError(f"unknown current_unit={current_unit!r}; use 'MA' or 'A'")


def _base_curve_samples(field_input: Any, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, int(samples), endpoint=False)
    curves = []
    d1s = []
    d2s = []
    for cx, cy, cz in zip(field_input.coeffs_x, field_input.coeffs_y, field_input.coeffs_z):
        x, x1, x2 = _curve_values_and_derivatives(cx, t)
        y, y1, y2 = _curve_values_and_derivatives(cy, t)
        z, z1, z2 = _curve_values_and_derivatives(cz, t)
        curves.append(np.column_stack([x, y, z]))
        d1s.append(np.column_stack([x1, y1, z1]))
        d2s.append(np.column_stack([x2, y2, z2]))
    return np.asarray(curves), np.asarray(d1s), np.asarray(d2s)


def _apply_symmetry(points: np.ndarray, nfp: int) -> tuple[np.ndarray, np.ndarray]:
    all_points = []
    labels = []
    label = 0
    for base in points:
        for stellarator_symmetry in (False, True):
            p = base.copy()
            if stellarator_symmetry:
                p[:, 1] *= -1.0
                p[:, 2] *= -1.0
            for k in range(int(nfp)):
                angle = TWOPI * k / int(nfp)
                ca = np.cos(angle)
                sa = np.sin(angle)
                q = p.copy()
                x = ca * p[:, 0] - sa * p[:, 1]
                y = sa * p[:, 0] + ca * p[:, 1]
                q[:, 0] = x
                q[:, 1] = y
                all_points.append(q)
                labels.extend([label] * len(q))
                label += 1
    return np.vstack(all_points), np.asarray(labels, dtype=int)


def coil_geometry_metrics(field_input: Any, *, current_unit: str = "MA", samples: int = 160) -> dict[str, float]:
    curves, d1s, d2s = _base_curve_samples(field_input, samples)
    speed = np.linalg.norm(d1s, axis=2)
    speed_periodic = np.concatenate([speed, speed[:, :1]], axis=1)
    if hasattr(np, "trapezoid"):
        length = np.trapezoid(speed_periodic, dx=1.0 / samples, axis=1)
    else:
        length = np.trapz(speed_periodic, dx=1.0 / samples, axis=1)
    cross = np.cross(d1s, d2s)
    curvature = np.linalg.norm(cross, axis=2) / np.maximum(speed**3, 1e-30)
    full_points, labels = _apply_symmetry(curves, field_input.nfp)
    tree = cKDTree(full_points)
    k = min(24, len(full_points))
    dists, idxs = tree.query(full_points, k=k)
    min_intercoil = np.inf
    for i in range(len(full_points)):
        other = labels[idxs[i]] != labels[i]
        if np.any(other):
            min_intercoil = min(min_intercoil, float(np.min(dists[i][other])))
    axis_distance = np.sqrt(full_points[:, 0] ** 2 + full_points[:, 1] ** 2)
    currents = np.abs(_normalize_currents(np.asarray(field_input.currents, dtype=float), current_unit))
    coeff_energy = 0.0
    high_energy = 0.0
    order = field_input.order
    high_start = max(1, int(np.floor(0.6 * order)))
    for block in (field_input.coeffs_x, field_input.coeffs_y, field_input.coeffs_z):
        coeff = np.asarray(block, dtype=float)
        for m in range(1, order + 1):
            e = float(np.sum(coeff[:, [2 * m - 1, 2 * m]] ** 2))
            coeff_energy += e
            if m >= high_start:
                high_energy += e
    return {
        "coil_length_mean": float(np.mean(length)),
        "coil_length_max": float(np.max(length)),
        "coil_curvature_p95": float(np.percentile(curvature, 95)),
        "coil_curvature_max": float(np.max(curvature)),
        "coil_min_intercoil_distance": float(min_intercoil) if np.isfinite(min_intercoil) else float("nan"),
        "coil_min_axis_distance": float(np.min(axis_distance)),
        "coil_current_abs_mean_a": float(np.mean(currents)) if currents.size else float("nan"),
        "coil_current_abs_max_a": float(np.max(currents)) if currents.size else float("nan"),
        "coil_current_cv": float(np.std(currents) / np.mean(currents)) if currents.size and np.mean(currents) > 0.0 else 0.0,
        "coil_high_mode_energy_fraction": float(high_energy / coeff_energy) if coeff_energy > 0.0 else 0.0,
    }


def _axis_score(result: dict, cfg: ScoreConfig) -> tuple[float, dict[str, float]]:
    axis = result.get("axis") or {}
    residual = axis.get("best_residual")
    residual_score = _q_down(residual, cfg.axis_residual_scale, cfg.axis_residual_power)
    topology = str(axis.get("topology_class") or "")
    if topology == "elliptic":
        stability_margin = _finite(axis.get("topology_stability_margin"), np.nan)
        if not np.isfinite(stability_margin):
            trace = _finite(axis.get("topology_trace"), np.nan)
            det = _finite(axis.get("topology_det"), np.nan)
            stability_margin = (
                2.0 - abs(trace) / np.sqrt(det)
                if np.isfinite(trace) and np.isfinite(det) and det > 0.0
                else np.nan
            )
        topology_score = _q_saturating_up(
            stability_margin, cfg.axis_topology_margin, default=1.0
        )
    elif topology in {"parabolic", "degenerate"}:
        topology_score = 0.45
    elif topology == "hyperbolic":
        topology_score = 0.10
    elif axis.get("has_axis"):
        topology_score = 0.80
    else:
        topology_score = 0.45
    aspect = axis.get("topology_ellipse_aspect")
    aspect_penalty = _q_down(max(_finite(aspect, 1.0) - 1.0, 0.0), 1.0, 1.2, default=0.8)
    score = _blend([(0.70, residual_score), (0.20, topology_score), (0.10, aspect_penalty)])
    return score, {
        "axis_residual_score": residual_score,
        "axis_topology_score": topology_score,
        "axis_aspect_score": aspect_penalty,
    }


def _psi_score(result: dict, cfg: ScoreConfig) -> tuple[float, dict[str, float]]:
    fit = ((result.get("psi") or {}).get("fit_info") or {})
    angle_p95 = fit.get("validation_angle_p95")
    angle_l2 = fit.get("validation_angle_l2")
    p95_score = _q_down(angle_p95, cfg.psi_angle_p95_scale, 1.1)
    l2_score = _q_down(angle_l2, cfg.psi_angle_l2_scale, 1.1)
    train_score = _q_down(fit.get("train_rms"), 5e-3, 1.0, default=0.45)
    score = _blend([(0.58, p95_score), (0.32, l2_score), (0.10, train_score)])
    return score, {
        "psi_angle_p95_score": p95_score,
        "psi_angle_l2_score": l2_score,
        "psi_train_score": train_score,
    }


def _surface_screen_stats(result: dict) -> dict[str, float]:
    levels = ((result.get("surface_screen") or {}).get("levels") or [])
    ok = [x for x in levels if x.get("ok")]
    best_level = max((_finite(x.get("psi_level"), np.nan) for x in ok), default=np.nan)
    rels = [_finite(x.get("rel_end_distance_p95"), np.nan) for x in levels]
    rels = [x for x in rels if np.isfinite(x)]
    dists = [_finite(x.get("end_distance_p95"), np.nan) for x in levels]
    dists = [x for x in dists if np.isfinite(x)]
    return {
        "screen_ok_count": float(len(ok)),
        "screen_level_count": float(len(levels)),
        "screen_best_psi_level": float(best_level),
        "screen_min_rel_distance_p95": float(min(rels)) if rels else float("nan"),
        "screen_min_distance_p95": float(min(dists)) if dists else float("nan"),
    }


def _surface_score(result: dict, cfg: ScoreConfig) -> tuple[float, dict[str, float]]:
    stats = _surface_screen_stats(result)
    level_score = _clip01(stats["screen_best_psi_level"] / cfg.max_psi_level_reference)
    level_score = float(level_score**0.45)
    drift_score = _q_down(stats["screen_min_rel_distance_p95"], cfg.drift_rel_scale, 1.0, default=0.25)
    count = stats["screen_ok_count"]
    count_score = _q_up(count, 2.0, 1.0, default=0.0)
    score = _blend([(0.52, level_score), (0.33, drift_score), (0.15, count_score)])
    return score, {
        "surface_level_score": level_score,
        "surface_drift_score": drift_score,
        "surface_count_score": count_score,
        **stats,
    }


def _best_candidate(result: dict) -> dict:
    best = result.get("best_surface")
    if best:
        return best
    candidates = result.get("surface_candidates") or []
    if candidates:
        return candidates[0]
    return {}


def _boozer_score(result: dict, cfg: ScoreConfig) -> tuple[float, dict[str, float]]:
    cand = _best_candidate(result)
    if not cand:
        return 0.20, {
            "boozer_initial_score": 0.0,
            "boozer_ls_score": 0.0,
            "boozer_newton_score": 0.0,
            "boozer_success_bonus": 0.0,
        }
    initial = _q_down(cand.get("initial_boozer_residual_norm"), cfg.boozer_initial_residual_scale, 1.0, default=0.35)
    ls = _q_down(cand.get("ls_residual_norm"), cfg.boozer_newton_residual_scale, 0.75, default=0.25)
    newton = _q_down(cand.get("newton_residual_norm"), cfg.boozer_newton_residual_scale, 0.75, default=0.20)
    success = 1.0 if cand.get("newton_success") else 0.35 if cand.get("ls_success") else 0.10
    score = _blend([(0.15, initial), (0.25, ls), (0.45, newton), (0.15, success)])
    return score, {
        "boozer_initial_score": initial,
        "boozer_ls_score": ls,
        "boozer_newton_score": newton,
        "boozer_success_bonus": success,
    }


def _select_qs_error(cand: dict, metadata: dict | None = None, target: str | None = None) -> tuple[float, str]:
    target_l = (target or "").upper()
    if not target_l and metadata is not None:
        helicity = metadata.get("helicity")
        if helicity is not None:
            target_l = "QA" if int(helicity) == 0 else "QH"
    keys = {
        "QA": "qs_error_QA_1_0",
        "QH": "qs_error_QH_1_1",
        "QP": "qs_error_QP_0_1",
    }
    if target_l in keys:
        return _finite(cand.get(keys[target_l]), np.nan), target_l
    vals = [(name, _finite(cand.get(key), np.nan)) for name, key in keys.items()]
    vals = [(name, val) for name, val in vals if np.isfinite(val)]
    if not vals:
        return float("nan"), "none"
    name, val = min(vals, key=lambda x: x[1])
    return val, name


def _physics_score(result: dict, cfg: ScoreConfig, metadata: dict | None, target: str | None) -> tuple[float, dict[str, float]]:
    cand = _best_candidate(result)
    qs_error, qs_target = _select_qs_error(cand, metadata, target)
    qs_score = _q_down(qs_error, cfg.qs_error_scale, 0.9, default=0.25)
    volume_score = _q_up(cand.get("volume"), cfg.volume_scale, 1.0, default=0.0)
    iota_score = 0.80
    if metadata is not None and metadata.get("mean_iota") is not None and cand.get("iota") is not None:
        iota_score = _q_down(abs(_finite(cand.get("iota")) - _finite(metadata.get("mean_iota"))), cfg.iota_target_scale, 2.0, default=0.8)
    score = _blend([(0.62, qs_score), (0.25, volume_score), (0.13, iota_score)])
    return score, {
        "physics_qs_score": qs_score,
        "physics_volume_score": volume_score,
        "physics_iota_score": iota_score,
        "selected_qs_error": float(qs_error) if np.isfinite(qs_error) else float("nan"),
        "selected_qs_target": qs_target,
    }


def _coil_score(metrics: dict[str, float] | None, cfg: ScoreConfig) -> tuple[float, dict[str, float]]:
    if not metrics:
        return cfg.missing_coil_score, {"coil_missing": 1.0}
    length = _q_down(metrics.get("coil_length_mean"), cfg.coil_length_scale, 1.4, default=0.6)
    curv_p95 = _q_down(metrics.get("coil_curvature_p95"), cfg.coil_curvature_p95_scale, 1.3, default=0.5)
    curv_max = _q_down(metrics.get("coil_curvature_max"), cfg.coil_curvature_max_scale, 1.2, default=0.5)
    spacing = _q_up(metrics.get("coil_min_intercoil_distance"), cfg.coil_min_distance_scale, 1.1, default=0.45)
    axis_distance = _q_up(metrics.get("coil_min_axis_distance"), cfg.coil_axis_distance_scale, 1.2, default=0.45)
    high_mode = _q_down(metrics.get("coil_high_mode_energy_fraction"), cfg.coil_high_mode_scale, 1.0, default=0.7)
    current = _q_down(metrics.get("coil_current_abs_max_a"), cfg.current_scale_a, 1.0, default=0.7)
    score = _blend(
        [
            (0.16, length),
            (0.20, curv_p95),
            (0.12, curv_max),
            (0.20, spacing),
            (0.12, axis_distance),
            (0.13, high_mode),
            (0.07, current),
        ]
    )
    return score, {
        "coil_length_score": length,
        "coil_curvature_p95_score": curv_p95,
        "coil_curvature_max_score": curv_max,
        "coil_spacing_score": spacing,
        "coil_axis_distance_score": axis_distance,
        "coil_high_mode_score": high_mode,
        "coil_current_score": current,
        **metrics,
    }


def evaluate_quality_score(
    result: dict,
    *,
    field_input: Any | None = None,
    current_unit: str = "MA",
    metadata: dict | None = None,
    target: str | None = None,
    config: ScoreConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ScoreConfig()
    axis_score, axis_parts = _axis_score(result, cfg)
    psi_score, psi_parts = _psi_score(result, cfg)
    surface_score, surface_parts = _surface_score(result, cfg)
    boozer_score, boozer_parts = _boozer_score(result, cfg)
    physics_score, physics_parts = _physics_score(result, cfg, metadata, target)
    coil_metrics = coil_geometry_metrics(field_input, current_unit=current_unit) if field_input is not None else None
    coil_score, coil_parts = _coil_score(coil_metrics, cfg)
    components = {
        "axis": axis_score,
        "psi": psi_score,
        "surface": surface_score,
        "boozer": boozer_score,
        "physics": physics_score,
        "coil": coil_score,
    }
    total_weight = sum(float(v) for v in cfg.weights.values())
    score = sum(float(cfg.weights[k]) * components[k] for k in components) / total_weight
    details = {}
    details.update(axis_parts)
    details.update(psi_parts)
    details.update(surface_parts)
    details.update(boozer_parts)
    details.update(physics_parts)
    details.update(coil_parts)
    status = "surface" if result.get("best_surface") is not None else "no_axis" if not (result.get("axis") or {}).get("has_axis") else "no_surface"
    return {
        "score": float(100.0 * _clip01(score)),
        "score_unit_interval": float(_clip01(score)),
        "status": status,
        "components": {k: float(100.0 * _clip01(v)) for k, v in components.items()},
        "raw_components": components,
        "details": details,
        "config": asdict(cfg),
    }
