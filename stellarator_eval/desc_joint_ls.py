from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class WeightedSampleSet:
    nodes: np.ndarray
    values: np.ndarray
    weight: float | np.ndarray = 1.0
    label: str = ""


@dataclass
class PhaseConstraintSet:
    nodes: np.ndarray
    theta: np.ndarray
    beta_ids: np.ndarray
    beta_groups: np.ndarray | None = None
    weight: float | np.ndarray = 1.0
    label: str = ""


@dataclass
class JointSpectralFitConfig:
    rz_ridge: float = 1e-8
    l_ridge: float = 1e-8
    beta_ridge: float = 1e-8
    iota_ridge: float = 1e-8
    beta_gauge_weight: float = 10.0
    radial_mode_scale: float = 0.35
    poloidal_mode_scale: float = 0.20
    toroidal_mode_scale: float = 0.20
    penalty_power: float = 1.5
    zero_mode_penalty: float = 1.0
    iota_powers: tuple[int, ...] = (0, 2, 4)
    normalize_columns: bool = False


@dataclass
class JointSpectralFitResult:
    R_lmn: np.ndarray
    Z_lmn: np.ndarray
    L_lmn: np.ndarray
    beta: np.ndarray
    iota_coeffs: np.ndarray
    iota_powers: tuple[int, ...]
    matrix_shape: tuple[int, int]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _as_float_array(values: np.ndarray | list[float], *, name: str = "values") -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1D array for {name}")
    return arr


def _as_int_array(values: np.ndarray | list[int], *, name: str = "values") -> np.ndarray:
    arr = np.asarray(values, dtype=int).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1D integer array for {name}")
    return arr


def _as_nodes(nodes: np.ndarray | list[list[float]]) -> np.ndarray:
    arr = np.asarray(nodes, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("expected nodes with shape (n, 3) in (rho, theta, zeta)")
    return arr


def _expand_weights(weight: float | np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(weight, dtype=float)
    if arr.ndim == 0:
        return np.full(size, float(arr), dtype=float)
    arr = arr.reshape(-1)
    if arr.size != size:
        raise ValueError(f"weight has size {arr.size}, expected {size}")
    return arr


def _stack_weighted_samples(
    basis: Any,
    datasets: list[WeightedSampleSet],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if not datasets:
        raise ValueError("at least one dataset is required")

    blocks = []
    rhs = []
    weights = []
    meta: list[dict[str, Any]] = []
    offset = 0
    for idx, dataset in enumerate(datasets):
        nodes = _as_nodes(dataset.nodes)
        values = _as_float_array(dataset.values)
        if len(nodes) != len(values):
            raise ValueError(f"dataset {idx} has {len(nodes)} nodes but {len(values)} values")
        block = np.asarray(basis.evaluate(nodes), dtype=float)
        if block.shape[0] != len(values):
            raise ValueError(f"basis block row count {block.shape[0]} does not match values {len(values)}")
        row_weights = _expand_weights(dataset.weight, len(values))
        blocks.append(block)
        rhs.append(values)
        weights.append(row_weights)
        meta.append(
            {
                "label": dataset.label or f"dataset_{idx}",
                "row_start": offset,
                "row_stop": offset + len(values),
                "weight_mean": float(np.mean(row_weights)),
                "count": int(len(values)),
            }
        )
        offset += len(values)

    A = np.vstack(blocks)
    b = np.concatenate(rhs)
    w = np.concatenate(weights)
    if np.any(w <= 0.0):
        raise ValueError("all sample weights must be positive")
    return A, b, w, meta


def _mode_penalty_diag(basis: Any, cfg: JointSpectralFitConfig) -> np.ndarray:
    modes = np.asarray(basis.modes, dtype=float)
    if modes.ndim != 2 or modes.shape[1] != 3:
        raise ValueError("expected DESC basis modes with shape (num_modes, 3)")
    l = np.abs(modes[:, 0])
    m = np.abs(modes[:, 1])
    n = np.abs(modes[:, 2])
    raw = (
        cfg.zero_mode_penalty
        + cfg.radial_mode_scale * l
        + cfg.poloidal_mode_scale * m
        + cfg.toroidal_mode_scale * n
    )
    return np.asarray(raw**cfg.penalty_power, dtype=float)


def _solve_weighted_regularized(
    A: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
    penalty: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, dict[str, float]]:
    sqrt_w = np.sqrt(weights)
    Aw = A * sqrt_w[:, None]
    bw = b * sqrt_w

    if ridge > 0.0:
        reg = ridge * np.diag(penalty)
        A_aug = np.vstack([Aw, reg])
        b_aug = np.concatenate([bw, np.zeros(reg.shape[0], dtype=float)])
    else:
        A_aug = Aw
        b_aug = bw

    coeffs, residuals, rank, singular = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    return coeffs, {
        "aug_rows": float(A_aug.shape[0]),
        "aug_cols": float(A_aug.shape[1]),
        "rank": float(rank),
        "residual_sum_squares": float(np.sum(residuals)) if residuals.size else 0.0,
        "sigma_max": float(singular[0]) if singular.size else 0.0,
        "sigma_min": float(singular[-1]) if singular.size else 0.0,
    }


def _column_scale(mats: list[np.ndarray], ncols: int, *, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.ones(ncols, dtype=float)
    mats = [mat for mat in mats if mat.size]
    if not mats:
        return np.ones(ncols, dtype=float)
    stacked = np.vstack(mats)
    norms = np.linalg.norm(stacked, axis=0)
    return np.where(norms > 0.0, norms, 1.0)


def _scale_info(scale: np.ndarray) -> dict[str, float]:
    if scale.size == 0:
        return {"column_scale_min": 0.0, "column_scale_max": 0.0}
    return {
        "column_scale_min": float(np.min(scale)),
        "column_scale_max": float(np.max(scale)),
    }


def _iota_matrix(nodes: np.ndarray, powers: tuple[int, ...]) -> np.ndarray:
    rho = np.asarray(nodes[:, 0], dtype=float)
    if not powers:
        return np.zeros((len(rho), 0), dtype=float)
    cols = [np.power(rho, int(p), dtype=float) for p in powers]
    return np.column_stack(cols)


def _stack_phase_constraints(
    basis: Any,
    datasets: list[PhaseConstraintSet],
    *,
    iota_powers: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if not datasets:
        return (
            np.zeros((0, basis.num_modes), dtype=float),
            np.zeros((0, 0), dtype=float),
            np.zeros((0, len(iota_powers)), dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            np.zeros((0, 0), dtype=float),
            [],
            {"beta_count": 0, "beta_group_count": 0},
        )

    prepared = []
    total_rows = 0
    total_beta = 0
    group_members: dict[tuple[int, int], list[int]] = {}
    meta: list[dict[str, Any]] = []
    for dataset_idx, dataset in enumerate(datasets):
        nodes = _as_nodes(dataset.nodes)
        theta = _as_float_array(dataset.theta, name="theta")
        beta_ids = _as_int_array(dataset.beta_ids, name="beta_ids")
        if len(nodes) != len(theta) or len(nodes) != len(beta_ids):
            raise ValueError("phase dataset nodes/theta/beta_ids must have the same length")
        if np.any(beta_ids < 0):
            raise ValueError("beta_ids must be non-negative")
        row_weights = _expand_weights(dataset.weight, len(theta))
        if np.any(row_weights <= 0.0):
            raise ValueError("all phase weights must be positive")
        local_beta_count = int(beta_ids.max()) + 1 if beta_ids.size else 0
        if dataset.beta_groups is None:
            beta_groups = np.zeros(local_beta_count, dtype=int)
        else:
            beta_groups = _as_int_array(dataset.beta_groups, name="beta_groups")
            if len(beta_groups) != local_beta_count:
                raise ValueError("beta_groups length must match the number of local beta ids")
        beta_offset = total_beta
        global_beta_ids = beta_ids + beta_offset
        A_L_local = np.asarray(basis.evaluate(nodes), dtype=float)
        A_iota_local = -np.asarray(nodes[:, 2:3], dtype=float) * _iota_matrix(nodes, iota_powers)
        prepared.append(
            {
                "nodes": nodes,
                "theta": theta,
                "weights": row_weights,
                "A_L": A_L_local,
                "A_iota": A_iota_local,
                "global_beta_ids": global_beta_ids,
                "local_beta_count": local_beta_count,
            }
        )
        for local_beta, group in enumerate(beta_groups):
            key = (dataset_idx, int(group))
            group_members.setdefault(key, []).append(beta_offset + local_beta)
        meta.append(
            {
                "label": dataset.label or f"phase_dataset_{dataset_idx}",
                "row_start": total_rows,
                "row_stop": total_rows + len(theta),
                "weight_mean": float(np.mean(row_weights)),
                "count": int(len(theta)),
                "beta_count": int(local_beta_count),
                "beta_group_count": int(len(np.unique(beta_groups))) if beta_groups.size else 0,
            }
        )
        total_rows += len(theta)
        total_beta += local_beta_count

    A_L = np.zeros((total_rows, basis.num_modes), dtype=float)
    A_beta = np.zeros((total_rows, total_beta), dtype=float)
    A_iota = np.zeros((total_rows, len(iota_powers)), dtype=float)
    b = np.zeros(total_rows, dtype=float)
    w = np.zeros(total_rows, dtype=float)
    row0 = 0
    for item in prepared:
        rows = len(item["theta"])
        row1 = row0 + rows
        A_L[row0:row1] = item["A_L"]
        A_iota[row0:row1] = item["A_iota"]
        A_beta[np.arange(row0, row1), item["global_beta_ids"]] = -1.0
        b[row0:row1] = -item["theta"]
        w[row0:row1] = item["weights"]
        row0 = row1

    gauge_rows = []
    for members in group_members.values():
        row = np.zeros(total_beta, dtype=float)
        row[np.asarray(members, dtype=int)] = 1.0 / max(len(members), 1)
        gauge_rows.append(row)
    if gauge_rows:
        A_gauge_beta = np.vstack(gauge_rows)
    else:
        A_gauge_beta = np.zeros((0, total_beta), dtype=float)

    return (
        A_L,
        A_beta,
        A_iota,
        b,
        w,
        A_gauge_beta,
        meta,
        {"beta_count": int(total_beta), "beta_group_count": int(len(group_members))},
    )


def fit_joint_rzl_data(
    eq: Any,
    *,
    R_datasets: list[WeightedSampleSet],
    Z_datasets: list[WeightedSampleSet],
    L_datasets: list[WeightedSampleSet] | None = None,
    phase_datasets: list[PhaseConstraintSet] | None = None,
    config: JointSpectralFitConfig | None = None,
) -> JointSpectralFitResult:
    cfg = config or JointSpectralFitConfig()
    L_datasets = L_datasets or []
    phase_datasets = phase_datasets or []

    A_R, b_R, w_R, meta_R = _stack_weighted_samples(eq.R_basis, R_datasets)
    A_Z, b_Z, w_Z, meta_Z = _stack_weighted_samples(eq.Z_basis, Z_datasets)
    if L_datasets:
        A_L_direct, b_L_direct, w_L_direct, meta_L = _stack_weighted_samples(eq.L_basis, L_datasets)
    else:
        A_L_direct = np.zeros((0, eq.L_basis.num_modes), dtype=float)
        b_L_direct = np.zeros(0, dtype=float)
        w_L_direct = np.zeros(0, dtype=float)
        meta_L = [
            {
                "label": "no_lambda_data",
                "row_start": 0,
                "row_stop": 0,
                "weight_mean": 0.0,
                "count": 0,
            }
        ]

    (
        A_L_phase,
        A_beta_phase,
        A_iota_phase,
        b_phase,
        w_phase,
        A_beta_gauge,
        meta_phase,
        phase_info,
    ) = _stack_phase_constraints(eq.L_basis, phase_datasets, iota_powers=cfg.iota_powers)

    nR = eq.R_basis.num_modes
    nZ = eq.Z_basis.num_modes
    nL = eq.L_basis.num_modes
    nBeta = A_beta_phase.shape[1]
    nIota = len(cfg.iota_powers)

    scale_R = _column_scale([A_R], nR, enabled=cfg.normalize_columns)
    scale_Z = _column_scale([A_Z], nZ, enabled=cfg.normalize_columns)
    scale_L = _column_scale([A_L_direct, A_L_phase], nL, enabled=cfg.normalize_columns)
    scale_beta = _column_scale([A_beta_phase, A_beta_gauge], nBeta, enabled=cfg.normalize_columns)
    scale_iota = _column_scale([A_iota_phase], nIota, enabled=cfg.normalize_columns)
    scale_big = np.concatenate([scale_R, scale_Z, scale_L, scale_beta, scale_iota])

    row_counts = [
        A_R.shape[0],
        A_Z.shape[0],
        A_L_direct.shape[0],
        A_L_phase.shape[0],
        A_beta_gauge.shape[0],
    ]
    total_rows = int(sum(row_counts))
    total_cols = int(nR + nZ + nL + nBeta + nIota)
    A_big = np.zeros((total_rows, total_cols), dtype=float)
    b_big = np.zeros(total_rows, dtype=float)
    w_big = np.zeros(total_rows, dtype=float)

    cR0 = 0
    cR1 = cR0 + nR
    cZ0 = cR1
    cZ1 = cZ0 + nZ
    cL0 = cZ1
    cL1 = cL0 + nL
    cB0 = cL1
    cB1 = cB0 + nBeta
    cI0 = cB1
    cI1 = cI0 + nIota

    row0 = 0
    row1 = row0 + A_R.shape[0]
    if row1 > row0:
        A_big[row0:row1, cR0:cR1] = A_R / scale_R[None, :]
        b_big[row0:row1] = b_R
        w_big[row0:row1] = w_R
    row0 = row1

    row1 = row0 + A_Z.shape[0]
    if row1 > row0:
        A_big[row0:row1, cZ0:cZ1] = A_Z / scale_Z[None, :]
        b_big[row0:row1] = b_Z
        w_big[row0:row1] = w_Z
    row0 = row1

    row1 = row0 + A_L_direct.shape[0]
    if row1 > row0:
        A_big[row0:row1, cL0:cL1] = A_L_direct / scale_L[None, :]
        b_big[row0:row1] = b_L_direct
        w_big[row0:row1] = w_L_direct
    row0 = row1

    row1 = row0 + A_L_phase.shape[0]
    if row1 > row0:
        A_big[row0:row1, cL0:cL1] = A_L_phase / scale_L[None, :]
        if nBeta:
            A_big[row0:row1, cB0:cB1] = A_beta_phase / scale_beta[None, :]
        if nIota:
            A_big[row0:row1, cI0:cI1] = A_iota_phase / scale_iota[None, :]
        b_big[row0:row1] = b_phase
        w_big[row0:row1] = w_phase
    row0 = row1

    row1 = row0 + A_beta_gauge.shape[0]
    if row1 > row0 and nBeta:
        A_big[row0:row1, cB0:cB1] = A_beta_gauge / scale_beta[None, :]
        b_big[row0:row1] = 0.0
        w_big[row0:row1] = float(cfg.beta_gauge_weight)

    penalty_R = _mode_penalty_diag(eq.R_basis, cfg) * np.sqrt(max(float(cfg.rz_ridge), 0.0))
    penalty_Z = _mode_penalty_diag(eq.Z_basis, cfg) * np.sqrt(max(float(cfg.rz_ridge), 0.0))
    penalty_L = _mode_penalty_diag(eq.L_basis, cfg) * np.sqrt(max(float(cfg.l_ridge), 0.0))
    penalty_beta = np.full(nBeta, np.sqrt(max(float(cfg.beta_ridge), 0.0)), dtype=float)
    penalty_iota = np.asarray([1.0 + abs(int(p)) for p in cfg.iota_powers], dtype=float) * np.sqrt(
        max(float(cfg.iota_ridge), 0.0)
    )
    penalty_big = np.concatenate([penalty_R, penalty_Z, penalty_L, penalty_beta, penalty_iota])

    coeffs_scaled, solve_info = _solve_weighted_regularized(A_big, b_big, w_big, penalty_big, ridge=1.0)
    coeffs_big = coeffs_scaled / np.where(scale_big > 0.0, scale_big, 1.0)

    cR = coeffs_big[cR0:cR1]
    cZ = coeffs_big[cZ0:cZ1]
    cL = coeffs_big[cL0:cL1]
    beta = coeffs_big[cB0:cB1]
    iota_coeffs = coeffs_big[cI0:cI1]

    diag = {
        "R_datasets": meta_R,
        "Z_datasets": meta_Z,
        "L_datasets": meta_L,
        "phase_datasets": meta_phase,
        "phase_info": phase_info,
        "R_column_scale": _scale_info(scale_R),
        "Z_column_scale": _scale_info(scale_Z),
        "L_column_scale": _scale_info(scale_L),
        "beta_column_scale": _scale_info(scale_beta),
        "iota_column_scale": _scale_info(scale_iota),
        "solve": solve_info,
        "iota_powers": [int(p) for p in cfg.iota_powers],
        "iota_coeffs": [float(x) for x in iota_coeffs],
    }

    for label, A, b, w, coeffs in (
        ("R", A_R, b_R, w_R, cR),
        ("Z", A_Z, b_Z, w_Z, cZ),
        ("L", A_L_direct, b_L_direct, w_L_direct, cL),
    ):
        if A.shape[0] == 0:
            diag[f"{label}_fit_rms"] = float("nan")
            diag[f"{label}_fit_weighted_rms"] = float("nan")
            continue
        resid = A @ coeffs - b
        diag[f"{label}_fit_rms"] = float(np.sqrt(np.mean(resid * resid)))
        diag[f"{label}_fit_weighted_rms"] = float(np.sqrt(np.mean(w * resid * resid)))
        diag[f"{label}_fit_max_abs"] = float(np.max(np.abs(resid)))

    if A_L_phase.shape[0]:
        phase_resid = A_L_phase @ cL
        if nBeta:
            phase_resid += A_beta_phase @ beta
        if nIota:
            phase_resid += A_iota_phase @ iota_coeffs
        phase_resid -= b_phase
        diag["phase_fit_rms"] = float(np.sqrt(np.mean(phase_resid * phase_resid)))
        diag["phase_fit_weighted_rms"] = float(np.sqrt(np.mean(w_phase * phase_resid * phase_resid)))
        diag["phase_fit_max_abs"] = float(np.max(np.abs(phase_resid)))
    else:
        diag["phase_fit_rms"] = float("nan")
        diag["phase_fit_weighted_rms"] = float("nan")

    if A_beta_gauge.shape[0] and nBeta:
        gauge_resid = A_beta_gauge @ beta
        diag["beta_gauge_rms"] = float(np.sqrt(np.mean(gauge_resid * gauge_resid)))
        diag["beta_gauge_max_abs"] = float(np.max(np.abs(gauge_resid)))
    else:
        diag["beta_gauge_rms"] = float("nan")

    return JointSpectralFitResult(
        R_lmn=np.asarray(cR, dtype=float),
        Z_lmn=np.asarray(cZ, dtype=float),
        L_lmn=np.asarray(cL, dtype=float),
        beta=np.asarray(beta, dtype=float),
        iota_coeffs=np.asarray(iota_coeffs, dtype=float),
        iota_powers=tuple(int(p) for p in cfg.iota_powers),
        matrix_shape=tuple(A_big.shape),
        diagnostics=diag,
    )


def apply_joint_fit(eq: Any, fit: JointSpectralFitResult) -> None:
    eq.R_lmn = np.asarray(fit.R_lmn, dtype=float)
    eq.Z_lmn = np.asarray(fit.Z_lmn, dtype=float)
    eq.L_lmn = np.asarray(fit.L_lmn, dtype=float)
