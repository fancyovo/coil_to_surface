from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from torch import nn

from flow_matching.model import FlowBlock


class LatentProxyTransformer(nn.Module):
    """Permutation-invariant classifier for flow latent coil sets."""

    def __init__(
        self,
        *,
        token_dim: int = 100,
        width: int = 256,
        layers: int = 4,
        heads: int = 8,
        hidden: int = 704,
        max_nfp: int = 16,
    ):
        super().__init__()
        self.config = {
            "token_dim": token_dim,
            "width": width,
            "layers": layers,
            "heads": heads,
            "hidden": hidden,
            "max_nfp": max_nfp,
        }
        self.input = nn.Linear(token_dim, width)
        self.nfp_embedding = nn.Embedding(max_nfp + 1, width)
        self.blocks = nn.ModuleList(
            [FlowBlock(width, heads, hidden) for _ in range(layers)]
        )
        self.final_norm = nn.RMSNorm(width, eps=1.0e-6)
        self.output = nn.Linear(width, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        nfp: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or nfp.shape != (tokens.shape[0],):
            raise ValueError("tokens and nfp batch dimensions must match")
        if torch.any((nfp < 1) | (nfp > self.config["max_nfp"])):
            raise ValueError("nfp is outside the model embedding range")
        condition = self.nfp_embedding(nfp)
        x = self.input(tokens)
        if mask is not None:
            if mask.shape != tokens.shape[:2]:
                raise ValueError("mask shape must match the token axes")
            x = x * mask[..., None]
        for block in self.blocks:
            x = block(x, condition, mask)
        x = self.final_norm(x)
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            weights = mask.to(dtype=x.dtype)[..., None]
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.output(pooled).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _as_binary_arrays(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    label = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probability.shape != label.shape or probability.size == 0:
        raise ValueError("probabilities and labels must be nonempty and have equal shape")
    if not np.isfinite(probability).all() or not np.isin(label, (0, 1)).all():
        raise ValueError("probabilities must be finite and labels must be binary")
    if not np.any(label == 0) or not np.any(label == 1):
        raise ValueError("both classes are required")
    return probability, label


def roc_auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probability, label = _as_binary_arrays(probabilities, labels)
    ranks = rankdata(probability, method="average")
    positive = label == 1
    n_positive = int(np.sum(positive))
    n_negative = int(len(label) - n_positive)
    numerator = np.sum(ranks[positive]) - n_positive * (n_positive + 1) / 2.0
    return float(numerator / (n_positive * n_negative))


def average_precision(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probability, label = _as_binary_arrays(probabilities, labels)
    order = np.argsort(-probability, kind="stable")
    sorted_label = label[order]
    true_positive = np.cumsum(sorted_label)
    precision = true_positive / np.arange(1, len(label) + 1)
    return float(np.sum(precision * sorted_label) / np.sum(sorted_label))


def validation_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Choose the threshold maximizing Youden's J on validation data."""
    probability, label = _as_binary_arrays(probabilities, labels)
    order = np.argsort(-probability, kind="stable")
    score = probability[order]
    sorted_label = label[order]
    true_positive = np.cumsum(sorted_label)
    false_positive = np.cumsum(1 - sorted_label)
    group_end = np.r_[score[:-1] != score[1:], True]
    indices = np.flatnonzero(group_end)
    sensitivity = true_positive[indices] / np.sum(label)
    specificity = 1.0 - false_positive[indices] / np.sum(1 - label)
    best = indices[int(np.argmax(sensitivity + specificity - 1.0))]
    return float(score[best])


def binary_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
    ece_bins: int = 15,
) -> dict[str, float | int | dict[str, int]]:
    probability, label = _as_binary_arrays(probabilities, labels)
    predicted = probability >= float(threshold)
    positive = label == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum(~predicted & negative))
    fp = int(np.sum(predicted & negative))
    fn = int(np.sum(~predicted & positive))
    clipped = np.clip(probability, 1.0e-7, 1.0 - 1.0e-7)
    log_loss = -np.mean(label * np.log(clipped) + (1 - label) * np.log(1 - clipped))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, int(ece_bins) + 1)
    for index in range(int(ece_bins)):
        selected = (probability >= edges[index]) & (
            probability < edges[index + 1]
            if index + 1 < ece_bins
            else probability <= edges[index + 1]
        )
        if np.any(selected):
            ece += np.mean(selected) * abs(
                float(np.mean(probability[selected])) - float(np.mean(label[selected]))
            )
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "count": int(len(label)),
        "positive_count": int(np.sum(positive)),
        "negative_count": int(np.sum(negative)),
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(label)),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "roc_auc": roc_auc(probability, label),
        "average_precision": average_precision(probability, label),
        "log_loss": float(log_loss),
        "brier": float(np.mean((probability - label) ** 2)),
        "ece": float(ece),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def enrichment_at_prior_rates(
    probabilities: np.ndarray,
    labels: np.ndarray,
    rates: tuple[float, ...] = (0.5, 0.1, 0.01, 0.001),
) -> list[dict[str, float]]:
    probability, label = _as_binary_arrays(probabilities, labels)
    positive = probability[label == 1]
    negative = probability[label == 0]
    rows = []
    for requested_rate in rates:
        if not 0.0 < requested_rate <= 1.0:
            raise ValueError("prior pass rates must lie in (0, 1]")
        threshold = float(
            np.quantile(negative, 1.0 - requested_rate, method="higher")
        )
        negative_rate = float(np.mean(negative >= threshold))
        positive_rate = float(np.mean(positive >= threshold))
        rows.append(
            {
                "requested_prior_pass_rate": float(requested_rate),
                "threshold": threshold,
                "actual_prior_pass_rate": negative_rate,
                "positive_retention_rate": positive_rate,
                "enrichment": positive_rate / max(negative_rate, 1.0 / len(negative)),
            }
        )
    return rows


def apply_logit_calibration(
    logits: np.ndarray,
    *,
    scale: float,
    bias: float,
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(scale) or not np.isfinite(bias):
        raise ValueError("logits and calibration parameters must be finite")
    if scale <= 0.0:
        raise ValueError("calibration scale must be positive")
    return expit(scale * values + bias)


def fit_logit_calibration(
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int | bool | str]:
    """Fit a monotone Platt calibration using validation labels only."""
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    label = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.shape != label.shape or values.size == 0:
        raise ValueError("logits and labels must be nonempty and have equal shape")
    if not np.isfinite(values).all() or not np.isin(label, (0, 1)).all():
        raise ValueError("logits must be finite and labels must be binary")
    if not np.any(label == 0) or not np.any(label == 1):
        raise ValueError("both classes are required")

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        log_scale, bias = parameters
        scale = np.exp(log_scale)
        calibrated_logit = scale * values + bias
        probability = expit(calibrated_logit)
        loss = np.mean(np.logaddexp(0.0, calibrated_logit) - label * calibrated_logit)
        residual = probability - label
        gradient = np.asarray(
            [np.mean(residual * scale * values), np.mean(residual)],
            dtype=np.float64,
        )
        return float(loss), gradient

    initial_loss, _ = objective(np.zeros(2, dtype=np.float64))
    result = minimize(
        objective,
        np.zeros(2, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=((-8.0, 8.0), (-50.0, 50.0)),
        options={"ftol": 1.0e-13, "gtol": 1.0e-10, "maxiter": 500},
    )
    scale = float(np.exp(result.x[0]))
    return {
        "method": "validation_platt_monotone",
        "scale": scale,
        "bias": float(result.x[1]),
        "initial_log_loss": float(initial_loss),
        "calibrated_log_loss": float(result.fun),
        "iterations": int(result.nit),
        "success": bool(result.success),
        "message": str(result.message),
    }
