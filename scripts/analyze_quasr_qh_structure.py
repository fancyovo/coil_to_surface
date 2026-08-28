from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_matching.data import file_sha256  # noqa: E402
from flow_matching.trajectory_dataset import atomic_savez_compressed, atomic_write_json  # noqa: E402


FORMAT = "quasr_qh_geometric_structure_atlas_v2"
CURVE_ORDER = 16
COEFF_COUNT = 2 * CURVE_ORDER + 1
TOKEN_DIM = 3 * COEFF_COUNT + 1


def group_name(nfp: int, ncoils: int) -> str:
    return f"nfp{nfp}_nc{ncoils}"


def dataset_groups(
    manifest: dict[str, Any], stratification: str = "condition"
) -> dict[tuple[int, ...], list[dict[str, Any]]]:
    groups: defaultdict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for shard in manifest["shards"]:
        if stratification == "condition":
            key = (int(shard["nfp"]), int(shard["n_coils"]))
        elif stratification == "ncoils":
            key = (int(shard["n_coils"]),)
        else:
            raise ValueError(f"unknown stratification: {stratification}")
        groups[key].append(shard)
    return {key: sorted(rows, key=lambda row: row["file"]) for key, rows in groups.items()}


def load_group(
    data_dir: Path,
    shards: list[dict[str, Any]],
    *,
    verify_hashes: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_parts = []
    id_parts = []
    nfp_parts = []
    for shard in shards:
        path = data_dir / shard["file"]
        if verify_hashes and file_sha256(path) != shard["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}")
        with np.load(path, allow_pickle=False) as payload:
            tokens = np.asarray(payload["tokens"], dtype=np.float32)
            ids = np.asarray(payload["ids"], dtype=np.int32)
        expected = (int(shard["count"]), int(shard["n_coils"]), TOKEN_DIM)
        if tokens.shape != expected or ids.shape != (expected[0],):
            raise ValueError(f"unexpected shard shape in {path.name}")
        token_parts.append(tokens)
        id_parts.append(ids)
        nfp_parts.append(np.full(len(ids), int(shard["nfp"]), dtype=np.int16))
    return (
        np.concatenate(token_parts),
        np.concatenate(id_parts),
        np.concatenate(nfp_parts),
    )


def curve_arrays(tokens: np.ndarray, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(tokens, dtype=np.float64)
    coefficients = values[..., :99].reshape(*values.shape[:-1], 3, COEFF_COUNT)
    t = np.arange(samples, dtype=np.float64) / samples
    modes = np.arange(1, CURVE_ORDER + 1, dtype=np.float64)
    angle = 2.0 * np.pi * modes[:, None] * t[None]
    sine = np.sin(angle)
    cosine = np.cos(angle)
    omega = 2.0 * np.pi * modes
    sine_coeff = coefficients[..., 1::2]
    cosine_coeff = coefficients[..., 2::2]
    position = (
        coefficients[..., 0, None]
        + np.einsum("...cm,mt->...ct", sine_coeff, sine, optimize=True)
        + np.einsum("...cm,mt->...ct", cosine_coeff, cosine, optimize=True)
    )
    first = (
        np.einsum(
            "...cm,mt->...ct", sine_coeff * omega, cosine, optimize=True
        )
        - np.einsum(
            "...cm,mt->...ct", cosine_coeff * omega, sine, optimize=True
        )
    )
    second = -(
        np.einsum(
            "...cm,mt->...ct", sine_coeff * omega**2, sine, optimize=True
        )
        + np.einsum(
            "...cm,mt->...ct", cosine_coeff * omega**2, cosine, optimize=True
        )
    )
    return (
        np.moveaxis(position, -2, -1),
        np.moveaxis(first, -2, -1),
        np.moveaxis(second, -2, -1),
    )


def coil_feature_names() -> list[str]:
    names = [
        "centroid_R",
        "centroid_Z",
        "curve_length",
        "radius_of_gyration",
    ]
    names.extend(f"R_q{quantile:02d}" for quantile in (5, 25, 50, 75, 95))
    names.extend(f"Z_q{quantile:02d}" for quantile in (5, 25, 50, 75, 95))
    names.extend(("speed_cv", "curvature_q50", "curvature_q90", "curvature_q99"))
    names.extend(f"xy_mode_energy_fraction_{mode:02d}" for mode in range(1, 17))
    names.extend(f"z_mode_energy_fraction_{mode:02d}" for mode in range(1, 17))
    names.extend(("signed_current_fraction", "absolute_current_fraction"))
    return names


def coil_features(tokens: np.ndarray, *, samples: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tokens, dtype=np.float64)
    position, first, second = curve_arrays(values, samples)
    centroid = np.mean(position, axis=-2)
    cylindrical_radius = np.linalg.norm(position[..., :2], axis=-1)
    speed = np.linalg.norm(first, axis=-1)
    curvature = np.linalg.norm(np.cross(first, second), axis=-1) / np.maximum(speed, 1.0e-10) ** 3
    centered = position - centroid[..., None, :]
    radius_of_gyration = np.sqrt(np.mean(np.sum(centered**2, axis=-1), axis=-1))
    coefficients = values[..., :99].reshape(*values.shape[:-1], 3, COEFF_COUNT)
    mode_energy = coefficients[..., 1::2] ** 2 + coefficients[..., 2::2] ** 2
    xy_energy = np.sum(mode_energy[..., :2, :], axis=-2)
    z_energy = mode_energy[..., 2, :]
    total_energy = np.maximum(np.sum(xy_energy + z_energy, axis=-1, keepdims=True), 1.0e-24)
    currents = values[..., -1]
    current_l1 = np.maximum(np.sum(np.abs(currents), axis=-1, keepdims=True), 1.0e-12)
    output = np.concatenate(
        (
            np.linalg.norm(centroid[..., :2], axis=-1)[..., None],
            centroid[..., 2, None],
            np.mean(speed, axis=-1)[..., None],
            radius_of_gyration[..., None],
            np.quantile(cylindrical_radius, (0.05, 0.25, 0.5, 0.75, 0.95), axis=-1).transpose(1, 2, 0),
            np.quantile(position[..., 2], (0.05, 0.25, 0.5, 0.75, 0.95), axis=-1).transpose(1, 2, 0),
            (np.std(speed, axis=-1) / np.maximum(np.mean(speed, axis=-1), 1.0e-12))[..., None],
            np.quantile(curvature, (0.5, 0.9, 0.99), axis=-1).transpose(1, 2, 0),
            xy_energy / total_energy,
            z_energy / total_energy,
            (currents / current_l1)[..., None],
            (np.abs(currents) / current_l1)[..., None],
        ),
        axis=-1,
    )
    if output.shape[-1] != len(coil_feature_names()) or not np.all(np.isfinite(output)):
        raise ValueError("nonfinite or inconsistent coil descriptors")
    return output, centroid


def set_feature_names(ncoils: int) -> list[str]:
    names = []
    for coil_index in range(ncoils):
        names.extend(f"coil{coil_index}_{name}" for name in coil_feature_names())
    pair_count = ncoils * (ncoils - 1) // 2
    for metric in ("centroid_distance_3d", "centroid_distance_xy", "centroid_angle", "centroid_abs_dz"):
        names.extend(f"pair{pair_index}_{metric}" for pair_index in range(pair_count))
    return names


def set_features(tokens: np.ndarray, *, samples: int = 96) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != TOKEN_DIM:
        raise ValueError("tokens must have shape (sample, coil, 100)")
    coil, centroid = coil_features(values, samples=samples)
    sort_key = coil[..., 0] + 1.0e-6 * coil[..., 2] + 1.0e-9 * coil[..., 1]
    order = np.argsort(sort_key, axis=1, kind="stable")
    ordered = np.take_along_axis(coil, order[..., None], axis=1)
    output = [ordered.reshape(len(values), -1)]
    pair_values: list[list[np.ndarray]] = [[], [], [], []]
    angles = np.arctan2(centroid[..., 1], centroid[..., 0])
    for first_index in range(values.shape[1]):
        for second_index in range(first_index + 1, values.shape[1]):
            delta = centroid[:, first_index] - centroid[:, second_index]
            angle_delta = np.abs(
                np.arctan2(
                    np.sin(angles[:, first_index] - angles[:, second_index]),
                    np.cos(angles[:, first_index] - angles[:, second_index]),
                )
            )
            pair_values[0].append(np.linalg.norm(delta, axis=-1))
            pair_values[1].append(np.linalg.norm(delta[:, :2], axis=-1))
            pair_values[2].append(angle_delta)
            pair_values[3].append(np.abs(delta[:, 2]))
    for values_for_metric in pair_values:
        if values_for_metric:
            output.append(np.sort(np.stack(values_for_metric, axis=-1), axis=-1))
    features = np.concatenate(output, axis=-1).astype(np.float32)
    if features.shape[1] != len(set_feature_names(values.shape[1])):
        raise RuntimeError("set descriptor name count does not match its width")
    if not np.all(np.isfinite(features)):
        raise ValueError("set descriptors contain nonfinite values")
    return features


def extract_group_features(
    tokens: np.ndarray,
    *,
    samples: int,
    batch_size: int,
) -> np.ndarray:
    parts = []
    for start in range(0, len(tokens), batch_size):
        parts.append(set_features(tokens[start : start + batch_size], samples=samples))
    return np.concatenate(parts)


def robust_scale(features: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = np.asarray(features, dtype=np.float64)
    median = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.3489795003921634
    standard = np.std(values, axis=0)
    scale = np.where(scale > 1.0e-10, scale, standard)
    keep = np.flatnonzero(scale > 1.0e-10)
    if len(keep) < 2:
        raise ValueError("fewer than two nonconstant geometric descriptors")
    scaled = (values[:, keep] - median[keep]) / scale[keep]
    return scaled.astype(np.float32), {"median": median, "scale": scale, "keep": keep}


def fit_pca(
    scaled: np.ndarray,
    fit_indices: np.ndarray,
    *,
    variance_target: float,
    max_components: int,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    fit = np.asarray(scaled[fit_indices], dtype=np.float64)
    mean = np.mean(fit, axis=0)
    centered = fit - mean
    covariance = centered.T @ centered / max(1, len(fit) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    cumulative = np.cumsum(eigenvalues) / max(np.sum(eigenvalues), 1.0e-30)
    retained = int(np.searchsorted(cumulative, variance_target) + 1)
    retained = max(2, min(retained, max_components, eigenvectors.shape[1]))
    components = eigenvectors[:, :retained]
    transformed = (np.asarray(scaled, dtype=np.float64) - mean) @ components
    return transformed.astype(np.float32), {
        "mean": mean.astype(np.float32),
        "components": components.astype(np.float32),
        "eigenvalues": eigenvalues[:retained].astype(np.float32),
        "retained_components": retained,
        "retained_variance_fraction": float(cumulative[retained - 1]),
    }


def squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    return np.maximum(
        np.sum(x * x, axis=1, keepdims=True)
        + np.sum(c * c, axis=1)[None]
        - 2.0 * x @ c.T,
        0.0,
    )


def kmeans_once(
    values: np.ndarray,
    k: int,
    *,
    seed: int,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    centers = np.empty((k, x.shape[1]), dtype=np.float64)
    centers[0] = x[int(rng.integers(len(x)))]
    closest = squared_distances(x, centers[:1])[:, 0]
    for index in range(1, k):
        total = float(np.sum(closest))
        selected = int(rng.integers(len(x))) if total <= 0.0 else int(rng.choice(len(x), p=closest / total))
        centers[index] = x[selected]
        closest = np.minimum(closest, squared_distances(x, centers[index : index + 1])[:, 0])
    labels = np.full(len(x), -1, dtype=np.int32)
    for iteration in range(1, max_iterations + 1):
        distances = squared_distances(x, centers)
        updated_labels = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(updated_labels, labels):
            labels = updated_labels
            break
        labels = updated_labels
        minimum = distances[np.arange(len(x)), labels]
        for cluster in range(k):
            selected = labels == cluster
            if np.any(selected):
                centers[cluster] = np.mean(x[selected], axis=0)
            else:
                centers[cluster] = x[int(np.argmax(minimum))]
    final_distances = squared_distances(x, centers)
    labels = np.argmin(final_distances, axis=1).astype(np.int32)
    inertia = float(np.sum(final_distances[np.arange(len(x)), labels]))
    return centers.astype(np.float32), labels, inertia, iteration


def fit_kmeans(
    values: np.ndarray,
    k: int,
    *,
    seed: int,
    n_init: int,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    candidates = [
        kmeans_once(values, k, seed=seed + 104729 * index, max_iterations=max_iterations)
        for index in range(n_init)
    ]
    return min(candidates, key=lambda result: result[2])


def adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.int64)
    b = np.asarray(second, dtype=np.int64)
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    contingency = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(contingency, (a, b), 1)

    def pairs(values: np.ndarray) -> float:
        return float(np.sum(values * (values - 1) // 2))

    observed = pairs(contingency)
    row_pairs = pairs(np.sum(contingency, axis=1))
    column_pairs = pairs(np.sum(contingency, axis=0))
    total_pairs = len(a) * (len(a) - 1) / 2
    if total_pairs == 0:
        return 1.0
    expected = row_pairs * column_pairs / total_pairs
    maximum = 0.5 * (row_pairs + column_pairs)
    return 1.0 if maximum == expected else (observed - expected) / (maximum - expected)


def silhouette_from_distances(distances: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    values = np.zeros(len(labels), dtype=np.float64)
    for index in range(len(labels)):
        same = labels == labels[index]
        same[index] = False
        if not np.any(same):
            continue
        a = float(np.mean(distances[index, same]))
        b = min(
            float(np.mean(distances[index, labels == cluster]))
            for cluster in unique
            if cluster != labels[index]
        )
        values[index] = (b - a) / max(a, b, 1.0e-12)
    return float(np.mean(values))


def choose_partition(
    coordinates: np.ndarray,
    fit_indices: np.ndarray,
    *,
    k_values: list[int],
    seed: int,
    max_iterations: int,
    silhouette_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fit = coordinates[fit_indices]
    rng = np.random.default_rng(seed + 17)
    evaluation_indices = np.sort(
        rng.choice(len(fit), size=min(silhouette_count, len(fit)), replace=False)
    )
    evaluation = np.asarray(fit[evaluation_indices], dtype=np.float64)
    pair_distances = np.sqrt(squared_distances(evaluation, evaluation))
    candidates: list[dict[str, Any]] = []
    models: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in k_values:
        if k >= len(fit):
            continue
        centers, labels, inertia, iterations = fit_kmeans(
            fit,
            k,
            seed=seed + 1009 * k,
            n_init=3,
            max_iterations=max_iterations,
        )
        alternate_centers, alternate_labels, alternate_inertia, _ = fit_kmeans(
            fit,
            k,
            seed=seed + 1009 * k + 500_009,
            n_init=2,
            max_iterations=max_iterations,
        )
        evaluation_labels = labels[evaluation_indices]
        silhouette = silhouette_from_distances(pair_distances, evaluation_labels)
        stability = adjusted_rand_index(labels, alternate_labels)
        counts = np.bincount(labels, minlength=k)
        score = silhouette * (0.5 + 0.5 * max(0.0, stability))
        candidates.append(
            {
                "k": k,
                "silhouette": silhouette,
                "stability_ari": stability,
                "mean_squared_radius": inertia / len(fit),
                "alternate_mean_squared_radius": alternate_inertia / len(fit),
                "smallest_cluster_fraction": float(np.min(counts) / len(labels)),
                "largest_cluster_fraction": float(np.max(counts) / len(labels)),
                "iterations": iterations,
                "selection_score": score,
            }
        )
        models[k] = (centers, labels)
    if not candidates:
        raise ValueError("no admissible k values")
    best_score = max(row["selection_score"] for row in candidates)
    near_best = [
        row
        for row in candidates
        if row["selection_score"] >= best_score - max(0.005, 0.05 * abs(best_score))
    ]
    selected_metric = min(near_best, key=lambda row: row["k"])
    centers, labels = models[int(selected_metric["k"])]
    return {
        "metrics": selected_metric,
        "centers": centers,
        "fit_labels": labels,
    }, candidates


def balanced_fit_indices(
    strata: np.ndarray,
    fit_count: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(strata)
    if len(values) <= fit_count:
        return np.arange(len(values), dtype=np.int64)
    unique = np.unique(values)
    base, remainder = divmod(fit_count, len(unique))
    selected: list[np.ndarray] = []
    leftovers: list[np.ndarray] = []
    for index, value in enumerate(unique):
        members = rng.permutation(np.flatnonzero(values == value))
        take = min(len(members), base + int(index < remainder))
        selected.append(members[:take])
        leftovers.append(members[take:])
    chosen = np.concatenate(selected)
    deficit = fit_count - len(chosen)
    if deficit:
        available = np.concatenate(leftovers)
        chosen = np.concatenate((chosen, rng.choice(available, deficit, replace=False)))
    return np.sort(chosen.astype(np.int64, copy=False))


def normalized_mutual_information(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first)
    right = np.asarray(second)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("mutual-information labels must be matching vectors")
    _, left_inverse = np.unique(left, return_inverse=True)
    _, right_inverse = np.unique(right, return_inverse=True)
    left_count = int(np.max(left_inverse)) + 1
    right_count = int(np.max(right_inverse)) + 1
    if left_count < 2 or right_count < 2:
        return 0.0
    contingency = np.zeros((left_count, right_count), dtype=np.float64)
    np.add.at(contingency, (left_inverse, right_inverse), 1.0)
    probability = contingency / len(left)
    left_probability = np.sum(probability, axis=1)
    right_probability = np.sum(probability, axis=0)
    expected = left_probability[:, None] * right_probability[None, :]
    occupied = probability > 0.0
    information = float(
        np.sum(probability[occupied] * np.log(probability[occupied] / expected[occupied]))
    )
    left_entropy = float(-np.sum(left_probability * np.log(left_probability)))
    right_entropy = float(-np.sum(right_probability * np.log(right_probability)))
    return information / math.sqrt(left_entropy * right_entropy)


def normalized_nearest_neighbor(coordinates: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    values = np.asarray(coordinates, dtype=np.float64)
    distances, _ = cKDTree(values).query(values, k=2, workers=-1)
    return distances[:, 1] / math.sqrt(values.shape[1])


def descriptor_duplicate_fraction(scaled: np.ndarray, decimals: int = 6) -> float:
    rounded = np.ascontiguousarray(np.round(scaled, decimals=decimals))
    row_type = np.dtype((np.void, rounded.dtype.itemsize * rounded.shape[1]))
    unique = np.unique(rounded.view(row_type).reshape(-1)).size
    return 1.0 - unique / len(rounded)


def quantiles(values: np.ndarray, probabilities: tuple[float, ...]) -> dict[str, float]:
    return {
        str(probability): float(value)
        for probability, value in zip(probabilities, np.quantile(values, probabilities), strict=True)
    }


def rotate_for_display(position: np.ndarray) -> np.ndarray:
    center = np.mean(position.reshape(-1, 3), axis=0)
    angle = math.atan2(float(center[1]), float(center[0]))
    cosine, sine = math.cos(-angle), math.sin(-angle)
    rotated = position.copy()
    rotated_x = cosine * position[..., 0] - sine * position[..., 1]
    rotated_y = sine * position[..., 0] + cosine * position[..., 1]
    rotated[..., 0] = rotated_x
    rotated[..., 1] = rotated_y
    return rotated


def equal_3d_limits(axis: Any, position: np.ndarray) -> None:
    low = np.min(position.reshape(-1, 3), axis=0)
    high = np.max(position.reshape(-1, 3), axis=0)
    center = 0.5 * (low + high)
    half = 0.55 * max(float(np.max(high - low)), 1.0e-6)
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)


def plot_gallery(
    path: Path,
    representative_tokens: np.ndarray,
    representative_ids: np.ndarray,
    representative_nfp: np.ndarray,
    cluster_sizes: np.ndarray,
    *,
    nfp_label: str,
    ncoils: int,
    curve_samples: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(representative_tokens)
    columns = min(4, count)
    rows = math.ceil(count / columns)
    figure = plt.figure(figsize=(3.7 * columns, 3.45 * rows), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for cluster, tokens in enumerate(representative_tokens):
        axis = figure.add_subplot(rows, columns, cluster + 1, projection="3d")
        position, _, _ = curve_arrays(tokens[None], curve_samples)
        position = rotate_for_display(position[0])
        for coil_index, coil in enumerate(position):
            closed = np.concatenate((coil, coil[:1]), axis=0)
            axis.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=colors(coil_index), linewidth=1.6)
        equal_3d_limits(axis, position)
        axis.set_title(
            f"C{cluster}: n={int(cluster_sizes[cluster])}\n"
            f"medoid nfp={int(representative_nfp[cluster])}, "
            f"QUASR {int(representative_ids[cluster])}",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])
        axis.view_init(elev=24, azim=-58)
    figure.suptitle(
        f"Representative base-coil sets: nfp={nfp_label}, base coils={ncoils}",
        fontsize=12,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_overview(
    groups: list[dict[str, Any]], output_dir: Path, *, stratification: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    counts = np.asarray([row["sample_count"] for row in groups])
    selected_k = np.asarray([row["selected_k"] for row in groups])
    hard_group_count = len(groups)
    leaf_cluster_count = sum(int(row["selected_k"]) for row in groups)
    if stratification == "condition":
        nfp = np.asarray([row["nfp"] for row in groups])
        ncoils = np.asarray([row["n_base_coils"] for row in groups])
        scatter = axes[0].scatter(
            nfp,
            ncoils,
            s=35.0 + 220.0 * np.sqrt(counts / np.max(counts)),
            c=selected_k,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.5,
        )
        for row in groups:
            axes[0].text(
                row["nfp"] + 0.06,
                row["n_base_coils"] + 0.06,
                str(row["selected_k"]),
                fontsize=7,
            )
        axes[0].set(
            xlabel="Field periods (nfp)",
            ylabel="Base-coil count",
            title="Hard condition groups and within-group partitions",
            xticks=range(2, 9),
            yticks=range(1, 6),
        )
        figure.colorbar(scatter, ax=axes[0], label="Within-group partition count")
        ordered = sorted(
            groups, key=lambda row: (-row["sample_count"], row["group"])
        )
        labels = [
            row["group"].replace("nfp", "").replace("_nc", "/")
            for row in ordered
        ]
        overview_title = (
            f"{hard_group_count} hard (nfp, base-coil count) groups; "
            f"{leaf_cluster_count} total leaf partitions"
        )
        concentration_title = "Partition concentration by condition"
        density_xlabel = "Condition groups, ordered by sample count"
    else:
        ordered = sorted(groups, key=lambda row: int(row["n_base_coils"]))
        x_left = np.arange(len(ordered))
        bars = axes[0].bar(
            x_left,
            [row["selected_k"] for row in ordered],
            color="#397c6b",
        )
        for bar, row in zip(bars, ordered, strict=True):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.08,
                f"k={row['selected_k']}\nn={row['sample_count']:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axes[0].set(
            xlabel="Base-coil count",
            ylabel="Cross-nfp partition count",
            title="Hard base-coil groups and cross-nfp partitions",
            xticks=x_left,
            xticklabels=[str(row["n_base_coils"]) for row in ordered],
            ylim=(0.0, max(row["selected_k"] for row in ordered) + 1.0),
        )
        labels = [str(row["n_base_coils"]) for row in ordered]
        overview_title = (
            f"{hard_group_count} hard base-coil-count groups; "
            f"{leaf_cluster_count} total cross-nfp leaf partitions"
        )
        concentration_title = "Cross-nfp partition concentration by base-coil count"
        density_xlabel = "Base-coil count"

    x = np.arange(len(ordered))
    effective = np.asarray([row["effective_cluster_count"] for row in ordered])
    largest = np.asarray([row["largest_cluster_fraction"] for row in ordered])
    axes[1].bar(x, effective, color="#397c6b", label="effective cluster count")
    twin = axes[1].twinx()
    twin.plot(x, largest, color="#a24d3f", marker=".", linewidth=1.0, label="largest-cluster share")
    axes[1].set(
        xlabel=density_xlabel,
        ylabel="Effective cluster count",
        title=concentration_title,
        xticks=x,
        xticklabels=labels,
    )
    axes[1].tick_params(axis="x", labelrotation=90, labelsize=6)
    twin.set_ylabel("Largest-cluster share")
    twin.set_ylim(0.0, 1.0)
    figure.suptitle(overview_title, fontsize=12)
    figure.savefig(output_dir / "group_structure_overview.png", dpi=190)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11.5, 5.0), constrained_layout=True)
    nn_data = [row["nearest_neighbor_normalized_quantiles"]["0.5"] for row in ordered]
    p95_data = [row["nearest_neighbor_normalized_quantiles"]["0.95"] for row in ordered]
    axis.plot(x, nn_data, marker="o", markersize=3, label="median")
    axis.plot(x, p95_data, marker=".", label="95th percentile")
    axis.set(
        yscale="log",
        xlabel=density_xlabel,
        ylabel="Nearest-neighbor distance / sqrt(PCA dimensions)",
        title="Local descriptor density",
        xticks=x,
        xticklabels=labels,
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=6)
    axis.legend()
    figure.savefig(output_dir / "nearest_neighbor_density.png", dpi=190)
    plt.close(figure)


def plot_cross_nfp_composition(
    groups: list[dict[str, Any]], output_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = [
        (row, composition)
        for row in sorted(groups, key=lambda item: int(item["n_base_coils"]))
        for composition in row["cluster_nfp_composition"]
    ]
    labels = [
        f"nc{row['n_base_coils']}/C{composition['cluster']}"
        for row, composition in entries
    ]
    totals = np.asarray(
        [sum(composition["counts"].values()) for _, composition in entries],
        dtype=np.float64,
    )
    nfp_values = sorted(
        {
            int(nfp)
            for _, composition in entries
            for nfp in composition["counts"]
        }
    )
    figure, axis = plt.subplots(
        figsize=(10.5, max(4.2, 0.48 * len(entries))), constrained_layout=True
    )
    left = np.zeros(len(entries), dtype=np.float64)
    colors = plt.get_cmap("tab10")
    for color_index, nfp in enumerate(nfp_values):
        fraction = np.asarray(
            [
                composition["counts"].get(str(nfp), 0) / total
                for (_, composition), total in zip(entries, totals, strict=True)
            ]
        )
        axis.barh(
            np.arange(len(entries)),
            fraction,
            left=left,
            color=colors(color_index),
            label=f"nfp={nfp}",
        )
        left += fraction
    for index, total in enumerate(totals):
        axis.text(1.01, index, f"n={int(total):,}", va="center", fontsize=8)
    axis.set(
        xlim=(0.0, 1.12),
        xlabel="Fraction of leaf members",
        ylabel="Cross-nfp leaf partition",
        title="Field-period composition of cross-nfp leaf partitions",
        yticks=np.arange(len(entries)),
        yticklabels=labels,
    )
    axis.invert_yaxis()
    axis.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    figure.savefig(output_dir / "cross_nfp_composition.png", dpi=190)
    plt.close(figure)


def analyze_group(
    tokens: np.ndarray,
    ids: np.ndarray,
    sample_nfp: np.ndarray,
    *,
    name: str,
    ncoils: int,
    args: argparse.Namespace,
    output_dir: Path,
    make_gallery: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    features = extract_group_features(
        tokens,
        samples=args.curve_samples,
        batch_size=args.feature_batch_size,
    )
    scaled, scaling = robust_scale(features)
    unique_nfp = np.unique(sample_nfp)
    if args.stratification == "condition":
        fit_seed = args.seed + 100 * int(unique_nfp[0]) + ncoils
        partition_seed = args.seed + 1009 * int(unique_nfp[0]) + ncoils
        rng = np.random.default_rng(fit_seed)
        fit_indices = np.sort(
            rng.choice(
                len(tokens), size=min(args.fit_count, len(tokens)), replace=False
            )
        )
        fit_sampling = "uniform within the (nfp,n_base_coils) hard group"
    else:
        fit_seed = args.seed + 10_000 + ncoils
        partition_seed = args.seed + 1_000_000 + ncoils
        rng = np.random.default_rng(fit_seed)
        fit_indices = balanced_fit_indices(sample_nfp, args.fit_count, rng=rng)
        fit_sampling = "nfp-balanced within the n_base_coils hard group"
    coordinates, pca = fit_pca(
        scaled,
        fit_indices,
        variance_target=args.pca_variance,
        max_components=args.max_pca_components,
    )
    k_values = [value for value in args.k_values if value < len(fit_indices)]
    selected, candidates = choose_partition(
        coordinates,
        fit_indices,
        k_values=k_values,
        seed=partition_seed,
        max_iterations=args.kmeans_iterations,
        silhouette_count=args.silhouette_count,
    )
    centers = np.asarray(selected["centers"], dtype=np.float32)
    labels = np.argmin(squared_distances(coordinates, centers), axis=1).astype(np.int16)
    distance_to_center = np.sqrt(
        squared_distances(coordinates, centers)[np.arange(len(tokens)), labels]
    ) / math.sqrt(coordinates.shape[1])
    cluster_count = len(centers)
    cluster_sizes = np.bincount(labels, minlength=cluster_count)
    probabilities = cluster_sizes / len(labels)
    effective_count = float(np.exp(-np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-30)))))
    representatives = []
    cluster_nfp_composition = []
    medoid_indices = []
    cluster_radius = []
    raw_center_distances = squared_distances(coordinates, centers)
    for cluster in range(cluster_count):
        member_indices = np.flatnonzero(labels == cluster)
        local = raw_center_distances[member_indices, cluster]
        medoid = int(member_indices[int(np.argmin(local))])
        medoid_indices.append(medoid)
        normalized = np.sqrt(local) / math.sqrt(coordinates.shape[1])
        radius = quantiles(normalized, (0.5, 0.9, 0.95, 0.99))
        cluster_radius.append(radius)
        member_nfp, member_nfp_counts = np.unique(
            sample_nfp[member_indices], return_counts=True
        )
        nfp_probabilities = member_nfp_counts / len(member_indices)
        nfp_effective_count = float(
            np.exp(
                -np.sum(
                    nfp_probabilities
                    * np.log(np.maximum(nfp_probabilities, 1.0e-30))
                )
            )
        )
        cluster_nfp_composition.append(
            {
                "cluster": cluster,
                "counts": {
                    str(int(nfp)): int(count)
                    for nfp, count in zip(
                        member_nfp, member_nfp_counts, strict=True
                    )
                },
                "nfp_value_count": len(member_nfp),
                "effective_nfp_count": nfp_effective_count,
                "largest_nfp_fraction": float(np.max(nfp_probabilities)),
            }
        )
        representatives.append(
            {
                "cluster": cluster,
                "sample_count": int(cluster_sizes[cluster]),
                "sample_fraction": float(probabilities[cluster]),
                "quasr_id": int(ids[medoid]),
                "nfp": int(sample_nfp[medoid]),
                "group_row_index": medoid,
                "normalized_center_distance": float(distance_to_center[medoid]),
                "member_center_distance_quantiles": radius,
            }
    )
    nearest = normalized_nearest_neighbor(coordinates)
    assignment_path = output_dir / "atlas_data" / f"{name}.npz"
    assignment_sha = atomic_savez_compressed(
        assignment_path,
        quasr_id=ids,
        nfp=sample_nfp.astype(np.int16, copy=False),
        cluster=labels,
        pca_coordinates=coordinates,
        normalized_center_distance=distance_to_center.astype(np.float32),
        normalized_nearest_neighbor_distance=nearest.astype(np.float32),
        representative_group_row_index=np.asarray(medoid_indices, dtype=np.int32),
    )
    if make_gallery:
        plot_gallery(
            output_dir / "galleries" / f"{name}_representatives.png",
            tokens[np.asarray(medoid_indices)],
            ids[np.asarray(medoid_indices)],
            sample_nfp[np.asarray(medoid_indices)],
            cluster_sizes,
            nfp_label=(
                str(int(unique_nfp[0]))
                if len(unique_nfp) == 1
                else f"{int(unique_nfp[0])}-{int(unique_nfp[-1])} mixed"
            ),
            ncoils=ncoils,
            curve_samples=max(192, args.curve_samples),
        )
    summary = {
        "group": name,
        "nfp": int(unique_nfp[0]) if len(unique_nfp) == 1 else None,
        "nfp_values": [int(value) for value in unique_nfp],
        "nfp_counts": {
            str(int(value)): int(np.sum(sample_nfp == value)) for value in unique_nfp
        },
        "n_base_coils": ncoils,
        "sample_count": len(tokens),
        "input_feature_count": features.shape[1],
        "nonconstant_feature_count": len(scaling["keep"]),
        "pca_component_count": int(pca["retained_components"]),
        "pca_retained_variance_fraction": float(pca["retained_variance_fraction"]),
        "fit_sampling": fit_sampling,
        "selected_k": cluster_count,
        "selected_metrics": selected["metrics"],
        "candidate_metrics": candidates,
        "cluster_sizes": cluster_sizes.tolist(),
        "largest_cluster_fraction": float(np.max(probabilities)),
        "effective_cluster_count": effective_count,
        "cluster_nfp_normalized_mutual_information": normalized_mutual_information(
            labels, sample_nfp
        ),
        "cluster_nfp_composition": cluster_nfp_composition,
        "descriptor_duplicate_fraction_round6": descriptor_duplicate_fraction(scaled),
        "nearest_neighbor_normalized_quantiles": quantiles(nearest, (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)),
        "near_neighbor_fraction": {
            str(threshold): float(np.mean(nearest <= threshold))
            for threshold in (0.001, 0.005, 0.01, 0.05)
        },
        "representatives": representatives,
        "assignment_file": str(assignment_path.relative_to(output_dir)),
        "assignment_sha256": assignment_sha,
        "wall_s": time.perf_counter() - started,
    }
    model = {
        "group": name,
        "nfp_values": [int(value) for value in unique_nfp],
        "n_base_coils": ncoils,
        "feature_names": set_feature_names(ncoils),
        "scaling_median": scaling["median"].tolist(),
        "scaling_scale": scaling["scale"].tolist(),
        "nonconstant_feature_indices": scaling["keep"].tolist(),
        "pca_mean": np.asarray(pca["mean"]).tolist(),
        "pca_components": np.asarray(pca["components"]).tolist(),
        "cluster_centers": centers.tolist(),
        "cluster_radius_quantiles": cluster_radius,
        "novelty_rule": (
            "Apply the same descriptors/scaling/PCA, then compare normalized "
            "distance to the nearest center with that center's saved member quantiles."
        ),
    }
    return summary, model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a physically stratified geometric atlas of QUASR QH coil sets."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument(
        "--stratification",
        choices=("condition", "ncoils"),
        default="condition",
        help=(
            "condition isolates each (nfp,n_base_coils) pair; ncoils permits "
            "cross-nfp clusters while keeping base-coil counts separate"
        ),
    )
    parser.add_argument("--curve-samples", type=int, default=96)
    parser.add_argument("--feature-batch-size", type=int, default=384)
    parser.add_argument("--fit-count", type=int, default=5000)
    parser.add_argument("--silhouette-count", type=int, default=600)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--max-pca-components", type=int, default=24)
    parser.add_argument("--k-values", type=lambda text: [int(value) for value in text.split(",")], default=[2, 3, 4, 5, 6, 8, 10, 12])
    parser.add_argument("--kmeans-iterations", type=int, default=40)
    parser.add_argument("--gallery-group-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026082803)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.curve_samples < 32 or args.fit_count < 20 or args.silhouette_count < 20:
        raise ValueError("curve-samples, fit-count, and silhouette-count are too small")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "atlas_data").mkdir()
    (args.output_dir / "galleries").mkdir()

    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "quasr_qh_flow_v1":
        raise ValueError("unexpected dataset format")
    groups = dataset_groups(manifest, args.stratification)
    expected_count = int(manifest["requested_count"])
    if sum(int(row["count"]) for rows in groups.values() for row in rows) != expected_count:
        raise ValueError("dataset shard counts do not sum to requested_count")
    gallery_keys = set(
        key
        for key, _ in sorted(
            groups.items(),
            key=lambda item: -sum(int(row["count"]) for row in item[1]),
        )[: args.gallery_group_count]
    )
    started = time.perf_counter()
    summaries = []
    models = {}
    for key, shards in sorted(groups.items()):
        if args.stratification == "condition":
            nfp, ncoils = key
            name = group_name(nfp, ncoils)
        else:
            (ncoils,) = key
            name = f"nc{ncoils}"
        tokens, ids, sample_nfp = load_group(
            args.data_dir, shards, verify_hashes=args.verify_hashes
        )
        print(
            json.dumps(
                {"event": "group_loaded", "group": name, "count": len(tokens)}
            ),
            flush=True,
        )
        summary, model = analyze_group(
            tokens,
            ids,
            sample_nfp,
            name=name,
            ncoils=ncoils,
            args=args,
            output_dir=args.output_dir,
            make_gallery=key in gallery_keys,
        )
        summaries.append(summary)
        models[summary["group"]] = model
        print(json.dumps({"event": "group_complete", **{name: summary[name] for name in ("group", "sample_count", "selected_k", "wall_s")}}), flush=True)
    plot_overview(summaries, args.output_dir, stratification=args.stratification)
    if args.stratification == "ncoils":
        plot_cross_nfp_composition(summaries, args.output_dir)
    leaf_cluster_count_by_n_base_coils = {
        str(ncoils): sum(
            int(row["selected_k"])
            for row in summaries
            if int(row["n_base_coils"]) == ncoils
        )
        for ncoils in sorted({int(row["n_base_coils"]) for row in summaries})
    }
    aggregate = {
        "sample_count": sum(row["sample_count"] for row in summaries),
        "hard_group_count": len(summaries),
        "nfp_values": sorted(
            {int(value) for row in summaries for value in row["nfp_values"]}
        ),
        "n_base_coils_values": sorted(
            {int(row["n_base_coils"]) for row in summaries}
        ),
        "leaf_cluster_count": sum(int(row["selected_k"]) for row in summaries),
        "leaf_cluster_count_by_n_base_coils": leaf_cluster_count_by_n_base_coils,
        "selected_k_per_hard_group_range": [
            min(row["selected_k"] for row in summaries),
            max(row["selected_k"] for row in summaries),
        ],
        "selected_k_range": [
            min(row["selected_k"] for row in summaries),
            max(row["selected_k"] for row in summaries),
        ],
        "sample_weighted_effective_cluster_count": float(
            np.average(
                [row["effective_cluster_count"] for row in summaries],
                weights=[row["sample_count"] for row in summaries],
            )
        ),
        "sample_weighted_largest_cluster_fraction": float(
            np.average(
                [row["largest_cluster_fraction"] for row in summaries],
                weights=[row["sample_count"] for row in summaries],
            )
        ),
    }
    if args.stratification == "ncoils":
        aggregate.update(
            {
                "cross_nfp_leaf_cluster_count": sum(
                    int(composition["nfp_value_count"] > 1)
                    for row in summaries
                    for composition in row["cluster_nfp_composition"]
                ),
                "sample_weighted_cluster_nfp_normalized_mutual_information": float(
                    np.average(
                        [
                            row["cluster_nfp_normalized_mutual_information"]
                            for row in summaries
                        ],
                        weights=[row["sample_count"] for row in summaries],
                    )
                ),
            }
        )
    atlas = {
        "format": FORMAT,
        "created_unix_s": time.time(),
        "dataset": {
            "root": str(args.data_dir.resolve()),
            "manifest_sha256": file_sha256(manifest_path),
            "sample_count": expected_count,
            "group_count": len(groups),
            "hashes_verified": bool(args.verify_hashes),
        },
        "method": {
            "scope": (
                "base-coil geometry and relative currents, hard-stratified by "
                + (
                    "(nfp,n_base_coils)"
                    if args.stratification == "condition"
                    else "n_base_coils with cross-nfp clustering allowed"
                )
            ),
            "stratification": args.stratification,
            "cross_nfp_clustering": args.stratification == "ncoils",
            "invariances": [
                "base-coil permutation",
                "global toroidal rotation",
                "curve-parameter phase approximately through sampled invariants and exactly through spectral energies",
                "curve-parameter reversal",
            ],
            "curve_samples": args.curve_samples,
            "robust_scaling": "median and Gaussian-equivalent IQR within each hard group",
            "fit_sampling": (
                "uniform within each condition group"
                if args.stratification == "condition"
                else "nfp-balanced within each n_base_coils hard group"
            ),
            "pca_variance_target": args.pca_variance,
            "pca_component_cap": args.max_pca_components,
            "fit_count_cap": args.fit_count,
            "candidate_k": args.k_values,
            "partition_selection": (
                "maximize approximate silhouette times initialization stability, then "
                "choose the smallest k within a 5%/0.005 near-best tolerance"
            ),
            "cluster_algorithm": "deterministic multi-start k-means++/Lloyd",
            "seed": args.seed,
        },
        "groups": summaries,
        "aggregate": aggregate,
        "wall_s": time.perf_counter() - started,
    }
    atomic_write_json(args.output_dir / "atlas_summary.json", atlas, allow_nan=True)
    atomic_write_json(
        args.output_dir / "novelty_reference.json",
        {
            "format": "quasr_qh_geometric_novelty_reference_v2",
            "stratification": args.stratification,
            "models": models,
        },
        allow_nan=True,
    )
    print(json.dumps({"event": "atlas_complete", **atlas["aggregate"], "wall_s": atlas["wall_s"]}), flush=True)


if __name__ == "__main__":
    main()
