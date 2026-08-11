from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("axis", "psi", "surface", "coordinate", "volume_qs", "iota", "coil")


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


def gradients(rows: list[dict], variant: str, field: str, scale: float) -> np.ndarray:
    by_direction: dict[int, dict[int, dict]] = {}
    for row in rows:
        metadata = row["metadata"]
        by_direction.setdefault(int(metadata["direction_index"]), {})[int(metadata["sign"])] = row["variants"][variant]
    output = []
    for direction in sorted(by_direction):
        pair = by_direction[direction]
        if field == "score":
            minus, plus = pair[-1]["score"], pair[1]["score"]
        else:
            minus = pair[-1]["components"][field]
            plus = pair[1]["components"][field]
        output.append((plus - minus) / (2.0 * scale))
    return np.asarray(output, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.005)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.input_dir.glob("rank_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle)
    if len(rows) != 600:
        raise ValueError(f"expected 600 endpoint rows, found {len(rows)}")
    endpoint_keys = {
        (int(row["metadata"]["direction_index"]), int(row["metadata"]["sign"]))
        for row in rows
    }
    expected_keys = {(direction, sign) for direction in range(300) for sign in (-1, 1)}
    if endpoint_keys != expected_keys:
        missing = sorted(expected_keys - endpoint_keys)
        extra = sorted(endpoint_keys - expected_keys)
        raise ValueError(f"incomplete endpoint coverage: missing={missing[:8]}, extra={extra[:8]}")
    variants = sorted(
        (name for name in rows[0]["variants"] if name.startswith("pcgls")),
        key=lambda name: int(name.removeprefix("pcgls")),
    )
    exact_gradient = gradients(rows, "exact", "score", args.scale)
    summary = []
    component_cosines = {}
    for variant in variants:
        gradient = gradients(rows, variant, "score", args.scale)
        exact_scores = np.asarray([row["variants"]["exact"]["score"] for row in rows])
        scores = np.asarray([row["variants"][variant]["score"] for row in rows])
        component_cosines[variant] = {
            name: cosine(
                gradients(rows, "exact", name, args.scale),
                gradients(rows, variant, name, args.scale),
            )
            for name in COMPONENTS
        }
        summary.append({
            "variant": variant,
            "all_status_ok": all(row["variants"][variant]["status"] == "ok" for row in rows),
            "score_gradient_cosine": cosine(exact_gradient, gradient),
            "score_gradient_norm_ratio": float(np.linalg.norm(gradient) / np.linalg.norm(exact_gradient)),
            "score_gradient_sign_fraction": float(np.mean(np.sign(gradient) == np.sign(exact_gradient))),
            "score_rmse": float(np.sqrt(np.mean(np.square(scores - exact_scores)))),
            "score_max_abs_error": float(np.max(np.abs(scores - exact_scores))),
            "psi_fit_s_p50": float(np.median([row["variants"][variant]["psi_fit_s"] for row in rows])),
            "total_s_p50": float(np.median([row["variants"][variant]["total_s"] for row in rows])),
        })
    exact_timing = {
        "all_status_ok": all(row["variants"]["exact"]["status"] == "ok" for row in rows),
        "psi_fit_s_p50": float(np.median([row["variants"]["exact"]["psi_fit_s"] for row in rows])),
        "total_s_p50": float(np.median([row["variants"]["exact"]["total_s"] for row in rows])),
    }
    output = {
        "format": "local_psi_pcgls_full300_v1",
        "scale": args.scale,
        "direction_count": 300,
        "endpoint_count": len(rows),
        "exact_timing": exact_timing,
        "summary": summary,
        "component_gradient_cosines": component_cosines,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, len(variants), figsize=(6.2 * len(variants), 5.2), squeeze=False)
    for axis, variant in zip(axes[0], variants, strict=True):
        gradient = gradients(rows, variant, "score", args.scale)
        axis.scatter(exact_gradient, gradient, s=15, alpha=0.65)
        limit = max(float(np.max(np.abs(exact_gradient))), float(np.max(np.abs(gradient))))
        axis.plot([-limit, limit], [-limit, limit], color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{variant}: cosine={cosine(exact_gradient, gradient):.3f}")
        axis.set_xlabel("endpoint QR score derivative")
        axis.set_ylabel("PCGLS score derivative")
        axis.grid(True, alpha=0.25)
    fig.suptitle("300-coordinate score derivatives at fixed h=0.005")
    fig.tight_layout()
    fig.savefig(args.output_dir / "full300_score_gradient_scatter.png", dpi=180)
    plt.close(fig)

    matrix = np.asarray([[component_cosines[variant][name] for variant in variants] for name in COMPONENTS])
    fig, axis = plt.subplots(figsize=(6.2, 4.8))
    image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    axis.set_xticks(range(len(variants)), labels=variants)
    axis.set_yticks(range(len(COMPONENTS)), labels=COMPONENTS)
    axis.set_title("Full300 component-gradient cosine vs endpoint QR")
    fig.colorbar(image, ax=axis, label="cosine")
    fig.tight_layout()
    fig.savefig(args.output_dir / "full300_component_gradient_cosines.png", dpi=180)
    plt.close(fig)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
