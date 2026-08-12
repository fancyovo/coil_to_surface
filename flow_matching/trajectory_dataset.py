from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from flow_matching.data import file_sha256


FORMAT = "qh_screen32_adam200_trajectory_v1"
COMPONENT_KEYS = (
    "axis",
    "psi",
    "surface",
    "coordinate",
    "volume_qs",
    "iota",
    "coil",
)


def atomic_savez_compressed(path: str | Path, **arrays: np.ndarray) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {target} or {partial}")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(partial, target)
    return file_sha256(target)


def atomic_write_json(
    path: str | Path, payload: dict[str, Any], *, allow_nan: bool = False
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {target} or {partial}")
    partial.write_text(
        json.dumps(payload, indent=2, allow_nan=allow_nan) + "\n", encoding="utf-8"
    )
    os.replace(partial, target)
    return file_sha256(target)


def atomic_write_jsonl_gzip(path: str | Path, rows: list[dict[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {target} or {partial}")
    with gzip.open(partial, "wt", encoding="utf-8", compresslevel=1) as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=True))
            stream.write("\n")
    os.replace(partial, target)
    return file_sha256(target)


def _numeric_keys(results: list[dict[str, Any]], section: str) -> tuple[str, ...]:
    keys: set[str] = set()
    for result in results:
        values = result.get(section, {})
        if not isinstance(values, dict):
            continue
        keys.update(
            key
            for key, value in values.items()
            if isinstance(value, (bool, int, float, np.number))
        )
    return tuple(sorted(keys))


def _numeric_matrix(
    results: list[dict[str, Any]], section: str, keys: tuple[str, ...]
) -> np.ndarray:
    values = np.full((len(results), len(keys)), np.nan, dtype=np.float64)
    for row_index, result in enumerate(results):
        section_values = result.get(section, {})
        if not isinstance(section_values, dict):
            continue
        for column_index, key in enumerate(keys):
            value = section_values.get(key)
            if isinstance(value, (bool, int, float, np.number)):
                values[row_index, column_index] = float(value)
    return values


class OptimizationTraceRecorder:
    """Collect one complete Adam trajectory before one atomic final write."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.initial: dict[str, Any] | None = None
        self.steps: list[dict[str, Any]] = []
        self.center_results: list[dict[str, Any]] = []

    def record_initial(
        self,
        noise: np.ndarray,
        tokens: np.ndarray,
        result: dict[str, Any],
    ) -> None:
        if self.initial is not None:
            raise RuntimeError("initial state was already recorded")
        self.initial = {
            "noise": np.asarray(noise, dtype=np.float32).copy(),
            "tokens": np.asarray(tokens, dtype=np.float32).copy(),
        }
        self.center_results.append(
            {"iteration": 0, "center_after_native_score": result}
        )

    def record_step(self, **values: Any) -> None:
        if self.initial is None:
            raise RuntimeError("record_initial must be called first")
        local_results = list(values.pop("local_results"))
        step = {
            key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
            for key, value in values.items()
        }
        step["local_results"] = local_results
        self.steps.append(step)
        self.center_results.append(
            {
                "iteration": int(step["iteration"]),
                "probe_captured_native_score": step.pop("probe_result"),
                "center_after_native_score": step.pop("center_result"),
            }
        )

    def finalize(self) -> dict[str, Any]:
        if self.initial is None or not self.steps:
            raise RuntimeError("cannot finalize an empty optimization trace")
        all_local_results = [
            result for step in self.steps for result in step["local_results"]
        ]
        diagnostic_keys = _numeric_keys(all_local_results, "diagnostics")
        timing_keys = _numeric_keys(all_local_results, "timing")
        endpoint_count = len(self.steps[0]["local_results"])
        if any(len(step["local_results"]) != endpoint_count for step in self.steps):
            raise RuntimeError("gradient endpoint count changed within one trajectory")

        arrays: dict[str, np.ndarray] = {
            "initial_noise": self.initial["noise"],
            "initial_tokens": self.initial["tokens"],
        }
        direct_arrays = (
            "iteration",
            "probe_noise",
            "probe_tokens",
            "directions",
            "endpoint_tokens",
            "raw_gradient",
            "first_moment_before",
            "second_moment_before",
            "first_moment_after",
            "second_moment_after",
            "proposed_update",
            "applied_update",
            "center_after_noise",
            "center_after_tokens",
            "gradient_step_applied",
            "center_update_accepted",
            "center_acceptance_fraction",
            "adam_step",
        )
        for key in direct_arrays:
            arrays[key] = np.asarray([step[key] for step in self.steps])

        arrays["endpoint_score"] = np.asarray(
            [
                [float(result.get("score", float("nan"))) for result in step["local_results"]]
                for step in self.steps
            ],
            dtype=np.float64,
        )
        arrays["endpoint_status"] = np.asarray(
            [
                [str(result.get("status", "missing")) for result in step["local_results"]]
                for step in self.steps
            ],
            dtype="U32",
        )
        arrays["endpoint_components"] = np.asarray(
            [
                [
                    [float(result.get("components", {}).get(key, float("nan"))) for key in COMPONENT_KEYS]
                    for result in step["local_results"]
                ]
                for step in self.steps
            ],
            dtype=np.float64,
        )
        arrays["endpoint_diagnostics"] = _numeric_matrix(
            all_local_results, "diagnostics", diagnostic_keys
        ).reshape(len(self.steps), endpoint_count, -1)
        arrays["endpoint_timing"] = _numeric_matrix(
            all_local_results, "timing", timing_keys
        ).reshape(len(self.steps), endpoint_count, -1)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.output_dir / "training_trace.npz"
        trace_sha = atomic_savez_compressed(trace_path, **arrays)
        center_path = self.output_dir / "center_native_results.jsonl.gz"
        center_sha = atomic_write_jsonl_gzip(center_path, self.center_results)
        schema = {
            "format": FORMAT,
            "completed_steps": len(self.steps),
            "endpoint_count_per_step": endpoint_count,
            "component_keys": list(COMPONENT_KEYS),
            "diagnostic_keys": list(diagnostic_keys),
            "timing_keys": list(timing_keys),
            "storage": {
                "latent_and_token_arrays": "float32",
                "scores_components_diagnostics": "float64",
                "endpoint_latent_reconstruction": (
                    "probe_noise +/- perturbation * directions, interleaved minus/plus"
                ),
                "center_results": center_path.name,
            },
            "files": {
                trace_path.name: trace_sha,
                center_path.name: center_sha,
            },
        }
        schema_path = self.output_dir / "training_trace_schema.json"
        schema_sha = atomic_write_json(schema_path, schema)
        schema["files"][schema_path.name] = schema_sha
        return schema
