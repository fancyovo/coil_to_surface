from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = REPO_ROOT / "gpu_backend" / "python"
if str(GPU_PYTHON) not in sys.path:
    sys.path.insert(0, str(GPU_PYTHON))

from stellarator_gpu import score_coils_native


def load_case(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data["raw"]
    unit = str(raw.get("current_unit", "A")).lower()
    scale = 1.0 if unit in {"a", "amp", "amps"} else 1.0e6
    metadata = raw.get("metadata", {})
    nfp = int(data.get("nfp", raw.get("nfp")))
    helicity = int(metadata.get("helicity", 1))
    return (
        raw["x"],
        raw["y"],
        raw["z"],
        [float(value) * scale for value in raw["current"]],
        nfp,
        (1, 0 if helicity == 0 else nfp),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    x, y, z, currents, nfp, target = load_case(args.case)

    def score(overrides: dict | None = None):
        return score_coils_native(
            args.lib,
            x,
            y,
            z,
            currents,
            nfp,
            target_helicity=target,
            config_overrides=overrides,
        )

    results = {"legacy": score()}
    for periods in (1, 2, 4):
        results[f"continuous_p{periods}"] = score(
            {
                "surface_selection_mode": 1,
                "surface_confidence_periods": periods,
            }
        )
    axis = results["legacy"]["diagnostics"]
    results["legacy_hint"] = score(
        {
            "axis_hint_enabled": 1,
            "axis_hint_require_continuation": 1,
            "axis_hint_R": float(axis["axis_R"]),
            "axis_hint_Z": float(axis["axis_Z"]),
        }
    )
    results["invalid_hint"] = score(
        {
            "axis_hint_enabled": 1,
            "axis_hint_require_continuation": 1,
            "axis_hint_R": 1.0e-3,
            "axis_hint_Z": 4.0,
            "axis_hint_max_distance": 1.0e-3,
        }
    )
    payload = {
        "case": str(args.case),
        "library": str(args.lib),
        "nfp": nfp,
        "target": list(target),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({name: [row["status"], row["score"], row["timing"]["total_s"]]
                      for name, row in results.items()}, allow_nan=True))


if __name__ == "__main__":
    main()
