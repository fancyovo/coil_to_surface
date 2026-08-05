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


SELECTED_CONTINUOUS = {
    "surface_selection_mode": 1,
    "surface_confidence_periods": 1,
    "surface_theta_count": 128,
    "surface_trace_steps": 400,
    "surface_flux_bisection_iters": 6,
}


def load_case(path: Path) -> tuple[tuple, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["raw"]
    unit = str(raw.get("current_unit", "A")).lower()
    scale = 1.0 if unit in {"a", "amp", "amps"} else 1.0e6
    arrays = (raw["x"], raw["y"], raw["z"], [scale * float(x) for x in raw["current"]])
    return arrays, int(payload.get("nfp", raw.get("nfp")))


def evaluate(lib: Path, path: Path, device: int) -> dict:
    arrays, nfp = load_case(path)
    legacy = score_coils_native(
        lib, *arrays, nfp, device_id=device, target_helicity=(1, nfp)
    )
    continuous = score_coils_native(
        lib, *arrays, nfp, device_id=device, target_helicity=(1, nfp),
        config_overrides=SELECTED_CONTINUOUS,
    )
    return {
        "path": str(path),
        "nfp": nfp,
        "legacy": legacy,
        "continuous": continuous,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("cases", nargs="+", type=Path)
    args = parser.parse_args()
    result = [evaluate(args.lib, path, args.device) for path in args.cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps([
        {
            "path": row["path"],
            "legacy": [row["legacy"]["status"], row["legacy"]["score"]],
            "continuous": [row["continuous"]["status"], row["continuous"]["score"]],
        }
        for row in result
    ], allow_nan=True))


if __name__ == "__main__":
    main()
