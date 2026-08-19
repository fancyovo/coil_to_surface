from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def solve_one(task: dict[str, str]) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "solve_qh_face_qs_surface_cpu.py"),
        "--case-file",
        task["case_file"],
        "--surface-npz",
        task["surface_npz"],
        "--output-dir",
        task["output_dir"],
    ]
    completed = subprocess.run(command, cwd="/", capture_output=True, text=True, env=os.environ.copy())
    summary_path = Path(task["output_dir"]) / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        status = summary["status"]
    else:
        status = "process_failed"
    return {
        **task,
        "status": status,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def build_tasks(experiment_root: Path) -> list[dict[str, str]]:
    tasks = []
    for case in read_json(experiment_root / "cases.json"):
        case_dir = experiment_root / "cases" / case["case_id"]
        prepared_path = case_dir / "face_qs" / "prepare_summary.json"
        if not prepared_path.is_file():
            continue
        prepared = read_json(prepared_path)
        if prepared.get("status") != "ok":
            continue
        for surface in prepared["surfaces"]:
            output = case_dir / "face_qs" / "cpu_solve" / surface["name"]
            tasks.append(
                {
                    "task_id": f"{case['case_id']}:{surface['name']}",
                    "case_id": case["case_id"],
                    "surface_name": surface["name"],
                    "case_file": str((case_dir / "case.json").resolve()),
                    "surface_npz": surface["surface_npz"],
                    "output_dir": str(output.resolve()),
                }
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent Simsopt surfaces in a bounded CPU process pool.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--residue-start", type=int, required=True)
    parser.add_argument("--residue-count", type=int, required=True)
    parser.add_argument("--residue-modulus", type=int, default=5)
    parser.add_argument("--pool-name", required=True)
    args = parser.parse_args()
    residues = set(range(args.residue_start, args.residue_start + args.residue_count))
    all_tasks = build_tasks(args.experiment_root)
    tasks = [task for index, task in enumerate(all_tasks) if index % args.residue_modulus in residues]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(solve_one, task): task for task in tasks}
        for completed_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(json.dumps({"event": "solved", "task_id": result["task_id"], "status": result["status"], "completed": completed_count, "count": len(tasks)}), flush=True)
    output = args.experiment_root / f"cpu_pool_{args.pool_name}.json"
    write_json(output, {"pool_name": args.pool_name, "workers": args.workers, "task_count": len(tasks), "results": results})
    failures = [row for row in results if row["status"] == "process_failed"]
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
