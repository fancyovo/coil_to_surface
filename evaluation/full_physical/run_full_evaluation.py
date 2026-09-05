#!/usr/bin/env python3
"""Run the fixed full physical evaluation inside one Slurm allocation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--case-file", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpu-lib", required=True, type=Path)
    parser.add_argument("--eval-env", required=True, type=Path)
    parser.add_argument("--a-values", default="0.04,0.05,0.06,0.08")
    parser.add_argument("--s-edges", default="0.12,0.24,0.36,0.49,0.64,0.81,1.0")
    parser.add_argument("--candidate-cpus", type=int, default=4)
    parser.add_argument("--desc-cpus", type=int, default=16)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_values(text: str, label: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{label} cannot be empty")
    for value in values:
        number = float(value)
        if not number > 0:
            raise ValueError(f"{label} must contain positive values: {value}")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate values")
    return values


def slug(value: str) -> str:
    return value.replace(".", "p")


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_logged(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, object]:
    started = time.monotonic()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return {
        "returncode": completed.returncode,
        "elapsed_s": time.monotonic() - started,
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
    }


def surface_expected_rejection(output_dir: Path, returncode: int) -> dict | None:
    """Classify exit 3 only when a surface stage wrote a physical rejection."""

    if returncode != 3:
        return None
    alpha_rejection = output_dir / "alpha" / "rejection.json"
    if alpha_rejection.is_file():
        try:
            payload = json.loads(alpha_rejection.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if payload.get("status") == "rejected":
            return {
                "kind": "physical_candidate_rejection",
                "record": str(alpha_rejection.resolve()),
                "details": payload,
            }
    standard_summary = output_dir / "standard_rho_1" / "summary.json"
    if standard_summary.is_file():
        try:
            payload = json.loads(standard_summary.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if payload.get("accepted_for_downstream") is False:
            return {
                "kind": "standard_surface_rejection",
                "record": str(standard_summary.resolve()),
                "details": {
                    "target_s": payload.get("target_s"),
                    "acceptance_checks": payload.get("acceptance_checks", {}),
                },
            }
    return None


def run_gpu_stage(
    *,
    project: Path,
    stage: str,
    values: list[str],
    gpu_ids: list[str],
    candidate_cpus: int,
    base_env: dict[str, str],
    command_for_value,
    env_for_value,
    output_for_value,
    log_dir: Path,
    classify_nonzero=None,
) -> list[dict[str, object]]:
    available: Queue[str] = Queue()
    for gpu_id in gpu_ids:
        available.put(gpu_id)

    def run_one(value: str) -> dict[str, object]:
        gpu_id = available.get()
        try:
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["SLURM_CPUS_PER_TASK"] = str(candidate_cpus)
            env.update(env_for_value(value))
            output_dir = output_for_value(value)
            env["OUTPUT_DIR"] = str(output_dir)
            result = run_logged(
                command_for_value(value),
                env=env,
                cwd=project,
                stdout_path=log_dir / f"{stage}_{slug(value)}.out",
                stderr_path=log_dir / f"{stage}_{slug(value)}.err",
            )
            result.update(
                {
                    "value": float(value),
                    "gpu": gpu_id,
                    "output_dir": str(output_dir.resolve()),
                }
            )
            if result["returncode"] == 0:
                result["outcome"] = "completed"
            else:
                rejection = (
                    classify_nonzero(output_dir, int(result["returncode"]))
                    if classify_nonzero is not None
                    else None
                )
                if rejection is None:
                    result["outcome"] = "failed"
                else:
                    result["outcome"] = "rejected"
                    result["rejection"] = rejection
            return result
        finally:
            available.put(gpu_id)

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {executor.submit(run_one, value): value for value in values}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: float(item["value"]))
    failed = [result for result in results if result["outcome"] == "failed"]
    if failed:
        details = ", ".join(
            f"{item['value']}:exit={item['returncode']}" for item in failed
        )
        raise RuntimeError(f"{stage} candidate failures: {details}")
    return results


def run_selector(
    command: list[str], *, cwd: Path, stdout_path: Path, stderr_path: Path
) -> tuple[str, float]:
    env = os.environ.copy()
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"selector failed with exit {completed.returncode}: {completed.stderr.strip()}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("selector returned no selected path")
    return lines[-1], time.monotonic() - started


def git_commit(project: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> int:
    args = parse_args()
    project = args.project.resolve()
    case_file = args.case_file.resolve()
    output_root = args.output_root.resolve()
    gpu_lib = args.gpu_lib.resolve()
    eval_env = args.eval_env.resolve()
    for path in (project, case_file, gpu_lib, eval_env):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    if args.candidate_cpus <= 0 or args.desc_cpus <= 0:
        raise ValueError("CPU counts must be positive")

    a_values = parse_values(args.a_values, "a-values")
    s_edges = parse_values(args.s_edges, "s-edges")
    gpu_ids = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if not gpu_ids:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose at least one allocated GPU")

    output_root.mkdir(parents=True)
    log_dir = output_root / "logs"
    log_dir.mkdir()
    started_wall = utc_now()
    started = time.monotonic()
    configuration: dict[str, object] = {
        "a_values_m": [float(value) for value in a_values],
        "s_edges": [float(value) for value in s_edges],
        "gpu_ids": gpu_ids,
        "candidate_cpus": args.candidate_cpus,
        "desc_cpus": args.desc_cpus,
    }
    status_path = output_root / "run_status.json"
    status: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "stage": "initializing",
        "started_at": started_wall,
        "updated_at": started_wall,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "case_file": str(case_file),
        "case_sha256": sha256(case_file),
        "gpu_library": str(gpu_lib),
        "gpu_library_sha256": sha256(gpu_lib),
        "code_commit": git_commit(project),
        "configuration": configuration,
    }
    atomic_json(status_path, status)

    base_env = os.environ.copy()
    base_env.update(
        {
            "PROJECT": str(project),
            "CASE_FILE": str(case_file),
            "GPU_LIB": str(gpu_lib),
            "EVAL_ENV": str(eval_env),
        }
    )
    timings: dict[str, float] = {}
    try:
        stage_start = time.monotonic()
        status.update(stage="source_psi_candidates", updated_at=utc_now())
        atomic_json(status_path, status)
        source_root = output_root / "source_psi_candidates"
        source_results = run_gpu_stage(
            project=project,
            stage="source_psi",
            values=a_values,
            gpu_ids=gpu_ids,
            candidate_cpus=args.candidate_cpus,
            base_env=base_env,
            command_for_value=lambda value: [
                "bash",
                str(project / "scripts" / "slurm_fit_source_psi.sh"),
            ],
            env_for_value=lambda value: {"A_VALUE": value},
            output_for_value=lambda value: source_root / f"a_{slug(value)}",
            log_dir=log_dir,
        )
        timings["source_psi_candidates"] = time.monotonic() - stage_start

        status.update(stage="source_psi_selection", updated_at=utc_now())
        status["source_psi_candidates"] = source_results
        atomic_json(status_path, status)
        source_selection_path = output_root / "source_psi_selection.json"
        selected_source, selector_time = run_selector(
            [
                sys.executable,
                str(project / "evaluation" / "full_physical" / "select_source_psi_candidate.py"),
                "--candidate-root",
                str(source_root),
                "--output",
                str(source_selection_path),
            ],
            cwd=project,
            stdout_path=log_dir / "source_psi_selection.out",
            stderr_path=log_dir / "source_psi_selection.err",
        )
        timings["source_psi_selection"] = selector_time
        selected_source_path = Path(selected_source)

        stage_start = time.monotonic()
        status.update(
            stage="surface_candidates",
            updated_at=utc_now(),
            selected_source_psi=str(selected_source_path),
        )
        atomic_json(status_path, status)
        surface_root = output_root / "candidates"
        surface_env = base_env.copy()
        surface_env["RUN_DIR"] = str(selected_source_path)
        surface_results = run_gpu_stage(
            project=project,
            stage="surface",
            values=s_edges,
            gpu_ids=gpu_ids,
            candidate_cpus=args.candidate_cpus,
            base_env=surface_env,
            command_for_value=lambda value: [
                "bash",
                str(project / "scripts" / "slurm_alpha_nu_guarded_boozer.sh"),
            ],
            env_for_value=lambda value: {"S_EDGE": value},
            output_for_value=lambda value: surface_root / f"s_{slug(value)}",
            log_dir=log_dir,
            classify_nonzero=surface_expected_rejection,
        )
        timings["surface_candidates"] = time.monotonic() - stage_start

        status.update(stage="surface_selection", updated_at=utc_now())
        status["surface_candidates"] = surface_results
        atomic_json(status_path, status)
        surface_selection_path = output_root / "selection.json"
        selected_surface, selector_time = run_selector(
            [
                sys.executable,
                str(project / "evaluation" / "full_physical" / "select_largest_standard_surface.py"),
                "--candidate-root",
                str(surface_root),
                "--output",
                str(surface_selection_path),
            ],
            cwd=project,
            stdout_path=log_dir / "surface_selection.out",
            stderr_path=log_dir / "surface_selection.err",
        )
        timings["surface_selection"] = selector_time

        stage_start = time.monotonic()
        status.update(
            stage="downstream",
            updated_at=utc_now(),
            selected_surface=selected_surface,
        )
        atomic_json(status_path, status)
        downstream_env = base_env.copy()
        downstream_env.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "SLURM_CPUS_PER_TASK": str(args.desc_cpus),
                "SURFACE_NPZ": selected_surface,
                "OUTPUT_DIR": str(output_root / "full"),
            }
        )
        downstream_result = run_logged(
            ["bash", str(project / "scripts" / "slurm_evaluate_saved_boozer_full_cpu.sh")],
            env=downstream_env,
            cwd=project,
            stdout_path=log_dir / "downstream.out",
            stderr_path=log_dir / "downstream.err",
        )
        if downstream_result["returncode"] != 0:
            raise RuntimeError(
                f"downstream evaluation failed with exit {downstream_result['returncode']}"
            )
        timings["downstream"] = time.monotonic() - stage_start

        full_summary_path = output_root / "full" / "full_summary.json"
        if not full_summary_path.is_file():
            raise FileNotFoundError(full_summary_path)
        final = {
            "schema_version": 1,
            "status": "completed",
            "started_at": started_wall,
            "completed_at": utc_now(),
            "elapsed_s": time.monotonic() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "case_file": str(case_file),
            "case_sha256": status["case_sha256"],
            "code_commit": status["code_commit"],
            "gpu_library_sha256": status["gpu_library_sha256"],
            "configuration": configuration,
            "timing_s": timings,
            "source_psi_selection": json.loads(
                source_selection_path.read_text(encoding="utf-8")
            ),
            "surface_selection": json.loads(
                surface_selection_path.read_text(encoding="utf-8")
            ),
            "downstream": json.loads(full_summary_path.read_text(encoding="utf-8")),
            "artifacts": {
                "run_status": str(status_path.resolve()),
                "source_psi_selection": str(source_selection_path.resolve()),
                "surface_selection": str(surface_selection_path.resolve()),
                "full_summary": str(full_summary_path.resolve()),
                "assets": str((output_root / "full" / "assets").resolve()),
                "desc": str((output_root / "full" / "desc").resolve()),
                "logs": str(log_dir.resolve()),
            },
        }
        atomic_json(output_root / "evaluation_summary.json", final)
        status.update(
            status="completed",
            stage="completed",
            updated_at=utc_now(),
            completed_at=final["completed_at"],
            elapsed_s=final["elapsed_s"],
            downstream=downstream_result,
            evaluation_summary=str((output_root / "evaluation_summary.json").resolve()),
        )
        atomic_json(status_path, status)
        print(output_root / "evaluation_summary.json")
        return 0
    except BaseException as exc:
        status.update(
            status="failed",
            stage=status.get("stage", "unknown"),
            updated_at=utc_now(),
            failed_at=utc_now(),
            elapsed_s=time.monotonic() - started,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        atomic_json(status_path, status)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
