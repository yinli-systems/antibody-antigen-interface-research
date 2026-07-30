#!/usr/bin/env python3
"""Run independent OAS materialization tasks with bounded CPU concurrency."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Sequence


FORMAT_VERSION = "oas_production_task_batch_receipt_v1"


class TaskBatchError(RuntimeError):
    """Report an invalid batch environment or incomplete task execution."""


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize one receipt with a stable ASCII representation."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    """Hash a regular file without following a symbolic link."""
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise TaskBatchError(f"Completion path is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_receipt() -> dict[str, Any]:
    """Prove the scheduler reservation while keeping CUDA hidden from children."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "NoDevFiles"}:
        raise TaskBatchError("The scheduler-reserved GPU must remain hidden")
    assigned = {
        key: value
        for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")
        if (value := os.environ.get(key))
    }
    assigned_value = assigned.get("SLURM_JOB_GPUS") or assigned.get(
        "SLURM_STEP_GPUS", ""
    )
    assigned_count = len([value for value in assigned_value.split(",") if value])
    if (
        assigned_count != 1
        or os.environ.get("OAS_SCHEDULER_RESERVED_GPU_COUNT") != "1"
    ):
        raise TaskBatchError("One scheduler-reserved idle GPU must be declared")
    return {
        "cuda_visible_devices": False,
        "gpu_compute_permitted": False,
        "scheduler_gpu_environment": assigned,
        "scheduler_reserved_gpu_count": 1,
        "scheduler_reserved_gpu_mode": "scheduler_mandated_idle_only",
    }


def _format_task_value(value: str, task_id: int) -> str:
    """Expand only the frozen task identifier placeholder."""
    try:
        rendered = value.format(task_id=task_id)
    except (IndexError, KeyError, ValueError) as error:
        raise TaskBatchError(f"Invalid task placeholder in {value!r}") from error
    if "{task_id" in rendered:
        raise TaskBatchError(f"Unexpanded task placeholder in {value!r}")
    return rendered


def _completion_path(template: str, task_id: int) -> Path:
    """Resolve one absolute completion receipt path."""
    path = Path(_format_task_value(template, task_id))
    if not path.is_absolute():
        raise TaskBatchError("Completion receipt paths must be absolute")
    return path


def _open_create_only(path: Path) -> Any:
    """Open one immutable-evidence file without replacing prior evidence."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    return os.fdopen(descriptor, "wb")


def _run_task(
    *,
    task_id: int,
    command_template: Sequence[str],
    completion_template: str,
    log_root: Path,
) -> dict[str, Any]:
    """Run or authenticate exactly one logical task."""
    completion = _completion_path(completion_template, task_id)
    if completion.exists() or completion.is_symlink():
        return {
            "completion_path": str(completion),
            "completion_sha256": _file_sha256(completion),
            "return_code": 0,
            "status": "authenticated_existing_completion",
            "task_id": task_id,
        }
    command = [_format_task_value(value, task_id) for value in command_template]
    stdout_path = log_root / f"task-{task_id:04d}.stdout"
    stderr_path = log_root / f"task-{task_id:04d}.stderr"
    started = time.monotonic_ns()
    with _open_create_only(stdout_path) as stdout_handle, _open_create_only(
        stderr_path
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            check=False,
            env=dict(os.environ),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    elapsed_ns = time.monotonic_ns() - started
    os.chmod(stdout_path, 0o400)
    os.chmod(stderr_path, 0o400)
    completion_sha256 = None
    if completed.returncode == 0:
        completion_sha256 = _file_sha256(completion)
    return {
        "command": command,
        "completion_path": str(completion),
        "completion_sha256": completion_sha256,
        "elapsed_monotonic_ns": elapsed_ns,
        "return_code": completed.returncode,
        "status": "completed" if completed.returncode == 0 else "failed",
        "stderr_path": str(stderr_path),
        "stderr_sha256": _file_sha256(stderr_path),
        "stdout_path": str(stdout_path),
        "stdout_sha256": _file_sha256(stdout_path),
        "task_id": task_id,
    }


def run_batch(
    *,
    stage: str,
    task_count: int,
    worker_count: int,
    completion_template: str,
    log_root: Path,
    command_template: Sequence[str],
) -> dict[str, Any]:
    """Run a complete logical stage and publish a terminal batch receipt."""
    if stage not in {"map", "reduce"}:
        raise TaskBatchError("Stage must be map or reduce")
    if task_count <= 0 or not 1 <= worker_count <= 8:
        raise TaskBatchError("Task and worker counts are outside the frozen domain")
    if not command_template or not any("{task_id" in item for item in command_template):
        raise TaskBatchError("The command must contain a task identifier placeholder")
    if not log_root.is_absolute():
        raise TaskBatchError("The batch log root must be absolute")
    resource_receipt = _resource_receipt()
    log_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                _run_task,
                task_id=task_id,
                command_template=command_template,
                completion_template=completion_template,
                log_root=log_root,
            ): task_id
            for task_id in range(task_count)
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                results.append(future.result())
            except BaseException as error:  # noqa: BLE001 - evidence must be terminal.
                results.append(
                    {
                        "error": f"{type(error).__name__}: {error}",
                        "return_code": 1,
                        "status": "supervisor_failure",
                        "task_id": task_id,
                    }
                )
    results.sort(key=lambda value: int(value["task_id"]))
    all_complete = all(result.get("return_code") == 0 for result in results)
    receipt = {
        "all_tasks_complete": all_complete,
        "claim_boundary": (
            "CPU materialization orchestration only; one scheduler-reserved GPU "
            "was hidden and performed no permitted compute"
        ),
        "command_template": list(command_template),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "completion_template": completion_template,
        "format_version": FORMAT_VERSION,
        "host": socket.gethostname(),
        "resource_receipt": resource_receipt,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stage": stage,
        "task_count": task_count,
        "task_results": results,
        "test_access": {
            "downstream_metrics_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "worker_count": worker_count,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    receipt_path = log_root / "receipt.json"
    with _open_create_only(receipt_path) as handle:
        handle.write(_canonical_json_bytes(receipt))
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(receipt_path, 0o400)
    os.chmod(log_root, 0o500)
    return receipt


def _parser() -> argparse.ArgumentParser:
    """Build the strict batch command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("map", "reduce"), required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--completion-template", required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    """Run the requested batch and emit its canonical terminal receipt."""
    arguments = _parser().parse_args()
    command = list(arguments.command)
    if command and command[0] == "--":
        command.pop(0)
    receipt = run_batch(
        stage=arguments.stage,
        task_count=arguments.task_count,
        worker_count=arguments.worker_count,
        completion_template=arguments.completion_template,
        log_root=arguments.log_root,
        command_template=command,
    )
    sys.stdout.buffer.write(_canonical_json_bytes(receipt))
    return 0 if receipt["all_tasks_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
