"""Exercise the bounded OAS production task supervisor."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts/run_oas_production_task_batch.py"


def _load_module():
    """Load the batch supervisor from its immutable source path."""
    specification = importlib.util.spec_from_file_location(
        "oas_production_task_batch", SUPERVISOR
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _resource_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the minimal scheduler evidence used by production v2."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OAS_SCHEDULER_RESERVED_GPU_COUNT", "1")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)


def test_supervisor_completes_and_authenticates_existing_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run bounded tasks once and authenticate their immutable receipts later."""
    _resource_environment(monkeypatch)
    module = _load_module()
    completion_template = str(tmp_path / "done-{task_id:04d}.json")
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')",
        completion_template,
    ]
    first = module.run_batch(
        stage="map",
        task_count=3,
        worker_count=2,
        completion_template=completion_template,
        log_root=tmp_path / "logs-first",
        command_template=command,
    )
    assert first["all_tasks_complete"] is True
    assert [item["status"] for item in first["task_results"]] == [
        "completed",
        "completed",
        "completed",
    ]
    second = module.run_batch(
        stage="map",
        task_count=3,
        worker_count=2,
        completion_template=completion_template,
        log_root=tmp_path / "logs-second",
        command_template=command,
    )
    assert second["all_tasks_complete"] is True
    assert all(
        item["status"] == "authenticated_existing_completion"
        for item in second["task_results"]
    )


def test_supervisor_records_failure_without_authorizing_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve a terminal failure receipt when one logical task exits nonzero."""
    _resource_environment(monkeypatch)
    module = _load_module()
    completion_template = str(tmp_path / "failed-{task_id:04d}.json")
    command = [
        sys.executable,
        "-c",
        "import sys; raise SystemExit(7)",
        "{task_id}",
    ]
    receipt = module.run_batch(
        stage="reduce",
        task_count=2,
        worker_count=2,
        completion_template=completion_template,
        log_root=tmp_path / "failure-logs",
        command_template=command,
    )
    assert receipt["all_tasks_complete"] is False
    assert [item["return_code"] for item in receipt["task_results"]] == [7, 7]
    assert (tmp_path / "failure-logs/receipt.json").is_file()


def test_supervisor_rejects_visible_or_missing_scheduler_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before execution if the idle-GPU evidence is not exact."""
    module = _load_module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("OAS_SCHEDULER_RESERVED_GPU_COUNT", "1")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    with pytest.raises(module.TaskBatchError, match="remain hidden"):
        module.run_batch(
            stage="map",
            task_count=1,
            worker_count=1,
            completion_template=str(tmp_path / "done-{task_id}.json"),
            log_root=tmp_path / "visible",
            command_template=[sys.executable, "-c", "pass", "{task_id}"],
        )
