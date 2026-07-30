"""Validate the fail-closed OAS production-materialization v2 amendment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "protocols/oas_production_materialization_v1.json"
V2 = ROOT / "protocols/oas_production_materialization_v2.json"
MAPPER = ROOT / "scripts/map_oas_production_records.py"
REDUCER = ROOT / "scripts/reduce_oas_production_bucket.py"
FINALIZER = ROOT / "scripts/finalize_oas_production_corpus.py"
AUDITOR = ROOT / "scripts/audit_oas_production_corpus_independent.py"
MAP_SLURM = ROOT / "slurm/oas_production_map_batched_v2.sbatch"
REDUCE_SLURM = ROOT / "slurm/oas_production_reduce_batched_v2.sbatch"
FINALIZE_SLURM = ROOT / "slurm/oas_production_finalize_v2.sbatch"
AUDIT_SLURM = ROOT / "slurm/oas_production_independent_audit_v1.sbatch"


def _load(path: Path, name: str):
    """Load one controller without importing it through another controller."""
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_v2_changes_only_execution_and_scheduler_resource_semantics() -> None:
    """Preserve the authenticated authority and every scientific boundary."""
    v1 = json.loads(V1.read_text(encoding="ascii"))
    v2 = json.loads(V2.read_text(encoding="ascii"))
    assert v2["format_version"] == "oas_production_materialization_protocol_v2"
    assert v2["parent_protocol_file_sha256"] == hashlib.sha256(
        V1.read_bytes()
    ).hexdigest()
    for field in (
        "authorization",
        "deduplication",
        "inputs",
        "map_record",
        "output",
        "quality_gates",
        "test_access",
    ):
        assert v2[field] == v1[field]
    assert v2["execution"]["map_task_count"] == 32
    assert v2["execution"]["reduce_task_count"] == 256
    assert v2["execution"]["bucket_count"] == 256
    assert v2["execution"]["map_concurrency_limit"] == 8
    assert v2["execution"]["reduce_concurrency_limit"] == 8
    assert v2["authorization"]["large_scale_pretraining"] is False
    assert v2["authorization"]["production_corpus_use"] is False


def test_v2_allows_only_one_hidden_scheduler_reserved_gpu() -> None:
    """Keep the scheduler reservation outside the materialization computation."""
    value = json.loads(V2.read_text(encoding="ascii"))
    resource = value["resource_policy"]
    assert resource == {
        "cuda_visible_devices_required_empty": True,
        "gpu_allocation_during_materialization": "scheduler_mandated_idle_only",
        "gpu_compute_permitted": False,
        "maximum_parallel_cpu_workers": 8,
        "scheduler_reserved_gpu_count": 1,
        "shared_storage_minimum_free_bytes": 1_000_000_000_000,
        "temporary_and_final_outputs_are_separate": True,
    }


@pytest.mark.parametrize(
    ("controller", "error_type"),
    ((MAPPER, "ProductionMapError"), (REDUCER, "ProductionReduceError")),
)
def test_v2_resource_receipt_requires_assignment_but_hides_cuda(
    controller: Path,
    error_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject both unassigned and CUDA-visible executions."""
    module = _load(controller, f"resource_{controller.stem}")
    protocol = json.loads(V2.read_text(encoding="ascii"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OAS_SCHEDULER_RESERVED_GPU_COUNT", "1")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    receipt = module._resource_receipt(protocol)
    assert receipt["scheduler_reserved_gpu_count"] == 1
    assert receipt["gpu_compute_permitted"] is False
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(getattr(module, error_type), match="must not expose"):
        module._resource_receipt(protocol)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1")
    with pytest.raises(getattr(module, error_type), match="evidence differs"):
        module._resource_receipt(protocol)
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    monkeypatch.delenv("SLURM_JOB_GPUS")
    with pytest.raises(getattr(module, error_type), match="evidence differs"):
        module._resource_receipt(protocol)


@pytest.mark.parametrize(
    ("controller", "error_type"),
    (
        (FINALIZER, "ProductionFinalizeError"),
        (AUDITOR, "IndependentCorpusAuditError"),
    ),
)
def test_v2_postprocessing_also_hides_reserved_gpu(
    controller: Path,
    error_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind finalization and replay to the same scheduler workaround."""
    module = _load(controller, f"postprocess_resource_{controller.stem}")
    protocol = json.loads(V2.read_text(encoding="ascii"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OAS_SCHEDULER_RESERVED_GPU_COUNT", "1")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    assert module._resource_receipt(protocol)["scheduler_reserved_gpu_count"] == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(getattr(module, error_type), match="must not expose"):
        module._resource_receipt(protocol)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1")
    with pytest.raises(getattr(module, error_type), match="evidence differs"):
        module._resource_receipt(protocol)
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    monkeypatch.delenv("SLURM_JOB_GPUS")
    with pytest.raises(getattr(module, error_type), match="evidence differs"):
        module._resource_receipt(protocol)


def test_batched_slurm_jobs_reserve_one_gpu_for_eight_cpu_workers() -> None:
    """Prevent array-scale GPU waste and accidental CUDA visibility."""
    for path, task_count in ((MAP_SLURM, 32), (REDUCE_SLURM, 256)):
        source = path.read_text(encoding="ascii")
        assert "#SBATCH --partition=gpu_5090" in source
        assert "#SBATCH --cpus-per-task=8" in source
        assert "#SBATCH --gres=gpu:1" in source
        assert "#SBATCH --array" not in source
        assert f"--task-count {task_count}" in source
        assert "--worker-count 8" in source
        assert "export CUDA_VISIBLE_DEVICES=" in source
        assert "export OAS_SCHEDULER_RESERVED_GPU_COUNT=1" in source
        assert "oas_production_materialization_v2.json" in source
        assert "OAS_SUPERVISOR_SHA256" in source


def test_postprocessing_slurm_jobs_reserve_but_hide_one_gpu() -> None:
    """Keep finalization and independent replay CPU-only and source-bound."""
    for path in (FINALIZE_SLURM, AUDIT_SLURM):
        source = path.read_text(encoding="ascii")
        assert "#SBATCH --partition=gpu_5090" in source
        assert "#SBATCH --cpus-per-task=8" in source
        assert "#SBATCH --gres=gpu:1" in source
        assert "export CUDA_VISIBLE_DEVICES=" in source
        assert "export OAS_SCHEDULER_RESERVED_GPU_COUNT=1" in source
        assert "oas_production_materialization_v2.json" in source
