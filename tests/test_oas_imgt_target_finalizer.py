"""Test complete OAS fixed-IMGT target finalization."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import stat

import pytest

import antibody_pretraining.imgt_position as imgt_position_module
import antibody_pretraining.oas_imgt_export as imgt_export_module
import antibody_pretraining.preprocessing as preprocessing_module
import antibody_pretraining.production_materialization as record_module


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/export_oas_imgt_targets.py"
FINALIZER = ROOT / "scripts/finalize_oas_imgt_targets.py"
MAPPER = ROOT / "scripts/map_oas_production_records.py"
MAPPER_TEST = ROOT / "tests/test_oas_production_mapper.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(stat.S_IRWXU)
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(stat.S_IRWXU)
        else:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _complete_task(tmp_path: Path):
    fixture_module = _load(MAPPER_TEST, f"oas_mapper_finalizer_fixture_{tmp_path.name}")
    mapper = _load(MAPPER, f"oas_mapper_for_finalizer_{tmp_path.name}")
    controller = _load(CONTROLLER, f"oas_imgt_controller_finalizer_{tmp_path.name}")
    fixture = fixture_module._fixture(tmp_path)
    mapper.map_production_task(
        task_id=0,
        task_count=1,
        download_root=fixture["download_root"],
        plan_path=fixture["plan"],
        download_audit_path=fixture["audit"],
        filter_policy_path=fixture["filter"],
        protocol_path=fixture["protocol"],
        protocol_file_sha256=fixture["protocol_sha256"],
        output_root=fixture["output"],
        expected_controller_source_sha256=hashlib.sha256(MAPPER.read_bytes()).hexdigest(),
        expected_preprocessing_source_sha256=hashlib.sha256(
            Path(preprocessing_module.__file__).read_bytes()
        ).hexdigest(),
        expected_record_source_sha256=hashlib.sha256(
            Path(record_module.__file__).read_bytes()
        ).hexdigest(),
    )
    original_root = fixture["output"] / "map/mapper-0000"
    output_root = tmp_path / "imgt-output"
    source_sha256 = {
        "controller": hashlib.sha256(CONTROLLER.read_bytes()).hexdigest(),
        "imgt_position": hashlib.sha256(
            Path(imgt_position_module.__file__).read_bytes()
        ).hexdigest(),
        "imgt_record": hashlib.sha256(
            Path(imgt_export_module.__file__).read_bytes()
        ).hexdigest(),
        "map_record": hashlib.sha256(
            Path(record_module.__file__).read_bytes()
        ).hexdigest(),
        "preprocessing": hashlib.sha256(
            Path(preprocessing_module.__file__).read_bytes()
        ).hexdigest(),
        "replay_mapper": hashlib.sha256(MAPPER.read_bytes()).hexdigest(),
    }
    controller.export_imgt_task(
        task_id=0,
        task_count=1,
        download_root=fixture["download_root"],
        plan_path=fixture["plan"],
        download_audit_path=fixture["audit"],
        filter_policy_path=fixture["filter"],
        materialization_protocol_path=fixture["protocol"],
        materialization_protocol_file_sha256=fixture["protocol_sha256"],
        normalization_protocol_path=None,
        normalization_audit_path=None,
        original_mapper_receipt_path=original_root / "receipt.json",
        mapper_path=MAPPER,
        output_root=output_root,
        expected_controller_source_sha256=source_sha256["controller"],
        expected_imgt_position_source_sha256=source_sha256["imgt_position"],
        expected_imgt_record_source_sha256=source_sha256["imgt_record"],
        expected_mapper_source_sha256=source_sha256["replay_mapper"],
        expected_preprocessing_source_sha256=source_sha256["preprocessing"],
        expected_record_source_sha256=source_sha256["map_record"],
    )
    return fixture_module, original_root, output_root, source_sha256


def test_global_finalizer_authenticates_exhaustive_two_way_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.delenv("OAS_SCHEDULER_RESERVED_GPU_COUNT", raising=False)
    fixture_module, original_root, output_root, source_sha256 = _complete_task(
        tmp_path
    )
    finalizer = _load(FINALIZER, "oas_imgt_finalizer_complete")
    try:
        receipt = finalizer.finalize_imgt_export(
            output_root=output_root,
            output_path=output_root / "global_receipt.json",
            task_count=1,
            worker_count=1,
            expected_accepted_records=2,
            expected_sources=source_sha256,
            controller_source_sha256=hashlib.sha256(FINALIZER.read_bytes()).hexdigest(),
        )
        assert receipt["counts"]["accepted_records"] == 2
        assert receipt["counts"]["representable_records"] == 2
        assert receipt["counts"]["unusual_records"] == 0
        assert receipt["claim_boundary"]["global_task_set_complete"] is True
        assert receipt["claim_boundary"]["near_identity_search_complete"] is False
        assert receipt["test_access"] == finalizer.TEST_ACCESS
        assert stat.S_IMODE(output_root.stat().st_mode) == 0o555
        assert stat.S_IMODE((output_root / "global_receipt.json").stat().st_mode) == 0o400
    finally:
        fixture_module._make_writable(original_root)
        _make_writable(output_root)


def test_global_finalizer_rejects_wrong_authoritative_total(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.delenv("OAS_SCHEDULER_RESERVED_GPU_COUNT", raising=False)
    fixture_module, original_root, output_root, source_sha256 = _complete_task(
        tmp_path
    )
    finalizer = _load(FINALIZER, "oas_imgt_finalizer_wrong_total")
    try:
        with pytest.raises(
            finalizer.OASIMGTFinalizeError,
            match="accepted-record count differs",
        ):
            finalizer.finalize_imgt_export(
                output_root=output_root,
                output_path=output_root / "global_receipt.json",
                task_count=1,
                worker_count=1,
                expected_accepted_records=3,
                expected_sources=source_sha256,
                controller_source_sha256=hashlib.sha256(
                    FINALIZER.read_bytes()
                ).hexdigest(),
            )
    finally:
        fixture_module._make_writable(original_root)
        _make_writable(output_root)
