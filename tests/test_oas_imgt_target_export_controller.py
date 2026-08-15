"""Test authenticated mapper replay into fixed-IMGT target evidence."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import stat

import antibody_pretraining.imgt_position as imgt_position_module
import antibody_pretraining.oas_imgt_export as imgt_export_module
import antibody_pretraining.preprocessing as preprocessing_module
import antibody_pretraining.production_materialization as record_module


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/export_oas_imgt_targets.py"
MAPPER = ROOT / "scripts/map_oas_production_records.py"
MAPPER_TEST = ROOT / "tests/test_oas_production_mapper.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_comparison_normalizes_only_explicit_zero_counts() -> None:
    controller = _load(CONTROLLER, "oas_imgt_controller_zero_count_test")
    original = {
        "unit_index": 3,
        "relative_unit": "study/unit.csv.gz",
        "counts": {"source_rows": 1, "accepted_records": 1},
        "status_flag_counts": {"shorter": 1},
        "metadata_normalization": "strict_json",
    }
    replayed = {
        **original,
        "counts": {
            **original["counts"],
            "plan_fraction_rejected": 0,
        },
    }
    changed = {
        **replayed,
        "counts": {
            **replayed["counts"],
            "plan_fraction_selected": 1,
        },
    }

    assert controller._canonicalize_unit_receipt_for_replay(replayed) == (
        controller._canonicalize_unit_receipt_for_replay(original)
    )
    assert controller._canonicalize_unit_receipt_for_replay(changed) != (
        controller._canonicalize_unit_receipt_for_replay(original)
    )


def test_complete_authenticated_replay_publishes_two_way_domain(tmp_path: Path) -> None:
    fixture_module = _load(MAPPER_TEST, "oas_mapper_fixture")
    mapper = _load(MAPPER, "oas_mapper_for_imgt_test")
    controller = _load(CONTROLLER, "oas_imgt_controller_test")
    fixture = fixture_module._fixture(tmp_path)
    original = mapper.map_production_task(
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
    try:
        receipt = controller.export_imgt_task(
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
            expected_controller_source_sha256=hashlib.sha256(
                CONTROLLER.read_bytes()
            ).hexdigest(),
            expected_imgt_position_source_sha256=hashlib.sha256(
                Path(imgt_position_module.__file__).read_bytes()
            ).hexdigest(),
            expected_imgt_record_source_sha256=hashlib.sha256(
                Path(imgt_export_module.__file__).read_bytes()
            ).hexdigest(),
            expected_mapper_source_sha256=hashlib.sha256(MAPPER.read_bytes()).hexdigest(),
            expected_preprocessing_source_sha256=hashlib.sha256(
                Path(preprocessing_module.__file__).read_bytes()
            ).hexdigest(),
            expected_record_source_sha256=hashlib.sha256(
                Path(record_module.__file__).read_bytes()
            ).hexdigest(),
        )
        published = output_root / "imgt-map/mapper-0000"
        assert original["counts"]["accepted_records"] == 2
        assert receipt["counts"]["accepted_records"] == 2
        assert receipt["counts"]["representable_records"] == 2
        assert receipt["counts"]["unusual_records"] == 0
        assert receipt["claim_boundary"]["fixed_imgt_task_domain_complete"] is True
        assert receipt["claim_boundary"]["near_identity_search_complete"] is False
        assert receipt["test_access"] == controller.TEST_ACCESS
        assert receipt["output_inventory"]["representable"]["records"] == 2
        assert receipt["output_inventory"]["unusual"]["records"] == 0
        assert stat.S_IMODE(published.stat().st_mode) == 0o555
    finally:
        fixture_module._make_writable(original_root)
        published = output_root / "imgt-map/mapper-0000"
        if published.exists():
            fixture_module._make_writable(published)
