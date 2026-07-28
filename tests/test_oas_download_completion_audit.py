from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_oas_download_completion.py"
SLURM = ROOT / "slurm/oas_download_completion_audit.sbatch"


def _load_module():
    spec = importlib.util.spec_from_file_location("oas_completion", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict[str, object], hash_field: str) -> str:
    module = _load_module()
    value[hash_field] = hashlib.sha256(module._canonical_json_bytes(value)).hexdigest()
    payload = module._canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path):
    module = _load_module()
    root = tmp_path / "downloads"
    authority_sha256 = "a" * 64
    units = []
    receipt_hashes = {}
    for index, split in enumerate(("train", "validation")):
        relative = f"study/csv/unit{index}.csv.gz"
        raw = f"raw-{index}".encode("ascii")
        raw_path = root / "raw/unpaired" / relative
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw)
        unit = {
            "canonical_url": f"https://example.test/{relative}",
            "expected_selected_records": index + 1,
            "relative_unit": relative,
            "selection_fraction_denominator": 1,
            "selection_fraction_numerator": 1,
            "selection_hash_namespace": str(index) * 64,
            "source_records": index + 2,
            "split": split,
            "stratum": "human_heavy",
            "study": "study",
        }
        units.append(unit)
        receipt = {
            "admission_status": "downloaded_and_admitted",
            "authority_manifest_payload_sha256": authority_sha256,
            "canonical_url": unit["canonical_url"],
            "compressed_bytes": len(raw),
            "compressed_sha256": hashlib.sha256(raw).hexdigest(),
            "controller_source_sha256": "b" * 64,
            "corpus_plan_membership": module._expected_membership(unit),
            "corpus_plan_payload_sha256": "pending",
            "data_row_count": unit["source_records"],
            "downloader_module_sha256": "c" * 64,
            "format_version": "oas_data_unit_download_receipt_v1",
            "relative_unit": relative,
            "source_class": "unpaired",
            "test_access": module.TEST_ACCESS,
            "unique_sequences": unit["source_records"],
        }
        receipt_hashes[relative] = receipt
    plan = {
        "authority": {"manifest_payload_sha256": authority_sha256},
        "format_version": "oas_unpaired_corpus_plan_v1",
        "selected_data_units": units,
    }
    plan_path = tmp_path / "plan.json"
    plan_file_sha256 = _write_json(plan_path, plan, "plan_sha256")
    plan_payload_sha256 = plan["plan_sha256"]
    summaries = []
    for index, unit in enumerate(units):
        receipt = receipt_hashes[unit["relative_unit"]]
        receipt["corpus_plan_payload_sha256"] = plan_payload_sha256
        receipt_path = (
            root
            / "receipts/unpaired"
            / f"{unit['relative_unit']}.receipt.json"
        )
        _write_json(receipt_path, receipt, "receipt_sha256")
        summary = {
            "authority_manifest_payload_sha256": authority_sha256,
            "corpus_plan_file_sha256": plan_file_sha256,
            "corpus_plan_payload_sha256": plan_payload_sha256,
            "failed_count": 0,
            "failures": [],
            "format_version": "oas_download_task_summary_v1",
            "selected_count": 1,
            "source_class": "unpaired",
            "split": "all",
            "success_count": 1,
            "successful_receipt_sha256": [receipt["receipt_sha256"]],
            "task_count": 2,
            "task_id": index,
            "test_access": module.TEST_ACCESS,
        }
        summary_path = root / "summaries" / f"task_{index}.json"
        _write_json(summary_path, summary, "summary_sha256")
        summaries.append(summary_path)
    return root, plan_path, plan_file_sha256, plan_payload_sha256, summaries


def test_complete_download_audit_rehashes_every_raw_file(tmp_path: Path) -> None:
    module = _load_module()
    root, plan, plan_file, plan_payload, summaries = _fixture(tmp_path)
    result = module.audit_download_completion(
        download_root=root,
        plan_path=plan,
        expected_plan_file_sha256=plan_file,
        expected_plan_payload_sha256=plan_payload,
        summary_paths=summaries,
        expected_task_count=2,
        expected_auditor_source_sha256=module._file_sha256(SCRIPT),
    )
    assert result["audit_passed"] is True
    assert result["receipt_count"] == 2
    assert result["counts"]["train_data_units"] == 1
    assert result["counts"]["validation_data_units"] == 1
    assert result["claim_boundary"]["dense_materialization_complete"] is False
    assert result["claim_boundary"]["production_pretraining_authorized"] is False


def test_complete_download_audit_rejects_raw_tampering(tmp_path: Path) -> None:
    module = _load_module()
    root, plan, plan_file, plan_payload, summaries = _fixture(tmp_path)
    raw = root / "raw/unpaired/study/csv/unit0.csv.gz"
    raw.write_bytes(raw.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="byte count"):
        module.audit_download_completion(
            download_root=root,
            plan_path=plan,
            expected_plan_file_sha256=plan_file,
            expected_plan_payload_sha256=plan_payload,
            summary_paths=summaries,
            expected_task_count=2,
            expected_auditor_source_sha256=module._file_sha256(SCRIPT),
        )


def test_completion_slurm_hides_scheduler_gpu_and_pins_four_summaries() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "export CUDA_VISIBLE_DEVICES=" in source
    assert "audit_oas_download_completion.py" in source
    assert source.count("--summary") == 4
    assert "--task-count 4" in source
    assert "--auditor-source-sha256" in source
