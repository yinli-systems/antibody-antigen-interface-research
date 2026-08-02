"""Test the narrow independent OAS metadata-normalization audit."""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_oas_metadata_normalization_independent.py"
PROTOCOL = ROOT / "protocols/oas_metadata_normalization_v1.json"


def _load():
    specification = importlib.util.spec_from_file_location(
        "independent_metadata_normalization", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _self_hashed(value: dict[str, object], field: str) -> bytes:
    value[field] = hashlib.sha256(_canonical(value)).hexdigest()
    return _canonical(value)


def _metadata_line(field: str) -> bytes:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerow([field])
    return buffer.getvalue().encode("utf-8")


def _gzip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(payload)
    return buffer.getvalue()


def _fixture(tmp_path: Path):
    module = _load()
    root = tmp_path / "download"
    units = []
    affected = []
    for index, field in enumerate(
        (
            '{"Age":42,"Chain":"Heavy","Unique sequences":1}',
            '{"Age":NaN,"Chain":"Heavy","Unique sequences":1}',
        )
    ):
        relative = f"study/csv/unit-{index}.csv.gz"
        metadata = _metadata_line(field)
        compressed = _gzip(metadata + b"sequence_alignment_aa\n")
        raw = root / "raw/unpaired" / relative
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(compressed)
        unit = {"relative_unit": relative, "source_records": 1}
        units.append(unit)
        receipt = {
            "data_row_count": 1,
            "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
            "relative_unit": relative,
        }
        receipt_path = root / "receipts/unpaired" / f"{relative}.receipt.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_bytes(_self_hashed(receipt, "receipt_sha256"))
        if index == 1:
            value, normalized = module._decode_metadata(field)
            assert normalized
            affected.append(
                {
                    "decoded_metadata_sha256": hashlib.sha256(
                        field.encode("utf-8")
                    ).hexdigest(),
                    "normalized_metadata_sha256": hashlib.sha256(
                        _canonical(value)
                    ).hexdigest(),
                    "raw_metadata_line_sha256": hashlib.sha256(metadata).hexdigest(),
                    "relative_unit": relative,
                }
            )
    plan = {
        "format_version": module.PLAN_FORMAT,
        "selected_data_units": units,
    }
    plan_path = tmp_path / "plan.json"
    plan_raw = _self_hashed(plan, "plan_sha256")
    plan_path.write_bytes(plan_raw)
    audit = {
        "audit_passed": True,
        "format_version": module.AUDIT_FORMAT,
    }
    audit_path = tmp_path / "audit.json"
    audit_raw = _self_hashed(audit, "receipt_sha256")
    audit_path.write_bytes(audit_raw)
    protocol = {
        "affected_units": affected,
        "authority": {
            "download_completion_audit_file_sha256": hashlib.sha256(
                audit_raw
            ).hexdigest(),
            "download_completion_audit_payload_sha256": audit["receipt_sha256"],
            "plan_file_sha256": hashlib.sha256(plan_raw).hexdigest(),
            "plan_payload_sha256": plan["plan_sha256"],
            "selected_data_unit_count": 2,
            "source_record_count": 2,
        },
        "authorization": {
            "large_scale_pretraining": False,
            "production_corpus_use": False,
            "scientific_claim": False,
        },
        "claim_boundary": "metadata only",
        "format_version": module.PROTOCOL_FORMAT,
        "normalization": {
            "affected_unit_count": 1,
            "strict_unit_count": 1,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(_canonical(protocol))
    return module, root, plan_path, audit_path, protocol_path


def test_frozen_protocol_binds_only_nine_age_nan_units() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="ascii"))
    assert value["format_version"] == "oas_metadata_normalization_protocol_v1"
    assert value["normalization"] == {
        "affected_unit_count": 9,
        "allowed_constant": "NaN",
        "allowed_json_path": ["Age"],
        "replacement_json_value": None,
        "strict_unit_count": 6364,
    }
    assert len(value["affected_units"]) == 9
    assert len({item["relative_unit"] for item in value["affected_units"]}) == 9
    assert all(
        item["relative_unit"].startswith("Kuri-Cervantes_2020/csv/")
        for item in value["affected_units"]
    )


def test_decoder_accepts_only_top_level_age_nan() -> None:
    module = _load()
    strict, normalized = module._decode_metadata('{"Age":42,"Chain":"Heavy"}')
    assert strict["Age"] == 42
    assert normalized is False
    missing, normalized = module._decode_metadata('{"Age":NaN,"Chain":"Heavy"}')
    assert missing["Age"] is None
    assert normalized is True
    for malformed in (
        '{"Age":Infinity}',
        '{"Age":-Infinity}',
        '{"Age":[NaN]}',
        '{"Other":NaN}',
        '{"Age":NaN,"Other":NaN}',
    ):
        with pytest.raises(module.MetadataAuditError, match="unauthorized"):
            module._decode_metadata(malformed)
    with pytest.raises(module.MetadataAuditError, match="Duplicate"):
        module._decode_metadata('{"Age":NaN,"Age":NaN}')


def test_complete_two_unit_audit_and_create_only_receipt(tmp_path: Path) -> None:
    module, root, plan, audit, protocol = _fixture(tmp_path)
    result = module.audit_metadata(
        download_root=root,
        plan_path=plan,
        download_audit_path=audit,
        protocol_path=protocol,
        worker_count=2,
    )
    assert result["audit_passed"] is True
    assert result["counts"] == {
        "affected_units": 1,
        "scanned_units": 2,
        "strict_units": 1,
    }
    output = tmp_path / "receipt.json"
    module._publish(output, result)
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        module._publish(output, result)


def test_independent_source_does_not_import_mapper() -> None:
    imported = []
    for node in ast.walk(ast.parse(SCRIPT.read_text(encoding="ascii"))):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert "scripts.map_oas_production_records" not in imported
    assert "antibody_pretraining.preprocessing" not in imported
