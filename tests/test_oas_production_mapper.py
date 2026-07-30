"""Test authenticated OAS production mapping on a complete synthetic gzip."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import io
import json
import stat
from pathlib import Path

import pytest

import antibody_pretraining.preprocessing as preprocessing_module
import antibody_pretraining.production_materialization as record_module
from antibody_pretraining.production_materialization import iter_map_records

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/map_oas_production_records.py"
FILTER = ROOT / "protocols/oas_sequence_filter_v1.json"
SEQUENCE = "ACDEFGHIKLMNPQRSTVWY" * 4


def _load_module():
    spec = importlib.util.spec_from_file_location("oas_production_mapper", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _self_hashed(value: dict[str, object], field: str) -> bytes:
    value[field] = hashlib.sha256(_canonical(value)).hexdigest()
    return _canonical(value)


def _oas_bytes(
    metadata_field: str | None = None,
) -> tuple[bytes, bytes, bytes, bytes]:
    metadata_buffer = io.StringIO(newline="")
    csv.writer(metadata_buffer, lineterminator="\n").writerow(
        [
            metadata_field
            if metadata_field is not None
            else json.dumps({"Chain": "Heavy", "Unique sequences": 2})
        ]
    )
    metadata = metadata_buffer.getvalue().encode("utf-8")
    header_fields = (
        "sequence_alignment_aa",
        "locus",
        "stop_codon",
        "vj_in_frame",
        "productive",
        "ANARCI_status",
    )
    body_buffer = io.StringIO(newline="")
    writer = csv.writer(body_buffer, lineterminator="\n")
    writer.writerow(header_fields)
    writer.writerow((SEQUENCE, "H", "F", "T", "T", ""))
    writer.writerow((SEQUENCE[::-1], "H", "F", "T", "T", "deletions"))
    body = body_buffer.getvalue().encode("utf-8")
    header, rows = body.split(b"\n", 1)
    header += b"\n"
    uncompressed = metadata + header + rows
    compressed_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed_buffer, mode="wb", mtime=0) as handle:
        handle.write(uncompressed)
    return compressed_buffer.getvalue(), uncompressed, metadata, header


def _fixture(tmp_path: Path, *, metadata_field: str | None = None):
    module = _load_module()
    download_root = tmp_path / "downloads"
    relative = "study/csv/unit.csv.gz"
    compressed, uncompressed, metadata, header = _oas_bytes(metadata_field)
    raw_path = download_root / "raw/unpaired" / relative
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(compressed)

    unit = {
        "canonical_url": f"https://example.test/{relative}",
        "expected_selected_records": 2,
        "relative_unit": relative,
        "selection_fraction_denominator": 1,
        "selection_fraction_numerator": 1,
        "selection_hash_namespace": "1" * 64,
        "source_records": 2,
        "split": "train",
        "stratum": "human_heavy",
        "study": "study",
    }
    plan = {
        "format_version": "oas_unpaired_corpus_plan_v1",
        "selected_data_units": [unit],
    }
    plan_path = tmp_path / "plan.json"
    plan_bytes = _self_hashed(plan, "plan_sha256")
    plan_path.write_bytes(plan_bytes)
    plan_file_sha256 = hashlib.sha256(plan_bytes).hexdigest()

    receipt = {
        "admission_status": "downloaded_and_admitted",
        "canonical_url": unit["canonical_url"],
        "compressed_bytes": len(compressed),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "corpus_plan_membership": module._expected_membership(unit),
        "corpus_plan_payload_sha256": plan["plan_sha256"],
        "data_row_count": 2,
        "format_version": "oas_data_unit_download_receipt_v1",
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "relative_unit": relative,
        "source_class": "unpaired",
        "test_access": {
            "downstream_heldout_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
        "uncompressed_bytes": len(uncompressed),
        "uncompressed_sha256": hashlib.sha256(uncompressed).hexdigest(),
        "unique_sequences": 2,
    }
    receipt_path = download_root / "receipts/unpaired" / f"{relative}.receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(_self_hashed(receipt, "receipt_sha256"))

    raw_inventory_sha256 = "a" * 64
    download_audit = {
        "audit_passed": True,
        "counts": {"compressed_bytes": len(compressed), "source_records": 2},
        "format_version": "oas_download_completion_audit_v1",
        "plan_file_sha256": plan_file_sha256,
        "plan_payload_sha256": plan["plan_sha256"],
        "raw_inventory_sha256": raw_inventory_sha256,
        "receipt_count": 1,
        "test_access": {
            "downstream_heldout_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    audit_path = tmp_path / "download-audit.json"
    audit_bytes = _self_hashed(download_audit, "receipt_sha256")
    audit_path.write_bytes(audit_bytes)

    filter_path = tmp_path / "filter.json"
    filter_bytes = FILTER.read_bytes()
    filter_path.write_bytes(filter_bytes)
    protocol = {
        "execution": {"map_task_count": 1},
        "format_version": "oas_production_materialization_protocol_v1",
        "inputs": {
            "download_completion_audit_file_sha256": hashlib.sha256(
                audit_bytes
            ).hexdigest(),
            "download_completion_audit_payload_sha256": download_audit[
                "receipt_sha256"
            ],
            "download_receipt_count": 1,
            "filter_policy_file_sha256": hashlib.sha256(filter_bytes).hexdigest(),
            "plan_file_sha256": plan_file_sha256,
            "plan_payload_sha256": plan["plan_sha256"],
            "raw_compressed_bytes": len(compressed),
            "raw_inventory_sha256": raw_inventory_sha256,
            "source_record_count": 2,
        },
        "resource_policy": {"shared_storage_minimum_free_bytes": 0},
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_bytes = _canonical(protocol)
    protocol_path.write_bytes(protocol_bytes)
    return {
        "audit": audit_path,
        "download_root": download_root,
        "output": tmp_path / "output",
        "plan": plan_path,
        "protocol": protocol_path,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "filter": filter_path,
    }


def _make_writable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(stat.S_IRWXU)
    root.chmod(stat.S_IRWXU)


def test_complete_map_task_streams_and_publishes_exact_records(tmp_path: Path) -> None:
    module = _load_module()
    fixture = _fixture(tmp_path)
    receipt = module.map_production_task(
        task_id=0,
        task_count=1,
        download_root=fixture["download_root"],
        plan_path=fixture["plan"],
        download_audit_path=fixture["audit"],
        filter_policy_path=fixture["filter"],
        protocol_path=fixture["protocol"],
        protocol_file_sha256=fixture["protocol_sha256"],
        output_root=fixture["output"],
        expected_controller_source_sha256=hashlib.sha256(
            SCRIPT.read_bytes()
        ).hexdigest(),
        expected_preprocessing_source_sha256=hashlib.sha256(
            Path(preprocessing_module.__file__).read_bytes()
        ).hexdigest(),
        expected_record_source_sha256=hashlib.sha256(
            Path(record_module.__file__).read_bytes()
        ).hexdigest(),
    )
    published = fixture["output"] / "map/mapper-0000"
    try:
        assert receipt["counts"] == {
            "accepted_records": 2,
            "plan_fraction_rejected": 0,
            "plan_fraction_selected": 2,
            "source_rows": 2,
        }
        assert receipt["claim_boundary"]["large_scale_pretraining_authorized"] is False
        assert receipt["test_access"] == module.TEST_ACCESS
        assert stat.S_IMODE(published.stat().st_mode) == 0o555
        bucket_files = sorted((published / "buckets").glob("*.bin"))
        records = []
        for path in bucket_files:
            with path.open("rb") as handle:
                records.extend(iter_map_records(handle))
        assert sorted(record.sequence for record in records) == sorted(
            (SEQUENCE, SEQUENCE[::-1])
        )
        assert all(record.study_id == 0 for record in records)
    finally:
        _make_writable(published)


def test_strict_json_and_existing_publication_fail_closed(tmp_path: Path) -> None:
    module = _load_module()
    with pytest.raises(module.ProductionMapError, match="strict ASCII JSON"):
        module._strict_json_bytes(b'{"a":1,"a":2}', "duplicate")
    fixture = _fixture(tmp_path)
    published = fixture["output"] / "map/mapper-0000"
    published.mkdir(parents=True)
    with pytest.raises(module.ProductionMapError, match="already exists"):
        module.map_production_task(
            task_id=0,
            task_count=1,
            download_root=fixture["download_root"],
            plan_path=fixture["plan"],
            download_audit_path=fixture["audit"],
            filter_policy_path=fixture["filter"],
            protocol_path=fixture["protocol"],
            protocol_file_sha256=fixture["protocol_sha256"],
            output_root=fixture["output"],
            expected_controller_source_sha256=hashlib.sha256(
                SCRIPT.read_bytes()
            ).hexdigest(),
            expected_preprocessing_source_sha256=hashlib.sha256(
                Path(preprocessing_module.__file__).read_bytes()
            ).hexdigest(),
            expected_record_source_sha256=hashlib.sha256(
                Path(record_module.__file__).read_bytes()
            ).hexdigest(),
        )


def test_v3_maps_only_independently_bound_age_nan(tmp_path: Path) -> None:
    module = _load_module()
    field = '{"Age":NaN,"Chain":"Heavy","Unique sequences":2}'
    fixture = _fixture(tmp_path, metadata_field=field)
    relative = "study/csv/unit.csv.gz"
    raw_path = fixture["download_root"] / "raw/unpaired" / relative
    with gzip.open(raw_path, "rt", encoding="utf-8", newline="") as handle:
        metadata_line = handle.readline()
    normalized = {"Age": None, "Chain": "Heavy", "Unique sequences": 2}
    normalization = {
        "affected_units": [
            {
                "decoded_metadata_sha256": hashlib.sha256(
                    field.encode("utf-8")
                ).hexdigest(),
                "normalized_metadata_sha256": hashlib.sha256(
                    _canonical(normalized)
                ).hexdigest(),
                "raw_metadata_line_sha256": hashlib.sha256(
                    metadata_line.encode("utf-8")
                ).hexdigest(),
                "relative_unit": relative,
            }
        ],
        "format_version": "oas_metadata_normalization_protocol_v1",
        "normalization": {"affected_unit_count": 1},
    }
    normalization_path = tmp_path / "normalization.json"
    normalization_raw = _canonical(normalization)
    normalization_path.write_bytes(normalization_raw)
    normalization_sha256 = hashlib.sha256(normalization_raw).hexdigest()
    normalization_audit = {
        "audit_passed": True,
        "authorization": {
            "large_scale_pretraining": False,
            "production_corpus_use": False,
            "scientific_claim": False,
        },
        "counts": {
            "affected_units": 1,
            "scanned_units": 1,
            "strict_units": 0,
        },
        "format_version": "oas_metadata_normalization_independent_audit_v1",
        "normalization_protocol_file_sha256": normalization_sha256,
        "test_access": module.TEST_ACCESS,
    }
    normalization_audit_path = tmp_path / "normalization-audit.json"
    normalization_audit_raw = _self_hashed(
        normalization_audit,
        "receipt_sha256",
    )
    normalization_audit_path.write_bytes(normalization_audit_raw)
    protocol = json.loads(fixture["protocol"].read_text(encoding="ascii"))
    protocol["format_version"] = "oas_production_materialization_protocol_v3"
    protocol["inputs"].update(
        {
            "metadata_normalization_audit_file_sha256": hashlib.sha256(
                normalization_audit_raw
            ).hexdigest(),
            "metadata_normalization_audit_payload_sha256": normalization_audit[
                "receipt_sha256"
            ],
            "metadata_normalization_protocol_file_sha256": normalization_sha256,
        }
    )
    protocol_raw = _canonical(protocol)
    fixture["protocol"].write_bytes(protocol_raw)
    fixture["protocol_sha256"] = hashlib.sha256(protocol_raw).hexdigest()
    receipt = module.map_production_task(
        task_id=0,
        task_count=1,
        download_root=fixture["download_root"],
        plan_path=fixture["plan"],
        download_audit_path=fixture["audit"],
        filter_policy_path=fixture["filter"],
        protocol_path=fixture["protocol"],
        protocol_file_sha256=fixture["protocol_sha256"],
        output_root=fixture["output"],
        expected_controller_source_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        expected_preprocessing_source_sha256=hashlib.sha256(
            Path(preprocessing_module.__file__).read_bytes()
        ).hexdigest(),
        expected_record_source_sha256=hashlib.sha256(
            Path(record_module.__file__).read_bytes()
        ).hexdigest(),
        normalization_protocol_path=normalization_path,
        normalization_audit_path=normalization_audit_path,
    )
    published = fixture["output"] / "map/mapper-0000"
    try:
        assert receipt["metadata_normalization_counts"] == {"age_nan_to_null": 1}
        assert receipt["unit_receipts"][0]["metadata_normalization"] == (
            "age_nan_to_null"
        )
        assert receipt["audit_bindings"][
            "metadata_normalization_audit_payload_sha256"
        ] == normalization_audit["receipt_sha256"]
    finally:
        _make_writable(published)


def test_metadata_exception_rejects_all_unregistered_nonfinite_values() -> None:
    module = _load_module()
    with pytest.raises(module.ProductionMapError, match="unauthorized"):
        module._decode_oas_metadata(
            '{"Age":NaN}',
            metadata_line='"{""Age"":NaN}"\n',
            relative_unit="study/csv/unit.csv.gz",
            exceptions={},
        )
    for payload in ('{"Age":Infinity}', '{"Age":[NaN]}', '{"Other":NaN}'):
        with pytest.raises(module.ProductionMapError, match="unauthorized"):
            module._decode_oas_metadata(
                payload,
                metadata_line=payload + "\n",
                relative_unit="study/csv/unit.csv.gz",
                exceptions={
                    "study/csv/unit.csv.gz": {
                        "decoded_metadata_sha256": "0" * 64,
                        "normalized_metadata_sha256": "0" * 64,
                        "raw_metadata_line_sha256": "0" * 64,
                        "relative_unit": "study/csv/unit.csv.gz",
                    }
                },
            )
