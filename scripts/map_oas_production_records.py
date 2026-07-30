#!/usr/bin/env python3
"""Map one authenticated OAS plan shard into exact-dedup binary buckets."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import socket
import sys
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import antibody_pretraining.preprocessing as preprocessing_module
import antibody_pretraining.production_materialization as record_module
from antibody_pretraining.preprocessing import (
    CompiledUnpairedRowEvaluator,
    load_filter_policy,
    selected_by_plan_fraction,
)
from antibody_pretraining.production_materialization import (
    BUCKET_COUNT,
    ProductionMapRecord,
    pack_map_record,
)

FORMAT_VERSION = "oas_production_map_task_receipt_v1"
TEST_ACCESS = {
    "downstream_metrics_used": False,
    "sabdab2_heldout_accessed": False,
}


class ProductionMapError(RuntimeError):
    """Report an authenticated-input or production-map failure."""


class _DigestingRawReader(io.RawIOBase):
    """Hash and count bytes while serving one forward-only binary stream."""

    def __init__(self, source: BinaryIO) -> None:
        super().__init__()
        self.source = source
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def readable(self) -> bool:
        """Declare the stream readable for BufferedReader."""
        return True

    def readinto(self, buffer: bytearray) -> int:
        """Read once from the source while updating the evidence digest."""
        payload = self.source.read(len(buffer))
        if not payload:
            return 0
        size = len(payload)
        buffer[:size] = payload
        self.digest.update(payload)
        self.byte_count += size
        return size


class _BucketWriter:
    """Lazily create fixed-record bucket files and track exact counts."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.handles: dict[int, BinaryIO] = {}
        self.counts: Counter[int] = Counter()

    def write(self, bucket: int, payload: bytes) -> None:
        """Append one validated record to its first-digest-byte bucket."""
        if not 0 <= bucket < BUCKET_COUNT:
            raise ProductionMapError("Bucket ID is outside the frozen domain")
        handle = self.handles.get(bucket)
        if handle is None:
            path = self.root / f"bucket-{bucket:03d}.bin"
            handle = path.open("xb")
            self.handles[bucket] = handle
        handle.write(payload)
        self.counts[bucket] += 1

    def close(self) -> None:
        """Flush and fsync every opened bucket before publication."""
        first_error: Exception | None = None
        for handle in self.handles.values():
            try:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            except OSError as error:
                if first_error is None:
                    first_error = error
        self.handles.clear()
        if first_error is not None:
            raise first_error


def _reject_constant(value: str) -> None:
    """Reject non-finite JSON constants."""
    raise ValueError(f"Non-finite JSON constant: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys at every nesting level."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    """Decode one duplicate-free, finite, ASCII JSON object."""
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ProductionMapError(f"{label} must be strict ASCII JSON") from error
    if not isinstance(value, dict):
        raise ProductionMapError(f"{label} must contain one JSON object")
    return value


def _strict_json_text(payload: str, label: str) -> dict[str, Any]:
    """Decode duplicate-free finite metadata JSON already decoded as UTF-8."""
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ProductionMapError(f"{label} must be strict JSON") from error
    if not isinstance(value, dict):
        raise ProductionMapError(f"{label} must contain one JSON object")
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    """Encode one canonical create-only evidence payload."""
    try:
        text = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ProductionMapError("Evidence payload is not canonical JSON") from error
    return (text + "\n").encode("ascii")


def _file_sha256(path: Path) -> str:
    """Hash a regular file without loading it into memory."""
    if not path.is_file() or path.is_symlink():
        raise ProductionMapError("Hashed input must be a regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    """Read one regular strict JSON input and preserve its exact bytes."""
    if not path.is_file() or path.is_symlink():
        raise ProductionMapError(f"{label} must be a regular non-symlink file")
    payload = path.read_bytes()
    return _strict_json_bytes(payload, label), payload


def _verify_self_hash(payload: Mapping[str, Any], field: str, label: str) -> str:
    """Recompute a canonical JSON self-hash after removing its hash field."""
    unsigned = dict(payload)
    claimed = unsigned.pop(field, None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise ProductionMapError(f"{label} self-hash differs")
    return observed


def _safe_relative_unit(value: object) -> str:
    """Validate one canonical relative OAS gzip path."""
    if not isinstance(value, str) or not value:
        raise ProductionMapError("Relative data-unit path must be nonempty text")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or str(relative) != value
        or not value.endswith(".csv.gz")
    ):
        raise ProductionMapError("Relative data-unit path is unsafe")
    return value


def _safe_input_path(root: Path, relative: str) -> Path:
    """Resolve a non-symlink file below the authenticated download root."""
    resolved_root = root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    current = resolved_root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ProductionMapError("Input path contains a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ProductionMapError("Input path escapes the authenticated root")
    if not resolved.is_file():
        raise ProductionMapError("Input path is not a regular file")
    return resolved


def _single_csv_field(line: str, label: str) -> str:
    """Parse one physical CSV line that must contain exactly one field."""
    try:
        row = next(csv.reader([line]))
    except (csv.Error, StopIteration) as error:
        raise ProductionMapError(f"{label} is not a valid CSV record") from error
    if len(row) != 1:
        raise ProductionMapError(f"{label} must contain one CSV field")
    return row[0]


def _expected_membership(unit: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact plan fields that every download receipt must bind."""
    return {
        "expected_selected_records": unit["expected_selected_records"],
        "selection_fraction_denominator": unit["selection_fraction_denominator"],
        "selection_fraction_numerator": unit["selection_fraction_numerator"],
        "selection_hash_namespace": unit["selection_hash_namespace"],
        "split": unit["split"],
        "stratum": unit["stratum"],
    }


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    unit: Mapping[str, Any],
    plan_payload_sha256: str,
) -> str:
    """Validate one download receipt against its canonical plan membership."""
    payload_sha256 = _verify_self_hash(receipt, "receipt_sha256", "Download receipt")
    expected = {
        "admission_status": "downloaded_and_admitted",
        "canonical_url": unit["canonical_url"],
        "corpus_plan_membership": _expected_membership(unit),
        "corpus_plan_payload_sha256": plan_payload_sha256,
        "data_row_count": unit["source_records"],
        "format_version": "oas_data_unit_download_receipt_v1",
        "relative_unit": unit["relative_unit"],
        "source_class": "unpaired",
        "test_access": {
            "downstream_heldout_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
        "unique_sequences": unit["source_records"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ProductionMapError("Download receipt disagrees with the corpus plan")
    return payload_sha256


def _validate_protocol_inputs(
    *,
    protocol_path: Path,
    protocol_file_sha256: str,
    plan_path: Path,
    download_audit_path: Path,
    filter_policy_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate the frozen protocol, plan, audit, and filter policy."""
    protocol, protocol_bytes = _load_json_file(protocol_path, "Protocol")
    if hashlib.sha256(protocol_bytes).hexdigest() != protocol_file_sha256:
        raise ProductionMapError("Protocol file digest differs")
    if protocol.get("format_version") != "oas_production_materialization_protocol_v1":
        raise ProductionMapError("Protocol format differs")
    inputs = protocol.get("inputs")
    if not isinstance(inputs, dict):
        raise ProductionMapError("Protocol input anchors are absent")

    plan, plan_bytes = _load_json_file(plan_path, "Corpus plan")
    if hashlib.sha256(plan_bytes).hexdigest() != inputs.get("plan_file_sha256"):
        raise ProductionMapError("Corpus-plan file digest differs")
    if _verify_self_hash(plan, "plan_sha256", "Corpus plan") != inputs.get(
        "plan_payload_sha256"
    ):
        raise ProductionMapError("Corpus-plan payload digest differs")
    if plan.get("format_version") != "oas_unpaired_corpus_plan_v1":
        raise ProductionMapError("Corpus-plan format differs")

    audit, audit_bytes = _load_json_file(download_audit_path, "Download audit")
    if hashlib.sha256(audit_bytes).hexdigest() != inputs.get(
        "download_completion_audit_file_sha256"
    ):
        raise ProductionMapError("Download-audit file digest differs")
    if _verify_self_hash(audit, "receipt_sha256", "Download audit") != inputs.get(
        "download_completion_audit_payload_sha256"
    ):
        raise ProductionMapError("Download-audit payload digest differs")
    if (
        audit.get("format_version") != "oas_download_completion_audit_v1"
        or audit.get("audit_passed") is not True
        or audit.get("plan_file_sha256") != inputs.get("plan_file_sha256")
        or audit.get("plan_payload_sha256") != inputs.get("plan_payload_sha256")
        or audit.get("raw_inventory_sha256") != inputs.get("raw_inventory_sha256")
        or audit.get("receipt_count") != inputs.get("download_receipt_count")
        or audit.get("test_access")
        != {
            "downstream_heldout_accessed": False,
            "sabdab2_heldout_accessed": False,
        }
    ):
        raise ProductionMapError("Download audit does not authorize mapping")
    counts = audit.get("counts")
    if (
        not isinstance(counts, dict)
        or counts.get("compressed_bytes") != inputs.get("raw_compressed_bytes")
        or counts.get("source_records") != inputs.get("source_record_count")
    ):
        raise ProductionMapError("Download-audit counts differ from the protocol")

    if not filter_policy_path.is_file() or filter_policy_path.is_symlink():
        raise ProductionMapError("Filter policy must be a regular non-symlink file")
    policy, filter_sha256 = load_filter_policy(filter_policy_path)
    if filter_sha256 != inputs.get("filter_policy_file_sha256"):
        raise ProductionMapError("Filter-policy file digest differs")
    return protocol, plan, audit, policy


def _validate_plan_units(
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate canonical unit order, fields, counts, and study vocabulary."""
    raw_units = plan.get("selected_data_units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ProductionMapError("Corpus plan has no selected data units")
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_unit in raw_units:
        if not isinstance(raw_unit, dict):
            raise ProductionMapError("Corpus-plan data unit is not an object")
        unit = dict(raw_unit)
        relative = _safe_relative_unit(unit.get("relative_unit"))
        if relative in seen:
            raise ProductionMapError("Corpus-plan data-unit path is duplicated")
        seen.add(relative)
        if unit.get("split") not in {"train", "validation"}:
            raise ProductionMapError("Corpus-plan split is invalid")
        if unit.get("stratum") not in record_module.STRATUM_TO_ID:
            raise ProductionMapError("Corpus-plan stratum is invalid")
        if not isinstance(unit.get("study"), str) or not unit["study"]:
            raise ProductionMapError("Corpus-plan study is invalid")
        for field in (
            "expected_selected_records",
            "selection_fraction_denominator",
            "selection_fraction_numerator",
            "source_records",
        ):
            value = unit.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProductionMapError(f"Corpus-plan {field} is invalid")
        if (
            unit["selection_fraction_numerator"] <= 0
            or unit["selection_fraction_denominator"] <= 0
            or unit["selection_fraction_numerator"]
            > unit["selection_fraction_denominator"]
            or unit["source_records"] >= 2**32
        ):
            raise ProductionMapError("Corpus-plan sampling domain is invalid")
        namespace = unit.get("selection_hash_namespace")
        if (
            not isinstance(namespace, str)
            or len(namespace) != 64
            or any(character not in "0123456789abcdef" for character in namespace)
        ):
            raise ProductionMapError("Corpus-plan selection namespace is invalid")
        units.append(unit)
    studies = sorted({unit["study"] for unit in units})
    if len(studies) >= 2**16:
        raise ProductionMapError("Study vocabulary exceeds uint16")
    if units != sorted(units, key=lambda unit: (unit["split"], unit["relative_unit"])):
        raise ProductionMapError("Corpus-plan data units are not canonically ordered")
    validation_studies = {
        entry.get("study")
        for entry in plan.get("validation_studies", [])
        if isinstance(entry, dict)
    }
    for unit in units:
        if (unit["study"] in validation_studies) != (unit["split"] == "validation"):
            raise ProductionMapError("Corpus-plan validation-study isolation differs")
    return units, studies


def _raw_sequence(row: Mapping[str, Any], priority: Sequence[str]) -> str:
    """Extract the exact pre-quality sequence used for plan sampling."""
    for field in priority:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return ""


def _map_one_unit(
    *,
    unit_index: int,
    unit: Mapping[str, Any],
    study_id: int,
    download_root: Path,
    plan_payload_sha256: str,
    evaluator: CompiledUnpairedRowEvaluator,
    sequence_field_priority: Sequence[str],
    bucket_writer: _BucketWriter,
    counters: Counter[str],
    status_counts: Counter[str],
) -> dict[str, Any]:
    """Authenticate, stream, filter, and map one complete OAS gzip unit."""
    relative = unit["relative_unit"]
    receipt_relative = f"receipts/unpaired/{relative}.receipt.json"
    raw_relative = f"raw/unpaired/{relative}"
    receipt_path = _safe_input_path(download_root, receipt_relative)
    raw_path = _safe_input_path(download_root, raw_relative)
    receipt, receipt_bytes = _load_json_file(receipt_path, "Download receipt")
    receipt_payload_sha256 = _validate_receipt(
        receipt,
        unit=unit,
        plan_payload_sha256=plan_payload_sha256,
    )
    if raw_path.stat().st_size != receipt.get("compressed_bytes"):
        raise ProductionMapError("Raw compressed byte count differs")

    unit_counts: Counter[str] = Counter()
    unit_status: Counter[str] = Counter()
    compressed_file = raw_path.open("rb")
    compressed_counter = _DigestingRawReader(compressed_file)
    compressed_buffer = io.BufferedReader(compressed_counter, 8 * 1024 * 1024)
    gzip_stream = gzip.GzipFile(fileobj=compressed_buffer, mode="rb")
    uncompressed_counter = _DigestingRawReader(gzip_stream)
    uncompressed_buffer = io.BufferedReader(uncompressed_counter, 8 * 1024 * 1024)
    text_stream = io.TextIOWrapper(
        uncompressed_buffer,
        encoding="utf-8",
        errors="strict",
        newline="",
    )
    try:
        metadata_line = text_stream.readline()
        header_line = text_stream.readline()
        if not metadata_line or not header_line:
            raise ProductionMapError("OAS unit lacks metadata or header")
        metadata = _strict_json_text(
            _single_csv_field(metadata_line, "OAS metadata line"),
            "OAS metadata",
        )
        try:
            header = tuple(next(csv.reader([header_line])))
        except (csv.Error, StopIteration) as error:
            raise ProductionMapError("OAS header is invalid") from error
        if not header or len(header) != len(set(header)):
            raise ProductionMapError("OAS header is empty or duplicated")
        required = {
            "ANARCI_status",
            "locus",
            "productive",
            "stop_codon",
            "vj_in_frame",
        }
        if not required.issubset(header) or not any(
            field in header for field in sequence_field_priority
        ):
            raise ProductionMapError("OAS header lacks frozen filter fields")
        authority_chain = metadata.get("Chain")
        if authority_chain not in {"Heavy", "Light"}:
            raise ProductionMapError("OAS metadata chain is invalid")
        if authority_chain != evaluator.authority_chain:
            raise ProductionMapError(
                "OAS metadata chain disagrees with the plan stratum"
            )
        if metadata.get("Unique sequences") != unit["source_records"]:
            raise ProductionMapError("OAS metadata row count differs from the plan")
        if hashlib.sha256(metadata_line.encode("utf-8")).hexdigest() != receipt.get(
            "metadata_sha256"
        ):
            raise ProductionMapError("OAS metadata line digest differs")
        if hashlib.sha256(header_line.encode("utf-8")).hexdigest() != receipt.get(
            "header_sha256"
        ):
            raise ProductionMapError("OAS header line digest differs")

        reader = csv.DictReader(text_stream, fieldnames=header)
        for physical_row_index, row in enumerate(reader):
            if None in row or any(value is None for value in row.values()):
                raise ProductionMapError("OAS data row width differs from the header")
            unit_counts["source_rows"] += 1
            sequence_text = _raw_sequence(row, sequence_field_priority)
            if not selected_by_plan_fraction(
                sequence_text,
                namespace_sha256=unit["selection_hash_namespace"],
                numerator=unit["selection_fraction_numerator"],
                denominator=unit["selection_fraction_denominator"],
            ):
                unit_counts["plan_fraction_rejected"] += 1
                continue
            unit_counts["plan_fraction_selected"] += 1
            decision = evaluator.evaluate(row)
            if decision.accepted is None:
                reason = decision.rejection_reason
                unit_counts[f"quality_rejected:{reason}"] += 1
                continue
            accepted = decision.accepted
            authority_ordinal = (unit_index << 32) | physical_row_index
            record = ProductionMapRecord(
                authority_ordinal=authority_ordinal,
                chain_type=accepted.chain_type,
                sequence=accepted.sequence,
                sequence_sha256=accepted.sequence_sha256,
                split=unit["split"],
                status_flags=accepted.status_flags,
                stratum=unit["stratum"],
                study_id=study_id,
            )
            payload = pack_map_record(record)
            bucket_writer.write(bytes.fromhex(accepted.sequence_sha256)[0], payload)
            unit_counts["accepted_records"] += 1
            for flag in accepted.status_flags:
                unit_status[flag] += 1
    except (OSError, EOFError, UnicodeError, csv.Error) as error:
        raise ProductionMapError("OAS gzip or CSV stream is invalid") from error
    finally:
        text_stream.close()

    if unit_counts["source_rows"] != receipt.get("data_row_count"):
        raise ProductionMapError("Mapped row count differs from the download receipt")
    observations = {
        "compressed_bytes": compressed_counter.byte_count,
        "compressed_sha256": compressed_counter.digest.hexdigest(),
        "uncompressed_bytes": uncompressed_counter.byte_count,
        "uncompressed_sha256": uncompressed_counter.digest.hexdigest(),
    }
    if any(observations[key] != receipt.get(key) for key in observations):
        raise ProductionMapError("Mapped byte stream differs from the download receipt")
    counters.update(unit_counts)
    status_counts.update(unit_status)
    return {
        "counts": dict(sorted(unit_counts.items())),
        "raw_compressed_bytes": observations["compressed_bytes"],
        "raw_compressed_sha256": observations["compressed_sha256"],
        "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "receipt_payload_sha256": receipt_payload_sha256,
        "relative_unit": relative,
        "status_flag_counts": dict(sorted(unit_status.items())),
        "unit_index": unit_index,
        "uncompressed_bytes": observations["uncompressed_bytes"],
        "uncompressed_sha256": observations["uncompressed_sha256"],
    }


def _bucket_inventory(root: Path, counts: Mapping[int, int]) -> list[dict[str, Any]]:
    """Hash every nonempty bucket and verify its fixed-record byte count."""
    inventory = []
    for bucket in sorted(counts):
        path = root / f"bucket-{bucket:03d}.bin"
        records = counts[bucket]
        expected_bytes = records * record_module.MAP_RECORD.size
        if path.stat().st_size != expected_bytes:
            raise ProductionMapError("Map bucket byte count differs")
        inventory.append(
            {
                "bucket": bucket,
                "bytes": expected_bytes,
                "file_sha256": _file_sha256(path),
                "records": records,
            }
        )
    return inventory


def _write_create_only(path: Path, payload: bytes) -> None:
    """Create one durable read-only evidence file and reject overwrite."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def _freeze_files(root: Path) -> None:
    """Remove write bits from evidence files before atomic publication."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            path.chmod(0o444)


def _freeze_directories(root: Path) -> None:
    """Remove directory write bits immediately after atomic publication."""
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def map_production_task(
    *,
    task_id: int,
    task_count: int,
    download_root: Path,
    plan_path: Path,
    download_audit_path: Path,
    filter_policy_path: Path,
    protocol_path: Path,
    protocol_file_sha256: str,
    output_root: Path,
    expected_controller_source_sha256: str,
    expected_preprocessing_source_sha256: str,
    expected_record_source_sha256: str,
    pilot_maximum_data_units: int | None = None,
) -> dict[str, Any]:
    """Execute one fail-closed modulo map task and atomically publish it."""
    source_checks = (
        (Path(__file__), expected_controller_source_sha256, "Controller"),
        (
            Path(preprocessing_module.__file__),
            expected_preprocessing_source_sha256,
            "Preprocessing module",
        ),
        (
            Path(record_module.__file__),
            expected_record_source_sha256,
            "Map-record module",
        ),
    )
    for path, expected, label in source_checks:
        if _file_sha256(path) != expected:
            raise ProductionMapError(f"{label} source digest differs")
    protocol, plan, audit, policy = _validate_protocol_inputs(
        protocol_path=protocol_path,
        protocol_file_sha256=protocol_file_sha256,
        plan_path=plan_path,
        download_audit_path=download_audit_path,
        filter_policy_path=filter_policy_path,
    )
    execution = protocol.get("execution")
    if not isinstance(execution, dict) or task_count != execution.get("map_task_count"):
        raise ProductionMapError("Map task count differs from the protocol")
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or not 0 <= task_id < task_count
    ):
        raise ProductionMapError("Map task ID is outside the task domain")
    if pilot_maximum_data_units is not None and (
        isinstance(pilot_maximum_data_units, bool)
        or not isinstance(pilot_maximum_data_units, int)
        or pilot_maximum_data_units <= 0
    ):
        raise ProductionMapError("Pilot maximum must be a positive integer")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "NoDevFiles"}:
        raise ProductionMapError("Production materialization must not consume a GPU")
    if any(os.environ.get(name) for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")):
        raise ProductionMapError("Slurm assigned a GPU to CPU-only materialization")
    minimum_free = protocol.get("resource_policy", {}).get(
        "shared_storage_minimum_free_bytes"
    )
    if (
        isinstance(minimum_free, bool)
        or not isinstance(minimum_free, int)
        or shutil.disk_usage(output_root.parent).free < minimum_free
    ):
        raise ProductionMapError("Shared storage does not satisfy the frozen reserve")

    units, studies = _validate_plan_units(plan)
    if len(units) != protocol["inputs"]["download_receipt_count"]:
        raise ProductionMapError("Plan data-unit count differs from the protocol")
    if (
        sum(unit["source_records"] for unit in units)
        != protocol["inputs"]["source_record_count"]
    ):
        raise ProductionMapError("Plan source-row total differs from the protocol")
    assigned = [
        (index, unit)
        for index, unit in enumerate(units)
        if index % task_count == task_id
    ]
    if pilot_maximum_data_units is not None:
        assigned = assigned[:pilot_maximum_data_units]
    if not assigned:
        raise ProductionMapError("Map task has no assigned data units")
    study_to_id = {study: index for index, study in enumerate(studies)}
    study_vocabulary_sha256 = hashlib.sha256(_canonical_json_bytes(studies)).hexdigest()

    final_root = output_root / "map" / f"mapper-{task_id:04d}"
    if final_root.exists():
        raise ProductionMapError("Published mapper output already exists")
    partial_root = (
        output_root / "work" / f"mapper-{task_id:04d}.partial-{uuid.uuid4().hex}"
    )
    partial_root.mkdir(parents=True, exist_ok=False)
    bucket_root = partial_root / "buckets"
    bucket_root.mkdir()
    bucket_writer = _BucketWriter(bucket_root)
    counters: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    unit_receipts = []
    try:
        for unit_index, unit in assigned:
            evaluator = CompiledUnpairedRowEvaluator.from_policy(
                policy,
                authority_chain=(
                    "Heavy" if unit["stratum"].endswith("heavy") else "Light"
                ),
            )
            unit_receipts.append(
                _map_one_unit(
                    unit_index=unit_index,
                    unit=unit,
                    study_id=study_to_id[unit["study"]],
                    download_root=download_root,
                    plan_payload_sha256=plan["plan_sha256"],
                    evaluator=evaluator,
                    sequence_field_priority=policy["sequence_field_priority"],
                    bucket_writer=bucket_writer,
                    counters=counters,
                    status_counts=status_counts,
                )
            )
        bucket_writer.close()
        bucket_inventory = _bucket_inventory(bucket_root, bucket_writer.counts)
        receipt = {
            "audit_bindings": {
                "download_audit_file_sha256": hashlib.sha256(
                    download_audit_path.read_bytes()
                ).hexdigest(),
                "download_audit_payload_sha256": audit["receipt_sha256"],
                "filter_policy_file_sha256": protocol["inputs"][
                    "filter_policy_file_sha256"
                ],
                "plan_file_sha256": protocol["inputs"]["plan_file_sha256"],
                "plan_payload_sha256": plan["plan_sha256"],
                "protocol_file_sha256": protocol_file_sha256,
            },
            "bucket_inventory": bucket_inventory,
            "claim_boundary": {
                "global_dedup_complete": False,
                "large_scale_pretraining_authorized": False,
                "map_task_complete": pilot_maximum_data_units is None,
                "pilot_only": pilot_maximum_data_units is not None,
                "scientific_claim": False,
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "counts": dict(sorted(counters.items())),
            "format_version": FORMAT_VERSION,
            "host": socket.gethostname(),
            "source_sha256": {
                "controller": expected_controller_source_sha256,
                "map_record": expected_record_source_sha256,
                "preprocessing": expected_preprocessing_source_sha256,
            },
            "status_flag_counts": dict(sorted(status_counts.items())),
            "study_count": len(studies),
            "study_vocabulary_sha256": study_vocabulary_sha256,
            "task_count": task_count,
            "task_id": task_id,
            "test_access": TEST_ACCESS,
            "unit_count": len(assigned),
            "unit_receipts": unit_receipts,
        }
        receipt["receipt_sha256"] = hashlib.sha256(
            _canonical_json_bytes(receipt)
        ).hexdigest()
        _write_create_only(
            partial_root / "receipt.json",
            _canonical_json_bytes(receipt),
        )
        _freeze_files(partial_root)
        final_root.parent.mkdir(parents=True, exist_ok=True)
        partial_root.rename(final_root)
        _freeze_directories(final_root)
        return receipt
    except Exception:  # noqa: BLE001 - every failed task must close bucket handles.
        try:
            bucket_writer.close()
        finally:
            raise


def _parser() -> argparse.ArgumentParser:
    """Build the strict production mapper command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--download-audit", type=Path, required=True)
    parser.add_argument("--filter-policy", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-file-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--controller-source-sha256", required=True)
    parser.add_argument("--preprocessing-source-sha256", required=True)
    parser.add_argument("--record-source-sha256", required=True)
    parser.add_argument("--pilot-maximum-data-units", type=int)
    return parser


def main() -> int:
    """Run one map task and print its canonical receipt."""
    arguments = _parser().parse_args()
    receipt = map_production_task(
        task_id=arguments.task_id,
        task_count=arguments.task_count,
        download_root=arguments.download_root,
        plan_path=arguments.plan,
        download_audit_path=arguments.download_audit,
        filter_policy_path=arguments.filter_policy,
        protocol_path=arguments.protocol,
        protocol_file_sha256=arguments.protocol_file_sha256,
        output_root=arguments.output_root,
        expected_controller_source_sha256=arguments.controller_source_sha256,
        expected_preprocessing_source_sha256=arguments.preprocessing_source_sha256,
        expected_record_source_sha256=arguments.record_source_sha256,
        pilot_maximum_data_units=arguments.pilot_maximum_data_units,
    )
    sys.stdout.buffer.write(_canonical_json_bytes(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
