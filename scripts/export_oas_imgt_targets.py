#!/usr/bin/env python3
"""Replay authenticated OAS admission and export fixed-IMGT search targets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from types import ModuleType
from typing import Any, BinaryIO, Mapping
import uuid

import antibody_pretraining.imgt_position as imgt_position_module
import antibody_pretraining.oas_imgt_export as imgt_export_module
import antibody_pretraining.preprocessing as preprocessing_module
import antibody_pretraining.production_materialization as record_module
from antibody_pretraining.imgt_position import (
    KASEARCH_CANONICAL_ALIGNMENT_SHA256,
    KASEARCH_IDENTITY_CALCULATIONS_SHA256,
    KASEARCH_PAPER_DOI,
    KASEARCH_PREPARE_OASDB_SHA256,
    KASEARCH_SOURCE_COMMIT,
)
from antibody_pretraining.oas_imgt_export import (
    IMGT_TARGET_RECORD,
    UNUSUAL_TARGET_RECORD,
    classify_oas_target,
    pack_imgt_target,
    pack_unusual_target,
)
from antibody_pretraining.preprocessing import CompiledUnpairedRowEvaluator


FORMAT_VERSION = "oas_imgt_target_export_task_v1"
TEST_ACCESS = {
    "downstream_metrics_used": False,
    "sabdab2_heldout_accessed": False,
}


class OASIMGTControllerError(RuntimeError):
    """Report an authenticated replay or publication failure."""


def _file_sha256(path: Path) -> str:
    """Hash one regular non-symlink file."""
    if not path.is_file() or path.is_symlink():
        raise OASIMGTControllerError("Source or evidence path is not a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    """Encode one finite canonical ASCII JSON object."""
    try:
        text = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise OASIMGTControllerError("Receipt is not canonical JSON") from error
    return (text + "\n").encode("ascii")


def _load_mapper(path: Path) -> ModuleType:
    """Load the source-pinned production mapper used for replay."""
    if not path.is_file() or path.is_symlink():
        raise OASIMGTControllerError("Production mapper path is invalid")
    spec = importlib.util.spec_from_file_location("oas_imgt_replay_mapper", path)
    if spec is None or spec.loader is None:
        raise OASIMGTControllerError("Production mapper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _NullBucketWriter:
    """Count admitted digest buckets without duplicating map-record storage."""

    def __init__(self) -> None:
        self.counts: Counter[int] = Counter()

    def write(self, bucket: int, payload: bytes) -> None:
        if not 0 <= bucket < record_module.BUCKET_COUNT:
            raise OASIMGTControllerError("Replay bucket is outside the frozen domain")
        if len(payload) != record_module.MAP_RECORD.size:
            raise OASIMGTControllerError("Replay map record width differs")
        self.counts[bucket] += 1


class _DigestingOutput:
    """Create, hash, count, flush, and fsync one fixed-record output."""

    def __init__(self, path: Path, record_size: int) -> None:
        self.path = path
        self.record_size = record_size
        self.handle: BinaryIO = path.open("xb")
        self.digest = hashlib.sha256()
        self.records = 0
        self.closed = False

    def write(self, payload: bytes) -> None:
        if self.closed or len(payload) != self.record_size:
            raise OASIMGTControllerError("Fixed-record output width differs")
        self.handle.write(payload)
        self.digest.update(payload)
        self.records += 1

    def close(self) -> None:
        if self.closed:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.closed = True

    def inventory(self) -> dict[str, Any]:
        if not self.closed:
            raise OASIMGTControllerError("Output inventory requested before close")
        size = self.path.stat().st_size
        if size != self.records * self.record_size:
            raise OASIMGTControllerError("Fixed-record output byte count differs")
        return {
            "bytes": size,
            "file_sha256": self.digest.hexdigest(),
            "record_bytes": self.record_size,
            "records": self.records,
        }


class _TargetOutputs:
    """Write an exclusive two-way target classification with exact counters."""

    def __init__(self, root: Path) -> None:
        self.representable = _DigestingOutput(
            root / "representable.bin",
            IMGT_TARGET_RECORD.size,
        )
        self.unusual = _DigestingOutput(
            root / "unusual.bin",
            UNUSUAL_TARGET_RECORD.size,
        )
        self.counts: Counter[str] = Counter(
            {
                "accepted_records": 0,
                "representable_records": 0,
                "unusual_records": 0,
            }
        )

    def observe(self, row: Mapping[str, Any], record: Any) -> None:
        if "ANARCI_numbering" not in row:
            raise OASIMGTControllerError("OAS row lacks ANARCI_numbering")
        classified = classify_oas_target(row["ANARCI_numbering"], record)
        classified.validate()
        self.counts["accepted_records"] += 1
        self.counts[f"split:{record.split}"] += 1
        self.counts[f"chain:{record.chain_type}"] += 1
        if classified.representable is not None:
            self.representable.write(pack_imgt_target(classified.representable))
            self.counts["representable_records"] += 1
        else:
            assert classified.unusual is not None
            self.unusual.write(pack_unusual_target(classified.unusual))
            self.counts["unusual_records"] += 1
            self.counts[f"unusual_reason:{classified.unusual.reason}"] += 1

    def close(self) -> None:
        first_error: Exception | None = None
        for output in (self.representable, self.unusual):
            try:
                output.close()
            except OSError as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def inventory(self) -> dict[str, dict[str, Any]]:
        return {
            "representable": self.representable.inventory(),
            "unusual": self.unusual.inventory(),
        }


def _aggregate_unit_receipts(
    unit_receipts: list[Mapping[str, Any]],
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    counters: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    normalizations: Counter[str] = Counter()
    for item in unit_receipts:
        counts = item.get("counts")
        status_counts = item.get("status_flag_counts")
        normalization = item.get("metadata_normalization")
        if (
            not isinstance(counts, Mapping)
            or not isinstance(status_counts, Mapping)
            or not isinstance(normalization, str)
        ):
            raise OASIMGTControllerError("Original unit receipt schema differs")
        counters.update(counts)
        statuses.update(status_counts)
        normalizations[normalization] += 1
    return counters, statuses, normalizations


def _verify_original_mapper_receipt(
    mapper: ModuleType,
    *,
    path: Path,
    task_id: int,
    task_count: int,
) -> tuple[dict[str, Any], str, str]:
    """Authenticate one immutable completed original mapper receipt."""
    receipt, payload = mapper._load_json_file(path, "Original mapper receipt")
    payload_sha256 = mapper._verify_self_hash(
        receipt,
        "receipt_sha256",
        "Original mapper receipt",
    )
    if (
        receipt.get("format_version") != "oas_production_map_task_receipt_v1"
        or receipt.get("task_id") != task_id
        or receipt.get("task_count") != task_count
        or receipt.get("test_access") != TEST_ACCESS
        or receipt.get("claim_boundary", {}).get("map_task_complete") is not True
        or receipt.get("claim_boundary", {}).get("pilot_only") is not False
        or not isinstance(receipt.get("unit_receipts"), list)
    ):
        raise OASIMGTControllerError("Original mapper receipt is not complete")
    return receipt, hashlib.sha256(payload).hexdigest(), payload_sha256


def _write_create_only(path: Path, payload: bytes) -> None:
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


def _freeze_and_publish(partial_root: Path, final_root: Path) -> None:
    for path in sorted(partial_root.rglob("*")):
        if path.is_file():
            path.chmod(0o444)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    partial_root.rename(final_root)
    for path in sorted(final_root.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
    final_root.chmod(0o555)


def export_imgt_task(
    *,
    task_id: int,
    task_count: int,
    download_root: Path,
    plan_path: Path,
    download_audit_path: Path,
    filter_policy_path: Path,
    materialization_protocol_path: Path,
    materialization_protocol_file_sha256: str,
    normalization_protocol_path: Path | None,
    normalization_audit_path: Path | None,
    original_mapper_receipt_path: Path,
    mapper_path: Path,
    output_root: Path,
    expected_controller_source_sha256: str,
    expected_imgt_position_source_sha256: str,
    expected_imgt_record_source_sha256: str,
    expected_mapper_source_sha256: str,
    expected_preprocessing_source_sha256: str,
    expected_record_source_sha256: str,
    pilot_maximum_data_units: int | None = None,
) -> dict[str, Any]:
    """Replay one mapper assignment and atomically publish IMGT targets."""
    mapper = _load_mapper(mapper_path)
    source_checks = (
        (Path(__file__), expected_controller_source_sha256, "Controller"),
        (
            Path(imgt_position_module.__file__),
            expected_imgt_position_source_sha256,
            "IMGT position module",
        ),
        (
            Path(imgt_export_module.__file__),
            expected_imgt_record_source_sha256,
            "IMGT record module",
        ),
        (mapper_path, expected_mapper_source_sha256, "Replay mapper"),
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
            raise OASIMGTControllerError(f"{label} source digest differs")
    original, original_file_sha256, original_payload_sha256 = (
        _verify_original_mapper_receipt(
            mapper,
            path=original_mapper_receipt_path,
            task_id=task_id,
            task_count=task_count,
        )
    )
    protocol, plan, audit, policy = mapper._validate_protocol_inputs(
        protocol_path=materialization_protocol_path,
        protocol_file_sha256=materialization_protocol_file_sha256,
        plan_path=plan_path,
        download_audit_path=download_audit_path,
        filter_policy_path=filter_policy_path,
    )
    metadata_exceptions, normalization_bindings = (
        mapper._metadata_normalization_exceptions(
            protocol=protocol,
            normalization_protocol_path=normalization_protocol_path,
            normalization_audit_path=normalization_audit_path,
        )
    )
    execution = protocol.get("execution")
    if not isinstance(execution, Mapping) or task_count != execution.get(
        "map_task_count"
    ):
        raise OASIMGTControllerError("Task count differs from materialization")
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or not 0 <= task_id < task_count
    ):
        raise OASIMGTControllerError("Task ID is outside the task domain")
    if pilot_maximum_data_units is not None and (
        isinstance(pilot_maximum_data_units, bool)
        or not isinstance(pilot_maximum_data_units, int)
        or pilot_maximum_data_units <= 0
    ):
        raise OASIMGTControllerError("Pilot maximum is invalid")
    resource_receipt = mapper._resource_receipt(protocol)
    minimum_free = protocol.get("resource_policy", {}).get(
        "shared_storage_minimum_free_bytes"
    )
    if (
        isinstance(minimum_free, bool)
        or not isinstance(minimum_free, int)
        or shutil.disk_usage(output_root.parent).free < minimum_free
    ):
        raise OASIMGTControllerError("Shared storage reserve is not satisfied")

    units, studies = mapper._validate_plan_units(plan)
    assigned = [
        (index, unit)
        for index, unit in enumerate(units)
        if index % task_count == task_id
    ]
    if pilot_maximum_data_units is not None:
        assigned = assigned[:pilot_maximum_data_units]
    if not assigned:
        raise OASIMGTControllerError("Replay task has no assigned data units")
    original_by_index = {
        item.get("unit_index"): item for item in original["unit_receipts"]
    }
    if len(original_by_index) != len(original["unit_receipts"]):
        raise OASIMGTControllerError("Original unit receipt index is duplicated")
    expected_units = []
    for unit_index, unit in assigned:
        item = original_by_index.get(unit_index)
        if not isinstance(item, Mapping) or item.get("relative_unit") != unit["relative_unit"]:
            raise OASIMGTControllerError("Original unit assignment differs")
        expected_units.append(dict(item))
    expected_counts, expected_statuses, expected_normalizations = (
        _aggregate_unit_receipts(expected_units)
    )

    final_root = output_root / "imgt-map" / f"mapper-{task_id:04d}"
    if final_root.exists():
        raise OASIMGTControllerError("Published IMGT mapper output already exists")
    partial_root = (
        output_root
        / "work"
        / f"imgt-mapper-{task_id:04d}.partial-{uuid.uuid4().hex}"
    )
    partial_root.mkdir(parents=True, exist_ok=False)
    outputs = _TargetOutputs(partial_root)
    bucket_writer = _NullBucketWriter()
    counters: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    normalizations: Counter[str] = Counter()
    replayed_units = []
    study_to_id = {study: index for index, study in enumerate(studies)}
    try:
        for unit_index, unit in assigned:
            evaluator = CompiledUnpairedRowEvaluator.from_policy(
                policy,
                authority_chain=(
                    "Heavy" if unit["stratum"].endswith("heavy") else "Light"
                ),
            )
            replayed_units.append(
                mapper._map_one_unit(
                    unit_index=unit_index,
                    unit=unit,
                    study_id=study_to_id[unit["study"]],
                    download_root=download_root,
                    plan_payload_sha256=plan["plan_sha256"],
                    evaluator=evaluator,
                    sequence_field_priority=policy["sequence_field_priority"],
                    bucket_writer=bucket_writer,
                    counters=counters,
                    status_counts=statuses,
                    metadata_exceptions=metadata_exceptions,
                    metadata_normalization_counts=normalizations,
                    accepted_row_observer=outputs.observe,
                )
            )
        outputs.close()
        if (
            replayed_units != expected_units
            or counters != expected_counts
            or statuses != expected_statuses
            or normalizations != expected_normalizations
        ):
            raise OASIMGTControllerError("Replay differs from original mapper receipt")
        if (
            outputs.counts["accepted_records"] != counters["accepted_records"]
            or outputs.counts["representable_records"]
            + outputs.counts["unusual_records"]
            != counters["accepted_records"]
            or sum(bucket_writer.counts.values()) != counters["accepted_records"]
        ):
            raise OASIMGTControllerError("Target classification count differs")
        inventory = outputs.inventory()
        receipt = {
            "audit_bindings": {
                "download_audit_file_sha256": hashlib.sha256(
                    download_audit_path.read_bytes()
                ).hexdigest(),
                "download_audit_payload_sha256": audit["receipt_sha256"],
                "materialization_protocol_file_sha256": (
                    materialization_protocol_file_sha256
                ),
                "original_mapper_receipt_file_sha256": original_file_sha256,
                "original_mapper_receipt_payload_sha256": original_payload_sha256,
                "plan_file_sha256": protocol["inputs"]["plan_file_sha256"],
                "plan_payload_sha256": plan["plan_sha256"],
                **normalization_bindings,
            },
            "claim_boundary": {
                "fixed_imgt_task_domain_complete": pilot_maximum_data_units is None,
                "gap_aware_handoff_preserved": True,
                "global_task_set_complete": False,
                "near_identity_search_complete": False,
                "pilot_only": pilot_maximum_data_units is not None,
                "scientific_claim": False,
            },
            "counts": dict(sorted(outputs.counts.items())),
            "format_version": FORMAT_VERSION,
            "host": socket.gethostname(),
            "kasearch_source": {
                "canonical_alignment_sha256": (
                    KASEARCH_CANONICAL_ALIGNMENT_SHA256
                ),
                "commit": KASEARCH_SOURCE_COMMIT,
                "identity_calculations_sha256": (
                    KASEARCH_IDENTITY_CALCULATIONS_SHA256
                ),
                "paper_doi": KASEARCH_PAPER_DOI,
                "prepare_oasdb_sha256": KASEARCH_PREPARE_OASDB_SHA256,
            },
            "output_inventory": inventory,
            "replay_counts": dict(sorted(counters.items())),
            "replay_metadata_normalization_counts": dict(
                sorted(normalizations.items())
            ),
            "replay_status_flag_counts": dict(sorted(statuses.items())),
            "replayed_unit_receipts": replayed_units,
            "resource_receipt": resource_receipt,
            "source_sha256": {
                "controller": expected_controller_source_sha256,
                "imgt_position": expected_imgt_position_source_sha256,
                "imgt_record": expected_imgt_record_source_sha256,
                "map_record": expected_record_source_sha256,
                "preprocessing": expected_preprocessing_source_sha256,
                "replay_mapper": expected_mapper_source_sha256,
            },
            "task_count": task_count,
            "task_id": task_id,
            "test_access": TEST_ACCESS,
            "unit_count": len(assigned),
        }
        receipt["receipt_sha256"] = hashlib.sha256(
            _canonical_json_bytes(receipt)
        ).hexdigest()
        _write_create_only(
            partial_root / "receipt.json",
            _canonical_json_bytes(receipt),
        )
        _freeze_and_publish(partial_root, final_root)
        return receipt
    finally:
        outputs.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--download-audit", type=Path, required=True)
    parser.add_argument("--filter-policy", type=Path, required=True)
    parser.add_argument("--materialization-protocol", type=Path, required=True)
    parser.add_argument("--materialization-protocol-file-sha256", required=True)
    parser.add_argument("--metadata-normalization-protocol", type=Path)
    parser.add_argument("--metadata-normalization-audit", type=Path)
    parser.add_argument("--original-mapper-receipt", type=Path, required=True)
    parser.add_argument("--mapper", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--controller-source-sha256", required=True)
    parser.add_argument("--imgt-position-source-sha256", required=True)
    parser.add_argument("--imgt-record-source-sha256", required=True)
    parser.add_argument("--mapper-source-sha256", required=True)
    parser.add_argument("--preprocessing-source-sha256", required=True)
    parser.add_argument("--record-source-sha256", required=True)
    parser.add_argument("--pilot-maximum-data-units", type=int)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    receipt = export_imgt_task(
        task_id=arguments.task_id,
        task_count=arguments.task_count,
        download_root=arguments.download_root,
        plan_path=arguments.plan,
        download_audit_path=arguments.download_audit,
        filter_policy_path=arguments.filter_policy,
        materialization_protocol_path=arguments.materialization_protocol,
        materialization_protocol_file_sha256=(
            arguments.materialization_protocol_file_sha256
        ),
        normalization_protocol_path=arguments.metadata_normalization_protocol,
        normalization_audit_path=arguments.metadata_normalization_audit,
        original_mapper_receipt_path=arguments.original_mapper_receipt,
        mapper_path=arguments.mapper,
        output_root=arguments.output_root,
        expected_controller_source_sha256=arguments.controller_source_sha256,
        expected_imgt_position_source_sha256=(
            arguments.imgt_position_source_sha256
        ),
        expected_imgt_record_source_sha256=arguments.imgt_record_source_sha256,
        expected_mapper_source_sha256=arguments.mapper_source_sha256,
        expected_preprocessing_source_sha256=(
            arguments.preprocessing_source_sha256
        ),
        expected_record_source_sha256=arguments.record_source_sha256,
        pilot_maximum_data_units=arguments.pilot_maximum_data_units,
    )
    sys.stdout.buffer.write(_canonical_json_bytes(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
