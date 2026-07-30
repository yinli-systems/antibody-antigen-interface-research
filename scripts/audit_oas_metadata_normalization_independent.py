#!/usr/bin/env python3
"""Independently audit the frozen OAS metadata normalization exception."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, NamedTuple


FORMAT_VERSION = "oas_metadata_normalization_independent_audit_v1"
PROTOCOL_FORMAT = "oas_metadata_normalization_protocol_v1"
PLAN_FORMAT = "oas_unpaired_corpus_plan_v1"
AUDIT_FORMAT = "oas_download_completion_audit_v1"
TEST_ACCESS = {
    "downstream_metrics_used": False,
    "sabdab2_heldout_accessed": False,
}


class MetadataAuditError(RuntimeError):
    """Report an authority, schema, normalization, or digest disagreement."""


class _NonFinite(NamedTuple):
    """Retain one non-finite token without creating a floating-point value."""

    spelling: str


def _canonical_json_bytes(value: Any) -> bytes:
    """Encode finite canonical ASCII JSON."""
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


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys at every JSON nesting level."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MetadataAuditError(f"Duplicate metadata key: {key}")
        result[key] = value
    return result


def _file_sha256(path: Path) -> str:
    """Hash one regular non-symlink file."""
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise MetadataAuditError(f"Input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    """Load duplicate-free, finite JSON from a regular input."""
    raw = path.read_bytes()

    def reject_constant(value: str) -> None:
        raise MetadataAuditError(f"Non-finite JSON constant: {value}")

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MetadataAuditError(f"Input is not strict ASCII JSON: {path}") from error
    if not isinstance(value, dict):
        raise MetadataAuditError(f"Input must contain one JSON object: {path}")
    return value, raw


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    """Verify one canonical payload self-hash."""
    unsigned = dict(value)
    claimed = unsigned.pop(field, None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise MetadataAuditError(f"Self hash differs: {field}")
    return observed


def _safe_input(
    root: Path,
    prefix: str,
    relative_unit: object,
    *,
    expected_suffix: str,
) -> Path:
    """Resolve one canonical regular input without following symlinks."""
    if not isinstance(relative_unit, str) or not relative_unit.endswith(expected_suffix):
        raise MetadataAuditError("Relative OAS unit is invalid")
    relative = PurePosixPath(relative_unit)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise MetadataAuditError("Relative OAS unit is unsafe")
    resolved_root = root.resolve(strict=True)
    path = root.joinpath(*PurePosixPath(prefix).parts, *relative.parts)
    current = resolved_root
    for part in (*PurePosixPath(prefix).parts, *relative.parts):
        current /= part
        if current.is_symlink():
            raise MetadataAuditError("OAS input path contains a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise MetadataAuditError("OAS input path escapes its authority root")
    return resolved


def _single_csv_field(line: str) -> str:
    """Extract the sole JSON field from one physical OAS metadata line."""
    try:
        row = next(csv.reader([line]))
    except (csv.Error, StopIteration) as error:
        raise MetadataAuditError("Metadata line is not a CSV record") from error
    if len(row) != 1:
        raise MetadataAuditError("Metadata line does not contain one CSV field")
    return row[0]


def _decode_metadata(field: str) -> tuple[dict[str, Any], bool]:
    """Decode strict metadata or the sole frozen top-level Age NaN exception."""
    try:
        value = json.loads(
            field,
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda spelling: _NonFinite(spelling),
        )
    except json.JSONDecodeError as error:
        raise MetadataAuditError("Metadata JSON syntax differs") from error
    if not isinstance(value, dict):
        raise MetadataAuditError("Metadata JSON is not an object")

    nonfinite_paths: list[tuple[str, ...]] = []

    def visit(item: Any, path: tuple[str, ...]) -> None:
        if isinstance(item, _NonFinite):
            nonfinite_paths.append(path)
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, (*path, key))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, (*path, str(index)))

    visit(value, ())
    if not nonfinite_paths:
        return value, False
    token = value.get("Age")
    if (
        nonfinite_paths != [("Age",)]
        or not isinstance(token, _NonFinite)
        or token.spelling != "NaN"
    ):
        raise MetadataAuditError("Metadata has an unauthorized non-finite value")
    value["Age"] = None
    return value, True


def _audit_unit(
    item: tuple[int, Mapping[str, Any]],
    *,
    download_root: Path,
) -> dict[str, Any]:
    """Audit one raw metadata line and its authenticated download receipt."""
    unit_index, unit = item
    relative = unit.get("relative_unit")
    raw_path = _safe_input(
        download_root,
        "raw/unpaired",
        relative,
        expected_suffix=".csv.gz",
    )
    receipt_path = _safe_input(
        download_root,
        "receipts/unpaired",
        f"{relative}.receipt.json",
        expected_suffix=".csv.gz.receipt.json",
    )
    receipt, receipt_raw = _load_json(receipt_path)
    _self_hash(receipt, "receipt_sha256")
    if (
        receipt.get("relative_unit") != relative
        or receipt.get("data_row_count") != unit.get("source_records")
        or receipt.get("metadata_sha256") is None
    ):
        raise MetadataAuditError("Download receipt disagrees with the plan")
    with gzip.open(raw_path, "rt", encoding="utf-8", errors="strict", newline="") as handle:
        metadata_line = handle.readline()
    if not metadata_line:
        raise MetadataAuditError("OAS unit has no metadata line")
    raw_line_sha256 = hashlib.sha256(metadata_line.encode("utf-8")).hexdigest()
    if raw_line_sha256 != receipt.get("metadata_sha256"):
        raise MetadataAuditError("Metadata line differs from its download receipt")
    field = _single_csv_field(metadata_line)
    value, normalized = _decode_metadata(field)
    return {
        "decoded_metadata_sha256": hashlib.sha256(field.encode("utf-8")).hexdigest(),
        "download_receipt_file_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "normalized": normalized,
        "normalized_metadata_sha256": hashlib.sha256(
            _canonical_json_bytes(value)
        ).hexdigest(),
        "raw_metadata_line_sha256": raw_line_sha256,
        "relative_unit": relative,
        "unit_index": unit_index,
    }


def audit_metadata(
    *,
    download_root: Path,
    plan_path: Path,
    download_audit_path: Path,
    protocol_path: Path,
    worker_count: int,
) -> dict[str, Any]:
    """Audit every authority metadata header against the frozen exception list."""
    if not 1 <= worker_count <= 8:
        raise MetadataAuditError("Worker count is outside the frozen domain")
    protocol, protocol_raw = _load_json(protocol_path)
    plan, plan_raw = _load_json(plan_path)
    download_audit, audit_raw = _load_json(download_audit_path)
    if protocol.get("format_version") != PROTOCOL_FORMAT:
        raise MetadataAuditError("Normalization protocol format differs")
    authority = protocol.get("authority")
    if not isinstance(authority, dict):
        raise MetadataAuditError("Normalization authority is absent")
    if (
        hashlib.sha256(plan_raw).hexdigest() != authority.get("plan_file_sha256")
        or _self_hash(plan, "plan_sha256") != authority.get("plan_payload_sha256")
        or plan.get("format_version") != PLAN_FORMAT
        or hashlib.sha256(audit_raw).hexdigest()
        != authority.get("download_completion_audit_file_sha256")
        or _self_hash(download_audit, "receipt_sha256")
        != authority.get("download_completion_audit_payload_sha256")
        or download_audit.get("format_version") != AUDIT_FORMAT
        or download_audit.get("audit_passed") is not True
    ):
        raise MetadataAuditError("Normalization authority anchors differ")
    units = plan.get("selected_data_units")
    if (
        not isinstance(units, list)
        or len(units) != authority.get("selected_data_unit_count")
        or sum(unit.get("source_records", -1) for unit in units)
        != authority.get("source_record_count")
    ):
        raise MetadataAuditError("Plan inventory differs from the protocol")
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        inventory = list(
            pool.map(
                lambda item: _audit_unit(item, download_root=download_root),
                enumerate(units),
            )
        )
    affected = [item for item in inventory if item.pop("normalized")]
    expected = protocol.get("affected_units")
    observed = [
        {
            key: item[key]
            for key in (
                "decoded_metadata_sha256",
                "normalized_metadata_sha256",
                "raw_metadata_line_sha256",
                "relative_unit",
            )
        }
        for item in affected
    ]
    normalization = protocol.get("normalization", {})
    if (
        observed != expected
        or len(affected) != normalization.get("affected_unit_count")
        or len(inventory) - len(affected) != normalization.get("strict_unit_count")
    ):
        raise MetadataAuditError("Observed metadata exceptions differ from the protocol")
    semantic = hashlib.sha256()
    for item in inventory:
        semantic.update(_canonical_json_bytes(item))
    return {
        "audit_passed": True,
        "authorization": dict(protocol["authorization"]),
        "claim_boundary": protocol["claim_boundary"],
        "counts": {
            "affected_units": len(affected),
            "scanned_units": len(inventory),
            "strict_units": len(inventory) - len(affected),
        },
        "format_version": FORMAT_VERSION,
        "inventory_semantic_sha256": semantic.hexdigest(),
        "normalization_protocol_file_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "test_access": TEST_ACCESS,
    }


def _publish(path: Path, result: dict[str, Any]) -> None:
    """Create one immutable self-hashed audit receipt."""
    payload = dict(result)
    payload["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--download-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Run the complete independent metadata scan and publish its receipt."""
    arguments = _parser().parse_args()
    result = audit_metadata(
        download_root=arguments.download_root,
        plan_path=arguments.plan,
        download_audit_path=arguments.download_audit,
        protocol_path=arguments.protocol,
        worker_count=arguments.worker_count,
    )
    _publish(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
