#!/usr/bin/env python3
"""Independently audit complete plan-bound OAS download materialization."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


TEST_ACCESS = {
    "downstream_heldout_accessed": False,
    "sabdab2_heldout_accessed": False,
}


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_regular_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} must be a regular non-symlink file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} must be ASCII JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object")
    return value, payload


def _self_hash(value: dict[str, Any], field: str, label: str) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop(field, None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise RuntimeError(f"{label} self-hash differs")
    return observed


def _safe_relative_unit(value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError("Planned relative unit is not a string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or value != str(relative):
        raise RuntimeError("Planned relative unit is unsafe")
    if not value.endswith(".csv.gz"):
        raise RuntimeError("Planned relative unit has an unexpected suffix")
    return value


def _expected_membership(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "expected_selected_records": unit["expected_selected_records"],
        "selection_fraction_denominator": unit["selection_fraction_denominator"],
        "selection_fraction_numerator": unit["selection_fraction_numerator"],
        "selection_hash_namespace": unit["selection_hash_namespace"],
        "split": unit["split"],
        "stratum": unit["stratum"],
    }


def audit_download_completion(
    *,
    download_root: Path,
    plan_path: Path,
    expected_plan_file_sha256: str,
    expected_plan_payload_sha256: str,
    summary_paths: Iterable[Path],
    expected_task_count: int,
    expected_auditor_source_sha256: str,
) -> dict[str, Any]:
    """Rehash all downloaded bytes and prove exact task and plan coverage."""
    source_path = Path(__file__)
    if _file_sha256(source_path) != expected_auditor_source_sha256:
        raise RuntimeError("Auditor source hash differs")
    plan, plan_bytes = _load_regular_json(plan_path, "Corpus plan")
    if hashlib.sha256(plan_bytes).hexdigest() != expected_plan_file_sha256:
        raise RuntimeError("Corpus-plan file hash differs")
    if _self_hash(plan, "plan_sha256", "Corpus plan") != (
        expected_plan_payload_sha256
    ):
        raise RuntimeError("Corpus-plan payload hash differs")
    if plan.get("format_version") != "oas_unpaired_corpus_plan_v1":
        raise RuntimeError("Corpus-plan format differs")
    units = plan.get("selected_data_units")
    if not isinstance(units, list) or not units:
        raise RuntimeError("Corpus plan has no selected data units")
    planned_by_relative: dict[str, dict[str, Any]] = {}
    for unit in units:
        if not isinstance(unit, dict):
            raise RuntimeError("Corpus-plan data unit is invalid")
        relative = _safe_relative_unit(unit.get("relative_unit"))
        if relative in planned_by_relative:
            raise RuntimeError("Corpus-plan data units are not unique")
        planned_by_relative[relative] = unit

    summaries: dict[int, dict[str, Any]] = {}
    summary_receipt_hashes: dict[int, list[str]] = {}
    summary_file_hashes: dict[str, str] = {}
    for path in summary_paths:
        summary, payload = _load_regular_json(path, "Task summary")
        _self_hash(summary, "summary_sha256", "Task summary")
        task_id = summary.get("task_id")
        if (
            not isinstance(task_id, int)
            or isinstance(task_id, bool)
            or task_id in summaries
        ):
            raise RuntimeError("Task summary ID is invalid or duplicated")
        summaries[task_id] = summary
        summary_receipt_hashes[task_id] = summary.get(
            "successful_receipt_sha256", []
        )
        summary_file_hashes[str(task_id)] = hashlib.sha256(payload).hexdigest()
    if set(summaries) != set(range(expected_task_count)):
        raise RuntimeError("Task summaries do not cover every task")

    receipt_root = download_root / "receipts" / "unpaired"
    discovered_receipts = sorted(receipt_root.rglob("*.receipt.json"))
    expected_receipt_paths = {
        receipt_root / f"{relative}.receipt.json"
        for relative in planned_by_relative
    }
    if set(discovered_receipts) != expected_receipt_paths:
        raise RuntimeError("Receipt inventory does not equal corpus-plan membership")

    receipt_payload_by_relative: dict[str, str] = {}
    receipt_inventory_digest = hashlib.sha256()
    raw_inventory_digest = hashlib.sha256()
    counts = Counter()
    controller_hashes: set[str] = set()
    downloader_hashes: set[str] = set()
    authority_hashes: set[str] = set()
    for receipt_path in discovered_receipts:
        receipt, receipt_bytes = _load_regular_json(receipt_path, "Download receipt")
        receipt_payload_sha256 = _self_hash(
            receipt, "receipt_sha256", "Download receipt"
        )
        relative = _safe_relative_unit(receipt.get("relative_unit"))
        unit = planned_by_relative.get(relative)
        if unit is None or relative in receipt_payload_by_relative:
            raise RuntimeError("Receipt identity is absent or duplicated")
        expected_fields = {
            "admission_status": "downloaded_and_admitted",
            "canonical_url": unit["canonical_url"],
            "corpus_plan_membership": _expected_membership(unit),
            "corpus_plan_payload_sha256": expected_plan_payload_sha256,
            "data_row_count": unit["source_records"],
            "format_version": "oas_data_unit_download_receipt_v1",
            "relative_unit": relative,
            "source_class": "unpaired",
            "test_access": TEST_ACCESS,
            "unique_sequences": unit["source_records"],
        }
        if any(receipt.get(key) != value for key, value in expected_fields.items()):
            raise RuntimeError("Download receipt disagrees with corpus-plan membership")
        receipt_payload_by_relative[relative] = receipt_payload_sha256
        controller_hashes.add(receipt.get("controller_source_sha256"))
        downloader_hashes.add(receipt.get("downloader_module_sha256"))
        authority_hashes.add(receipt.get("authority_manifest_payload_sha256"))
        raw_path = download_root / "raw" / "unpaired" / Path(relative)
        if not raw_path.is_file() or raw_path.is_symlink():
            raise RuntimeError("Raw data unit is missing or is a symlink")
        if raw_path.stat().st_size != receipt.get("compressed_bytes"):
            raise RuntimeError("Raw compressed byte count differs")
        observed_raw_sha256 = _file_sha256(raw_path)
        if observed_raw_sha256 != receipt.get("compressed_sha256"):
            raise RuntimeError("Raw compressed hash differs")
        receipt_file_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        receipt_inventory_digest.update(
            f"{relative}\t{receipt_file_sha256}\t{receipt_payload_sha256}\n".encode(
                "ascii"
            )
        )
        raw_inventory_digest.update(
            f"{relative}\t{raw_path.stat().st_size}\t{observed_raw_sha256}\n".encode(
                "ascii"
            )
        )
        counts["compressed_bytes"] += raw_path.stat().st_size
        counts["source_records"] += receipt["data_row_count"]
        counts[f"{unit['split']}_data_units"] += 1
        counts[f"{unit['split']}_expected_prefilter_records"] += unit[
            "expected_selected_records"
        ]

    for task_id, summary in summaries.items():
        expected_relatives = sorted(
            unit["relative_unit"]
            for index, unit in enumerate(units)
            if index % expected_task_count == task_id
        )
        expected_hashes = [
            receipt_payload_by_relative[relative] for relative in expected_relatives
        ]
        summary_expected = {
            "authority_manifest_payload_sha256": next(iter(authority_hashes)),
            "corpus_plan_file_sha256": expected_plan_file_sha256,
            "corpus_plan_payload_sha256": expected_plan_payload_sha256,
            "failed_count": 0,
            "failures": [],
            "format_version": "oas_download_task_summary_v1",
            "selected_count": len(expected_relatives),
            "source_class": "unpaired",
            "split": "all",
            "success_count": len(expected_relatives),
            "successful_receipt_sha256": expected_hashes,
            "task_count": expected_task_count,
            "task_id": task_id,
            "test_access": TEST_ACCESS,
        }
        if any(summary.get(key) != value for key, value in summary_expected.items()):
            raise RuntimeError("Task summary does not reproduce task membership")
    if not all(
        len(values) == 1
        for values in (controller_hashes, downloader_hashes, authority_hashes)
    ):
        raise RuntimeError("Download source provenance differs across receipts")
    if next(iter(authority_hashes)) != plan.get("authority", {}).get(
        "manifest_payload_sha256"
    ):
        raise RuntimeError("Download authority and corpus plan disagree")

    receipt = {
        "audit_passed": True,
        "auditor_source_sha256": expected_auditor_source_sha256,
        "authority_manifest_payload_sha256": next(iter(authority_hashes)),
        "claim_boundary": {
            "dense_materialization_complete": False,
            "download_completion_only": True,
            "production_pretraining_authorized": False,
            "raw_compressed_bytes_rehashed": True,
        },
        "controller_source_sha256": next(iter(controller_hashes)),
        "counts": dict(sorted(counts.items())),
        "downloader_module_sha256": next(iter(downloader_hashes)),
        "format_version": "oas_download_completion_audit_v1",
        "plan_file_sha256": expected_plan_file_sha256,
        "plan_payload_sha256": expected_plan_payload_sha256,
        "raw_inventory_sha256": raw_inventory_digest.hexdigest(),
        "receipt_count": len(discovered_receipts),
        "receipt_inventory_sha256": receipt_inventory_digest.hexdigest(),
        "summary_file_sha256": dict(sorted(summary_file_hashes.items())),
        "task_count": expected_task_count,
        "test_access": TEST_ACCESS,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-file-sha256", required=True)
    parser.add_argument("--plan-payload-sha256", required=True)
    parser.add_argument("--summary", type=Path, action="append", default=[])
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--auditor-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit_download_completion(
        download_root=args.download_root,
        plan_path=args.plan,
        expected_plan_file_sha256=args.plan_file_sha256,
        expected_plan_payload_sha256=args.plan_payload_sha256,
        summary_paths=args.summary,
        expected_task_count=args.task_count,
        expected_auditor_source_sha256=args.auditor_source_sha256,
    )
    encoded = _canonical_json_bytes(receipt)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)


if __name__ == "__main__":
    main()
