"""Resumable, integrity-checked OAS data-unit downloads."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from .corpus_plan import CorpusPlanError, validate_corpus_plan
from .oas_authority import (
    OASAuthorityError,
    canonical_json_bytes,
    validate_oas_authority_manifest,
)


DOWNLOAD_RECEIPT_FORMAT = "oas_data_unit_download_receipt_v1"
TASK_SUMMARY_FORMAT = "oas_download_task_summary_v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class OASDownloadError(RuntimeError):
    """Report a fail-closed download or data-unit validation failure."""


@dataclass(frozen=True)
class GzipDataUnitInspection:
    """Integrity and schema observations from one complete gzip stream."""

    compressed_bytes: int
    compressed_sha256: str
    data_row_count: int
    header: tuple[str, ...]
    header_sha256: str
    metadata: Mapping[str, Any]
    metadata_sha256: str
    uncompressed_bytes: int
    uncompressed_sha256: str


def file_sha256(path: Path) -> str:
    """Hash a regular file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_authority_manifest(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    """Load one byte-anchored, semantically validated authority manifest."""
    for name, value in (
        ("expected_file_sha256", expected_file_sha256),
        ("expected_payload_sha256", expected_payload_sha256),
    ):
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise OASDownloadError(f"{name} must be a lowercase SHA-256")
    if not path.is_file() or path.is_symlink():
        raise OASDownloadError("Authority manifest must be a regular file")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_file_sha256:
        raise OASDownloadError("Authority manifest file digest differs")
    try:
        manifest = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OASDownloadError("Authority manifest is not ASCII JSON") from error
    try:
        validate_oas_authority_manifest(manifest)
    except OASAuthorityError as error:
        raise OASDownloadError("Authority manifest validation failed") from error
    if manifest["manifest_sha256"] != expected_payload_sha256:
        raise OASDownloadError("Authority manifest payload digest differs")
    return manifest


def load_corpus_plan(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    """Load one byte-anchored and semantically validated corpus plan."""
    for name, value in (
        ("expected_file_sha256", expected_file_sha256),
        ("expected_payload_sha256", expected_payload_sha256),
    ):
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise OASDownloadError(f"{name} must be a lowercase SHA-256")
    if not path.is_file() or path.is_symlink():
        raise OASDownloadError("Corpus plan must be a regular file")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_file_sha256:
        raise OASDownloadError("Corpus plan file digest differs")
    try:
        plan = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OASDownloadError("Corpus plan is not ASCII JSON") from error
    try:
        validate_corpus_plan(plan)
    except CorpusPlanError as error:
        raise OASDownloadError("Corpus plan validation failed") from error
    if plan["plan_sha256"] != expected_payload_sha256:
        raise OASDownloadError("Corpus plan payload digest differs")
    return plan


def select_task_data_units(
    data_units: Sequence[Mapping[str, Any]],
    *,
    task_id: int,
    task_count: int,
    maximum_units: int | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Select a deterministic modulo shard without materializing a permutation."""
    if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count <= 0:
        raise OASDownloadError("task_count must be a positive integer")
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or not 0 <= task_id < task_count
    ):
        raise OASDownloadError("task_id must be in [0, task_count)")
    if maximum_units is not None and (
        isinstance(maximum_units, bool)
        or not isinstance(maximum_units, int)
        or maximum_units <= 0
    ):
        raise OASDownloadError("maximum_units must be a positive integer")
    selected = tuple(
        entry for index, entry in enumerate(data_units) if index % task_count == task_id
    )
    if maximum_units is not None:
        selected = selected[:maximum_units]
    return selected


def select_planned_data_units(
    authority_data_units: Sequence[Mapping[str, Any]],
    planned_data_units: Sequence[Mapping[str, Any]],
    *,
    split: str,
    task_id: int,
    task_count: int,
    maximum_units: int | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Resolve plan memberships to authority entries and select one modulo shard."""
    if split not in {"all", "train", "validation"}:
        raise OASDownloadError("split must be all, train, or validation")
    authority_by_relative: dict[str, Mapping[str, Any]] = {}
    for entry in authority_data_units:
        relative = entry.get("relative_unit")
        if not isinstance(relative, str) or relative in authority_by_relative:
            raise OASDownloadError("Authority data-unit identities are not unique")
        authority_by_relative[relative] = entry
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for membership in planned_data_units:
        if membership.get("split") not in {"train", "validation"}:
            raise OASDownloadError("Planned data-unit split is invalid")
        if split != "all" and membership["split"] != split:
            continue
        relative = membership.get("relative_unit")
        if not isinstance(relative, str) or relative in seen:
            raise OASDownloadError("Planned data-unit identities are not unique")
        seen.add(relative)
        authority = authority_by_relative.get(relative)
        if authority is None:
            raise OASDownloadError("Planned data unit is absent from authority")
        if (
            authority.get("canonical_url") != membership.get("canonical_url")
            or authority.get("study") != membership.get("study")
            or authority.get("unique_sequences") != membership.get("source_records")
        ):
            raise OASDownloadError("Planned data unit disagrees with authority")
        resolved_entry = dict(authority)
        resolved_entry["corpus_plan_membership"] = {
            "expected_selected_records": membership["expected_selected_records"],
            "selection_fraction_denominator": membership[
                "selection_fraction_denominator"
            ],
            "selection_fraction_numerator": membership[
                "selection_fraction_numerator"
            ],
            "selection_hash_namespace": membership["selection_hash_namespace"],
            "split": membership["split"],
            "stratum": membership["stratum"],
        }
        resolved.append(resolved_entry)
    return select_task_data_units(
        resolved,
        task_id=task_id,
        task_count=task_count,
        maximum_units=maximum_units,
    )


def _single_csv_record(line: bytes, label: str) -> str:
    """Decode one physical CSV line that must contain exactly one record field."""
    try:
        decoded = line.decode("utf-8")
        row = next(csv.reader([decoded]))
    except (UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise OASDownloadError(f"{label} is not a valid UTF-8 CSV record") from error
    if len(row) != 1:
        raise OASDownloadError(f"{label} must contain exactly one CSV field")
    return row[0]


def inspect_gzip_data_unit(
    path: Path,
    *,
    expected_unique_sequences: int,
    source_class: str,
) -> GzipDataUnitInspection:
    """Read a complete gzip stream and validate its metadata, header, and row count."""
    if (
        isinstance(expected_unique_sequences, bool)
        or not isinstance(expected_unique_sequences, int)
        or expected_unique_sequences < 0
    ):
        raise OASDownloadError("expected_unique_sequences must be non-negative")
    if source_class not in {"unpaired", "paired"}:
        raise OASDownloadError("source_class must be unpaired or paired")
    if not path.is_file() or path.is_symlink():
        raise OASDownloadError("Downloaded data unit must be a regular file")

    compressed_bytes = path.stat().st_size
    compressed_sha256 = file_sha256(path)
    uncompressed_digest = hashlib.sha256()
    uncompressed_bytes = 0
    data_row_count = 0
    try:
        with gzip.open(path, "rb") as handle:
            metadata_line = handle.readline()
            header_line = handle.readline()
            for prefix in (metadata_line, header_line):
                uncompressed_digest.update(prefix)
                uncompressed_bytes += len(prefix)
            for line in handle:
                uncompressed_digest.update(line)
                uncompressed_bytes += len(line)
                data_row_count += 1
    except (OSError, EOFError) as error:
        raise OASDownloadError("Downloaded data unit failed gzip validation") from error
    if not metadata_line or not header_line:
        raise OASDownloadError("Downloaded data unit is missing metadata or header")

    metadata_text = _single_csv_record(metadata_line, "metadata line")
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as error:
        raise OASDownloadError("OAS metadata field is not JSON") from error
    if not isinstance(metadata, dict):
        raise OASDownloadError("OAS metadata must be a JSON object")
    metadata_unique = metadata.get("Unique sequences")
    if metadata_unique != expected_unique_sequences:
        raise OASDownloadError(
            "OAS metadata unique count differs from the authority manifest"
        )

    try:
        header = tuple(next(csv.reader([header_line.decode("utf-8")])))
    except (UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise OASDownloadError("OAS header is not valid UTF-8 CSV") from error
    if len(header) != len(set(header)) or not header:
        raise OASDownloadError("OAS header is empty or contains duplicate columns")
    required_columns = (
        {"sequence_alignment_aa", "productive", "ANARCI_status"}
        if source_class == "unpaired"
        else {
            "sequence_alignment_aa_heavy",
            "sequence_alignment_aa_light",
            "productive_heavy",
            "productive_light",
        }
    )
    missing_columns = sorted(required_columns - set(header))
    if missing_columns:
        raise OASDownloadError(
            f"OAS data unit is missing required columns: {missing_columns}"
        )
    if data_row_count != expected_unique_sequences:
        raise OASDownloadError(
            "OAS physical data-row count differs from the authority manifest"
        )
    return GzipDataUnitInspection(
        compressed_bytes=compressed_bytes,
        compressed_sha256=compressed_sha256,
        data_row_count=data_row_count,
        header=header,
        header_sha256=hashlib.sha256(header_line).hexdigest(),
        metadata=metadata,
        metadata_sha256=hashlib.sha256(metadata_line).hexdigest(),
        uncompressed_bytes=uncompressed_bytes,
        uncompressed_sha256=uncompressed_digest.hexdigest(),
    )


def _safe_mirrored_path(root: Path, relative_unit: str) -> Path:
    """Resolve a manifest unit below a fixed root and reject path traversal."""
    if not isinstance(relative_unit, str) or not relative_unit:
        raise OASDownloadError("relative_unit must be nonempty text")
    destination = root.joinpath(*relative_unit.split("/"))
    resolved_root = root.resolve()
    resolved_destination = destination.resolve()
    if not resolved_destination.is_relative_to(resolved_root):
        raise OASDownloadError("Data-unit destination escapes its output root")
    return destination


def _write_create_only(path: Path, payload: bytes) -> None:
    """Create one read-only evidence artifact and reject existing targets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def _curl_version(curl_binary: str) -> str:
    """Capture the first curl version line used by the downloader."""
    completed = subprocess.run(
        [curl_binary, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    first_line = completed.stdout.splitlines()[0].strip()
    if not first_line:
        raise OASDownloadError("curl returned no version identity")
    return first_line


def download_data_unit(
    entry: Mapping[str, Any],
    *,
    raw_root: Path,
    work_root: Path,
    receipt_root: Path,
    authority_manifest_sha256: str,
    controller_source_sha256: str,
    downloader_module_sha256: str,
    source_class: str,
    curl_binary: str,
    curl_version: str,
    corpus_plan_payload_sha256: str | None = None,
) -> dict[str, Any]:
    """Download, validate, and atomically admit one manifest-bound data unit."""
    for name, value in (
        ("authority_manifest_sha256", authority_manifest_sha256),
        ("controller_source_sha256", controller_source_sha256),
        ("downloader_module_sha256", downloader_module_sha256),
    ):
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise OASDownloadError(f"{name} must be a lowercase SHA-256")
    relative_unit = entry["relative_unit"]
    destination = _safe_mirrored_path(raw_root, relative_unit)
    partial = _safe_mirrored_path(work_root / "partial", relative_unit + ".partial")
    receipt_path = _safe_mirrored_path(
        receipt_root,
        relative_unit + ".receipt.json",
    )
    lock_digest = hashlib.sha256(entry["canonical_url"].encode("ascii")).hexdigest()
    lock_path = work_root / "locks" / f"{lock_digest}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise OASDownloadError(f"Data unit is already locked: {relative_unit}") from error

        if destination.exists():
            if destination.is_symlink() or not destination.is_file():
                raise OASDownloadError("Existing destination is not a regular file")
            inspection = inspect_gzip_data_unit(
                destination,
                expected_unique_sequences=entry["unique_sequences"],
                source_class=source_class,
            )
            status = "verified_existing"
        else:
            partial.parent.mkdir(parents=True, exist_ok=True)
            command = [
                curl_binary,
                "--silent",
                "--proto",
                "=https",
                "--tlsv1.2",
                "--location",
                "--fail",
                "--show-error",
                "--retry",
                "8",
                "--retry-all-errors",
                "--retry-delay",
                "5",
                "--connect-timeout",
                "30",
                "--speed-limit",
                "1024",
                "--speed-time",
                "120",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                entry["canonical_url"],
            ]
            subprocess.run(command, check=True)
            inspection = inspect_gzip_data_unit(
                partial,
                expected_unique_sequences=entry["unique_sequences"],
                source_class=source_class,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(partial, destination)
            except FileExistsError as error:
                raise OASDownloadError(
                    "Destination appeared during atomic admission"
                ) from error
            partial.unlink()
            destination.chmod(0o444)
            status = "downloaded_and_admitted"

        receipt = {
            "format_version": DOWNLOAD_RECEIPT_FORMAT,
            "admission_status": status,
            "authority_manifest_payload_sha256": authority_manifest_sha256,
            "canonical_url": entry["canonical_url"],
            "compressed_bytes": inspection.compressed_bytes,
            "compressed_sha256": inspection.compressed_sha256,
            "controller_source_sha256": controller_source_sha256,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "curl_version": curl_version,
            "data_row_count": inspection.data_row_count,
            "downloader_module_sha256": downloader_module_sha256,
            "header_sha256": inspection.header_sha256,
            "metadata_sha256": inspection.metadata_sha256,
            "relative_unit": relative_unit,
            "source_class": source_class,
            "test_access": {
                "downstream_heldout_accessed": False,
                "sabdab2_heldout_accessed": False,
            },
            "uncompressed_bytes": inspection.uncompressed_bytes,
            "uncompressed_sha256": inspection.uncompressed_sha256,
            "unique_sequences": entry["unique_sequences"],
        }
        if corpus_plan_payload_sha256 is not None:
            if not _SHA256_PATTERN.fullmatch(corpus_plan_payload_sha256):
                raise OASDownloadError("Corpus plan payload digest is invalid")
            membership = entry.get("corpus_plan_membership")
            if not isinstance(membership, Mapping):
                raise OASDownloadError("Planned download has no plan membership")
            receipt["corpus_plan_membership"] = dict(membership)
            receipt["corpus_plan_payload_sha256"] = corpus_plan_payload_sha256
        receipt["receipt_sha256"] = hashlib.sha256(
            canonical_json_bytes(receipt)
        ).hexdigest()
        if receipt_path.exists():
            try:
                existing = json.loads(receipt_path.read_text(encoding="ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise OASDownloadError("Existing receipt is invalid JSON") from error
            immutable_fields = (
                "authority_manifest_payload_sha256",
                "canonical_url",
                "compressed_bytes",
                "compressed_sha256",
                "data_row_count",
                "downloader_module_sha256",
                "header_sha256",
                "metadata_sha256",
                "relative_unit",
                "source_class",
                "uncompressed_bytes",
                "uncompressed_sha256",
                "unique_sequences",
            )
            if corpus_plan_payload_sha256 is not None:
                immutable_fields += (
                    "corpus_plan_membership",
                    "corpus_plan_payload_sha256",
                )
            if any(existing.get(field) != receipt[field] for field in immutable_fields):
                raise OASDownloadError("Existing receipt disagrees with verified bytes")
        else:
            _write_create_only(receipt_path, canonical_json_bytes(receipt))
        return receipt


def locate_curl() -> tuple[str, str]:
    """Locate curl once and bind its version for task receipts."""
    curl_binary = shutil.which("curl")
    if curl_binary is None:
        raise OASDownloadError("curl is required for resumable HTTPS downloads")
    return curl_binary, _curl_version(curl_binary)
