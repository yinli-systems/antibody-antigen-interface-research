"""Fail-closed helpers for the sharded MMseqs2 gap-aware OAS audit.

MMseqs2 is a sensitive heuristic search engine, not an exhaustive oracle.  The
helpers in this module therefore make a narrower claim: every authenticated
target shard was searched, every development query was represented in the raw
prefilter result database, no raw hit list reached the configured cap, and all
reported alignments independently satisfy the frozen 90/90/90 rule.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import mmap
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping, Optional, Sequence, TextIO, Tuple

from .imgt_query_export import ASEP_QUERY_EXPORT_FORMAT
from .near_identity_export import publish_export_receipt
from .near_identity_overlap import normalize_fraction
from .oas_imgt_export import (
    IMGT_TARGET_RECORD,
    UNUSUAL_TARGET_RECORD,
    sequence_from_alignment,
    unpack_imgt_target,
    unpack_unusual_target,
)


GLOBAL_TARGET_FORMAT = "oas_imgt_target_export_global_v1"
QUERY_MANIFEST_FORMAT = "oas_asep_development_query_manifest_v1"
TARGET_FASTA_FORMAT = "oas_mmseqs_gap_target_fasta_v1"
PREFILTER_AUDIT_FORMAT = "oas_mmseqs_gap_prefilter_audit_v1"
ALIGNMENT_AUDIT_FORMAT = "oas_mmseqs_gap_alignment_audit_v1"
TARGET_ID_PREFIX = "oasgap"
TARGET_SPLIT_CODE = {"train": "t", "validation": "v"}
TARGET_CODE_SPLIT = {value: key for key, value in TARGET_SPLIT_CODE.items()}
TARGET_CHAIN_CODE = {"heavy": "h", "kappa": "k", "lambda": "l"}
TARGET_CODE_CHAIN = {value: key for key, value in TARGET_CHAIN_CODE.items()}
ALIGNMENT_COLUMNS = (
    "query",
    "target",
    "fident",
    "qcov",
    "tcov",
    "alnlen",
    "nident",
    "qstart",
    "qend",
    "qlen",
    "tstart",
    "tend",
    "tlen",
    "qaln",
    "taln",
)
TEST_ACCESS = {
    "development_sequence_fields_accessed": True,
    "frozen_official_external_test_accessed": False,
    "model_predictions_or_scores_accessed": False,
    "task_label_values_used": False,
}


class MMseqsGapAwareError(RuntimeError):
    """Report incomplete inputs, truncation, or contradictory search output."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        body = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise MMseqsGapAwareError("Payload is not finite canonical JSON") from error
    return (body + "\n").encode("ascii")


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constant: " + value)


def _reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Mapping[str, Any]:
    output = {}  # type: dict[str, Any]
    for key, value in pairs:
        if key in output:
            raise ValueError("Duplicate JSON key: " + key)
        output[key] = value
    return output


def regular_file(path: Path, label: str) -> Path:
    candidate = Path(path).expanduser().absolute()
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise MMseqsGapAwareError("Cannot stat " + label) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise MMseqsGapAwareError(label + " is absent, symbolic, or non-regular")
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with regular_file(path, "hashed file").open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MMseqsGapAwareError(label + " is not lowercase SHA-256")
    return value


def nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MMseqsGapAwareError(label + " is not a non-negative integer")
    return value


def signed_json(path: Path, label: str) -> Tuple[Mapping[str, Any], str]:
    path = regular_file(path, label)
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise MMseqsGapAwareError(label + " is not strict ASCII JSON") from error
    if not isinstance(value, Mapping) or raw != canonical_json_bytes(value):
        raise MMseqsGapAwareError(label + " is not canonical JSON")
    unsigned = dict(value)
    claimed = require_sha256(unsigned.pop("receipt_sha256", None), label + " hash")
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != claimed:
        raise MMseqsGapAwareError(label + " self hash differs")
    return value, hashlib.sha256(raw).hexdigest()


def _inventory_file(
    path: Path,
    inventory: Mapping[str, Any],
    *,
    record_bytes: int,
    label: str,
) -> Tuple[Path, int]:
    path = regular_file(path, label)
    records = nonnegative_integer(inventory.get("records"), label + " records")
    byte_count = nonnegative_integer(inventory.get("bytes"), label + " bytes")
    expected_hash = require_sha256(inventory.get("file_sha256"), label + " hash")
    if (
        inventory.get("record_bytes") != record_bytes
        or byte_count != records * record_bytes
        or path.stat().st_size != byte_count
        or sha256_file(path) != expected_hash
    ):
        raise MMseqsGapAwareError(label + " inventory differs")
    return path, records


def target_identifier(
    *, task_id: int, split: str, chain_type: str, authority_ordinal: int
) -> str:
    try:
        split_code = TARGET_SPLIT_CODE[split]
        chain_code = TARGET_CHAIN_CODE[chain_type]
    except KeyError as error:
        raise MMseqsGapAwareError("Target header enum is invalid") from error
    if not 0 <= task_id < 10_000 or not 0 <= authority_ordinal < 2**64:
        raise MMseqsGapAwareError("Target header integer is outside its domain")
    return (
        f"{TARGET_ID_PREFIX}_{task_id:04d}_{split_code}_{chain_code}_"
        f"{authority_ordinal:016x}"
    )


def parse_target_identifier(value: str) -> Mapping[str, Any]:
    fields = value.split("_")
    if len(fields) != 5 or fields[0] != TARGET_ID_PREFIX:
        raise MMseqsGapAwareError("MMseqs target identifier has the wrong schema")
    try:
        task_id = int(fields[1], 10)
        split = TARGET_CODE_SPLIT[fields[2]]
        chain_type = TARGET_CODE_CHAIN[fields[3]]
        ordinal = int(fields[4], 16)
    except (KeyError, ValueError) as error:
        raise MMseqsGapAwareError("MMseqs target identifier enum is invalid") from error
    if target_identifier(
        task_id=task_id,
        split=split,
        chain_type=chain_type,
        authority_ordinal=ordinal,
    ) != value:
        raise MMseqsGapAwareError("MMseqs target identifier is not canonical")
    return {
        "authority_ordinal": ordinal,
        "chain_type": chain_type,
        "split": split,
        "task_id": task_id,
    }


def authenticate_query_fasta(query_root: Path) -> Mapping[str, Any]:
    query_root = Path(query_root).expanduser().absolute()
    receipt, receipt_file_sha256 = signed_json(
        query_root / "EXPORT_RECEIPT.json", "query receipt"
    )
    manifest, manifest_file_sha256 = signed_json(
        query_root / "sequence_manifest.json", "query sequence manifest"
    )
    outputs = receipt.get("outputs")
    if (
        receipt.get("format_version") != ASEP_QUERY_EXPORT_FORMAT
        or receipt.get("gap_aware_query_count") != 3_106
        or receipt.get("query_count") != 3_106
        or receipt.get("test_access") != TEST_ACCESS
        or manifest.get("format_version") != QUERY_MANIFEST_FORMAT
        or manifest.get("query_count") != 3_106
        or not isinstance(outputs, Mapping)
        or not isinstance(manifest.get("records"), list)
    ):
        raise MMseqsGapAwareError("Query gap-aware scope differs")
    fasta_path = regular_file(query_root / "queries.fasta", "query FASTA")
    for name, path in (
        ("queries.fasta", fasta_path),
        ("sequence_manifest.json", query_root / "sequence_manifest.json"),
    ):
        inventory = outputs.get(name)
        if (
            not isinstance(inventory, Mapping)
            or inventory.get("bytes") != path.stat().st_size
            or inventory.get("sha256") != sha256_file(path)
        ):
            raise MMseqsGapAwareError("Query inventory differs for " + name)
    records = manifest["records"]
    query_lengths = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise MMseqsGapAwareError("Query manifest row is not an object")
        key = row.get("query_reference_key")
        length = row.get("sequence_length")
        if (
            not isinstance(key, str)
            or not key.startswith("asep_")
            or len(key) != 69
            or key in query_lengths
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
        ):
            raise MMseqsGapAwareError("Query manifest row identity differs")
        query_lengths[key] = length
    if len(query_lengths) != 3_106:
        raise MMseqsGapAwareError("Query manifest accounting differs")
    return {
        "fasta_path": fasta_path,
        "fasta_sha256": sha256_file(fasta_path),
        "manifest_file_sha256": manifest_file_sha256,
        "query_lengths": query_lengths,
        "query_receipt_file_sha256": receipt_file_sha256,
        "query_receipt_sha256": receipt["receipt_sha256"],
    }


def export_authenticated_target_task(
    *, target_output_root: Path, task_id: int, output_fasta: Path
) -> Mapping[str, Any]:
    """Stream one finalized target task into a digest-bound MMseqs FASTA."""
    target_output_root = Path(target_output_root).expanduser().absolute()
    global_receipt, global_file_sha256 = signed_json(
        target_output_root / "global_receipt.json", "global target receipt"
    )
    boundary = global_receipt.get("claim_boundary")
    tasks = global_receipt.get("task_receipts")
    if (
        global_receipt.get("format_version") != GLOBAL_TARGET_FORMAT
        or not isinstance(boundary, Mapping)
        or boundary.get("global_task_set_complete") is not True
        or boundary.get("accepted_oas_domain_partition_complete") is not True
        or boundary.get("near_identity_search_complete") is not False
        or not isinstance(tasks, list)
        or global_receipt.get("task_count") != len(tasks)
        or len(tasks) != 32
        or not 0 <= task_id < len(tasks)
    ):
        raise MMseqsGapAwareError("Global target receipt is incomplete or out of scope")
    task = tasks[task_id]
    if not isinstance(task, Mapping) or task.get("task_id") != task_id:
        raise MMseqsGapAwareError("Target task ordering differs")
    inventory = task.get("output_inventory")
    counts = task.get("counts")
    if not isinstance(inventory, Mapping) or not isinstance(counts, Mapping):
        raise MMseqsGapAwareError("Target task inventory is absent")
    representable = inventory.get("representable")
    unusual = inventory.get("unusual")
    if not isinstance(representable, Mapping) or not isinstance(unusual, Mapping):
        raise MMseqsGapAwareError("Target task streams are absent")
    task_root = target_output_root / "imgt-map" / f"mapper-{task_id:04d}"
    representable_path, representable_count = _inventory_file(
        task_root / "representable.bin",
        representable,
        record_bytes=IMGT_TARGET_RECORD.size,
        label="representable target binary",
    )
    unusual_path, unusual_count = _inventory_file(
        task_root / "unusual.bin",
        unusual,
        record_bytes=UNUSUAL_TARGET_RECORD.size,
        label="unusual target binary",
    )
    accepted = nonnegative_integer(counts.get("accepted_records"), "accepted targets")
    if (
        accepted != representable_count + unusual_count
        or counts.get("representable_records") != representable_count
        or counts.get("unusual_records") != unusual_count
    ):
        raise MMseqsGapAwareError("Target task count accounting differs")

    output_fasta = Path(output_fasta).expanduser().absolute()
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output_fasta, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    observed_ordinals = set()
    observed = Counter()  # type: Counter[str]
    fasta_digest = hashlib.sha256()
    fasta_bytes = 0
    try:
        with os.fdopen(descriptor, "wb", buffering=16 * 1024 * 1024) as output:
            streams = (
                (representable_path, IMGT_TARGET_RECORD.size, unpack_imgt_target),
                (unusual_path, UNUSUAL_TARGET_RECORD.size, unpack_unusual_target),
            )
            for path, width, unpacker in streams:
                with path.open("rb", buffering=16 * 1024 * 1024) as handle:
                    while True:
                        payload = handle.read(width)
                        if not payload:
                            break
                        if len(payload) != width:
                            raise MMseqsGapAwareError("Target stream ended mid-record")
                        record = unpacker(payload)
                        ordinal = record.authority_ordinal
                        if ordinal in observed_ordinals:
                            raise MMseqsGapAwareError("Target ordinal is duplicated in task")
                        observed_ordinals.add(ordinal)
                        if hasattr(record, "alignment"):
                            sequence = sequence_from_alignment(record.alignment)
                        else:
                            sequence = record.sequence
                        identifier = target_identifier(
                            task_id=task_id,
                            split=record.split,
                            chain_type=record.chain_type,
                            authority_ordinal=ordinal,
                        )
                        row = f">{identifier}\n{sequence}\n".encode("ascii")
                        output.write(row)
                        fasta_digest.update(row)
                        fasta_bytes += len(row)
                        observed["records"] += 1
                        observed["split:" + record.split] += 1
                        observed["chain:" + record.chain_type] += 1
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        output_fasta.unlink(missing_ok=True)
        raise
    if observed["records"] != accepted or len(observed_ordinals) != accepted:
        output_fasta.unlink(missing_ok=True)
        raise MMseqsGapAwareError("Exported target count differs")
    expected_counts = {
        key: value
        for key, value in counts.items()
        if key.startswith("split:") or key.startswith("chain:")
    }
    if any(observed[key] != value for key, value in expected_counts.items()):
        output_fasta.unlink(missing_ok=True)
        raise MMseqsGapAwareError("Exported target split or chain count differs")
    return {
        "format_version": TARGET_FASTA_FORMAT,
        "counts": {key: observed[key] for key in sorted(observed)},
        "fasta_bytes": fasta_bytes,
        "fasta_path": str(output_fasta),
        "fasta_sha256": fasta_digest.hexdigest(),
        "global_target_receipt_file_sha256": global_file_sha256,
        "global_target_receipt_sha256": global_receipt["receipt_sha256"],
        "task_id": task_id,
        "test_access": TEST_ACCESS,
    }


def _prefilter_data_path(prefix: Path) -> Path:
    prefix = Path(prefix).expanduser().absolute()
    if prefix.exists() and prefix.is_file() and not prefix.is_symlink():
        if Path(str(prefix) + ".0").exists():
            raise MMseqsGapAwareError("Split prefilter data files are unsupported")
        return prefix
    candidates = sorted(prefix.parent.glob(prefix.name + ".[0-9]*"))
    if len(candidates) != 1 or candidates[0].name != prefix.name + ".0":
        raise MMseqsGapAwareError("Prefilter data file is absent or split unexpectedly")
    return regular_file(candidates[0], "prefilter data")


def audit_prefilter_database(
    *, prefix: Path, expected_query_count: int, maximum_sequences: int
) -> Mapping[str, Any]:
    """Count every raw prefilter record and reject possible cap saturation."""
    if expected_query_count <= 0 or maximum_sequences <= 0:
        raise MMseqsGapAwareError("Prefilter audit configuration is invalid")
    prefix = Path(prefix).expanduser().absolute()
    index_path = regular_file(Path(str(prefix) + ".index"), "prefilter index")
    data_path = _prefilter_data_path(prefix)
    entries = []
    with index_path.open("r", encoding="ascii", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise MMseqsGapAwareError(
                    f"Prefilter index row {line_number} has the wrong width"
                )
            try:
                key, offset, size = (int(field, 10) for field in fields)
            except ValueError as error:
                raise MMseqsGapAwareError("Prefilter index integer is invalid") from error
            if key < 0 or offset < 0 or size <= 0:
                raise MMseqsGapAwareError("Prefilter index value is outside its domain")
            entries.append((key, offset, size))
    keys = [entry[0] for entry in entries]
    if keys != list(range(expected_query_count)):
        raise MMseqsGapAwareError("Prefilter query accounting is incomplete or unordered")
    file_size = data_path.stat().st_size
    hit_counts = []
    with data_path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as payload:
            for _, offset, size in entries:
                if offset + size > file_size:
                    raise MMseqsGapAwareError("Prefilter record exceeds the data file")
                record = payload[offset : offset + size]
                if not record.endswith(b"\0"):
                    raise MMseqsGapAwareError("Prefilter record lacks its NUL terminator")
                body = record[:-1]
                if body and not body.endswith(b"\n"):
                    raise MMseqsGapAwareError("Prefilter record lacks its final newline")
                count = body.count(b"\n")
                if count >= maximum_sequences:
                    raise MMseqsGapAwareError(
                        "Prefilter hit count reached --max-seqs; truncation is possible"
                    )
                hit_counts.append(count)
    ordered = sorted(hit_counts)
    quantile = lambda fraction: ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    return {
        "format_version": PREFILTER_AUDIT_FORMAT,
        "data_bytes": file_size,
        "data_sha256": sha256_file(data_path),
        "hit_count_maximum": max(hit_counts),
        "hit_count_minimum": min(hit_counts),
        "hit_count_p50": quantile(0.50),
        "hit_count_p95": quantile(0.95),
        "hit_count_p99": quantile(0.99),
        "hit_count_total": sum(hit_counts),
        "index_bytes": index_path.stat().st_size,
        "index_sha256": sha256_file(index_path),
        "maximum_sequences": maximum_sequences,
        "query_count": len(entries),
        "saturation_detected": False,
    }


def _parse_int(value: str, label: str) -> int:
    try:
        output = int(value, 10)
    except ValueError as error:
        raise MMseqsGapAwareError(label + " is not an integer") from error
    if output < 0:
        raise MMseqsGapAwareError(label + " is negative")
    return output


def audit_alignment_rows(
    lines: Iterable[str],
    *,
    query_lengths: Mapping[str, int],
    expected_task_id: int,
    hits_output: Optional[TextIO] = None,
    threshold: float = 0.9,
) -> Mapping[str, Any]:
    """Independently replay exact identity and bidirectional coverage."""
    if not 0 < threshold <= 1:
        raise MMseqsGapAwareError("Alignment threshold is invalid")
    pairs = set()
    queries = set()
    targets = set()
    split_counts = Counter()  # type: Counter[str]
    rows = 0
    # convertalis rounds its printed fractions.  Exact thresholding below uses
    # the replayed alignment strings, while this tolerance only authenticates
    # that the rounded text describes the same alignment.
    tolerance = 5.1e-4
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) != len(ALIGNMENT_COLUMNS):
            raise MMseqsGapAwareError(
                f"Alignment row {line_number} does not have {len(ALIGNMENT_COLUMNS)} columns"
            )
        (
            query,
            target,
            raw_identity,
            raw_qcov,
            raw_tcov,
            raw_alnlen,
            raw_nident,
            raw_qstart,
            raw_qend,
            raw_qlen,
            raw_tstart,
            raw_tend,
            raw_tlen,
            query_alignment,
            target_alignment,
        ) = fields
        if query not in query_lengths:
            raise MMseqsGapAwareError("Alignment contains an unknown query")
        identity = normalize_fraction(raw_identity)
        query_coverage = normalize_fraction(raw_qcov)
        target_coverage = normalize_fraction(raw_tcov)
        alnlen = _parse_int(raw_alnlen, "alignment length")
        nident = _parse_int(raw_nident, "identical count")
        qstart = _parse_int(raw_qstart, "query start")
        qend = _parse_int(raw_qend, "query end")
        qlen = _parse_int(raw_qlen, "query length")
        tstart = _parse_int(raw_tstart, "target start")
        tend = _parse_int(raw_tend, "target end")
        tlen = _parse_int(raw_tlen, "target length")
        target_metadata = parse_target_identifier(target)
        pair = (query, target)
        if pair in pairs or target_metadata["task_id"] != expected_task_id:
            raise MMseqsGapAwareError("Alignment pair is duplicated or cross-shard")
        pairs.add(pair)
        if (
            qlen != query_lengths[query]
            or qlen <= 0
            or tlen <= 0
            or alnlen <= 0
            or len(query_alignment) != alnlen
            or len(target_alignment) != alnlen
            or qstart <= 0
            or tstart <= 0
            or qend < qstart
            or tend < tstart
            or qend > qlen
            or tend > tlen
        ):
            raise MMseqsGapAwareError("Alignment coordinates or lengths differ")
        observed_q = sum(value != "-" for value in query_alignment)
        observed_t = sum(value != "-" for value in target_alignment)
        observed_identical = sum(
            left == right and left != "-"
            for left, right in zip(query_alignment, target_alignment)
        )
        replay_identity = observed_identical / alnlen
        replay_qcov = observed_q / qlen
        replay_tcov = observed_t / tlen
        if (
            observed_q != qend - qstart + 1
            or observed_t != tend - tstart + 1
            or observed_identical != nident
            or abs(identity - replay_identity) > tolerance
            or abs(query_coverage - replay_qcov) > tolerance
            or abs(target_coverage - replay_tcov) > tolerance
        ):
            raise MMseqsGapAwareError("Independent alignment metric replay differs")
        if min(replay_identity, replay_qcov, replay_tcov) + 1e-12 < threshold:
            raise MMseqsGapAwareError("Alignment output violates the frozen threshold")
        if hits_output is not None:
            hits_output.write(
                f"{query}\t{target}\t{replay_identity:.12f}\t"
                f"{replay_qcov:.12f}\t{replay_tcov:.12f}\n"
            )
        rows += 1
        queries.add(query)
        targets.add(target)
        split_counts[str(target_metadata["split"])] += 1
    return {
        "format_version": ALIGNMENT_AUDIT_FORMAT,
        "alignment_columns": list(ALIGNMENT_COLUMNS),
        "alignment_rows": rows,
        "matched_query_count": len(queries),
        "matched_target_count": len(targets),
        "split_hit_counts": {
            "train": split_counts["train"],
            "validation": split_counts["validation"],
        },
        "thresholds": {
            "identity_minimum_inclusive": threshold,
            "query_coverage_minimum_inclusive": threshold,
            "target_coverage_minimum_inclusive": threshold,
        },
    }


__all__ = [
    "ALIGNMENT_COLUMNS",
    "ALIGNMENT_AUDIT_FORMAT",
    "MMseqsGapAwareError",
    "PREFILTER_AUDIT_FORMAT",
    "TARGET_FASTA_FORMAT",
    "TEST_ACCESS",
    "audit_alignment_rows",
    "audit_prefilter_database",
    "authenticate_query_fasta",
    "canonical_json_bytes",
    "export_authenticated_target_task",
    "parse_target_identifier",
    "publish_export_receipt",
    "sha256_file",
    "signed_json",
    "target_identifier",
]
