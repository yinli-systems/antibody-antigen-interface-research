"""Annotate AVIDa IL-6 constructs from pinned UniProt authority bytes."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from interaction_first.avida_annotation_protocol import (
    load_avida_annotation_protocol,
)


SOURCE_ID = "avida_hil6_2024_11"
INPUT_STATUS = "raw_construct_untrimmed_signal_and_tag_boundaries_pending"
ACCEPTED_STATUS = "mature_chain_verified_single_variant"
QUARANTINED_STATUS = "quarantined_antigen_boundary"
_ANTIGEN_ID = re.compile(r"^ag_sha256:[0-9a-f]{64}$")
_MUTANT_LABEL = re.compile(r"^IL-6_([A-Z])(\d+)([A-Z])$")


class AvidaAntigenAnnotationError(ValueError):
    """Raised when antigen annotation cannot produce authenticated evidence."""


def annotate_avida_antigen_boundaries(
    *,
    candidate_root: Path,
    protocol_path: Path,
    uniprot_json: Path,
    output_root: Path,
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Derive mature IL-6 chains while preserving raw construct provenance."""

    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    _utc(timestamp)
    protocol, protocol_sha256 = load_avida_annotation_protocol(protocol_path)
    reference_sequence, authority = _load_uniprot_authority(
        uniprot_json,
        protocol=protocol,
    )
    candidate_root = _directory(candidate_root, "candidate root")
    candidate_receipt_path = _regular_file(
        candidate_root / "CANDIDATE_RECEIPT.json",
        "candidate receipt",
    )
    candidate_receipt_sha256 = _file_sha256(candidate_receipt_path)
    if candidate_receipt_sha256 != protocol["input_binding"][
        "candidate_receipt_sha256"
    ]:
        raise AvidaAntigenAnnotationError("candidate receipt digest differs from protocol")
    candidate_receipt = _strict_json(candidate_receipt_path, "candidate receipt")
    antigen_path = _authenticate_candidate(candidate_root, candidate_receipt)
    inputs = _load_antigens(
        antigen_path,
        expected_count=protocol["input_binding"]["expected_antigens"],
    )
    output_root = Path(output_root).expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise AvidaAntigenAnnotationError("annotation output root already exists")
    if not output_root.parent.is_dir() or output_root.parent.is_symlink():
        raise AvidaAntigenAnnotationError("annotation output parent is invalid")
    output_root.mkdir(mode=0o750)
    try:
        annotations = [
            _interpret_antigen(
                row=row,
                reference_sequence=reference_sequence,
                protocol=protocol,
            )
            for row in inputs
        ]
        table_path = output_root / "ANTIGEN_BOUNDARY_ANNOTATIONS.parquet"
        _write_table(table_path, annotations)
        status_counts = Counter(row["annotation_status"] for row in annotations)
        reason_counts = Counter(
            reason for row in annotations for reason in row["quarantine_reasons"]
        )
        wild_types = sum(row["mutation_count"] == 0 for row in annotations)
        mutants = sum(row["mutation_count"] == 1 for row in annotations)
        counts = {
            "accepted": status_counts[ACCEPTED_STATUS],
            "input_antigens": len(inputs),
            "mutant_constructs": mutants,
            "quarantined": status_counts[QUARANTINED_STATUS],
            "quarantine_reasons": dict(sorted(reason_counts.items())),
            "wild_type_constructs": wild_types,
        }
        expected_variants = protocol["antigen_boundary"]["variant_policy"]
        if (
            counts["accepted"] + counts["quarantined"] != len(inputs)
            or wild_types != expected_variants["expected_wild_type_constructs"]
            or mutants != expected_variants["expected_mutant_constructs"]
        ):
            raise AvidaAntigenAnnotationError("antigen annotation accounting differs")
        table_record = _file_record(table_path, output_root)
        audit = {
            "annotation_counts": counts,
            "annotator_source_sha256": _file_sha256(Path(__file__).resolve()),
            "authority": authority,
            "candidate_receipt_sha256": candidate_receipt_sha256,
            "created_at_utc": timestamp,
            "format_version": "pairbench_avida_antigen_annotation_audit_v1",
            "input_antigen_table_sha256": _file_sha256(antigen_path),
            "normalization_eligible": False,
            "normalization_blockers": [
                "independent_antigen_annotation_audit_pending",
                "vhh_domain_boundary_annotation_pending",
            ],
            "protocol_sha256": protocol_sha256,
            "source_id": SOURCE_ID,
            "table": table_record,
            "unknown_or_dropped_antigens": 0,
        }
        audit_path = output_root / "AUDIT_SUMMARY.json"
        _write_create_only_json(audit_path, audit)
        receipt = {
            "annotation_counts": counts,
            "candidate_receipt_sha256": candidate_receipt_sha256,
            "created_at_utc": timestamp,
            "format_version": "pairbench_avida_antigen_annotation_receipt_v1",
            "normalization_eligible": False,
            "normalization_blockers": audit["normalization_blockers"],
            "outputs": sorted(
                [table_record, _file_record(audit_path, output_root)],
                key=lambda value: value["path"],
            ),
            "protocol_sha256": protocol_sha256,
            "source_id": SOURCE_ID,
            "status": "annotation_complete_with_explicit_quarantine",
        }
        _write_create_only_json(output_root / "ANNOTATION_RECEIPT.json", receipt)
        return receipt
    except Exception as exc:
        failure_path = output_root / "FAILURE_RECEIPT.json"
        if not failure_path.exists():
            _write_create_only_json(
                failure_path,
                {
                    "created_at_utc": timestamp,
                    "error_message": str(exc),
                    "error_type": type(exc).__name__,
                    "format_version": "pairbench_avida_antigen_annotation_failure_v1",
                    "protocol_sha256": protocol_sha256,
                    "source_id": SOURCE_ID,
                },
            )
        raise


def _interpret_antigen(
    *,
    row: Mapping[str, Any],
    reference_sequence: str,
    protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    reasons: list[str] = []
    construct_policy = protocol["antigen_boundary"]["construct"]
    variant_policy = protocol["antigen_boundary"]["variant_policy"]
    construct = row["construct_sequences"][0]
    tag = construct_policy["expected_c_terminal_tag"]
    precursor_length = protocol["antigen_boundary"]["authority"]["sequence_length"]
    if len(construct) != construct_policy["expected_construct_length"]:
        reasons.append("construct_length_differs")
    if not construct.endswith(tag):
        reasons.append("construct_tag_differs")
    precursor = construct[:precursor_length] if len(construct) >= precursor_length else None
    suffix = construct[precursor_length:] if len(construct) >= precursor_length else None
    if suffix != tag:
        reasons.append("precursor_suffix_differs")
    mutations = (
        [
            (position + 1, reference, observed)
            for position, (reference, observed) in enumerate(
                zip(reference_sequence, precursor, strict=True)
            )
            if reference != observed
        ]
        if precursor is not None and len(precursor) == len(reference_sequence)
        else []
    )
    label = row["variant_id"]
    label_match = _MUTANT_LABEL.fullmatch(label)
    if label == "IL-6_WTs":
        if mutations:
            reasons.append("wild_type_sequence_differs")
    elif label_match is None:
        reasons.append("variant_label_invalid")
    elif len(mutations) != variant_policy["allowed_mutant_count_per_construct"]:
        reasons.append("variant_mutation_count_differs")
    else:
        position, reference, observed = mutations[0]
        if (
            reference != label_match.group(1)
            or position != int(label_match.group(2))
            or observed != label_match.group(3)
            or observed != variant_policy["allowed_mutant_residue"]
        ):
            reasons.append("variant_label_sequence_disagrees")
    mature_start = construct_policy["mature_chain_start_one_based_inclusive"]
    mature_end = construct_policy["mature_chain_end_one_based_inclusive"]
    if any(not mature_start <= position <= mature_end for position, _, _ in mutations):
        reasons.append("mutation_outside_mature_chain")
    signal_sequence = precursor[: construct_policy["signal_end_one_based_inclusive"]] if precursor else None
    mature_sequence = precursor[mature_start - 1 : mature_end] if precursor else None
    canonical_reasons = sorted(set(reasons))
    mutation = mutations[0] if len(mutations) == 1 else None
    return {
        "annotation_status": ACCEPTED_STATUS if not canonical_reasons else QUARANTINED_STATUS,
        "antigen_id": row["antigen_id"],
        "construct_sequence_digest": row["sequence_digest"],
        "mature_chain_sequence": mature_sequence,
        "mature_chain_sequence_digest": (
            hashlib.sha256(mature_sequence.encode()).hexdigest()
            if mature_sequence is not None
            else None
        ),
        "mutation_count": len(mutations),
        "mutation_position_uniprot_one_based": mutation[0] if mutation else None,
        "mutant_residue": mutation[2] if mutation else None,
        "precursor_sequence": precursor,
        "quarantine_reasons": canonical_reasons,
        "reference_residue": mutation[1] if mutation else None,
        "signal_sequence": signal_sequence,
        "terminal_tag_sequence": suffix,
        "uniprot_accession": protocol["antigen_boundary"]["authority"]["accession"],
        "variant_id": label,
    }


def _load_uniprot_authority(
    path: Path,
    *,
    protocol: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    path = _regular_file(path, "UniProt JSON")
    authority = protocol["antigen_boundary"]["authority"]
    digest = _file_sha256(path)
    if digest != authority["json_sha256"]:
        raise AvidaAntigenAnnotationError("UniProt JSON digest differs")
    payload = _strict_json(path, "UniProt JSON")
    audit = payload.get("entryAudit")
    sequence = payload.get("sequence")
    if type(audit) is not dict or type(sequence) is not dict:
        raise AvidaAntigenAnnotationError("UniProt authority schema differs")
    reference_sequence = sequence.get("value")
    if (
        payload.get("primaryAccession") != authority["accession"]
        or audit.get("entryVersion") != authority["entry_version"]
        or audit.get("sequenceVersion") != authority["sequence_version"]
        or audit.get("lastAnnotationUpdateDate")
        != authority["last_annotation_update_date"]
        or sequence.get("length") != authority["sequence_length"]
        or type(reference_sequence) is not str
        or len(reference_sequence) != authority["sequence_length"]
    ):
        raise AvidaAntigenAnnotationError("UniProt authority identity differs")
    construct = protocol["antigen_boundary"]["construct"]
    signal_hits = [
        feature
        for feature in payload.get("features", [])
        if type(feature) is dict and feature.get("type") == "Signal"
    ]
    chain_hits = [
        feature
        for feature in payload.get("features", [])
        if type(feature) is dict
        and feature.get("type") == "Chain"
        and feature.get("featureId") == construct["mature_chain_feature_id"]
    ]
    if len(signal_hits) != 1 or len(chain_hits) != 1:
        raise AvidaAntigenAnnotationError("UniProt boundary feature multiplicity differs")
    _validate_location(
        signal_hits[0].get("location"),
        start=construct["signal_start_one_based_inclusive"],
        end=construct["signal_end_one_based_inclusive"],
        label="signal",
    )
    _validate_location(
        chain_hits[0].get("location"),
        start=construct["mature_chain_start_one_based_inclusive"],
        end=construct["mature_chain_end_one_based_inclusive"],
        label="mature chain",
    )
    return reference_sequence, {
        "accession": authority["accession"],
        "entry_version": authority["entry_version"],
        "json_sha256": digest,
        "last_annotation_update_date": authority["last_annotation_update_date"],
        "mature_chain_feature_id": construct["mature_chain_feature_id"],
        "sequence_version": authority["sequence_version"],
    }


def _validate_location(value: Any, *, start: int, end: int, label: str) -> None:
    if type(value) is not dict:
        raise AvidaAntigenAnnotationError(f"UniProt {label} location differs")
    observed_start = value.get("start")
    observed_end = value.get("end")
    if (
        type(observed_start) is not dict
        or type(observed_end) is not dict
        or observed_start != {"value": start, "modifier": "EXACT"}
        or observed_end != {"value": end, "modifier": "EXACT"}
    ):
        raise AvidaAntigenAnnotationError(f"UniProt {label} location differs")


def _authenticate_candidate(
    candidate_root: Path,
    receipt: Mapping[str, Any],
) -> Path:
    if (
        receipt.get("format_version") != "pairbench_avida_candidate_receipt_v1"
        or receipt.get("source_id") != SOURCE_ID
        or receipt.get("status") != "candidate_complete_blocked"
        or receipt.get("registry_normalization_eligible") is not False
    ):
        raise AvidaAntigenAnnotationError("candidate receipt authority differs")
    records = receipt.get("outputs")
    if type(records) is not list:
        raise AvidaAntigenAnnotationError("candidate output inventory differs")
    by_path = {
        record.get("path"): record for record in records if type(record) is dict
    }
    record = by_path.get("entities_antigen.parquet")
    if type(record) is not dict or set(record) != {"path", "sha256", "size_bytes"}:
        raise AvidaAntigenAnnotationError("candidate antigen record differs")
    path = _regular_file(candidate_root / record["path"], "candidate antigen table")
    if path.stat().st_size != record["size_bytes"] or _file_sha256(path) != record[
        "sha256"
    ]:
        raise AvidaAntigenAnnotationError("candidate antigen table bytes differ")
    return path


def _load_antigens(path: Path, *, expected_count: int) -> list[Mapping[str, Any]]:
    table = pq.read_table(
        path,
        columns=[
            "annotation_status",
            "antigen_id",
            "construct_sequences",
            "sequence_digest",
            "source_id",
            "variant_id",
        ],
    )
    if table.num_rows != expected_count:
        raise AvidaAntigenAnnotationError("candidate antigen count differs")
    rows = table.to_pylist()
    identities: list[str] = []
    variants: list[str] = []
    for row in rows:
        constructs = row["construct_sequences"]
        if type(constructs) is not list or len(constructs) != 1 or type(constructs[0]) is not str:
            raise AvidaAntigenAnnotationError("candidate antigen construct differs")
        construct = constructs[0]
        digest = hashlib.sha256(construct.encode()).hexdigest()
        identity = row["antigen_id"]
        if (
            _ANTIGEN_ID.fullmatch(identity or "") is None
            or identity != f"ag_sha256:{digest}"
            or row["sequence_digest"] != digest
            or row["annotation_status"] != INPUT_STATUS
            or row["source_id"] != SOURCE_ID
            or type(row["variant_id"]) is not str
        ):
            raise AvidaAntigenAnnotationError("candidate antigen semantics differ")
        identities.append(identity)
        variants.append(row["variant_id"])
    if len(set(identities)) != len(identities) or variants != sorted(set(variants)):
        raise AvidaAntigenAnnotationError("candidate antigens are not canonical and unique")
    return rows


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("annotation_status", pa.string(), nullable=False),
            pa.field("antigen_id", pa.string(), nullable=False),
            pa.field("construct_sequence_digest", pa.string(), nullable=False),
            pa.field("mature_chain_sequence", pa.string()),
            pa.field("mature_chain_sequence_digest", pa.string()),
            pa.field("mutation_count", pa.int8(), nullable=False),
            pa.field("mutation_position_uniprot_one_based", pa.int16()),
            pa.field("mutant_residue", pa.string()),
            pa.field("precursor_sequence", pa.string()),
            pa.field("quarantine_reasons", pa.list_(pa.string()), nullable=False),
            pa.field("reference_residue", pa.string()),
            pa.field("signal_sequence", pa.string()),
            pa.field("terminal_tag_sequence", pa.string()),
            pa.field("uniprot_accession", pa.string(), nullable=False),
            pa.field("variant_id", pa.string(), nullable=False),
        ]
    )


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("xb") as handle:
        pq.write_table(
            pa.Table.from_pylist(list(rows), schema=_schema()),
            handle,
            compression="zstd",
            compression_level=9,
            use_dictionary=True,
            write_statistics=True,
        )


def _file_record(path: Path, root: Path) -> Mapping[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _strict_json(path: Path, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise AvidaAntigenAnnotationError(
            f"non-finite JSON constant in {label}: {value}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AvidaAntigenAnnotationError(f"duplicate JSON key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AvidaAntigenAnnotationError(f"cannot parse {label}") from exc
    if type(value) is not dict:
        raise AvidaAntigenAnnotationError(f"{label} must be an object")
    return value


def _directory(path: Path, label: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_dir():
        raise AvidaAntigenAnnotationError(f"{label} is invalid")
    return result


def _regular_file(path: Path, label: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_file():
        raise AvidaAntigenAnnotationError(f"{label} is invalid")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def _utc(value: str) -> None:
    if type(value) is not str or not value.endswith("Z"):
        raise AvidaAntigenAnnotationError("annotation timestamp must use UTC Z form")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AvidaAntigenAnnotationError("annotation timestamp is invalid") from exc
