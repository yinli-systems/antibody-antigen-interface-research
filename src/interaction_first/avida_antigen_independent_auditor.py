"""Independently reconstruct and audit AVIDa IL-6 boundary annotations."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
from typing import Any, Iterator, Mapping

import pyarrow.parquet as pq


SOURCE_ID = "avida_hil6_2024_11"
PROTOCOL_SHA256 = "63d88b7ff3f37c80d9478cad27cb11d9099258f83e15b49c6313684a2c2d0cca"
CANDIDATE_RECEIPT_SHA256 = "eceb95ab6e94e711fd3a7ad48e6bf42bf05f6b0d2a455a7921597ec2b0832dbe"
UNIPROT_JSON_SHA256 = "e507d4a8f6c19d58ccedeaec1ca72435a18dbe18f9bd4d901b3f07776323d0e2"
INPUT_TABLE_SHA256 = "8e7b58bc07f2dba2cbaade1ed7f458aa3305a43b9c7ebd17877d72a7af5904b4"
ACCEPTED_STATUS = "mature_chain_verified_single_variant"
_LABEL = re.compile(r"^IL-6_([A-Z])(\d+)([A-Z])$")


class AvidaAntigenIndependentAuditError(ValueError):
    """Raised when independently reconstructed antigen evidence disagrees."""


def audit_avida_antigen_annotations(
    *,
    candidate_root: Path,
    annotation_root: Path,
    protocol_path: Path,
    uniprot_json: Path,
    annotator_source: Path,
    output_receipt: Path,
    audited_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Reconstruct every mature chain without importing the primary annotator."""

    timestamp = audited_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    _utc(timestamp)
    candidate_root = _directory(candidate_root, "candidate root")
    annotation_root = _directory(annotation_root, "annotation root")
    protocol_path = _regular_file(protocol_path, "annotation protocol")
    uniprot_json = _regular_file(uniprot_json, "UniProt JSON")
    annotator_source = _regular_file(annotator_source, "annotator source")
    if _file_sha256(protocol_path) != PROTOCOL_SHA256:
        raise AvidaAntigenIndependentAuditError("annotation protocol digest differs")
    protocol = _strict_json(protocol_path, "annotation protocol")
    _audit_protocol(protocol)
    if _file_sha256(uniprot_json) != UNIPROT_JSON_SHA256:
        raise AvidaAntigenIndependentAuditError("UniProt JSON digest differs")
    reference = _audit_uniprot(_strict_json(uniprot_json, "UniProt JSON"))
    candidate_receipt_path = _regular_file(
        candidate_root / "CANDIDATE_RECEIPT.json",
        "candidate receipt",
    )
    if _file_sha256(candidate_receipt_path) != CANDIDATE_RECEIPT_SHA256:
        raise AvidaAntigenIndependentAuditError("candidate receipt digest differs")
    candidate_receipt = _strict_json(candidate_receipt_path, "candidate receipt")
    candidate_table = _candidate_antigen_table(candidate_root, candidate_receipt)
    annotation_receipt_path = _regular_file(
        annotation_root / "ANNOTATION_RECEIPT.json",
        "annotation receipt",
    )
    annotation_receipt = _strict_json(annotation_receipt_path, "annotation receipt")
    outputs = _audit_annotation_root(annotation_root, annotation_receipt)
    audit_summary = _strict_json(
        annotation_root / "AUDIT_SUMMARY.json",
        "annotation audit summary",
    )
    if audit_summary.get("annotator_source_sha256") != _file_sha256(annotator_source):
        raise AvidaAntigenIndependentAuditError("annotator source digest differs")
    annotation_table = annotation_root / "ANTIGEN_BOUNDARY_ANNOTATIONS.parquet"
    rows = 0
    wild_types = 0
    mutants = 0
    sentinel = object()
    for candidate, observed in zip_longest(
        _parquet_rows(candidate_table),
        _parquet_rows(annotation_table),
        fillvalue=sentinel,
    ):
        if candidate is sentinel or observed is sentinel:
            raise AvidaAntigenIndependentAuditError("antigen row counts differ")
        expected = _expected_annotation(candidate, reference)
        if observed != expected:
            raise AvidaAntigenIndependentAuditError(
                f"antigen annotation differs: {candidate['variant_id']}"
            )
        rows += 1
        wild_types += expected["mutation_count"] == 0
        mutants += expected["mutation_count"] == 1
    counts = {
        "accepted": rows,
        "input_antigens": rows,
        "mutant_constructs": mutants,
        "quarantine_reasons": {},
        "quarantined": 0,
        "wild_type_constructs": wild_types,
    }
    if rows != 31 or wild_types != 1 or mutants != 30:
        raise AvidaAntigenIndependentAuditError("antigen counts differ")
    if (
        annotation_receipt.get("annotation_counts") != counts
        or audit_summary.get("annotation_counts") != counts
        or annotation_receipt.get("candidate_receipt_sha256")
        != CANDIDATE_RECEIPT_SHA256
        or annotation_receipt.get("protocol_sha256") != PROTOCOL_SHA256
        or annotation_receipt.get("normalization_eligible") is not False
        or annotation_receipt.get("status")
        != "annotation_complete_with_explicit_quarantine"
        or audit_summary.get("unknown_or_dropped_antigens") != 0
        or audit_summary.get("table") != outputs["ANTIGEN_BOUNDARY_ANNOTATIONS.parquet"]
    ):
        raise AvidaAntigenIndependentAuditError("annotation receipt semantics differ")
    result = {
        "annotation_counts": counts,
        "annotation_receipt_sha256": _file_sha256(annotation_receipt_path),
        "audited_at_utc": timestamp,
        "auditor_source_sha256": _file_sha256(Path(__file__).resolve()),
        "candidate_receipt_sha256": CANDIDATE_RECEIPT_SHA256,
        "format_version": "pairbench_avida_antigen_independent_audit_v1",
        "normalization_eligible": False,
        "normalization_blockers": [
            "vhh_domain_boundary_annotation_pending",
            "vhh_independent_annotation_audit_pending",
        ],
        "protocol_sha256": PROTOCOL_SHA256,
        "semantic_comparison": "exact_row_order_and_content",
        "source_id": SOURCE_ID,
        "status": "pass_antigen_annotation_vhh_pending",
        "uniprot_json_sha256": UNIPROT_JSON_SHA256,
    }
    output_receipt = Path(output_receipt).expanduser().absolute()
    if not output_receipt.parent.is_dir() or output_receipt.parent.is_symlink():
        raise AvidaAntigenIndependentAuditError("audit receipt parent is invalid")
    _write_create_only_json(output_receipt, result)
    return result


def _audit_protocol(protocol: Mapping[str, Any]) -> None:
    antigen = protocol.get("antigen_boundary")
    if type(antigen) is not dict:
        raise AvidaAntigenIndependentAuditError("protocol antigen section differs")
    authority = antigen.get("authority")
    construct = antigen.get("construct")
    variants = antigen.get("variant_policy")
    if (
        protocol.get("format_version")
        != "pairscale_avida_hil6_annotation_protocol_v1"
        or protocol.get("source_id") != SOURCE_ID
        or protocol.get("score_free") is not True
        or type(authority) is not dict
        or authority.get("accession") != "P05231"
        or authority.get("entry_version") != 258
        or authority.get("sequence_version") != 1
        or authority.get("sequence_length") != 212
        or authority.get("json_sha256") != UNIPROT_JSON_SHA256
        or type(construct) is not dict
        or construct.get("signal_start_one_based_inclusive") != 1
        or construct.get("signal_end_one_based_inclusive") != 29
        or construct.get("mature_chain_start_one_based_inclusive") != 30
        or construct.get("mature_chain_end_one_based_inclusive") != 212
        or construct.get("expected_c_terminal_tag") != "HHHHHH"
        or type(variants) is not dict
        or variants.get("expected_total_constructs") != 31
        or variants.get("expected_mutant_constructs") != 30
    ):
        raise AvidaAntigenIndependentAuditError("protocol antigen semantics differ")


def _audit_uniprot(payload: Mapping[str, Any]) -> str:
    audit = payload.get("entryAudit")
    sequence = payload.get("sequence")
    if (
        payload.get("primaryAccession") != "P05231"
        or type(audit) is not dict
        or audit.get("entryVersion") != 258
        or audit.get("sequenceVersion") != 1
        or audit.get("lastAnnotationUpdateDate") != "2026-06-10"
        or type(sequence) is not dict
        or sequence.get("length") != 212
        or type(sequence.get("value")) is not str
        or len(sequence["value"]) != 212
    ):
        raise AvidaAntigenIndependentAuditError("UniProt identity differs")
    signal = [
        feature
        for feature in payload.get("features", [])
        if type(feature) is dict and feature.get("type") == "Signal"
    ]
    chain = [
        feature
        for feature in payload.get("features", [])
        if type(feature) is dict
        and feature.get("type") == "Chain"
        and feature.get("featureId") == "PRO_0000015582"
    ]
    if len(signal) != 1 or len(chain) != 1:
        raise AvidaAntigenIndependentAuditError("UniProt feature multiplicity differs")
    if not _location_is(signal[0].get("location"), 1, 29) or not _location_is(
        chain[0].get("location"), 30, 212
    ):
        raise AvidaAntigenIndependentAuditError("UniProt feature coordinates differ")
    return sequence["value"]


def _location_is(value: Any, start: int, end: int) -> bool:
    return type(value) is dict and value.get("start") == {
        "value": start,
        "modifier": "EXACT",
    } and value.get("end") == {"value": end, "modifier": "EXACT"}


def _candidate_antigen_table(
    root: Path,
    receipt: Mapping[str, Any],
) -> Path:
    if (
        receipt.get("format_version") != "pairbench_avida_candidate_receipt_v1"
        or receipt.get("source_id") != SOURCE_ID
        or receipt.get("counts", {}).get("antigens") != 31
    ):
        raise AvidaAntigenIndependentAuditError("candidate receipt semantics differ")
    records = receipt.get("outputs")
    if type(records) is not list:
        raise AvidaAntigenIndependentAuditError("candidate inventory differs")
    matches = [
        record
        for record in records
        if type(record) is dict and record.get("path") == "entities_antigen.parquet"
    ]
    if len(matches) != 1 or matches[0].get("sha256") != INPUT_TABLE_SHA256:
        raise AvidaAntigenIndependentAuditError("candidate antigen record differs")
    path = _regular_file(root / "entities_antigen.parquet", "candidate antigen table")
    if (
        path.stat().st_size != matches[0].get("size_bytes")
        or _file_sha256(path) != INPUT_TABLE_SHA256
    ):
        raise AvidaAntigenIndependentAuditError("candidate antigen table bytes differ")
    return path


def _audit_annotation_root(
    root: Path,
    receipt: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    records = receipt.get("outputs")
    if type(records) is not list:
        raise AvidaAntigenIndependentAuditError("annotation inventory differs")
    by_path = {
        record.get("path"): record for record in records if type(record) is dict
    }
    expected = {"ANTIGEN_BOUNDARY_ANNOTATIONS.parquet", "AUDIT_SUMMARY.json"}
    if set(by_path) != expected:
        raise AvidaAntigenIndependentAuditError("annotation output records differ")
    observed: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise AvidaAntigenIndependentAuditError(
                "annotation root contains a non-regular entry"
            )
        observed.add(path.relative_to(root).as_posix())
    if observed != expected | {"ANNOTATION_RECEIPT.json"}:
        raise AvidaAntigenIndependentAuditError("annotation root file set differs")
    for name, record in by_path.items():
        if set(record) != {"path", "sha256", "size_bytes"}:
            raise AvidaAntigenIndependentAuditError("annotation output record differs")
        path = _regular_file(root / name, f"annotation output {name}")
        if path.stat().st_size != record["size_bytes"] or _file_sha256(path) != record[
            "sha256"
        ]:
            raise AvidaAntigenIndependentAuditError("annotation output bytes differ")
    return by_path


def _expected_annotation(row: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    constructs = row["construct_sequences"]
    if (
        row["annotation_status"]
        != "raw_construct_untrimmed_signal_and_tag_boundaries_pending"
        or row["source_id"] != SOURCE_ID
        or type(constructs) is not list
        or len(constructs) != 1
    ):
        raise AvidaAntigenIndependentAuditError("candidate antigen row differs")
    construct = constructs[0]
    digest = hashlib.sha256(construct.encode()).hexdigest()
    if (
        len(construct) != 218
        or not construct.endswith("HHHHHH")
        or row["sequence_digest"] != digest
        or row["antigen_id"] != f"ag_sha256:{digest}"
    ):
        raise AvidaAntigenIndependentAuditError("candidate antigen construct differs")
    precursor = construct[:212]
    mutations = [
        (index + 1, expected, observed)
        for index, (expected, observed) in enumerate(zip(reference, precursor, strict=True))
        if expected != observed
    ]
    label = row["variant_id"]
    if label == "IL-6_WTs":
        if mutations:
            raise AvidaAntigenIndependentAuditError("wild-type sequence differs")
    else:
        match = _LABEL.fullmatch(label)
        if match is None or len(mutations) != 1:
            raise AvidaAntigenIndependentAuditError("mutant label shape differs")
        position, expected, observed = mutations[0]
        if (
            expected != match.group(1)
            or position != int(match.group(2))
            or observed != match.group(3)
            or observed != "A"
            or not 30 <= position <= 212
        ):
            raise AvidaAntigenIndependentAuditError("mutant label disagrees")
    mutation = mutations[0] if mutations else None
    mature = precursor[29:212]
    return {
        "annotation_status": ACCEPTED_STATUS,
        "antigen_id": row["antigen_id"],
        "construct_sequence_digest": digest,
        "mature_chain_sequence": mature,
        "mature_chain_sequence_digest": hashlib.sha256(mature.encode()).hexdigest(),
        "mutation_count": len(mutations),
        "mutation_position_uniprot_one_based": mutation[0] if mutation else None,
        "mutant_residue": mutation[2] if mutation else None,
        "precursor_sequence": precursor,
        "quarantine_reasons": [],
        "reference_residue": mutation[1] if mutation else None,
        "signal_sequence": precursor[:29],
        "terminal_tag_sequence": construct[212:],
        "uniprot_accession": "P05231",
        "variant_id": label,
    }


def _parquet_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    path = _regular_file(path, "Parquet table")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1_024):
        yield from batch.to_pylist()


def _strict_json(path: Path, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise AvidaAntigenIndependentAuditError(
            f"non-finite JSON constant in {label}: {value}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AvidaAntigenIndependentAuditError(f"duplicate JSON key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AvidaAntigenIndependentAuditError(f"cannot parse {label}") from exc
    if type(value) is not dict:
        raise AvidaAntigenIndependentAuditError(f"{label} must be an object")
    return value


def _directory(path: Path, label: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_dir():
        raise AvidaAntigenIndependentAuditError(f"{label} is invalid")
    return result


def _regular_file(path: Path, label: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_file():
        raise AvidaAntigenIndependentAuditError(f"{label} is invalid")
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
        raise AvidaAntigenIndependentAuditError("audit timestamp must use UTC Z form")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AvidaAntigenIndependentAuditError("audit timestamp is invalid") from exc
