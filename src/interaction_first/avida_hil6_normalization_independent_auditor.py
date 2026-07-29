"""Independently audit materialized AVIDa Tier-1 normalization outputs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import pyarrow.parquet as pq

from interaction_first.avida_normalization_protocol import (
    load_avida_normalization_protocol,
)


SOURCE_ID = "avida_hil6_2024_11"
VHH_ACCEPTED = "imgt_heavy_domain_verified"
ANTIGEN_ACCEPTED = "mature_chain_verified_single_variant"


class AvidaNormalizationAuditError(ValueError):
    """Raised when normalized AVIDa evidence cannot be independently verified."""


def audit_avida_hil6_normalization(
    *,
    candidate_root: Path,
    vhh_annotation_root: Path,
    antigen_annotation_root: Path,
    normalization_root: Path,
    protocol_path: Path,
    output_receipt: Path,
    audited_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Recompute normalized identities and compare every source pair exactly once."""

    timestamp = audited_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    _utc(timestamp)
    protocol, protocol_sha256 = load_avida_normalization_protocol(protocol_path)
    candidate_root = _directory(candidate_root, "candidate root")
    vhh_root = _directory(vhh_annotation_root, "VHH annotation root")
    antigen_root = _directory(antigen_annotation_root, "antigen annotation root")
    normalization_root = _directory(normalization_root, "normalization root")
    candidate_receipt = _load_json(candidate_root / "CANDIDATE_RECEIPT.json")
    if _file_sha256(candidate_root / "CANDIDATE_RECEIPT.json") != protocol[
        "input_bindings"
    ]["candidate_receipt_sha256"]:
        raise AvidaNormalizationAuditError("candidate receipt digest differs")
    candidate_files = _authenticate_inventory(candidate_root, candidate_receipt)
    vhh_receipt = _load_json(vhh_root / "ANNOTATION_RECEIPT.json")
    antigen_receipt = _load_json(antigen_root / "ANNOTATION_RECEIPT.json")
    vhh_files = _authenticate_inventory(vhh_root, vhh_receipt)
    antigen_files = _authenticate_inventory(antigen_root, antigen_receipt)
    receipt_path = normalization_root / "NORMALIZATION_RECEIPT.json"
    normalization_receipt = _load_json(receipt_path)
    if (
        normalization_receipt.get("format_version")
        != "pairbench_avida_normalization_receipt_v1"
        or normalization_receipt.get("protocol_sha256") != protocol_sha256
        or normalization_receipt.get("source_id") != SOURCE_ID
        or normalization_receipt.get("status")
        != "normalized_tier1_with_explicit_quarantine"
        or normalization_receipt.get("pairbench_training_eligible") is not True
    ):
        raise AvidaNormalizationAuditError("normalization receipt semantics differ")
    normalized_files = _authenticate_inventory(
        normalization_root,
        normalization_receipt,
    )
    candidate_antibodies = _keyed_table(
        candidate_files["entities_antibody.parquet"],
        "antibody_id",
    )
    candidate_antigens = _keyed_table(
        candidate_files["entities_antigen.parquet"],
        "antigen_id",
    )
    vhh_annotations = _keyed_table(
        vhh_files["VHH_BOUNDARY_ANNOTATIONS.parquet"],
        "antibody_id",
    )
    antigen_annotations = _keyed_table(
        antigen_files["ANTIGEN_BOUNDARY_ANNOTATIONS.parquet"],
        "antigen_id",
    )
    normalized_antibodies = _keyed_table(
        normalized_files["entities_antibody.parquet"],
        "antibody_id",
    )
    normalized_antigens = _keyed_table(
        normalized_files["entities_antigen.parquet"],
        "antigen_id",
    )
    quarantine_antibodies = _keyed_table(
        normalized_files["quarantined_antibodies.parquet"],
        "source_antibody_id",
    )
    rekeys = pq.read_table(normalized_files["entity_rekey.parquet"]).to_pylist()
    rekey_by_source: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rekeys:
        key = (row["entity_type"], row["source_entity_id"])
        if key in rekey_by_source:
            raise AvidaNormalizationAuditError("rekey identity is duplicated")
        rekey_by_source[key] = row
    antibody_rekey = _audit_antibody_entities(
        candidate_antibodies,
        vhh_annotations,
        normalized_antibodies,
        quarantine_antibodies,
        rekey_by_source,
    )
    antigen_rekey = _audit_antigen_entities(
        candidate_antigens,
        antigen_annotations,
        normalized_antigens,
        rekey_by_source,
    )
    if len(rekey_by_source) != len(rekeys):
        raise AvidaNormalizationAuditError("rekey table contains unknown entities")
    counts = _audit_pairs_and_assays(
        candidate_pair_path=candidate_files["pairs.parquet"],
        candidate_assay_path=candidate_files["assays.parquet"],
        normalized_pair_path=normalized_files["pairs.parquet"],
        normalized_assay_path=normalized_files["assays.parquet"],
        quarantined_pair_path=normalized_files["quarantined_pairs.parquet"],
        antibody_rekey=antibody_rekey,
        antigen_rekey=antigen_rekey,
    )
    duplicate_table = pq.read_table(normalized_files["duplicate_groups.parquet"])
    if duplicate_table.num_rows != 0:
        raise AvidaNormalizationAuditError("normalized duplicate table is not empty")
    counts.update(
        {
            "admitted_antibodies": len(normalized_antibodies),
            "admitted_antigens": len(normalized_antigens),
            "quarantined_antibodies": len(quarantine_antibodies),
        }
    )
    if counts != normalization_receipt["counts"]:
        raise AvidaNormalizationAuditError("receipt accounting differs")
    expected = protocol["admission"]
    if counts != {
        "admitted_antibodies": expected["vhh"]["expected_accepted"],
        "admitted_antigens": expected["antigen"]["expected_accepted"],
        "admitted_binders": expected["pair"]["expected_admitted_binders"],
        "admitted_nonbinders": expected["pair"]["expected_admitted_nonbinders"],
        "admitted_pairs": expected["pair"]["expected_admitted_rows"],
        "quarantined_antibodies": expected["vhh"]["expected_quarantined"],
        "quarantined_binders": expected["pair"][
            "expected_quarantined_binders"
        ],
        "quarantined_nonbinders": expected["pair"][
            "expected_quarantined_nonbinders"
        ],
        "quarantined_pairs": expected["pair"]["expected_quarantined_rows"],
    }:
        raise AvidaNormalizationAuditError("protocol accounting differs")
    result = {
        "audited_at_utc": timestamp,
        "auditor_source_sha256": _file_sha256(Path(__file__)),
        "counts": counts,
        "format_version": "pairbench_avida_normalization_independent_audit_v1",
        "normalization_receipt_sha256": _file_sha256(receipt_path),
        "protocol_sha256": protocol_sha256,
        "semantic_comparison": (
            "independent_entity_rekey_and_exact_source_order_pair_assay_replay"
        ),
        "source_id": SOURCE_ID,
        "status": "pass_tier1_normalization_independent_audit",
        "tier2_admitted": False,
        "unknown_cross_pairs_created": 0,
    }
    output_receipt = Path(output_receipt).expanduser().absolute()
    if (
        output_receipt.exists()
        or output_receipt.is_symlink()
        or not output_receipt.parent.is_dir()
        or output_receipt.parent.is_symlink()
    ):
        raise AvidaNormalizationAuditError("output receipt must be create-only")
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    with output_receipt.open("xb") as handle:
        handle.write(encoded)
    return result


def _audit_antibody_entities(
    candidates: Mapping[str, Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    normalized: Mapping[str, Mapping[str, Any]],
    quarantined: Mapping[str, Mapping[str, Any]],
    rekeys: Mapping[tuple[str, str], Mapping[str, Any]],
) -> Mapping[str, str]:
    if set(candidates) != set(annotations):
        raise AvidaNormalizationAuditError("antibody annotation inventory differs")
    result: dict[str, str] = {}
    expected_normalized: set[str] = set()
    expected_quarantined: set[str] = set()
    for source_id, candidate in candidates.items():
        annotation = annotations[source_id]
        if candidate["sequence_digest"] != annotation["construct_sequence_digest"]:
            raise AvidaNormalizationAuditError("VHH construct digest differs")
        reasons = annotation["quarantine_reasons"]
        rekey = rekeys.get(("antibody", source_id))
        if rekey is None:
            raise AvidaNormalizationAuditError("antibody rekey is missing")
        if annotation["annotation_status"] == VHH_ACCEPTED and reasons == []:
            domain = annotation["domain_sequence"]
            digest = hashlib.sha256(domain.encode()).hexdigest()
            normalized_id = f"ab_sha256:{digest}"
            row = normalized.get(normalized_id)
            if (
                row is None
                or digest != annotation["domain_sequence_digest"]
                or row["vh_sequence"] != domain
                or row["sequence_digest"] != digest
                or row["construct_sequence"] != candidate["construct_sequence"]
                or row["annotation_status"]
                != "imgt_heavy_domain_verified_tier1"
                or row["antibody_format"] != "vhh"
                or row["license_route"] != "noncommercial_only"
                or row["vl_sequence"] is not None
            ):
                raise AvidaNormalizationAuditError("normalized VHH content differs")
            numbering_residues = "".join(
                item.split(":", 1)[1]
                for item in row["imgt_numbering"]
                if item.split(":", 1)[1] != "-"
            )
            if numbering_residues != domain:
                raise AvidaNormalizationAuditError("normalized IMGT numbering differs")
            if (
                rekey["admission_status"] != "admitted_tier1"
                or rekey["normalized_entity_id"] != normalized_id
                or rekey["chain_sequence_digest"] != digest
                or rekey["quarantine_reasons"] != []
            ):
                raise AvidaNormalizationAuditError("admitted VHH rekey differs")
            expected_normalized.add(normalized_id)
            result[source_id] = normalized_id
        else:
            row = quarantined.get(source_id)
            if (
                row is None
                or row["construct_sequence"] != candidate["construct_sequence"]
                or row["construct_sequence_digest"] != candidate["sequence_digest"]
                or row["quarantine_reasons"] != reasons
                or rekey["admission_status"] != "quarantined"
                or rekey["normalized_entity_id"] is not None
                or rekey["quarantine_reasons"] != reasons
            ):
                raise AvidaNormalizationAuditError("quarantined VHH content differs")
            expected_quarantined.add(source_id)
    if set(normalized) != expected_normalized or set(quarantined) != expected_quarantined:
        raise AvidaNormalizationAuditError("normalized VHH inventory differs")
    return result


def _audit_antigen_entities(
    candidates: Mapping[str, Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    normalized: Mapping[str, Mapping[str, Any]],
    rekeys: Mapping[tuple[str, str], Mapping[str, Any]],
) -> Mapping[str, str]:
    if set(candidates) != set(annotations):
        raise AvidaNormalizationAuditError("antigen annotation inventory differs")
    result: dict[str, str] = {}
    expected_ids: set[str] = set()
    for source_id, candidate in candidates.items():
        annotation = annotations[source_id]
        chain = annotation["mature_chain_sequence"]
        digest = hashlib.sha256(chain.encode()).hexdigest()
        normalized_id = f"ag_sha256:{digest}"
        row = normalized.get(normalized_id)
        rekey = rekeys.get(("antigen", source_id))
        if (
            annotation["annotation_status"] != ANTIGEN_ACCEPTED
            or annotation["quarantine_reasons"] != []
            or digest != annotation["mature_chain_sequence_digest"]
            or candidate["sequence_digest"] != annotation["construct_sequence_digest"]
            or row is None
            or row["chain_sequences"] != [chain]
            or row["construct_sequences"] != candidate["construct_sequences"]
            or row["sequence_digest"] != digest
            or row["uniprot_ids"] != ["P05231"]
            or row["annotation_status"]
            != "mature_chain_verified_single_variant_tier1"
            or rekey is None
            or rekey["normalized_entity_id"] != normalized_id
            or rekey["admission_status"] != "admitted_tier1"
        ):
            raise AvidaNormalizationAuditError("normalized antigen content differs")
        expected_ids.add(normalized_id)
        result[source_id] = normalized_id
    if set(normalized) != expected_ids:
        raise AvidaNormalizationAuditError("normalized antigen inventory differs")
    return result


def _audit_pairs_and_assays(
    *,
    candidate_pair_path: Path,
    candidate_assay_path: Path,
    normalized_pair_path: Path,
    normalized_assay_path: Path,
    quarantined_pair_path: Path,
    antibody_rekey: Mapping[str, str],
    antigen_rekey: Mapping[str, str],
) -> dict[str, int]:
    candidate_pairs = _row_iterator(candidate_pair_path)
    candidate_assays = _row_iterator(candidate_assay_path)
    normalized_pairs = _row_iterator(normalized_pair_path)
    normalized_assays = _row_iterator(normalized_assay_path)
    quarantined_pairs = _row_iterator(quarantined_pair_path)
    counts: Counter[str] = Counter()
    normalized_pair_ids: set[str] = set()
    for source_pair, source_assay in zip(
        candidate_pairs,
        candidate_assays,
        strict=True,
    ):
        if (
            source_pair["pair_id"] != source_assay["pair_id"]
            or source_pair["assay_id"] != source_assay["assay_id"]
        ):
            raise AvidaNormalizationAuditError("source pair and assay differ")
        antigen_id = antigen_rekey.get(source_pair["antigen_id"])
        if antigen_id is None:
            raise AvidaNormalizationAuditError("source pair antigen is unknown")
        antibody_id = antibody_rekey.get(source_pair["antibody_id"])
        label = source_pair["binary_binding_target"]
        suffix = "binders" if label == 1 else "nonbinders"
        if antibody_id is None:
            row = _next(quarantined_pairs, "quarantined pair")
            if row != {
                "antigen_id": antigen_id,
                "binary_binding_target": label,
                "candidate_antibody_id": source_pair["antibody_id"],
                "candidate_pair_id": source_pair["pair_id"],
                "quarantine_reasons": [
                    "antibody_not_admitted_by_tier1_protocol"
                ],
                "source_record_id": source_pair["source_record_id"],
            }:
                raise AvidaNormalizationAuditError("quarantined pair content differs")
            counts["quarantined_pairs"] += 1
            counts[f"quarantined_{suffix}"] += 1
            continue
        expected_pair_id = _content_id(
            "pair",
            f"{antibody_id}\n{antigen_id}",
        )
        pair = _next(normalized_pairs, "normalized pair")
        assay = _next(normalized_assays, "normalized assay")
        expected_pair = dict(source_pair)
        expected_pair.update(
            {
                "antibody_id": antibody_id,
                "antigen_id": antigen_id,
                "pair_id": expected_pair_id,
            }
        )
        expected_assay = dict(source_assay)
        expected_assay["pair_id"] = expected_pair_id
        if pair != expected_pair or assay != expected_assay:
            raise AvidaNormalizationAuditError("normalized pair or assay differs")
        if expected_pair_id in normalized_pair_ids:
            raise AvidaNormalizationAuditError("normalized pair identity is duplicated")
        normalized_pair_ids.add(expected_pair_id)
        counts["admitted_pairs"] += 1
        counts[f"admitted_{suffix}"] += 1
    for iterator, name in (
        (normalized_pairs, "normalized pair"),
        (normalized_assays, "normalized assay"),
        (quarantined_pairs, "quarantined pair"),
    ):
        try:
            next(iterator)
        except StopIteration:
            continue
        raise AvidaNormalizationAuditError(f"{name} output has extra rows")
    return dict(counts)


def _row_iterator(path: Path, batch_size: int = 16_384) -> Iterator[dict[str, Any]]:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _next(iterator: Iterator[dict[str, Any]], name: str) -> dict[str, Any]:
    try:
        return next(iterator)
    except StopIteration as exc:
        raise AvidaNormalizationAuditError(f"{name} output ended early") from exc


def _keyed_table(path: Path, key: str) -> Mapping[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in pq.read_table(path).to_pylist():
        identifier = row[key]
        if identifier in result:
            raise AvidaNormalizationAuditError(f"duplicate table identity: {key}")
        result[identifier] = row
    return result


def _authenticate_inventory(
    root: Path,
    receipt: Mapping[str, Any],
) -> Mapping[str, Path]:
    records = receipt.get("outputs")
    if type(records) is not list or not records:
        raise AvidaNormalizationAuditError("receipt output inventory is invalid")
    result: dict[str, Path] = {}
    for record in records:
        if type(record) is not dict or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise AvidaNormalizationAuditError("receipt output record differs")
        relative = record["path"]
        if type(relative) is not str or Path(relative).name != relative:
            raise AvidaNormalizationAuditError("receipt output path is invalid")
        path = _regular_file(root / relative, f"receipt output {relative}")
        if (
            path.stat().st_size != record["size_bytes"]
            or _file_sha256(path) != record["sha256"]
        ):
            raise AvidaNormalizationAuditError("receipt output digest differs")
        result[relative] = path
    return result


def _load_json(path: Path) -> Mapping[str, Any]:
    path = _regular_file(path, "JSON receipt")

    def reject_constant(value: str) -> None:
        raise AvidaNormalizationAuditError(f"non-finite JSON value: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AvidaNormalizationAuditError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as exc:
        raise AvidaNormalizationAuditError("cannot parse JSON receipt") from exc
    if type(value) is not dict:
        raise AvidaNormalizationAuditError("JSON receipt must be an object")
    return value


def _content_id(kind: str, value: str) -> str:
    return f"{kind}_sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _directory(path: Path, name: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_dir():
        raise AvidaNormalizationAuditError(f"{name} is invalid")
    return result


def _regular_file(path: Path, name: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_file():
        raise AvidaNormalizationAuditError(f"{name} is invalid")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> None:
    if type(value) is not str or not value.endswith("Z"):
        raise AvidaNormalizationAuditError("audited_at_utc must use UTC Z form")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AvidaNormalizationAuditError("audited_at_utc is invalid") from exc


__all__ = ["AvidaNormalizationAuditError", "audit_avida_hil6_normalization"]
