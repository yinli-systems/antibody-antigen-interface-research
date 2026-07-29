"""Materialize admitted AVIDa Tier-1 tables with explicit quarantine ledgers."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from interaction_first.avida_normalization_protocol import (
    load_avida_normalization_protocol,
)
from interaction_first.pairbench_schema import validate_pair_record


SOURCE_ID = "avida_hil6_2024_11"
VHH_ACCEPTED = "imgt_heavy_domain_verified"
ANTIGEN_ACCEPTED = "mature_chain_verified_single_variant"
NORMALIZED_VHH_STATUS = "imgt_heavy_domain_verified_tier1"
NORMALIZED_ANTIGEN_STATUS = "mature_chain_verified_single_variant_tier1"


class AvidaNormalizationError(ValueError):
    """Raised when authenticated AVIDa annotations cannot be normalized."""


def normalize_avida_hil6_tier1(
    *,
    candidate_root: Path,
    vhh_annotation_root: Path,
    antigen_annotation_root: Path,
    vhh_independent_receipt: Path,
    antigen_independent_receipt: Path,
    protocol_path: Path,
    output_root: Path,
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Create content-addressed admitted tables and complete quarantine ledgers."""

    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    _utc(timestamp)
    protocol, protocol_sha256 = load_avida_normalization_protocol(protocol_path)
    bindings = protocol["input_bindings"]
    candidate_root = _directory(candidate_root, "candidate root")
    vhh_annotation_root = _directory(vhh_annotation_root, "VHH annotation root")
    antigen_annotation_root = _directory(
        antigen_annotation_root,
        "antigen annotation root",
    )
    vhh_independent_receipt = _regular_file(
        vhh_independent_receipt,
        "VHH independent receipt",
    )
    antigen_independent_receipt = _regular_file(
        antigen_independent_receipt,
        "antigen independent receipt",
    )
    candidate_receipt = _bound_json(
        candidate_root / "CANDIDATE_RECEIPT.json",
        bindings["candidate_receipt_sha256"],
        "candidate receipt",
    )
    vhh_receipt = _bound_json(
        vhh_annotation_root / "ANNOTATION_RECEIPT.json",
        bindings["vhh_annotation_receipt_sha256"],
        "VHH annotation receipt",
    )
    _bound_json(
        vhh_annotation_root / "AUDIT_SUMMARY.json",
        bindings["vhh_audit_summary_sha256"],
        "VHH audit summary",
    )
    vhh_independent = _bound_json(
        vhh_independent_receipt,
        bindings["vhh_independent_replay_sha256"],
        "VHH independent receipt",
    )
    antigen_receipt = _bound_json(
        antigen_annotation_root / "ANNOTATION_RECEIPT.json",
        bindings["antigen_annotation_receipt_sha256"],
        "antigen annotation receipt",
    )
    _bound_json(
        antigen_annotation_root / "AUDIT_SUMMARY.json",
        bindings["antigen_audit_summary_sha256"],
        "antigen audit summary",
    )
    antigen_independent = _bound_json(
        antigen_independent_receipt,
        bindings["antigen_independent_audit_sha256"],
        "antigen independent receipt",
    )
    _validate_receipts(
        protocol=protocol,
        candidate=candidate_receipt,
        vhh=vhh_receipt,
        vhh_independent=vhh_independent,
        antigen=antigen_receipt,
        antigen_independent=antigen_independent,
    )
    candidate_files = _authenticate_output_inventory(
        candidate_root,
        candidate_receipt["outputs"],
    )
    vhh_files = _authenticate_output_inventory(
        vhh_annotation_root,
        vhh_receipt["outputs"],
    )
    antigen_files = _authenticate_output_inventory(
        antigen_annotation_root,
        antigen_receipt["outputs"],
    )
    output_root = Path(output_root).expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise AvidaNormalizationError("normalization output root already exists")
    if not output_root.parent.is_dir() or output_root.parent.is_symlink():
        raise AvidaNormalizationError("normalization output parent is invalid")
    output_root.mkdir(mode=0o750)
    try:
        result = _materialize(
            candidate_files=candidate_files,
            vhh_annotation_path=vhh_files["VHH_BOUNDARY_ANNOTATIONS.parquet"],
            antigen_annotation_path=antigen_files[
                "ANTIGEN_BOUNDARY_ANNOTATIONS.parquet"
            ],
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            output_root=output_root,
            created_at_utc=timestamp,
        )
        _write_create_only_json(output_root / "NORMALIZATION_RECEIPT.json", result)
        return result
    except Exception as exc:
        failure_path = output_root / "FAILURE_RECEIPT.json"
        if not failure_path.exists():
            _write_create_only_json(
                failure_path,
                {
                    "created_at_utc": timestamp,
                    "error_message": str(exc),
                    "error_type": type(exc).__name__,
                    "format_version": "pairbench_avida_normalization_failure_v1",
                    "protocol_sha256": protocol_sha256,
                    "source_id": SOURCE_ID,
                },
            )
        raise


def _materialize(
    *,
    candidate_files: Mapping[str, Path],
    vhh_annotation_path: Path,
    antigen_annotation_path: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    output_root: Path,
    created_at_utc: str,
) -> Mapping[str, Any]:
    candidate_antibody = pq.read_table(candidate_files["entities_antibody.parquet"])
    candidate_antigen = pq.read_table(candidate_files["entities_antigen.parquet"])
    vhh_annotations = pq.read_table(vhh_annotation_path)
    antigen_annotations = pq.read_table(antigen_annotation_path)
    vhh_by_id = _unique_rows(vhh_annotations, "antibody_id", "VHH annotations")
    antigen_by_id = _unique_rows(
        antigen_annotations,
        "antigen_id",
        "antigen annotations",
    )
    admitted_antibodies: list[dict[str, Any]] = []
    quarantined_antibodies: list[dict[str, Any]] = []
    entity_rekeys: list[dict[str, Any]] = []
    antibody_rekey: dict[str, str] = {}
    for candidate in candidate_antibody.to_pylist():
        annotation = vhh_by_id.pop(candidate["antibody_id"], None)
        if annotation is None:
            raise AvidaNormalizationError("VHH annotation inventory differs")
        normalized, rekey, quarantined = _normalize_antibody(candidate, annotation)
        entity_rekeys.append(rekey)
        if normalized is None:
            quarantined_antibodies.append(quarantined)
        else:
            admitted_antibodies.append(normalized)
            antibody_rekey[candidate["antibody_id"]] = normalized["antibody_id"]
    if vhh_by_id:
        raise AvidaNormalizationError("VHH annotation contains unknown antibodies")
    admitted_antigens: list[dict[str, Any]] = []
    antigen_rekey: dict[str, str] = {}
    for candidate in candidate_antigen.to_pylist():
        annotation = antigen_by_id.pop(candidate["antigen_id"], None)
        if annotation is None:
            raise AvidaNormalizationError("antigen annotation inventory differs")
        normalized, rekey = _normalize_antigen(candidate, annotation)
        admitted_antigens.append(normalized)
        entity_rekeys.append(rekey)
        antigen_rekey[candidate["antigen_id"]] = normalized["antigen_id"]
    if antigen_by_id:
        raise AvidaNormalizationError("antigen annotation contains unknown antigens")
    _unique_field(admitted_antibodies, "antibody_id", "normalized antibodies")
    _unique_field(admitted_antigens, "antigen_id", "normalized antigens")
    admitted_antibodies.sort(key=lambda row: row["antibody_id"])
    admitted_antigens.sort(key=lambda row: row["antigen_id"])
    entity_rekeys.sort(key=lambda row: (row["entity_type"], row["source_entity_id"]))
    quarantined_antibodies.sort(key=lambda row: row["source_antibody_id"])
    antibody_out = output_root / "entities_antibody.parquet"
    antigen_out = output_root / "entities_antigen.parquet"
    rekey_out = output_root / "entity_rekey.parquet"
    quarantine_antibody_out = output_root / "quarantined_antibodies.parquet"
    _write_table(
        antibody_out,
        admitted_antibodies,
        candidate_antibody.schema,
    )
    _write_table(
        antigen_out,
        admitted_antigens,
        candidate_antigen.schema,
    )
    _write_table(rekey_out, entity_rekeys, _rekey_schema())
    _write_table(
        quarantine_antibody_out,
        quarantined_antibodies,
        _quarantined_antibody_schema(),
    )
    pair_counts = _materialize_pairs_and_assays(
        candidate_pair_path=candidate_files["pairs.parquet"],
        candidate_assay_path=candidate_files["assays.parquet"],
        antibody_rekey=antibody_rekey,
        antigen_rekey=antigen_rekey,
        output_root=output_root,
    )
    duplicate_table = pq.read_table(candidate_files["duplicate_groups.parquet"])
    if duplicate_table.num_rows != 0:
        raise AvidaNormalizationError("candidate duplicate table is not empty")
    _write_table(
        output_root / "duplicate_groups.parquet",
        [],
        duplicate_table.schema,
    )
    admission = protocol["admission"]
    counts = {
        "admitted_antibodies": len(admitted_antibodies),
        "admitted_antigens": len(admitted_antigens),
        "quarantined_antibodies": len(quarantined_antibodies),
        **pair_counts,
    }
    expected_counts = {
        "admitted_antibodies": admission["vhh"]["expected_accepted"],
        "admitted_antigens": admission["antigen"]["expected_accepted"],
        "quarantined_antibodies": admission["vhh"]["expected_quarantined"],
        "admitted_binders": admission["pair"]["expected_admitted_binders"],
        "admitted_nonbinders": admission["pair"]["expected_admitted_nonbinders"],
        "admitted_pairs": admission["pair"]["expected_admitted_rows"],
        "quarantined_binders": admission["pair"]["expected_quarantined_binders"],
        "quarantined_nonbinders": admission["pair"][
            "expected_quarantined_nonbinders"
        ],
        "quarantined_pairs": admission["pair"]["expected_quarantined_rows"],
    }
    if counts != expected_counts:
        raise AvidaNormalizationError("normalization accounting differs")
    output_names = [
        "assays.parquet",
        "duplicate_groups.parquet",
        "entities_antibody.parquet",
        "entities_antigen.parquet",
        "entity_rekey.parquet",
        "pairs.parquet",
        "quarantined_antibodies.parquet",
        "quarantined_pairs.parquet",
    ]
    inventory = [_file_record(output_root / name, output_root) for name in output_names]
    audit = {
        "counts": counts,
        "created_at_utc": created_at_utc,
        "format_version": "pairbench_avida_normalization_audit_v1",
        "identity_semantics": {
            "antibody": "sha256_of_verified_vhh_domain_sequence",
            "antigen": "sha256_of_verified_mature_chain_sequence",
            "pair": "sha256_of_normalized_antibody_id_newline_antigen_id",
        },
        "license_route": "noncommercial_only",
        "pairbench_training_eligible": True,
        "protocol_sha256": protocol_sha256,
        "source_id": SOURCE_ID,
        "table_inventory": inventory,
        "tier2_admitted": False,
        "unknown_cross_pairs_created": 0,
    }
    audit_path = output_root / "AUDIT_SUMMARY.json"
    _write_create_only_json(audit_path, audit)
    outputs = sorted(
        [*inventory, _file_record(audit_path, output_root)],
        key=lambda row: row["path"],
    )
    return {
        "counts": counts,
        "created_at_utc": created_at_utc,
        "format_version": "pairbench_avida_normalization_receipt_v1",
        "normalizer_source_sha256": _file_sha256(Path(__file__)),
        "outputs": outputs,
        "pairbench_training_eligible": True,
        "protocol_sha256": protocol_sha256,
        "source_id": SOURCE_ID,
        "status": "normalized_tier1_with_explicit_quarantine",
    }


def _normalize_antibody(
    candidate: Mapping[str, Any],
    annotation: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any] | None]:
    if (
        candidate["annotation_status"]
        != "raw_construct_untrimmed_signal_and_tag_boundaries_pending"
        or candidate["sequence_digest"] != annotation["construct_sequence_digest"]
        or hashlib.sha256(candidate["construct_sequence"].encode()).hexdigest()
        != candidate["sequence_digest"]
    ):
        raise AvidaNormalizationError("candidate VHH provenance differs")
    reasons = annotation["quarantine_reasons"]
    if type(reasons) is not list or reasons != sorted(set(reasons)):
        raise AvidaNormalizationError("VHH quarantine reasons are not canonical")
    accepted = annotation["annotation_status"] == VHH_ACCEPTED and not reasons
    normalized_id = None
    if accepted:
        domain = annotation["domain_sequence"]
        domain_digest = annotation["domain_sequence_digest"]
        if (
            type(domain) is not str
            or not domain
            or hashlib.sha256(domain.encode()).hexdigest() != domain_digest
            or candidate["construct_sequence"][
                annotation["query_start_zero_based_inclusive"] :
                annotation["query_end_zero_based_inclusive"] + 1
            ]
            != domain
        ):
            raise AvidaNormalizationError("accepted VHH domain provenance differs")
        numbering = _canonical_numbering(annotation["imgt_numbering"], domain)
        normalized_id = f"ab_sha256:{domain_digest}"
        normalized = dict(candidate)
        normalized.update(
            {
                "annotation_status": NORMALIZED_VHH_STATUS,
                "antibody_format": "vhh",
                "antibody_id": normalized_id,
                "cdr_boundaries": [
                    "imgt_cdr1:27-38",
                    "imgt_cdr2:56-65",
                    "imgt_cdr3:105-117",
                ],
                "imgt_numbering": numbering,
                "sequence_digest": domain_digest,
                "vh_sequence": domain,
                "vl_sequence": None,
            }
        )
        quarantined = None
        admission_status = "admitted_tier1"
    else:
        if annotation["annotation_status"] == VHH_ACCEPTED:
            raise AvidaNormalizationError("accepted VHH has quarantine reasons")
        normalized = None
        admission_status = "quarantined"
        quarantined = {
            "annotation_status": annotation["annotation_status"],
            "construct_sequence": candidate["construct_sequence"],
            "construct_sequence_digest": candidate["sequence_digest"],
            "domain_sequence": annotation["domain_sequence"],
            "domain_sequence_digest": annotation["domain_sequence_digest"],
            "quarantine_reasons": reasons,
            "query_end_zero_based_inclusive": annotation[
                "query_end_zero_based_inclusive"
            ],
            "query_start_zero_based_inclusive": annotation[
                "query_start_zero_based_inclusive"
            ],
            "source_antibody_id": candidate["antibody_id"],
            "tool_error": annotation["tool_error"],
        }
    rekey = {
        "admission_status": admission_status,
        "chain_sequence_digest": annotation["domain_sequence_digest"],
        "construct_sequence_digest": candidate["sequence_digest"],
        "entity_type": "antibody",
        "normalized_entity_id": normalized_id,
        "quarantine_reasons": reasons,
        "source_entity_id": candidate["antibody_id"],
    }
    return normalized, rekey, quarantined


def _normalize_antigen(
    candidate: Mapping[str, Any],
    annotation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reasons = annotation["quarantine_reasons"]
    chain = annotation["mature_chain_sequence"]
    chain_digest = annotation["mature_chain_sequence_digest"]
    if (
        annotation["annotation_status"] != ANTIGEN_ACCEPTED
        or reasons != []
        or candidate["sequence_digest"] != annotation["construct_sequence_digest"]
        or candidate["variant_id"] != annotation["variant_id"]
        or hashlib.sha256(chain.encode()).hexdigest() != chain_digest
    ):
        raise AvidaNormalizationError("antigen admission provenance differs")
    normalized_id = f"ag_sha256:{chain_digest}"
    normalized = dict(candidate)
    normalized.update(
        {
            "annotation_status": NORMALIZED_ANTIGEN_STATUS,
            "antigen_id": normalized_id,
            "chain_sequences": [chain],
            "sequence_digest": chain_digest,
            "uniprot_ids": [annotation["uniprot_accession"]],
        }
    )
    rekey = {
        "admission_status": "admitted_tier1",
        "chain_sequence_digest": chain_digest,
        "construct_sequence_digest": candidate["sequence_digest"],
        "entity_type": "antigen",
        "normalized_entity_id": normalized_id,
        "quarantine_reasons": [],
        "source_entity_id": candidate["antigen_id"],
    }
    return normalized, rekey


def _materialize_pairs_and_assays(
    *,
    candidate_pair_path: Path,
    candidate_assay_path: Path,
    antibody_rekey: Mapping[str, str],
    antigen_rekey: Mapping[str, str],
    output_root: Path,
) -> Mapping[str, int]:
    pair_file = pq.ParquetFile(candidate_pair_path)
    assay_file = pq.ParquetFile(candidate_assay_path)
    if pair_file.metadata.num_rows != assay_file.metadata.num_rows:
        raise AvidaNormalizationError("candidate pair and assay counts differ")
    pair_schema = pair_file.schema_arrow
    assay_schema = assay_file.schema_arrow
    quarantine_schema = _quarantined_pair_schema()
    pair_counts: Counter[str] = Counter()
    observed_pair_ids: set[str] = set()
    with ExitStack() as stack:
        pair_writer = stack.enter_context(
            pq.ParquetWriter(
                output_root / "pairs.parquet",
                pair_schema,
                compression="zstd",
                compression_level=9,
                use_dictionary=False,
                write_statistics=True,
            )
        )
        assay_writer = stack.enter_context(
            pq.ParquetWriter(
                output_root / "assays.parquet",
                assay_schema,
                compression="zstd",
                compression_level=9,
                use_dictionary=False,
                write_statistics=True,
            )
        )
        quarantine_writer = stack.enter_context(
            pq.ParquetWriter(
                output_root / "quarantined_pairs.parquet",
                quarantine_schema,
                compression="zstd",
                compression_level=9,
                use_dictionary=False,
                write_statistics=True,
            )
        )
        pair_batches = pair_file.iter_batches(batch_size=16_384)
        assay_batches = assay_file.iter_batches(batch_size=16_384)
        for pair_batch, assay_batch in zip(pair_batches, assay_batches, strict=True):
            pairs = pair_batch.to_pylist()
            assays = assay_batch.to_pylist()
            admitted_pairs: list[dict[str, Any]] = []
            admitted_assays: list[dict[str, Any]] = []
            quarantined_pairs: list[dict[str, Any]] = []
            for pair, assay in zip(pairs, assays, strict=True):
                if (
                    pair["assay_id"] != assay["assay_id"]
                    or pair["pair_id"] != assay["pair_id"]
                    or pair["source_id"] != SOURCE_ID
                    or assay["source_id"] != SOURCE_ID
                ):
                    raise AvidaNormalizationError("candidate pair and assay rows differ")
                normalized_antigen = antigen_rekey.get(pair["antigen_id"])
                if normalized_antigen is None:
                    raise AvidaNormalizationError("pair references excluded antigen")
                normalized_antibody = antibody_rekey.get(pair["antibody_id"])
                label = pair["binary_binding_target"]
                label_name = "binders" if label == 1 else "nonbinders"
                if normalized_antibody is None:
                    pair_counts[f"quarantined_{label_name}"] += 1
                    pair_counts["quarantined_pairs"] += 1
                    quarantined_pairs.append(
                        {
                            "antigen_id": normalized_antigen,
                            "binary_binding_target": label,
                            "candidate_antibody_id": pair["antibody_id"],
                            "candidate_pair_id": pair["pair_id"],
                            "quarantine_reasons": [
                                "antibody_not_admitted_by_tier1_protocol"
                            ],
                            "source_record_id": pair["source_record_id"],
                        }
                    )
                    continue
                normalized_pair_id = _content_id(
                    "pair",
                    f"{normalized_antibody}\n{normalized_antigen}",
                )
                if normalized_pair_id in observed_pair_ids:
                    raise AvidaNormalizationError("normalized pair identity is duplicated")
                observed_pair_ids.add(normalized_pair_id)
                normalized_pair = dict(pair)
                normalized_pair.update(
                    {
                        "antibody_id": normalized_antibody,
                        "antigen_id": normalized_antigen,
                        "pair_id": normalized_pair_id,
                    }
                )
                validate_pair_record(normalized_pair)
                normalized_assay = dict(assay)
                normalized_assay["pair_id"] = normalized_pair_id
                admitted_pairs.append(normalized_pair)
                admitted_assays.append(normalized_assay)
                pair_counts[f"admitted_{label_name}"] += 1
                pair_counts["admitted_pairs"] += 1
            if admitted_pairs:
                pair_writer.write_table(pa.Table.from_pylist(admitted_pairs, pair_schema))
                assay_writer.write_table(
                    pa.Table.from_pylist(admitted_assays, assay_schema)
                )
            if quarantined_pairs:
                quarantine_writer.write_table(
                    pa.Table.from_pylist(quarantined_pairs, quarantine_schema)
                )
    return dict(pair_counts)


def _validate_receipts(
    *,
    protocol: Mapping[str, Any],
    candidate: Mapping[str, Any],
    vhh: Mapping[str, Any],
    vhh_independent: Mapping[str, Any],
    antigen: Mapping[str, Any],
    antigen_independent: Mapping[str, Any],
) -> None:
    bindings = protocol["input_bindings"]
    if (
        candidate.get("format_version") != "pairbench_avida_candidate_receipt_v1"
        or candidate.get("source_id") != SOURCE_ID
        or candidate.get("pairbench_contract_sha256")
        != bindings["pairbench_contract_sha256"]
        or candidate.get("status") != "candidate_complete_blocked"
    ):
        raise AvidaNormalizationError("candidate receipt semantics differ")
    if (
        vhh.get("status") != "annotation_complete_with_explicit_quarantine"
        or vhh.get("protocol_sha256") != bindings["annotation_protocol_sha256"]
        or vhh_independent.get("status") != "pass_vhh_annotation_strict_tier1"
        or vhh_independent.get("annotation_receipt_sha256")
        != bindings["vhh_annotation_receipt_sha256"]
        or antigen.get("status") != "annotation_complete_with_explicit_quarantine"
        or antigen.get("protocol_sha256")
        != bindings["annotation_protocol_sha256"]
        or antigen_independent.get("status")
        != "pass_antigen_annotation_vhh_pending"
        or antigen_independent.get("annotation_receipt_sha256")
        != bindings["antigen_annotation_receipt_sha256"]
    ):
        raise AvidaNormalizationError("annotation receipt semantics differ")


def _authenticate_output_inventory(
    root: Path,
    records: Any,
) -> Mapping[str, Path]:
    if type(records) is not list or not records:
        raise AvidaNormalizationError("output inventory is invalid")
    result: dict[str, Path] = {}
    for record in records:
        if type(record) is not dict or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise AvidaNormalizationError("output record fields differ")
        relative = record["path"]
        if type(relative) is not str or Path(relative).name != relative:
            raise AvidaNormalizationError("output path is invalid")
        path = _regular_file(root / relative, f"output {relative}")
        if (
            path.stat().st_size != record["size_bytes"]
            or _file_sha256(path) != record["sha256"]
        ):
            raise AvidaNormalizationError(f"output authentication failed: {relative}")
        result[relative] = path
    return result


def _unique_rows(table: pa.Table, key: str, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        identifier = row[key]
        if identifier in result:
            raise AvidaNormalizationError(f"{name} contains duplicate identities")
        result[identifier] = row
    return result


def _unique_field(rows: list[dict[str, Any]], key: str, name: str) -> None:
    values = [row[key] for row in rows]
    if len(values) != len(set(values)):
        raise AvidaNormalizationError(f"{name} contains duplicate identities")


def _canonical_numbering(numbering: Any, domain: str) -> list[str]:
    if type(numbering) is not list or not numbering:
        raise AvidaNormalizationError("accepted VHH numbering is missing")
    encoded: list[str] = []
    residues: list[str] = []
    coordinates: set[tuple[int, str]] = set()
    for row in numbering:
        if type(row) is not dict or set(row) != {"insertion", "position", "residue"}:
            raise AvidaNormalizationError("VHH numbering fields differ")
        coordinate = (row["position"], row["insertion"])
        if coordinate in coordinates:
            raise AvidaNormalizationError("VHH numbering coordinate is duplicated")
        coordinates.add(coordinate)
        encoded.append(f"{row['position']}{row['insertion']}:{row['residue']}")
        if row["residue"] != "-":
            residues.append(row["residue"])
    if "".join(residues) != domain:
        raise AvidaNormalizationError("VHH numbering does not reconstruct domain")
    return encoded


def _rekey_schema() -> pa.Schema:
    text = pa.string()
    return pa.schema(
        [
            ("admission_status", text, False),
            ("chain_sequence_digest", text),
            ("construct_sequence_digest", text, False),
            ("entity_type", text, False),
            ("normalized_entity_id", text),
            ("quarantine_reasons", pa.list_(text), False),
            ("source_entity_id", text, False),
        ]
    )


def _quarantined_antibody_schema() -> pa.Schema:
    text = pa.string()
    return pa.schema(
        [
            ("annotation_status", text, False),
            ("construct_sequence", text, False),
            ("construct_sequence_digest", text, False),
            ("domain_sequence", text),
            ("domain_sequence_digest", text),
            ("quarantine_reasons", pa.list_(text), False),
            ("query_end_zero_based_inclusive", pa.int32()),
            ("query_start_zero_based_inclusive", pa.int32()),
            ("source_antibody_id", text, False),
            ("tool_error", text),
        ]
    )


def _quarantined_pair_schema() -> pa.Schema:
    text = pa.string()
    return pa.schema(
        [
            ("antigen_id", text, False),
            ("binary_binding_target", pa.int8(), False),
            ("candidate_antibody_id", text, False),
            ("candidate_pair_id", text, False),
            ("quarantine_reasons", pa.list_(text), False),
            ("source_record_id", text, False),
        ]
    )


def _write_table(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
    )


def _bound_json(path: Path, expected_sha256: str, name: str) -> Mapping[str, Any]:
    path = _regular_file(path, name)
    if _file_sha256(path) != expected_sha256:
        raise AvidaNormalizationError(f"{name} SHA-256 differs")
    return _strict_json(path, name)


def _strict_json(path: Path, name: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise AvidaNormalizationError(f"{name} contains {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AvidaNormalizationError(f"{name} contains duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as exc:
        raise AvidaNormalizationError(f"cannot parse {name}") from exc
    if type(value) is not dict:
        raise AvidaNormalizationError(f"{name} must be an object")
    return value


def _content_id(kind: str, value: str) -> str:
    return f"{kind}_sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _directory(path: Path, name: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_dir():
        raise AvidaNormalizationError(f"{name} is invalid")
    return result


def _regular_file(path: Path, name: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_file():
        raise AvidaNormalizationError(f"{name} is invalid")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(encoded)


def _utc(value: str) -> None:
    if type(value) is not str or not value.endswith("Z"):
        raise AvidaNormalizationError("created_at_utc must use UTC Z form")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AvidaNormalizationError("created_at_utc is invalid") from exc


__all__ = ["AvidaNormalizationError", "normalize_avida_hil6_tier1"]
