"""Independently audit AVIDa raw bytes against PAIRBench candidate Parquet rows."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import pyarrow.parquet as pq


SOURCE_ID = "avida_hil6_2024_11"
PAIR_HEADER = (
    "VHH_sequence",
    "label",
    "Ag_label",
    "subject_species",
    "subject_name",
    "subject_sex",
)
ANTIGEN_HEADER = ("Ag_label", "Ag_sequence")
BLOCKERS = (
    "antigen_chain_boundary_annotation_pending",
    "vhh_domain_boundary_annotation_pending",
)
EXPECTED_COUNTS = {
    "antibodies": 38_599,
    "antigens": 31,
    "binders": 20_980,
    "conflicting_pairs": 0,
    "exact_duplicate_groups": 0,
    "exact_duplicate_records": 0,
    "nonbinders": 552_911,
    "rows": 573_891,
    "subject_names": ["wizzy"],
    "subject_sexes": ["male"],
    "subject_species": ["alpaca"],
    "unique_pairs": 573_891,
}


class AvidaIndependentAuditError(ValueError):
    """Raised when independently reconstructed AVIDa semantics disagree."""


def audit_avida_hil6_candidate(
    *,
    raw_root: Path,
    registered_raw_inventory: Path,
    contract_path: Path,
    candidate_root: Path,
    normalizer_source_path: Path,
    output_receipt: Path,
    audited_at_utc: str | None = None,
    expected_counts: Mapping[str, Any] = EXPECTED_COUNTS,
) -> Mapping[str, Any]:
    """Reconstruct every candidate row from authenticated raw CSV bytes."""

    raw_root = _directory(raw_root, "raw root")
    candidate_root = _directory(candidate_root, "candidate root")
    timestamp = audited_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    _utc(timestamp)
    registered_raw_inventory = _regular_file(
        registered_raw_inventory,
        "registered raw inventory",
    )
    materialized_inventory = _regular_file(
        raw_root / "raw_sha256.json",
        "materialized raw inventory",
    )
    inventory_bytes = registered_raw_inventory.read_bytes()
    if materialized_inventory.read_bytes() != inventory_bytes:
        raise AvidaIndependentAuditError("raw inventory bytes disagree")
    raw_inventory = _strict_json(inventory_bytes, "raw inventory")
    _authenticate_raw_files(raw_root, raw_inventory)
    raw_inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    contract_path = _regular_file(contract_path, "PAIRBench contract")
    contract_sha256 = _file_sha256(contract_path)
    normalizer_source_path = _regular_file(
        normalizer_source_path,
        "normalizer source",
    )
    normalizer_sha256 = _file_sha256(normalizer_source_path)
    receipt_path = _regular_file(
        candidate_root / "CANDIDATE_RECEIPT.json",
        "candidate receipt",
    )
    candidate_receipt = _strict_json(receipt_path.read_bytes(), "candidate receipt")
    _validate_candidate_receipt(
        candidate_receipt,
        raw_inventory_sha256=raw_inventory_sha256,
        contract_sha256=contract_sha256,
        normalizer_sha256=normalizer_sha256,
        expected_counts=expected_counts,
    )
    output_records = {
        record["path"]: record for record in candidate_receipt["outputs"]
    }
    _authenticate_candidate_files(candidate_root, output_records)
    audit_summary = _strict_json(
        (candidate_root / "AUDIT_SUMMARY.json").read_bytes(),
        "candidate audit summary",
    )
    _validate_audit_summary(
        audit_summary,
        output_records=output_records,
        raw_inventory_sha256=raw_inventory_sha256,
        contract_sha256=contract_sha256,
        expected_counts=expected_counts,
    )
    antigen_by_label = _raw_antigens(raw_root / "antigen_sequences.csv")
    antibodies = _audit_antibody_entities(
        candidate_root / "entities_antibody.parquet"
    )
    antigen_ids = _audit_antigen_entities(
        candidate_root / "entities_antigen.parquet",
        antigen_by_label,
    )
    counts = _audit_pairs_and_assays(
        raw_path=raw_root / "AVIDa-hIL6.csv",
        pair_path=candidate_root / "pairs.parquet",
        assay_path=candidate_root / "assays.parquet",
        antibodies=antibodies,
        antigen_ids=antigen_ids,
    )
    if pq.ParquetFile(candidate_root / "duplicate_groups.parquet").metadata.num_rows:
        raise AvidaIndependentAuditError("duplicate group table is not empty")
    counts.update(
        {
            "antibodies": len(antibodies),
            "antigens": len(antigen_ids),
            "conflicting_pairs": 0,
            "exact_duplicate_groups": 0,
            "exact_duplicate_records": 0,
        }
    )
    if counts != expected_counts or counts != candidate_receipt["counts"]:
        raise AvidaIndependentAuditError("independent counts disagree")
    result = {
        "audited_at_utc": timestamp,
        "auditor_source_sha256": _file_sha256(Path(__file__).resolve()),
        "candidate_receipt_sha256": _file_sha256(receipt_path),
        "counts": counts,
        "format_version": "pairbench_avida_independent_audit_v1",
        "input_raw_inventory_sha256": raw_inventory_sha256,
        "normalizer_source_sha256": normalizer_sha256,
        "pairbench_contract_sha256": contract_sha256,
        "registry_normalization_eligible": False,
        "semantic_comparison": "exact_row_order_and_content",
        "source_id": SOURCE_ID,
        "status": "pass_candidate_remains_blocked",
    }
    output_receipt = Path(output_receipt).expanduser().absolute()
    if not output_receipt.parent.is_dir() or output_receipt.parent.is_symlink():
        raise AvidaIndependentAuditError("audit receipt parent is invalid")
    _write_create_only_json(output_receipt, result)
    return result


def _validate_candidate_receipt(
    payload: Mapping[str, Any],
    *,
    raw_inventory_sha256: str,
    contract_sha256: str,
    normalizer_sha256: str,
    expected_counts: Mapping[str, Any],
) -> None:
    expected = {
        "blocking_reasons",
        "counts",
        "created_at_utc",
        "format_version",
        "input_raw_inventory_sha256",
        "normalizer_source_sha256",
        "outputs",
        "pairbench_contract_sha256",
        "registry_normalization_eligible",
        "source_id",
        "status",
    }
    if set(payload) != expected:
        raise AvidaIndependentAuditError("candidate receipt fields differ")
    if (
        payload["format_version"] != "pairbench_avida_candidate_receipt_v1"
        or payload["source_id"] != SOURCE_ID
        or payload["status"] != "candidate_complete_blocked"
        or payload["registry_normalization_eligible"] is not False
        or tuple(payload["blocking_reasons"]) != BLOCKERS
        or payload["input_raw_inventory_sha256"] != raw_inventory_sha256
        or payload["pairbench_contract_sha256"] != contract_sha256
        or payload["normalizer_source_sha256"] != normalizer_sha256
    ):
        raise AvidaIndependentAuditError("candidate receipt authority differs")
    if payload["counts"] != expected_counts:
        raise AvidaIndependentAuditError("candidate receipt counts differ")
    outputs = payload["outputs"]
    paths = [record.get("path") for record in outputs if type(record) is dict]
    if len(outputs) != 6 or paths != sorted(set(paths)):
        raise AvidaIndependentAuditError("candidate output inventory differs")
    for record in outputs:
        if set(record) != {"path", "sha256", "size_bytes"}:
            raise AvidaIndependentAuditError("candidate output record fields differ")


def _authenticate_candidate_files(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    observed: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AvidaIndependentAuditError("candidate root contains a symlink")
        if path.is_file():
            observed.add(path.relative_to(root).as_posix())
    expected = set(records) | {"CANDIDATE_RECEIPT.json"}
    if observed != expected:
        raise AvidaIndependentAuditError("candidate root file set differs")
    for relative, record in records.items():
        path = _regular_file(root / relative, f"candidate output {relative}")
        if path.stat().st_size != record["size_bytes"]:
            raise AvidaIndependentAuditError("candidate output size differs")
        if _file_sha256(path) != record["sha256"]:
            raise AvidaIndependentAuditError("candidate output digest differs")


def _validate_audit_summary(
    payload: Mapping[str, Any],
    *,
    output_records: Mapping[str, Mapping[str, Any]],
    raw_inventory_sha256: str,
    contract_sha256: str,
    expected_counts: Mapping[str, Any],
) -> None:
    expected_tables = [
        output_records[name]
        for name in sorted(output_records)
        if name.endswith(".parquet")
    ]
    if (
        payload.get("format_version") != "pairbench_avida_candidate_audit_v1"
        or payload.get("source_id") != SOURCE_ID
        or payload.get("counts") != expected_counts
        or tuple(payload.get("blocking_reasons", ())) != BLOCKERS
        or payload.get("registry_normalization_eligible") is not False
        or payload.get("input_raw_inventory_sha256") != raw_inventory_sha256
        or payload.get("pairbench_contract_sha256") != contract_sha256
        or payload.get("table_inventory") != expected_tables
        or payload.get("unknown_cross_pairs_created") != 0
    ):
        raise AvidaIndependentAuditError("candidate audit summary differs")


def _audit_antibody_entities(path: Path) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for row in _parquet_rows(path):
        sequence = row["construct_sequence"]
        antibody_id = _content_id("ab", sequence)
        if (
            row["antibody_id"] != antibody_id
            or row["sequence_digest"] != hashlib.sha256(sequence.encode()).hexdigest()
            or row["annotation_status"]
            != "raw_construct_untrimmed_signal_and_tag_boundaries_pending"
            or row["antibody_format"] != "vhh_construct_untrimmed"
            or row["vh_sequence"] is not None
            or row["vl_sequence"] is not None
            or row["source_id"] != SOURCE_ID
            or row["license_route"] != "noncommercial_only"
            or antibody_id in result
        ):
            raise AvidaIndependentAuditError("antibody entity semantics differ")
        result[antibody_id] = sequence
    return result


def _audit_antigen_entities(
    path: Path,
    raw_antigens: Mapping[str, str],
) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for row in _parquet_rows(path):
        label = row["variant_id"]
        sequence = raw_antigens.get(label)
        antigen_id = _content_id("ag", sequence or "")
        if (
            sequence is None
            or row["antigen_id"] != antigen_id
            or row["construct_sequences"] != [sequence]
            or row["chain_sequences"] is not None
            or row["sequence_digest"] != hashlib.sha256(sequence.encode()).hexdigest()
            or row["annotation_status"]
            != "raw_construct_untrimmed_signal_and_tag_boundaries_pending"
            or row["source_id"] != SOURCE_ID
            or antigen_id in result
        ):
            raise AvidaIndependentAuditError("antigen entity semantics differ")
        result[antigen_id] = label
    if set(result.values()) != set(raw_antigens):
        raise AvidaIndependentAuditError("antigen entity coverage differs")
    return result


def _audit_pairs_and_assays(
    *,
    raw_path: Path,
    pair_path: Path,
    assay_path: Path,
    antibodies: Mapping[str, str],
    antigen_ids: Mapping[str, str],
) -> dict[str, Any]:
    antigen_by_label = {label: identity for identity, label in antigen_ids.items()}
    pair_ids: set[str] = set()
    binders = 0
    nonbinders = 0
    subjects = {"names": set(), "sexes": set(), "species": set()}
    sentinel = object()
    streams = zip_longest(
        _raw_rows(raw_path, PAIR_HEADER),
        _parquet_rows(pair_path),
        _parquet_rows(assay_path),
        fillvalue=sentinel,
    )
    rows = 0
    for raw_value, pair, assay in streams:
        if raw_value is sentinel or pair is sentinel or assay is sentinel:
            raise AvidaIndependentAuditError("raw, pair, and assay row counts differ")
        row_number, raw = raw_value
        rows += 1
        label = int(raw["label"])
        binders += label
        nonbinders += 1 - label
        antibody_id = _content_id("ab", raw["VHH_sequence"])
        antigen_id = antigen_by_label.get(raw["Ag_label"])
        if antibody_id not in antibodies or antigen_id is None:
            raise AvidaIndependentAuditError("pair entity reference differs")
        pair_id = _content_id("pair", f"{antibody_id}\n{antigen_id}")
        source_record_id = f"{SOURCE_ID}:row:{row_number:06d}"
        assay_id = _content_id("assay", source_record_id)
        evidence = "measured_binder" if label else "measured_nonbinder"
        expected_pair = {
            "antibody_id": antibody_id,
            "antigen_id": antigen_id,
            "assay_id": assay_id,
            "binary_binding_target": label,
            "evidence_semantics": evidence,
            "pair_id": pair_id,
            "source_id": SOURCE_ID,
            "source_record_id": source_record_id,
        }
        if pair != expected_pair or pair_id in pair_ids:
            raise AvidaIndependentAuditError("pair row semantics differ")
        pair_ids.add(pair_id)
        expected_assay = {
            "assay_id": assay_id,
            "assay_type": "binary_binding_unspecified_in_raw_release",
            "construct_context": {
                "antigen_label": raw["Ag_label"],
                "subject_name": raw["subject_name"],
                "subject_sex": raw["subject_sex"],
                "subject_species": raw["subject_species"],
            },
            "experiment_id": f"{SOURCE_ID}:released_binary_labels",
            "measurement_uncertainty": None,
            "pair_id": pair_id,
            "raw_unit": "binary_label",
            "raw_value": float(label),
            "source_id": SOURCE_ID,
            "temperature_kelvin": None,
        }
        if assay != expected_assay:
            raise AvidaIndependentAuditError("assay row semantics differ")
        subjects["names"].add(raw["subject_name"])
        subjects["sexes"].add(raw["subject_sex"])
        subjects["species"].add(raw["subject_species"])
    return {
        "binders": binders,
        "nonbinders": nonbinders,
        "rows": rows,
        "subject_names": sorted(subjects["names"]),
        "subject_sexes": sorted(subjects["sexes"]),
        "subject_species": sorted(subjects["species"]),
        "unique_pairs": len(pair_ids),
    }


def _raw_antigens(path: Path) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for _, row in _raw_rows(path, ANTIGEN_HEADER):
        if row["Ag_label"] in result:
            raise AvidaIndependentAuditError("raw antigen labels are duplicated")
        result[row["Ag_label"]] = row["Ag_sequence"]
    return result


def _raw_rows(
    path: Path,
    expected_header: tuple[str, ...],
) -> Iterator[tuple[int, Mapping[str, str]]]:
    path = _regular_file(path, "raw CSV")
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise AvidaIndependentAuditError("raw CSV header differs")
        for row_number, row in enumerate(reader, start=2):
            if None in row or set(row) != set(expected_header):
                raise AvidaIndependentAuditError("raw CSV row shape differs")
            yield row_number, row


def _parquet_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    path = _regular_file(path, "candidate Parquet")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16_384):
        yield from batch.to_pylist()


def _authenticate_raw_files(root: Path, inventory: Mapping[str, Any]) -> None:
    if (
        inventory.get("format_version") != "pairscale_raw_sha256_inventory_v1"
        or inventory.get("source_id") != SOURCE_ID
        or inventory.get("status") != "materialized"
    ):
        raise AvidaIndependentAuditError("raw inventory authority differs")
    records = inventory.get("files")
    if type(records) is not list:
        raise AvidaIndependentAuditError("raw inventory file records differ")
    expected_paths = {record.get("path") for record in records}
    expected_paths.add("raw_sha256.json")
    observed_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AvidaIndependentAuditError("raw root contains a symlink")
        if path.is_file():
            observed_paths.add(path.relative_to(root).as_posix())
    if observed_paths != expected_paths:
        raise AvidaIndependentAuditError("raw root file set differs")
    for record in records:
        path = _regular_file(root / record["path"], "raw inventory file")
        if path.stat().st_size != record["size_bytes"] or _file_sha256(path) != record[
            "sha256"
        ]:
            raise AvidaIndependentAuditError("raw file bytes differ")


def _strict_json(raw: bytes, name: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise AvidaIndependentAuditError(f"non-finite JSON constant in {name}: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AvidaIndependentAuditError(f"duplicate JSON key in {name}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AvidaIndependentAuditError(f"cannot parse {name}") from exc
    if type(value) is not dict:
        raise AvidaIndependentAuditError(f"{name} must be an object")
    return value


def _content_id(kind: str, value: str) -> str:
    return f"{kind}_sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _directory(path: Path, name: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_dir():
        raise AvidaIndependentAuditError(f"{name} is invalid")
    return result


def _regular_file(path: Path, name: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_file():
        raise AvidaIndependentAuditError(f"{name} is invalid")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(data)


def _utc(value: str) -> None:
    if type(value) is not str or not value.endswith("Z"):
        raise AvidaIndependentAuditError("audit timestamp must use UTC Z form")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AvidaIndependentAuditError("audit timestamp is invalid") from exc
