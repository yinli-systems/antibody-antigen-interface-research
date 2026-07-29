"""Build fail-closed PAIRBench candidate tables from pinned AVIDa-hIL6 bytes."""

from __future__ import annotations

from contextlib import ExitStack
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from interaction_first.data_registry import load_source_registry
from interaction_first.pairbench_schema import load_pairbench_contract


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
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
RAW_CONSTRUCT_STATUS = "raw_construct_untrimmed_signal_and_tag_boundaries_pending"
BLOCKING_REASONS = (
    "antigen_chain_boundary_annotation_pending",
    "vhh_domain_boundary_annotation_pending",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AvidaCandidateError(ValueError):
    """Raised when authenticated AVIDa bytes cannot produce a safe candidate."""


@dataclass(frozen=True)
class AvidaExpectations:
    """Pinned byte-level invariants established before candidate generation."""

    rows: int
    antibodies: int
    antigens: int
    binders: int
    nonbinders: int
    exact_duplicate_groups: int
    exact_duplicate_records: int
    conflicting_pairs: int
    subject_species: tuple[str, ...]
    subject_names: tuple[str, ...]
    subject_sexes: tuple[str, ...]


PINNED_EXPECTATIONS = AvidaExpectations(
    rows=573_891,
    antibodies=38_599,
    antigens=31,
    binders=20_980,
    nonbinders=552_911,
    exact_duplicate_groups=0,
    exact_duplicate_records=0,
    conflicting_pairs=0,
    subject_species=("alpaca",),
    subject_names=("wizzy",),
    subject_sexes=("male",),
)


def build_avida_hil6_candidate(
    *,
    raw_root: Path,
    registry_root: Path,
    contract_path: Path,
    output_root: Path,
    expectations: AvidaExpectations = PINNED_EXPECTATIONS,
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Authenticate raw bytes and create a non-admitted normalization candidate."""

    raw_root = _directory(raw_root, "raw root")
    registry = load_source_registry(Path(registry_root) / SOURCE_ID)
    if registry.source_id != SOURCE_ID or registry.raw["status"] != "materialized":
        raise AvidaCandidateError("AVIDa registry is not materialized")
    raw_inventory_sha256 = _authenticate_raw_root(raw_root, registry)
    contract, contract_sha256 = load_pairbench_contract(contract_path)
    _validate_schemas_against_contract(contract)
    created_at = created_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    _utc(created_at)
    output_root = Path(output_root).expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise AvidaCandidateError("candidate output root already exists")
    if not output_root.parent.is_dir() or output_root.parent.is_symlink():
        raise AvidaCandidateError("candidate output parent is invalid")
    output_root.mkdir(mode=0o750)
    try:
        receipt = _build_authenticated_candidate(
            raw_root=raw_root,
            output_root=output_root,
            raw_inventory_sha256=raw_inventory_sha256,
            contract_sha256=contract_sha256,
            expectations=expectations,
            created_at=created_at,
        )
        _write_create_only_json(output_root / "CANDIDATE_RECEIPT.json", receipt)
        return receipt
    except Exception as exc:
        failure = {
            "created_at_utc": created_at,
            "error_message": str(exc),
            "error_type": type(exc).__name__,
            "format_version": "pairbench_avida_candidate_failure_v1",
            "source_id": SOURCE_ID,
        }
        failure_path = output_root / "FAILURE_RECEIPT.json"
        if not failure_path.exists():
            _write_create_only_json(failure_path, failure)
        raise


def _build_authenticated_candidate(
    *,
    raw_root: Path,
    output_root: Path,
    raw_inventory_sha256: str,
    contract_sha256: str,
    expectations: AvidaExpectations,
    created_at: str,
) -> Mapping[str, Any]:
    antigens = _read_antigens(raw_root / "antigen_sequences.csv")
    antibody_rows: dict[str, dict[str, Any]] = {}
    pair_observations: dict[str, tuple[int, str]] = {}
    duplicate_rows: dict[str, dict[str, Any]] = {}
    subjects = {"species": set(), "names": set(), "sexes": set()}
    counters = {
        "rows": 0,
        "binders": 0,
        "nonbinders": 0,
        "conflicts": 0,
        "duplicate_records": 0,
    }
    schemas = _schemas()
    output_paths = {
        "antibodies": output_root / "entities_antibody.parquet",
        "antigens": output_root / "entities_antigen.parquet",
        "pairs": output_root / "pairs.parquet",
        "assays": output_root / "assays.parquet",
        "duplicates": output_root / "duplicate_groups.parquet",
    }
    with ExitStack() as stack:
        pair_writer = stack.enter_context(_writer(output_paths["pairs"], schemas["pairs"]))
        assay_writer = stack.enter_context(
            _writer(output_paths["assays"], schemas["assays"])
        )
        pair_batch: list[dict[str, Any]] = []
        assay_batch: list[dict[str, Any]] = []
        for row_number, row in _read_pair_rows(raw_root / "AVIDa-hIL6.csv"):
            sequence = _sequence(row["VHH_sequence"], f"row {row_number} VHH")
            antigen_label = _canonical_text(row["Ag_label"], "antigen label")
            if antigen_label not in antigens:
                raise AvidaCandidateError(f"row {row_number} references unknown antigen")
            if row["label"] not in {"0", "1"}:
                raise AvidaCandidateError(f"row {row_number} has a non-binary label")
            label = int(row["label"])
            counters["rows"] += 1
            counters["binders" if label else "nonbinders"] += 1
            species = _canonical_text(row["subject_species"], "subject species")
            subject_name = _canonical_text(row["subject_name"], "subject name")
            subject_sex = _canonical_text(row["subject_sex"], "subject sex")
            subjects["species"].add(species)
            subjects["names"].add(subject_name)
            subjects["sexes"].add(subject_sex)
            antibody_id = _content_id("ab", sequence)
            antigen_id = antigens[antigen_label]["antigen_id"]
            pair_id = _content_id("pair", f"{antibody_id}\n{antigen_id}")
            source_record_id = f"{SOURCE_ID}:row:{row_number:06d}"
            assay_id = _content_id("assay", source_record_id)
            if antibody_id not in antibody_rows:
                antibody_rows[antibody_id] = _antibody_record(
                    antibody_id=antibody_id,
                    sequence=sequence,
                )
            previous = pair_observations.get(pair_id)
            if previous is not None:
                counters["duplicate_records"] += 1
                if previous[0] != label:
                    counters["conflicts"] += 1
                group = duplicate_rows.setdefault(
                    pair_id,
                    {
                        "duplicate_group_id": _content_id("duplicate", pair_id),
                        "duplicate_type": "exact_construct_pair",
                        "labels": [previous[0]],
                        "pair_id": pair_id,
                        "source_record_ids": [previous[1]],
                    },
                )
                group["labels"].append(label)
                group["source_record_ids"].append(source_record_id)
            else:
                pair_observations[pair_id] = (label, source_record_id)
            evidence = "measured_binder" if label else "measured_nonbinder"
            pair_batch.append(
                {
                    "antibody_id": antibody_id,
                    "antigen_id": antigen_id,
                    "assay_id": assay_id,
                    "binary_binding_target": label,
                    "evidence_semantics": evidence,
                    "pair_id": pair_id,
                    "source_id": SOURCE_ID,
                    "source_record_id": source_record_id,
                }
            )
            assay_batch.append(
                {
                    "assay_id": assay_id,
                    "assay_type": "binary_binding_unspecified_in_raw_release",
                    "construct_context": {
                        "antigen_label": antigen_label,
                        "subject_name": subject_name,
                        "subject_sex": subject_sex,
                        "subject_species": species,
                    },
                    "experiment_id": f"{SOURCE_ID}:released_binary_labels",
                    "measurement_uncertainty": None,
                    "pair_id": pair_id,
                    "raw_unit": "binary_label",
                    "raw_value": float(label),
                    "source_id": SOURCE_ID,
                    "temperature_kelvin": None,
                }
            )
            if len(pair_batch) == 16_384:
                _write_batch(pair_writer, pair_batch, schemas["pairs"])
                _write_batch(assay_writer, assay_batch, schemas["assays"])
                pair_batch.clear()
                assay_batch.clear()
        if pair_batch:
            _write_batch(pair_writer, pair_batch, schemas["pairs"])
            _write_batch(assay_writer, assay_batch, schemas["assays"])
    _write_table(
        output_paths["antibodies"],
        [antibody_rows[key] for key in sorted(antibody_rows)],
        schemas["antibodies"],
    )
    _write_table(
        output_paths["antigens"],
        [antigens[key]["record"] for key in sorted(antigens)],
        schemas["antigens"],
    )
    _write_table(
        output_paths["duplicates"],
        [duplicate_rows[key] for key in sorted(duplicate_rows)],
        schemas["duplicates"],
    )
    observed = {
        "antibodies": len(antibody_rows),
        "antigens": len(antigens),
        "binders": counters["binders"],
        "conflicting_pairs": counters["conflicts"],
        "exact_duplicate_groups": len(duplicate_rows),
        "exact_duplicate_records": counters["duplicate_records"],
        "nonbinders": counters["nonbinders"],
        "rows": counters["rows"],
        "subject_names": sorted(subjects["names"]),
        "subject_sexes": sorted(subjects["sexes"]),
        "subject_species": sorted(subjects["species"]),
        "unique_pairs": len(pair_observations),
    }
    _check_expectations(observed, expectations)
    table_inventory = [_file_record(path) for path in output_paths.values()]
    table_inventory.sort(key=lambda value: value["path"])
    audit = {
        "blocking_reasons": list(BLOCKING_REASONS),
        "counts": observed,
        "created_at_utc": created_at,
        "format_version": "pairbench_avida_candidate_audit_v1",
        "input_raw_inventory_sha256": raw_inventory_sha256,
        "pairbench_contract_sha256": contract_sha256,
        "raw_antigen_sequence_semantics": "untrimmed_construct_not_chain_sequence",
        "raw_vhh_sequence_semantics": "untrimmed_construct_not_vh_domain",
        "registry_normalization_eligible": False,
        "source_id": SOURCE_ID,
        "table_inventory": table_inventory,
        "unknown_cross_pairs_created": 0,
    }
    audit_path = output_root / "AUDIT_SUMMARY.json"
    _write_create_only_json(audit_path, audit)
    outputs = table_inventory + [_file_record(audit_path)]
    outputs.sort(key=lambda value: value["path"])
    return {
        "blocking_reasons": list(BLOCKING_REASONS),
        "counts": observed,
        "created_at_utc": created_at,
        "format_version": "pairbench_avida_candidate_receipt_v1",
        "input_raw_inventory_sha256": raw_inventory_sha256,
        "normalizer_source_sha256": _source_sha256(),
        "outputs": outputs,
        "pairbench_contract_sha256": contract_sha256,
        "registry_normalization_eligible": False,
        "source_id": SOURCE_ID,
        "status": "candidate_complete_blocked",
    }


def _schemas() -> Mapping[str, pa.Schema]:
    text = pa.string()
    return {
        "antibodies": pa.schema(
            [
                ("annotation_status", text),
                ("antibody_format", text),
                ("antibody_id", text),
                ("cdr_boundaries", pa.list_(text)),
                ("clonotype_id", text),
                ("construct_sequence", text),
                ("germline_j", text),
                ("germline_v", text),
                ("imgt_numbering", pa.list_(text)),
                ("license_route", text),
                ("lineage_id", text),
                ("sequence_digest", text),
                ("source_id", text),
                ("vh_sequence", text),
                ("vl_sequence", text),
            ]
        ),
        "antigens": pa.schema(
            [
                ("annotation_status", text),
                ("antigen_family", text),
                ("antigen_id", text),
                ("chain_sequences", pa.list_(text)),
                ("construct_sequences", pa.list_(text)),
                ("fold_cluster_id", text),
                ("license_route", text),
                ("pdb_mappings", pa.list_(text)),
                ("release_date", pa.date32()),
                ("sequence_digest", text),
                ("source_id", text),
                ("structure_digest", text),
                ("uniprot_ids", pa.list_(text)),
                ("variant_id", text),
            ]
        ),
        "pairs": pa.schema(
            [
                ("antibody_id", text),
                ("antigen_id", text),
                ("assay_id", text),
                ("binary_binding_target", pa.int8()),
                ("evidence_semantics", text),
                ("pair_id", text),
                ("source_id", text),
                ("source_record_id", text),
            ]
        ),
        "assays": pa.schema(
            [
                ("assay_id", text),
                ("assay_type", text),
                (
                    "construct_context",
                    pa.struct(
                        [
                            ("antigen_label", text),
                            ("subject_name", text),
                            ("subject_sex", text),
                            ("subject_species", text),
                        ]
                    ),
                ),
                ("experiment_id", text),
                ("measurement_uncertainty", pa.float64()),
                ("pair_id", text),
                ("raw_unit", text),
                ("raw_value", pa.float64()),
                ("source_id", text),
                ("temperature_kelvin", pa.float64()),
            ]
        ),
        "duplicates": pa.schema(
            [
                ("duplicate_group_id", text),
                ("duplicate_type", text),
                ("labels", pa.list_(pa.int8())),
                ("pair_id", text),
                ("source_record_ids", pa.list_(text)),
            ]
        ),
    }


def _read_antigens(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row_number, row in _read_csv(path, ANTIGEN_HEADER):
        label = _canonical_text(row["Ag_label"], "antigen label")
        sequence = _sequence(row["Ag_sequence"], f"antigen row {row_number}")
        if label in result:
            raise AvidaCandidateError("duplicate antigen label")
        antigen_id = _content_id("ag", sequence)
        result[label] = {
            "antigen_id": antigen_id,
            "record": {
                "annotation_status": RAW_CONSTRUCT_STATUS,
                "antigen_family": "human_interleukin_6",
                "antigen_id": antigen_id,
                "chain_sequences": None,
                "construct_sequences": [sequence],
                "fold_cluster_id": None,
                "license_route": "noncommercial_only",
                "pdb_mappings": [],
                "release_date": None,
                "sequence_digest": hashlib.sha256(sequence.encode()).hexdigest(),
                "source_id": SOURCE_ID,
                "structure_digest": None,
                "uniprot_ids": [],
                "variant_id": label,
            },
        }
    if not result:
        raise AvidaCandidateError("antigen table is empty")
    return result


def _antibody_record(*, antibody_id: str, sequence: str) -> dict[str, Any]:
    return {
        "annotation_status": RAW_CONSTRUCT_STATUS,
        "antibody_format": "vhh_construct_untrimmed",
        "antibody_id": antibody_id,
        "cdr_boundaries": None,
        "clonotype_id": None,
        "construct_sequence": sequence,
        "germline_j": None,
        "germline_v": None,
        "imgt_numbering": None,
        "license_route": "noncommercial_only",
        "lineage_id": None,
        "sequence_digest": hashlib.sha256(sequence.encode()).hexdigest(),
        "source_id": SOURCE_ID,
        "vh_sequence": None,
        "vl_sequence": None,
    }


def _read_pair_rows(path: Path) -> Sequence[tuple[int, Mapping[str, str]]]:
    return _read_csv(path, PAIR_HEADER)


def _read_csv(path: Path, header: tuple[str, ...]) -> Any:
    if path.is_symlink() or not path.is_file():
        raise AvidaCandidateError(f"CSV input is invalid: {path.name}")

    def generate() -> Any:
        with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != header:
                raise AvidaCandidateError(f"CSV header differs: {path.name}")
            for row_number, row in enumerate(reader, start=2):
                if None in row or set(row) != set(header):
                    raise AvidaCandidateError(f"CSV row shape differs at {row_number}")
                yield row_number, row

    return generate()


def _authenticate_raw_root(raw_root: Path, registry: Any) -> str:
    registry_inventory = registry.root / "raw_sha256.json"
    materialized_inventory = raw_root / "raw_sha256.json"
    if materialized_inventory.is_symlink() or not materialized_inventory.is_file():
        raise AvidaCandidateError("materialized raw inventory is missing")
    if materialized_inventory.read_bytes() != registry_inventory.read_bytes():
        raise AvidaCandidateError("materialized raw inventory differs from registry")
    allowed_files = {record["path"] for record in registry.raw["files"]}
    allowed_files.add("raw_sha256.json")
    observed_files: set[str] = set()
    for path in raw_root.rglob("*"):
        if path.is_symlink():
            raise AvidaCandidateError("raw root contains a symlink")
        if path.is_file():
            observed_files.add(path.relative_to(raw_root).as_posix())
    if observed_files != allowed_files:
        raise AvidaCandidateError("raw root files differ from registered inventory")
    for record in registry.raw["files"]:
        path = raw_root / record["path"]
        if path.is_symlink() or not path.is_file():
            raise AvidaCandidateError(f"raw file is missing: {record['path']}")
        if path.stat().st_size != record["size_bytes"]:
            raise AvidaCandidateError(f"raw file size differs: {record['path']}")
        if _file_sha256(path) != record["sha256"]:
            raise AvidaCandidateError(f"raw file digest differs: {record['path']}")
    return _file_sha256(registry_inventory)


def _check_expectations(observed: Mapping[str, Any], expected: AvidaExpectations) -> None:
    expected_payload = {
        "antibodies": expected.antibodies,
        "antigens": expected.antigens,
        "binders": expected.binders,
        "conflicting_pairs": expected.conflicting_pairs,
        "exact_duplicate_groups": expected.exact_duplicate_groups,
        "exact_duplicate_records": expected.exact_duplicate_records,
        "nonbinders": expected.nonbinders,
        "rows": expected.rows,
        "subject_names": list(expected.subject_names),
        "subject_sexes": list(expected.subject_sexes),
        "subject_species": list(expected.subject_species),
        "unique_pairs": expected.rows - expected.exact_duplicate_records,
    }
    if dict(observed) != expected_payload:
        raise AvidaCandidateError("observed AVIDa invariants differ from pinned expectations")


def _writer(path: Path, schema: pa.Schema) -> pq.ParquetWriter:
    return pq.ParquetWriter(
        path,
        schema,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
    )


def _validate_schemas_against_contract(contract: Mapping[str, Any]) -> None:
    schemas = _schemas()
    mapping = {
        "assays.parquet": "assays",
        "entities_antibody.parquet": "antibodies",
        "entities_antigen.parquet": "antigens",
        "pairs.parquet": "pairs",
    }
    for filename, schema_name in mapping.items():
        expected = contract["tables"][filename]["required_fields"]
        if schemas[schema_name].names != expected:
            raise AvidaCandidateError(f"physical schema differs from {filename} contract")


def _write_batch(
    writer: pq.ParquetWriter,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
) -> None:
    writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def _write_table(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
    )


def _content_id(kind: str, value: str) -> str:
    return f"{kind}_sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _sequence(value: Any, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise AvidaCandidateError(f"{name} is not a canonical sequence")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise AvidaCandidateError(f"{name} is not ASCII") from exc
    if value != value.upper() or not set(value) <= AMINO_ACIDS:
        raise AvidaCandidateError(f"{name} has an invalid amino-acid alphabet")
    return value


def _canonical_text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise AvidaCandidateError(f"{name} is not canonical text")
    return value


def _directory(path: Path, name: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_dir():
        raise AvidaCandidateError(f"{name} is invalid")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _source_sha256() -> str:
    return _file_sha256(Path(__file__).resolve())


def _write_create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(data)


def _utc(value: str) -> None:
    if type(value) is not str or not value.endswith("Z"):
        raise AvidaCandidateError("created_at_utc must use UTC Z form")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AvidaCandidateError("created_at_utc is invalid") from exc
