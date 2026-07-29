"""Strict PAIRBench v0 content semantics and input-track authorization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


CONTRACT_FORMAT = "pairbench_content_contract_v0"
PAIR_FIELDS = frozenset(
    {
        "antibody_id",
        "antigen_id",
        "assay_id",
        "binary_binding_target",
        "evidence_semantics",
        "pair_id",
        "source_id",
        "source_record_id",
    }
)
TABLE_NAMES = frozenset(
    {
        "entities_antibody.parquet",
        "entities_antigen.parquet",
        "pairs.parquet",
        "assays.parquet",
        "structures.parquet",
        "residue_geometry.zarr",
    }
)
EVIDENCE_NAMES = frozenset(
    {
        "observed_structural_positive",
        "measured_binder",
        "measured_nonbinder",
        "measured_affinity",
        "measured_mutation_effect",
        "unknown_cross_pair",
        "synthetic_decoy",
        "predicted_complex_candidate",
    }
)
CANONICALIZATION_CONTRACT = {
    "digest_algorithm": "sha256",
    "digest_input_encoding": "utf-8",
    "digest_object_encoding": "rfc8785_json_canonicalization_scheme",
    "entity_identity_scope": "biological_sequence_content_not_source_row",
    "pair_identity_scope": "antibody_id_plus_antigen_id",
    "sequence_alphabet": "ACDEFGHIKLMNPQRSTVWY",
    "sequence_case": "uppercase",
}
GLOBAL_PROHIBITIONS = (
    "complex_coordinates_in_track_s_or_m",
    "cross_chain_distance_in_track_s_or_m",
    "predicted_complex_candidate_as_experimental_binding_truth",
    "raw_cross_assay_value_regression_without_unit_and_context",
    "test_threshold_selection",
    "unknown_cross_pair_as_binary_nonbinder",
)
EVIDENCE_CONTRACT = {
    "measured_affinity": (
        "excluded",
        ("affinity_ranking", "assay_specific_calibration"),
    ),
    "measured_binder": (
        "positive",
        ("binary_binding", "specificity_retrieval"),
    ),
    "measured_mutation_effect": (
        "excluded",
        ("mutation_effect_ranking", "mutation_sign"),
    ),
    "measured_nonbinder": (
        "negative",
        ("binary_binding", "specificity_retrieval"),
    ),
    "observed_structural_positive": (
        "not_measured_binding_evidence",
        ("distance", "orientation", "residue_pair_contact"),
    ),
    "predicted_complex_candidate": (
        "excluded",
        ("complex_quality_ranking", "structure_candidate_classification"),
    ),
    "synthetic_decoy": ("excluded", ("ranking_control",)),
    "unknown_cross_pair": (
        "excluded",
        ("positive_unlabeled_learning", "ranking_control"),
    ),
}
TRACK_LABELS = {
    "S": "sequence_only",
    "M": "monomer_structure_aware",
    "C": "complex_candidate_aware",
}
TRACK_FEATURES = {
    "S": (
        "antibody_sequence",
        "antigen_sequence",
        "sequence_derived_annotation",
    ),
    "M": (
        "antibody_monomer_structure",
        "antibody_sequence",
        "antigen_monomer_structure",
        "antigen_sequence",
        "monomer_surface_feature",
        "monomer_uncertainty",
        "sequence_derived_annotation",
    ),
    "C": (
        "antibody_monomer_structure",
        "antibody_sequence",
        "antigen_monomer_structure",
        "antigen_sequence",
        "candidate_confidence",
        "complex_candidate_coordinates",
        "cross_chain_distance",
        "monomer_surface_feature",
        "monomer_uncertainty",
        "pairformer_contact_consistency",
        "sequence_derived_annotation",
    ),
}
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,255}$")


class PairbenchSchemaError(ValueError):
    """Raised when PAIRBench content or feature authorization is invalid."""


def load_pairbench_contract(path: Path) -> tuple[Mapping[str, Any], str]:
    """Load the machine contract with duplicate-key and semantic checks."""

    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise PairbenchSchemaError("content contract must be a non-symlink file")
    raw = path.read_bytes()
    payload = _strict_json(raw)
    expected_fields = {
        "canonicalization",
        "evidence_semantics",
        "format_version",
        "global_prohibitions",
        "input_tracks",
        "schema_name",
        "schema_status",
        "tables",
    }
    if set(payload) != expected_fields:
        raise PairbenchSchemaError("content contract fields differ")
    if payload["format_version"] != CONTRACT_FORMAT:
        raise PairbenchSchemaError("content contract version differs")
    if payload["schema_name"] != "PAIRBench v0 content contract":
        raise PairbenchSchemaError("content contract schema name differs")
    if payload["schema_status"] != "development_not_frozen":
        raise PairbenchSchemaError("content contract status is not fail-closed")
    if _object(payload["canonicalization"], "canonicalization") != (
        CANONICALIZATION_CONTRACT
    ):
        raise PairbenchSchemaError("canonicalization contract differs")
    prohibitions = _string_list(
        payload["global_prohibitions"],
        "global_prohibitions",
    )
    if tuple(prohibitions) != GLOBAL_PROHIBITIONS:
        raise PairbenchSchemaError("global prohibition set differs")
    tables = _object(payload["tables"], "tables")
    if set(tables) != TABLE_NAMES:
        raise PairbenchSchemaError("content table set differs")
    for name, value in tables.items():
        table = _object(value, f"table {name}")
        if set(table) != {"primary_key", "required_fields"}:
            raise PairbenchSchemaError(f"table fields differ for {name}")
        primary_key = _string_list(table["primary_key"], f"primary key {name}")
        required = _string_list(
            table["required_fields"],
            f"required fields {name}",
        )
        if not primary_key:
            raise PairbenchSchemaError(f"primary key is empty for {name}")
        if not set(primary_key) <= set(required):
            raise PairbenchSchemaError(f"primary key is not required for {name}")
    evidence = _object(payload["evidence_semantics"], "evidence_semantics")
    if set(evidence) != EVIDENCE_NAMES:
        raise PairbenchSchemaError("evidence semantic set differs")
    for name, (role, permitted_tasks) in EVIDENCE_CONTRACT.items():
        semantic = _object(evidence[name], name)
        if set(semantic) != {"binary_binding_role", "permitted_tasks"}:
            raise PairbenchSchemaError(f"evidence fields differ for {name}")
        if semantic["binary_binding_role"] != role:
            raise PairbenchSchemaError(f"binary role differs for {name}")
        tasks = _string_list(semantic["permitted_tasks"], f"tasks {name}")
        if tuple(tasks) != permitted_tasks:
            raise PairbenchSchemaError(f"permitted tasks differ for {name}")
    tracks = _object(payload["input_tracks"], "input_tracks")
    if set(tracks) != {"S", "M", "C"}:
        raise PairbenchSchemaError("input track set differs")
    track_features: dict[str, set[str]] = {}
    for name, label in TRACK_LABELS.items():
        track = _object(tracks[name], f"track {name}")
        if set(track) != {"allowed_feature_groups", "label"}:
            raise PairbenchSchemaError(f"track fields differ for {name}")
        if track["label"] != label:
            raise PairbenchSchemaError(f"track label differs for {name}")
        allowed = _string_list(
            track["allowed_feature_groups"],
            f"allowed feature groups {name}",
        )
        if tuple(allowed) != TRACK_FEATURES[name]:
            raise PairbenchSchemaError(f"allowed feature groups differ for {name}")
        track_features[name] = set(allowed)
    if not track_features["S"] < track_features["M"] < track_features["C"]:
        raise PairbenchSchemaError("input tracks are not strictly nested")
    return payload, hashlib.sha256(raw).hexdigest()


def binding_target_for_evidence(evidence_semantics: str) -> int | None:
    """Return the only authorized binary label for one evidence semantic."""

    if evidence_semantics == "measured_binder":
        return 1
    if evidence_semantics == "measured_nonbinder":
        return 0
    if evidence_semantics not in EVIDENCE_NAMES:
        raise PairbenchSchemaError("unknown evidence semantics")
    return None


def validate_pair_record(record: Mapping[str, Any]) -> None:
    """Validate a pair row and prevent unknown or synthetic negative promotion."""

    if type(record) is not dict or set(record) != PAIR_FIELDS:
        raise PairbenchSchemaError("pair record fields differ")
    for name in (
        "antibody_id",
        "antigen_id",
        "pair_id",
        "source_id",
        "source_record_id",
    ):
        value = record[name]
        if type(value) is not str or not _IDENTIFIER.fullmatch(value):
            raise PairbenchSchemaError(f"{name} is not a canonical identifier")
    if record["assay_id"] is not None and (
        type(record["assay_id"]) is not str
        or not _IDENTIFIER.fullmatch(record["assay_id"])
    ):
        raise PairbenchSchemaError("assay_id is invalid")
    evidence = record["evidence_semantics"]
    expected_target = binding_target_for_evidence(evidence)
    if record["binary_binding_target"] != expected_target:
        raise PairbenchSchemaError("binary target contradicts evidence semantics")
    if evidence.startswith("measured_") and record["assay_id"] is None:
        raise PairbenchSchemaError("measured evidence requires an assay")


def authorize_input_track(
    contract: Mapping[str, Any],
    *,
    track: str,
    feature_groups: Sequence[str],
) -> tuple[str, ...]:
    """Authorize an exact feature group set for one declared input track."""

    tracks = _object(contract.get("input_tracks"), "input_tracks")
    if track not in tracks:
        raise PairbenchSchemaError("unknown input track")
    if type(feature_groups) not in {list, tuple}:
        raise PairbenchSchemaError("feature_groups must be an ordered sequence")
    if list(feature_groups) != sorted(set(feature_groups)):
        raise PairbenchSchemaError("feature_groups must be sorted and unique")
    allowed = set(_object(tracks[track], f"track {track}")["allowed_feature_groups"])
    unauthorized = sorted(set(feature_groups) - allowed)
    if unauthorized:
        raise PairbenchSchemaError(
            f"feature groups are unauthorized for track {track}: {unauthorized}"
        )
    return tuple(feature_groups)


def _strict_json(raw: bytes) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise PairbenchSchemaError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PairbenchSchemaError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PairbenchSchemaError("cannot parse content contract") from exc
    if type(value) is not dict:
        raise PairbenchSchemaError("content contract must be an object")
    return value


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise PairbenchSchemaError(f"{name} must be an object")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise PairbenchSchemaError(f"{name} must be a string list")
    if value != sorted(set(value)):
        raise PairbenchSchemaError(f"{name} must be sorted and unique")
    return value
