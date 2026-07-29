"""Validate the frozen AVIDa-hIL6 construct-boundary protocol."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


FORMAT_VERSION = "pairscale_avida_hil6_annotation_protocol_v1"
PROTOCOL_ID = "avida_hil6_construct_boundaries_v1"
SOURCE_ID = "avida_hil6_2024_11"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AvidaAnnotationProtocolError(ValueError):
    """Raised when an AVIDa annotation protocol is unsafe or ambiguous."""


def load_avida_annotation_protocol(path: Path) -> tuple[Mapping[str, Any], str]:
    """Load one exact protocol and return its authenticated SHA-256 digest."""

    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise AvidaAnnotationProtocolError("annotation protocol must be a regular file")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AvidaAnnotationProtocolError("annotation protocol is not valid JSON") from exc
    if type(payload) is not dict:
        raise AvidaAnnotationProtocolError("annotation protocol root must be an object")
    _validate_protocol(payload)
    return payload, hashlib.sha256(raw).hexdigest()


def _validate_protocol(payload: Mapping[str, Any]) -> None:
    _exact_fields(
        payload,
        {
            "admission_policy",
            "antigen_boundary",
            "format_version",
            "frozen_at_utc",
            "input_binding",
            "protocol_id",
            "references",
            "score_free",
            "source_id",
            "vhh_boundary",
        },
        "protocol",
    )
    if payload["format_version"] != FORMAT_VERSION:
        raise AvidaAnnotationProtocolError("unsupported annotation protocol format")
    if payload["protocol_id"] != PROTOCOL_ID or payload["source_id"] != SOURCE_ID:
        raise AvidaAnnotationProtocolError("annotation protocol identity differs")
    if payload["score_free"] is not True:
        raise AvidaAnnotationProtocolError("annotation protocol must remain score-free")
    _timestamp(payload["frozen_at_utc"])
    _validate_input_binding(_mapping(payload["input_binding"], "input_binding"))
    _validate_vhh(_mapping(payload["vhh_boundary"], "vhh_boundary"))
    _validate_antigen(_mapping(payload["antigen_boundary"], "antigen_boundary"))
    _validate_admission(_mapping(payload["admission_policy"], "admission_policy"))
    _validate_references(payload["references"])


def _validate_input_binding(value: Mapping[str, Any]) -> None:
    _exact_fields(
        value,
        {
            "candidate_receipt_sha256",
            "expected_antibodies",
            "expected_antigens",
            "raw_inventory_sha256",
        },
        "input_binding",
    )
    _digest(value["candidate_receipt_sha256"], "candidate receipt")
    _digest(value["raw_inventory_sha256"], "raw inventory")
    if value["expected_antibodies"] != 38_599 or value["expected_antigens"] != 31:
        raise AvidaAnnotationProtocolError("expected entity counts differ")


def _validate_vhh(value: Mapping[str, Any]) -> None:
    _exact_fields(
        value,
        {"acceptance", "algorithm", "coordinate_semantics", "imgt_cdr_ranges_one_based_inclusive"},
        "vhh_boundary",
    )
    algorithm = _mapping(value["algorithm"], "vhh algorithm")
    _exact_fields(
        algorithm,
        {
            "license",
            "license_file_sha256",
            "mode",
            "model_parameters_path",
            "model_parameters_sha256",
            "model_weights_path",
            "model_weights_sha256",
            "package",
            "package_version",
            "rejected_seq_type_aliases",
            "seq_type_argument",
            "source_commit",
            "wheel_sha256",
        },
        "vhh algorithm",
    )
    locked = {
        "license": "BSD-3-Clause",
        "mode": "accuracy",
        "model_parameters_path": "anarcii/models/antibody/antibody_4_2_128_512.json",
        "model_weights_path": "anarcii/models/antibody/antibody_4_2_128_512.pt",
        "package": "anarcii",
        "package_version": "2.0.8",
        "seq_type_argument": "antibody",
        "source_commit": "e0d8f192f5a861e03a50918f114d0f5735e42333",
    }
    for field, expected in locked.items():
        if algorithm[field] != expected:
            raise AvidaAnnotationProtocolError(f"locked VHH algorithm field differs: {field}")
    if algorithm["rejected_seq_type_aliases"] != ["vhh", "vnar"]:
        raise AvidaAnnotationProtocolError("unsafe VHH alias policy differs")
    for field in (
        "license_file_sha256",
        "model_parameters_sha256",
        "model_weights_sha256",
        "wheel_sha256",
    ):
        _digest(algorithm[field], field)
    acceptance = _mapping(value["acceptance"], "vhh acceptance")
    expected_acceptance = {
        "chain_type": "H",
        "expected_construct_prefix": "MKYLLPTAAAGLLLLAAQPAMA",
        "expected_construct_suffix": "HHHHHH",
        "expected_query_start_zero_based_inclusive": 22,
        "numbered_residues_must_equal_construct_slice": True,
        "scheme": "imgt",
        "suffix_must_begin_after_query_end_inclusive": True,
        "tool_error_must_be_null": True,
    }
    if acceptance != expected_acceptance:
        raise AvidaAnnotationProtocolError("VHH acceptance policy differs")
    coordinates = _mapping(value["coordinate_semantics"], "VHH coordinates")
    expected_coordinates = {
        "domain_python_slice": "construct_sequence[query_start:query_end+1]",
        "query_end": "zero_based_inclusive",
        "query_start": "zero_based_inclusive",
    }
    if coordinates != expected_coordinates:
        raise AvidaAnnotationProtocolError("VHH coordinate semantics differ")
    cdrs = _mapping(value["imgt_cdr_ranges_one_based_inclusive"], "IMGT CDRs")
    if cdrs != {"cdr1": [27, 38], "cdr2": [56, 65], "cdr3": [105, 117]}:
        raise AvidaAnnotationProtocolError("IMGT CDR ranges differ")


def _validate_antigen(value: Mapping[str, Any]) -> None:
    _exact_fields(value, {"authority", "construct", "variant_policy"}, "antigen_boundary")
    authority = _mapping(value["authority"], "antigen authority")
    _exact_fields(
        authority,
        {
            "accession",
            "entry_version",
            "json_sha256",
            "json_url",
            "last_annotation_update_date",
            "sequence_length",
            "sequence_version",
        },
        "antigen authority",
    )
    expected_authority = {
        "accession": "P05231",
        "entry_version": 258,
        "json_url": "https://rest.uniprot.org/uniprotkb/P05231.json",
        "last_annotation_update_date": "2026-06-10",
        "sequence_length": 212,
        "sequence_version": 1,
    }
    for field, expected in expected_authority.items():
        if authority[field] != expected:
            raise AvidaAnnotationProtocolError(f"antigen authority field differs: {field}")
    _digest(authority["json_sha256"], "UniProt JSON")
    construct = _mapping(value["construct"], "antigen construct")
    expected_construct = {
        "expected_construct_length": 218,
        "expected_c_terminal_tag": "HHHHHH",
        "mature_chain_end_one_based_inclusive": 212,
        "mature_chain_feature_id": "PRO_0000015582",
        "mature_chain_start_one_based_inclusive": 30,
        "signal_end_one_based_inclusive": 29,
        "signal_start_one_based_inclusive": 1,
    }
    if construct != expected_construct:
        raise AvidaAnnotationProtocolError("antigen construct boundary differs")
    variants = _mapping(value["variant_policy"], "antigen variants")
    expected_variants = {
        "allowed_mutant_count_per_construct": 1,
        "allowed_mutant_residue": "A",
        "expected_mutant_constructs": 30,
        "expected_total_constructs": 31,
        "expected_wild_type_constructs": 1,
        "mutation_coordinates": "uniprot_precursor_one_based",
    }
    if variants != expected_variants:
        raise AvidaAnnotationProtocolError("antigen variant policy differs")


def _validate_admission(value: Mapping[str, Any]) -> None:
    expected = {
        "accepted_antigen_status": "mature_chain_verified_single_variant",
        "accepted_vhh_status": "imgt_heavy_domain_verified",
        "failure_action": "quarantine_with_explicit_reason_no_fallback",
        "registry_admission_before_independent_audit": False,
    }
    if value != expected:
        raise AvidaAnnotationProtocolError("annotation admission policy differs")


def _validate_references(value: Any) -> None:
    if type(value) is not list or len(value) != 5:
        raise AvidaAnnotationProtocolError("annotation references differ")
    roles = []
    for item in value:
        reference = _mapping(item, "reference")
        _exact_fields(reference, {"role", "url"}, "reference")
        if type(reference["role"]) is not str or not reference["role"]:
            raise AvidaAnnotationProtocolError("reference role is invalid")
        if type(reference["url"]) is not str or not reference["url"].startswith("https://"):
            raise AvidaAnnotationProtocolError("reference URL must use HTTPS")
        roles.append(reference["role"])
    if roles != sorted(set(roles)):
        raise AvidaAnnotationProtocolError("reference roles must be sorted and unique")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise AvidaAnnotationProtocolError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise AvidaAnnotationProtocolError(
            f"{label} fields differ; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _digest(value: Any, label: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise AvidaAnnotationProtocolError(f"{label} SHA-256 is invalid")


def _timestamp(value: Any) -> None:
    if type(value) is not str:
        raise AvidaAnnotationProtocolError("frozen_at_utc must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AvidaAnnotationProtocolError("frozen_at_utc is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AvidaAnnotationProtocolError("frozen_at_utc must include a UTC offset")
