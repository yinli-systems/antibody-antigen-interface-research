"""Load and validate the frozen AVIDa Tier-1 normalization protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


FORMAT = "pairbench_avida_tier1_admission_protocol_v1"
SOURCE_ID = "avida_hil6_2024_11"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_FIELDS = {
    "admission",
    "canonical_outputs",
    "format_version",
    "input_bindings",
    "license_route",
    "prohibitions",
    "source_id",
    "status",
}
_PROHIBITIONS = (
    "admit_tier2_without_new_protocol",
    "convert_unknown_cross_pair_to_nonbinder",
    "drop_quarantine_accounting",
    "infer_failed_vhh_boundaries",
    "merge_construct_and_domain_identity",
    "use_avida_for_commercial_training",
)


class AvidaNormalizationProtocolError(ValueError):
    """Raised when the AVIDa Tier-1 normalization protocol differs."""


def load_avida_normalization_protocol(
    path: Path,
) -> tuple[Mapping[str, Any], str]:
    """Load one non-symlink protocol and validate every frozen semantic field."""

    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise AvidaNormalizationProtocolError("protocol must be a regular file")
    raw = path.read_bytes()
    payload = _strict_json(raw)
    if set(payload) != _TOP_FIELDS:
        raise AvidaNormalizationProtocolError("protocol fields differ")
    if (
        payload["format_version"] != FORMAT
        or payload["source_id"] != SOURCE_ID
        or payload["status"] != "frozen_tier1_only"
        or payload["license_route"] != "noncommercial_only"
    ):
        raise AvidaNormalizationProtocolError("protocol identity differs")
    prohibitions = _string_list(payload["prohibitions"], "prohibitions")
    if tuple(prohibitions) != _PROHIBITIONS:
        raise AvidaNormalizationProtocolError("prohibitions differ")
    bindings = _object(payload["input_bindings"], "input_bindings")
    expected_bindings = {
        "antigen_annotation_receipt_sha256",
        "antigen_audit_summary_sha256",
        "antigen_independent_audit_sha256",
        "annotation_protocol_sha256",
        "candidate_receipt_sha256",
        "pairbench_contract_sha256",
        "vhh_annotation_receipt_sha256",
        "vhh_audit_summary_sha256",
        "vhh_independent_replay_sha256",
    }
    if set(bindings) != expected_bindings or any(
        type(value) is not str or _SHA256.fullmatch(value) is None
        for value in bindings.values()
    ):
        raise AvidaNormalizationProtocolError("input bindings differ")
    admission = _object(payload["admission"], "admission")
    if set(admission) != {"antigen", "pair", "vhh"}:
        raise AvidaNormalizationProtocolError("admission fields differ")
    _validate_entity_admission(
        admission["antigen"],
        name="antigen",
        status="mature_chain_verified_single_variant",
        identity="mature_chain_sequence_digest",
        accepted=31,
        quarantined=0,
    )
    vhh = _object(admission["vhh"], "vhh admission")
    _validate_entity_admission(
        vhh,
        name="vhh",
        status="imgt_heavy_domain_verified",
        identity="domain_sequence_digest",
        accepted=37_200,
        quarantined=1_399,
        extra={"tier2_policy"},
    )
    if vhh["tier2_policy"] != (
        "not_admitted_future_separately_frozen_ablation_only"
    ):
        raise AvidaNormalizationProtocolError("Tier-2 policy differs")
    pair = _object(admission["pair"], "pair admission")
    expected_pair = {
        "admit_only_if_both_normalized_entities_are_admitted": True,
        "expected_admitted_binders": 20_275,
        "expected_admitted_nonbinders": 535_666,
        "expected_admitted_rows": 555_941,
        "expected_quarantined_binders": 705,
        "expected_quarantined_nonbinders": 17_245,
        "expected_quarantined_rows": 17_950,
        "preserve_measured_binary_target": True,
        "unknown_cross_pairs_created": 0,
    }
    if pair != expected_pair:
        raise AvidaNormalizationProtocolError("pair admission differs")
    outputs = _object(payload["canonical_outputs"], "canonical_outputs")
    if outputs != {
        "admitted_tables": [
            "assays.parquet",
            "duplicate_groups.parquet",
            "entities_antibody.parquet",
            "entities_antigen.parquet",
            "pairs.parquet",
        ],
        "entity_rekey_table": "entity_rekey.parquet",
        "quarantine_tables": [
            "quarantined_antibodies.parquet",
            "quarantined_pairs.parquet",
        ],
        "raw_construct_provenance_preserved": True,
    }:
        raise AvidaNormalizationProtocolError("canonical outputs differ")
    return payload, hashlib.sha256(raw).hexdigest()


def _validate_entity_admission(
    value: Any,
    *,
    name: str,
    status: str,
    identity: str,
    accepted: int,
    quarantined: int,
    extra: set[str] | None = None,
) -> None:
    entity = _object(value, f"{name} admission")
    expected_fields = {
        "accepted_annotation_status",
        "content_identity_field",
        "expected_accepted",
        "expected_quarantined",
        "require_empty_quarantine_reasons",
    } | (extra or set())
    if set(entity) != expected_fields:
        raise AvidaNormalizationProtocolError(f"{name} admission fields differ")
    expected = {
        "accepted_annotation_status": status,
        "content_identity_field": identity,
        "expected_accepted": accepted,
        "expected_quarantined": quarantined,
        "require_empty_quarantine_reasons": True,
    }
    if any(entity[key] != value for key, value in expected.items()):
        raise AvidaNormalizationProtocolError(f"{name} admission differs")


def _strict_json(raw: bytes) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise AvidaNormalizationProtocolError(f"non-finite value: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AvidaNormalizationProtocolError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AvidaNormalizationProtocolError("cannot parse protocol") from exc
    if type(value) is not dict:
        raise AvidaNormalizationProtocolError("protocol must be an object")
    return value


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise AvidaNormalizationProtocolError(f"{name} must be an object")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise AvidaNormalizationProtocolError(f"{name} must be a string list")
    if value != sorted(set(value)):
        raise AvidaNormalizationProtocolError(f"{name} must be sorted and unique")
    return value


__all__ = [
    "AvidaNormalizationProtocolError",
    "load_avida_normalization_protocol",
]
