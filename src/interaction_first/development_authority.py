"""Fail-closed authority for train-only nonconfirmatory development.

This authority is deliberately separate from confirmatory Gate 0. It permits a
small set of debugging and systems purposes on an exact authority-train subset,
but it cannot authorize model selection, held-out evaluation, test access, or
promotion into the confirmatory evidence chain.

Digests are provenance bindings rather than signatures. Launchers must receive
the expected policy and decision digests from an immutable control plane that
is outside the worker output directory.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PERMISSION_NAME = "development_only_nonconfirmatory"
POLICY_SCHEMA_VERSION = "pairscale_development_policy_v1"
REQUEST_SCHEMA_VERSION = "pairscale_development_request_v1"
DECISION_SCHEMA_VERSION = "pairscale_development_decision_v1"
CLAIM_BOUNDARY = (
    "Authority-train-only development evidence for debugging and systems "
    "engineering; no validation, test, model-selection, confirmatory, "
    "biological generalization, SOTA, or world-leading claim."
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._:@+-]{0,254}[A-Za-z0-9])?$"
)
_UTC_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_FORBIDDEN_PERMISSIONS = frozenset(
    {
        "checkpoint_promotion",
        "confirmatory_claim",
        "contact_training",
        "external_test_inference",
        "model_selection",
        "outer_evaluation",
        "paper_result_claim",
        "sota_claim",
    }
)


class DevelopmentAuthorityError(ValueError):
    """Raised when a development authority artifact fails closed."""


class DevelopmentPurpose(str, Enum):
    """The complete set of nonconfirmatory development purposes."""

    OVERFIT_SANITY = "overfit_sanity"
    ARCHITECTURE_ABLATION = "architecture_ablation"
    LOSS_DEBUGGING = "loss_debugging"
    SYSTEMS_PROFILING = "systems_profiling"
    BASELINE_DEVELOPMENT = "baseline_development"


@dataclass(frozen=True)
class AuthorityCounts:
    """Frozen inventory counts for the authority-train universe."""

    source_instances: int
    exact_pairs: int
    antibody_entities: int
    antigen_entities: int

    def __post_init__(self) -> None:
        for name in (
            "source_instances",
            "exact_pairs",
            "antibody_entities",
            "antigen_entities",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DevelopmentAuthorityError(f"{name} must be a positive integer")

    def canonical_payload(self) -> Mapping[str, int]:
        """Return the canonical count representation."""

        return {
            "source_instances": self.source_instances,
            "exact_pairs": self.exact_pairs,
            "antibody_entities": self.antibody_entities,
            "antigen_entities": self.antigen_entities,
        }


@dataclass(frozen=True)
class DevelopmentPolicy:
    """Externally anchored policy for one train-only development authority."""

    protocol_sha256: str
    authority_manifest_sha256: str
    cache_authorization_manifest_sha256: str
    cache_manifest_sha256: str
    authority_membership_sha256: str
    counts: AuthorityCounts
    allowed_purposes: tuple[DevelopmentPurpose, ...] = tuple(
        sorted(DevelopmentPurpose, key=lambda value: value.value)
    )
    prohibited_permissions: tuple[str, ...] = tuple(sorted(_FORBIDDEN_PERMISSIONS))

    def __post_init__(self) -> None:
        for name in (
            "protocol_sha256",
            "authority_manifest_sha256",
            "cache_authorization_manifest_sha256",
            "cache_manifest_sha256",
            "authority_membership_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if not isinstance(self.counts, AuthorityCounts):
            raise DevelopmentAuthorityError("counts must be AuthorityCounts")
        if not isinstance(self.allowed_purposes, tuple) or set(
            self.allowed_purposes
        ) != set(DevelopmentPurpose):
            raise DevelopmentAuthorityError(
                "allowed_purposes must equal the complete locked development set"
            )
        if len(set(self.allowed_purposes)) != len(self.allowed_purposes):
            raise DevelopmentAuthorityError("allowed_purposes contains duplicates")
        if not all(
            isinstance(value, DevelopmentPurpose) for value in self.allowed_purposes
        ):
            raise DevelopmentAuthorityError(
                "allowed_purposes must contain only DevelopmentPurpose values"
            )
        if not isinstance(self.prohibited_permissions, tuple) or set(
            self.prohibited_permissions
        ) != _FORBIDDEN_PERMISSIONS:
            raise DevelopmentAuthorityError(
                "prohibited_permissions must equal the complete locked set"
            )
        if len(set(self.prohibited_permissions)) != len(
            self.prohibited_permissions
        ):
            raise DevelopmentAuthorityError(
                "prohibited_permissions contains duplicates"
            )
        object.__setattr__(
            self,
            "allowed_purposes",
            tuple(sorted(self.allowed_purposes, key=lambda value: value.value)),
        )
        object.__setattr__(
            self,
            "prohibited_permissions",
            tuple(sorted(self.prohibited_permissions)),
        )

    def canonical_payload(self) -> Mapping[str, object]:
        """Return the canonical policy representation."""

        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "permission_name": PERMISSION_NAME,
            "protocol_sha256": self.protocol_sha256,
            "data_boundary": {
                "source_split": "train",
                "authority_manifest_sha256": self.authority_manifest_sha256,
                "cache_authorization_manifest_sha256": (
                    self.cache_authorization_manifest_sha256
                ),
                "cache_manifest_sha256": self.cache_manifest_sha256,
                "authority_membership_sha256": (
                    self.authority_membership_sha256
                ),
                "counts": self.counts.canonical_payload(),
            },
            "allowed_purposes": sorted(value.value for value in self.allowed_purposes),
            "prohibited_permissions": sorted(self.prohibited_permissions),
            "sealed_data_access_authorized": False,
            "validation_data_access_authorized": False,
            "test_data_access_authorized": False,
            "checkpoint_promotion_authorized": False,
            "confirmatory_claim_authorized": False,
            "claim_boundary": CLAIM_BOUNDARY,
        }

    @property
    def policy_sha256(self) -> str:
        """Return the digest of the canonical policy."""

        return canonical_sha256(self.canonical_payload())


@dataclass(frozen=True)
class DevelopmentRequest:
    """One exact request for train-only development authority."""

    run_id: str
    purpose: DevelopmentPurpose
    subset_instance_ids: tuple[str, ...]
    subset_manifest_sha256: str
    requested_at_utc: str
    source_split: str = "train"
    sealed_data_accessed: bool = False
    validation_data_accessed: bool = False
    test_data_accessed: bool = False
    model_selection_requested: bool = False
    checkpoint_promotion_requested: bool = False
    confirmatory_claim_requested: bool = False

    def __post_init__(self) -> None:
        _validate_identifier(self.run_id, "run_id")
        if not isinstance(self.purpose, DevelopmentPurpose):
            raise DevelopmentAuthorityError(
                "purpose must be a recognized DevelopmentPurpose"
            )
        if not isinstance(self.subset_instance_ids, tuple):
            raise DevelopmentAuthorityError(
                "subset_instance_ids must be an immutable tuple"
            )
        if not self.subset_instance_ids:
            raise DevelopmentAuthorityError("subset_instance_ids must be nonempty")
        for value in self.subset_instance_ids:
            _validate_identifier(value, "subset instance identifier")
        if len(set(self.subset_instance_ids)) != len(self.subset_instance_ids):
            raise DevelopmentAuthorityError(
                "subset_instance_ids contains duplicates"
            )
        _validate_sha256(self.subset_manifest_sha256, "subset_manifest_sha256")
        _validate_utc_timestamp(self.requested_at_utc)
        if not isinstance(self.source_split, str) or not self.source_split.isascii():
            raise DevelopmentAuthorityError("source_split must be ASCII text")
        for name in (
            "sealed_data_accessed",
            "validation_data_accessed",
            "test_data_accessed",
            "model_selection_requested",
            "checkpoint_promotion_requested",
            "confirmatory_claim_requested",
        ):
            if not isinstance(getattr(self, name), bool):
                raise DevelopmentAuthorityError(f"{name} must be an explicit boolean")

    def canonical_payload(self) -> Mapping[str, object]:
        """Return the canonical request representation."""

        return {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "purpose": self.purpose.value,
            "source_split": self.source_split,
            "subset_instance_ids": sorted(self.subset_instance_ids),
            "subset_membership_sha256": membership_sha256(
                self.subset_instance_ids
            ),
            "subset_manifest_sha256": self.subset_manifest_sha256,
            "requested_at_utc": self.requested_at_utc,
            "sealed_data_accessed": self.sealed_data_accessed,
            "validation_data_accessed": self.validation_data_accessed,
            "test_data_accessed": self.test_data_accessed,
            "model_selection_requested": self.model_selection_requested,
            "checkpoint_promotion_requested": self.checkpoint_promotion_requested,
            "confirmatory_claim_requested": self.confirmatory_claim_requested,
        }

    @property
    def request_sha256(self) -> str:
        """Return the digest of the canonical request."""

        return canonical_sha256(self.canonical_payload())


@dataclass(frozen=True)
class DevelopmentDecision:
    """Deterministic authorization result for one development request."""

    policy_sha256: str
    request_sha256: str
    run_id: str
    purpose: DevelopmentPurpose
    subset_manifest_sha256: str
    subset_membership_sha256: str
    subset_instance_count: int
    evaluated_at_utc: str
    authorized: bool
    blocking_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "policy_sha256",
            "request_sha256",
            "subset_manifest_sha256",
            "subset_membership_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        _validate_identifier(self.run_id, "run_id")
        if not isinstance(self.purpose, DevelopmentPurpose):
            raise DevelopmentAuthorityError("decision purpose is invalid")
        if (
            isinstance(self.subset_instance_count, bool)
            or not isinstance(self.subset_instance_count, int)
            or self.subset_instance_count <= 0
        ):
            raise DevelopmentAuthorityError(
                "subset_instance_count must be a positive integer"
            )
        _validate_utc_timestamp(self.evaluated_at_utc)
        if not isinstance(self.authorized, bool):
            raise DevelopmentAuthorityError("authorized must be an explicit boolean")
        if not isinstance(self.blocking_reasons, tuple):
            raise DevelopmentAuthorityError(
                "blocking_reasons must be an immutable tuple"
            )
        if tuple(sorted(set(self.blocking_reasons))) != self.blocking_reasons:
            raise DevelopmentAuthorityError(
                "blocking_reasons must be sorted and unique"
            )
        for reason in self.blocking_reasons:
            _validate_reason(reason)
        if self.authorized == bool(self.blocking_reasons):
            raise DevelopmentAuthorityError(
                "authorized must be true exactly when blocking_reasons is empty"
            )

    def canonical_payload(self) -> Mapping[str, object]:
        """Return the canonical decision representation."""

        return {
            "schema_version": DECISION_SCHEMA_VERSION,
            "permission_name": PERMISSION_NAME,
            "policy_sha256": self.policy_sha256,
            "request_sha256": self.request_sha256,
            "run_id": self.run_id,
            "purpose": self.purpose.value,
            "subset_manifest_sha256": self.subset_manifest_sha256,
            "subset_membership_sha256": self.subset_membership_sha256,
            "subset_instance_count": self.subset_instance_count,
            "evaluated_at_utc": self.evaluated_at_utc,
            "authorized": self.authorized,
            "blocking_reasons": list(self.blocking_reasons),
            "sealed_data_access_authorized": False,
            "validation_data_access_authorized": False,
            "test_data_access_authorized": False,
            "model_selection_authorized": False,
            "checkpoint_promotion_authorized": False,
            "confirmatory_claim_authorized": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "implementation_source_sha256": implementation_source_sha256(),
        }

    @property
    def decision_sha256(self) -> str:
        """Return the digest of the canonical decision."""

        return canonical_sha256(self.canonical_payload())


def decide_development_authority(
    *,
    policy: DevelopmentPolicy,
    request: DevelopmentRequest,
    authority_instance_ids: Iterable[str],
    expected_policy_sha256: str,
    evaluated_at_utc: str,
) -> DevelopmentDecision:
    """Evaluate one request against exact authority-train membership."""

    _validate_sha256(expected_policy_sha256, "expected_policy_sha256")
    _validate_utc_timestamp(evaluated_at_utc)
    authority = tuple(authority_instance_ids)
    for value in authority:
        _validate_identifier(value, "authority instance identifier")
    if len(set(authority)) != len(authority):
        raise DevelopmentAuthorityError("authority_instance_ids contains duplicates")

    reasons: list[str] = []
    if policy.policy_sha256 != expected_policy_sha256:
        reasons.append("policy_digest_mismatch")
    if len(authority) != policy.counts.exact_pairs:
        reasons.append("authority_count_mismatch")
    if membership_sha256(authority) != policy.authority_membership_sha256:
        reasons.append("authority_membership_digest_mismatch")
    if request.purpose not in policy.allowed_purposes:
        reasons.append("purpose_not_authorized")
    if request.source_split != "train":
        reasons.append("source_split_not_train")
    for name in (
        "sealed_data_accessed",
        "validation_data_accessed",
        "test_data_accessed",
        "model_selection_requested",
        "checkpoint_promotion_requested",
        "confirmatory_claim_requested",
    ):
        if getattr(request, name):
            reasons.append(name)
    authority_set = set(authority)
    if not set(request.subset_instance_ids).issubset(authority_set):
        reasons.append("subset_contains_unknown_instance")

    unique_reasons = tuple(sorted(set(reasons)))
    return DevelopmentDecision(
        policy_sha256=policy.policy_sha256,
        request_sha256=request.request_sha256,
        run_id=request.run_id,
        purpose=request.purpose,
        subset_manifest_sha256=request.subset_manifest_sha256,
        subset_membership_sha256=membership_sha256(request.subset_instance_ids),
        subset_instance_count=len(request.subset_instance_ids),
        evaluated_at_utc=evaluated_at_utc,
        authorized=not unique_reasons,
        blocking_reasons=unique_reasons,
    )


def verify_development_decision(
    decision: DevelopmentDecision,
    *,
    expected_policy_sha256: str,
    expected_decision_sha256: str,
    expected_run_id: str,
    expected_purpose: DevelopmentPurpose,
    expected_subset_manifest_sha256: str,
    required_permission: str = PERMISSION_NAME,
) -> None:
    """Verify external anchors and the exact nonconfirmatory permission."""

    _validate_sha256(expected_policy_sha256, "expected_policy_sha256")
    _validate_sha256(expected_decision_sha256, "expected_decision_sha256")
    _validate_identifier(expected_run_id, "expected_run_id")
    _validate_sha256(
        expected_subset_manifest_sha256,
        "expected_subset_manifest_sha256",
    )
    if required_permission != PERMISSION_NAME:
        raise DevelopmentAuthorityError(
            "development authority cannot grant the requested permission"
        )
    if decision.policy_sha256 != expected_policy_sha256:
        raise DevelopmentAuthorityError("decision policy digest mismatch")
    if decision.decision_sha256 != expected_decision_sha256:
        raise DevelopmentAuthorityError("decision digest mismatch")
    if decision.run_id != expected_run_id:
        raise DevelopmentAuthorityError("decision run_id mismatch")
    if decision.purpose != expected_purpose:
        raise DevelopmentAuthorityError("decision purpose mismatch")
    if decision.subset_manifest_sha256 != expected_subset_manifest_sha256:
        raise DevelopmentAuthorityError("decision subset manifest digest mismatch")
    if not decision.authorized:
        raise DevelopmentAuthorityError(
            "development request is blocked: " + ",".join(decision.blocking_reasons)
        )


def development_policy_from_payload(payload: Mapping[str, Any]) -> DevelopmentPolicy:
    """Parse a policy mapping using an exact fail-closed schema."""

    expected_top = {
        "schema_version",
        "permission_name",
        "protocol_sha256",
        "data_boundary",
        "allowed_purposes",
        "prohibited_permissions",
        "sealed_data_access_authorized",
        "validation_data_access_authorized",
        "test_data_access_authorized",
        "checkpoint_promotion_authorized",
        "confirmatory_claim_authorized",
        "claim_boundary",
    }
    _require_exact_keys(payload, expected_top, "development policy")
    if payload["schema_version"] != POLICY_SCHEMA_VERSION:
        raise DevelopmentAuthorityError("unsupported development policy schema")
    if payload["permission_name"] != PERMISSION_NAME:
        raise DevelopmentAuthorityError("development permission name is not locked")
    for name in (
        "sealed_data_access_authorized",
        "validation_data_access_authorized",
        "test_data_access_authorized",
        "checkpoint_promotion_authorized",
        "confirmatory_claim_authorized",
    ):
        if payload[name] is not False:
            raise DevelopmentAuthorityError(f"{name} must be exactly false")
    if payload["claim_boundary"] != CLAIM_BOUNDARY:
        raise DevelopmentAuthorityError("development claim boundary is not locked")

    boundary = _require_mapping(payload["data_boundary"], "data_boundary")
    _require_exact_keys(
        boundary,
        {
            "source_split",
            "authority_manifest_sha256",
            "cache_authorization_manifest_sha256",
            "cache_manifest_sha256",
            "authority_membership_sha256",
            "counts",
        },
        "data_boundary",
    )
    if boundary["source_split"] != "train":
        raise DevelopmentAuthorityError("policy source_split must be exactly train")
    counts = _require_mapping(boundary["counts"], "counts")
    _require_exact_keys(
        counts,
        {
            "source_instances",
            "exact_pairs",
            "antibody_entities",
            "antigen_entities",
        },
        "counts",
    )
    purposes = _require_string_list(payload["allowed_purposes"], "allowed_purposes")
    prohibited = _require_string_list(
        payload["prohibited_permissions"],
        "prohibited_permissions",
    )
    try:
        typed_purposes = tuple(DevelopmentPurpose(value) for value in purposes)
    except ValueError as error:
        raise DevelopmentAuthorityError("unknown development purpose") from error
    return DevelopmentPolicy(
        protocol_sha256=payload["protocol_sha256"],
        authority_manifest_sha256=boundary["authority_manifest_sha256"],
        cache_authorization_manifest_sha256=(
            boundary["cache_authorization_manifest_sha256"]
        ),
        cache_manifest_sha256=boundary["cache_manifest_sha256"],
        authority_membership_sha256=boundary["authority_membership_sha256"],
        counts=AuthorityCounts(
            source_instances=counts["source_instances"],
            exact_pairs=counts["exact_pairs"],
            antibody_entities=counts["antibody_entities"],
            antigen_entities=counts["antigen_entities"],
        ),
        allowed_purposes=typed_purposes,
        prohibited_permissions=tuple(prohibited),
    )


def development_policy_from_json(payload_bytes: bytes) -> DevelopmentPolicy:
    """Parse strict ASCII JSON and reject duplicate keys or non-finite values."""

    payload = _strict_json_object(payload_bytes, "development policy")
    return development_policy_from_payload(payload)


def canonical_json_bytes(payload: object) -> bytes:
    """Encode one deterministic ASCII JSON document."""

    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def canonical_sha256(payload: object) -> str:
    """Hash one canonical JSON payload."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def membership_sha256(values: Iterable[str]) -> str:
    """Hash sorted unique identifiers using unambiguous length framing."""

    canonical = sorted(values)
    if len(set(canonical)) != len(canonical):
        raise DevelopmentAuthorityError("membership input contains duplicates")
    digest = hashlib.sha256()
    for value in canonical:
        _validate_identifier(value, "membership identifier")
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def implementation_source_sha256() -> str:
    """Hash the exact authority implementation bytes."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DevelopmentAuthorityError(
            f"{label} must be a lowercase 64-character SHA-256"
        )
    return value


def _validate_identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or _IDENTIFIER_RE.fullmatch(value) is None
    ):
        raise DevelopmentAuthorityError(f"{label} is not a canonical identifier")
    return value


def _validate_reason(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or any(character.isspace() for character in value)
    ):
        raise DevelopmentAuthorityError("blocking reason is not canonical ASCII")
    return value


def _validate_utc_timestamp(value: Any) -> str:
    if not isinstance(value, str) or _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise DevelopmentAuthorityError(
            "timestamp must use second-precision UTC YYYY-MM-DDTHH:MM:SSZ"
        )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise DevelopmentAuthorityError("timestamp is not a valid UTC time") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise DevelopmentAuthorityError("timestamp is not canonically formatted")
    return value


def _strict_json_object(payload_bytes: bytes, label: str) -> dict[str, Any]:
    if not isinstance(payload_bytes, bytes) or not payload_bytes:
        raise DevelopmentAuthorityError(f"{label} must be nonempty bytes")
    if payload_bytes.startswith(b"\xef\xbb\xbf"):
        raise DevelopmentAuthorityError(f"{label} must not contain a BOM")
    try:
        text = payload_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise DevelopmentAuthorityError(f"{label} must be ASCII JSON") from error

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DevelopmentAuthorityError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DevelopmentAuthorityError(
                    f"{label} contains non-finite JSON constant {token}"
                )
            ),
        )
    except json.JSONDecodeError as error:
        raise DevelopmentAuthorityError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise DevelopmentAuthorityError(f"{label} must be a JSON object")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DevelopmentAuthorityError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise DevelopmentAuthorityError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DevelopmentAuthorityError(f"{label} must be a JSON string array")
    return value
