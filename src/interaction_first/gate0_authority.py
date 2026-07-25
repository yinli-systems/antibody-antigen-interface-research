"""Fail-closed Gate 0 authorization for prospective PAIRFormer runs.

This module does not inspect scientific data. It combines independently
produced, content-addressed receipts into a deterministic authorization
decision that training launchers can verify before reading any dataset.

Authority identifiers and content digests are provenance bindings, not digital
signatures and not proof of legal or organizational independence. A launcher
must obtain the expected policy and decision digests from a trusted,
out-of-band immutable control plane. This module cannot make a self-authored
policy trustworthy.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Mapping, Sequence


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._:@/-]{0,126}[a-z0-9])?$")
_LOGICAL_NAME_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9._:/-]{0,126}[a-z0-9])?$"
)
_UTC_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_PERMISSIONS = frozenset(
    {
        "contact_training",
        "model_selection",
        "outer_evaluation",
        "external_test_inference",
    }
)
_POST_GATE0_PERMISSIONS = frozenset(
    {"outer_evaluation", "external_test_inference"}
)


class Gate0Error(ValueError):
    """Raised when a Gate 0 contract or receipt is malformed."""


class GateName(str, Enum):
    """Required evidence classes for scientific-training authorization."""

    DATA_LICENSE_PROVENANCE = "data_license_provenance"
    JOINT_NESTED_SPLIT = "joint_nested_split"
    INDEPENDENT_LEAKAGE_AUDIT = "independent_leakage_audit"
    FOLD_CACHE_EQUALITY = "fold_cache_equality"
    RUNTIME_LOCK = "runtime_lock"
    IMMUTABLE_MANIFESTS = "immutable_manifests"


_REQUIRED_INDEPENDENT_AUDIT_GATES = frozenset(
    {
        GateName.JOINT_NESTED_SPLIT,
        GateName.INDEPENDENT_LEAKAGE_AUDIT,
        GateName.FOLD_CACHE_EQUALITY,
    }
)


class ReceiptStatus(str, Enum):
    """A receipt's explicit decision state."""

    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"


@dataclass(frozen=True, order=True)
class DigestBinding:
    """Bind a logical artifact name to its SHA-256 digest."""

    name: str
    sha256: str

    def __post_init__(self) -> None:
        _validate_logical_name(self.name, "digest binding name")
        _validate_sha256(self.sha256, f"digest binding {self.name!r}")


@dataclass(frozen=True)
class GateReceipt:
    """One producer-issued and, where required, independently audited receipt."""

    gate: GateName
    status: ReceiptStatus
    producer_id: str
    auditor_id: str | None
    protocol_sha256: str
    artifacts: tuple[DigestBinding, ...]
    assertion: str
    issued_at_utc: str
    sealed_or_test_data_accessed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.gate, GateName):
            raise Gate0Error("receipt gate must be a GateName")
        if not isinstance(self.status, ReceiptStatus):
            raise Gate0Error("receipt status must be a ReceiptStatus")
        _validate_identity(self.producer_id, "producer_id")
        if self.auditor_id is not None:
            _validate_identity(self.auditor_id, "auditor_id")
        _validate_sha256(self.protocol_sha256, "receipt protocol_sha256")
        _validate_artifact_tuple(self.artifacts)
        _validate_assertion(self.assertion)
        _validate_utc_timestamp(self.issued_at_utc)
        if not isinstance(self.sealed_or_test_data_accessed, bool):
            raise Gate0Error(
                "sealed_or_test_data_accessed must be an explicit boolean"
            )

    def canonical_payload(self) -> Mapping[str, object]:
        """Return the canonical, hashable representation of this receipt."""

        return {
            "schema_version": "pairformer_gate0_receipt_v1",
            "gate": self.gate.value,
            "status": self.status.value,
            "producer_id": self.producer_id,
            "auditor_id": self.auditor_id,
            "protocol_sha256": self.protocol_sha256,
            "artifacts": [
                {"name": item.name, "sha256": item.sha256}
                for item in sorted(self.artifacts)
            ],
            "assertion": self.assertion,
            "issued_at_utc": self.issued_at_utc,
            "sealed_or_test_data_accessed": self.sealed_or_test_data_accessed,
        }

    @property
    def receipt_sha256(self) -> str:
        """Return the SHA-256 digest of the canonical receipt."""

        return _canonical_sha256(self.canonical_payload())


@dataclass(frozen=True, order=True)
class ExpectedArtifact:
    """An artifact digest required from a particular gate."""

    gate: GateName
    name: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.gate, GateName):
            raise Gate0Error("expected artifact gate must be a GateName")
        _validate_logical_name(self.name, "expected artifact name")
        _validate_sha256(self.sha256, f"expected artifact {self.name!r}")


@dataclass(frozen=True, order=True)
class ExpectedReceipt:
    """Externally freeze the complete receipt expected for one gate."""

    gate: GateName
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.gate, GateName):
            raise Gate0Error("expected receipt gate must be a GateName")
        _validate_sha256(self.sha256, f"expected receipt for {self.gate.value}")


@dataclass(frozen=True)
class Gate0Policy:
    """Frozen policy that defines the complete authorization condition."""

    protocol_sha256: str
    required_gates: tuple[GateName, ...] = tuple(GateName)
    independent_audit_gates: tuple[GateName, ...] = (
        GateName.JOINT_NESTED_SPLIT,
        GateName.INDEPENDENT_LEAKAGE_AUDIT,
        GateName.FOLD_CACHE_EQUALITY,
    )
    expected_artifacts: tuple[ExpectedArtifact, ...] = ()
    expected_receipts: tuple[ExpectedReceipt, ...] = ()
    authorized_permissions: tuple[str, ...] = (
        "contact_training",
        "model_selection",
    )

    def __post_init__(self) -> None:
        _validate_sha256(self.protocol_sha256, "policy protocol_sha256")
        for field, value in (
            ("required_gates", self.required_gates),
            ("independent_audit_gates", self.independent_audit_gates),
            ("expected_artifacts", self.expected_artifacts),
            ("expected_receipts", self.expected_receipts),
            ("authorized_permissions", self.authorized_permissions),
        ):
            if not isinstance(value, tuple):
                raise Gate0Error(f"{field} must be an immutable tuple")
        if not all(isinstance(gate, GateName) for gate in self.required_gates):
            raise Gate0Error("required_gates must contain only GateName values")
        if not all(
            isinstance(gate, GateName) for gate in self.independent_audit_gates
        ):
            raise Gate0Error(
                "independent_audit_gates must contain only GateName values"
            )
        if not all(
            isinstance(item, ExpectedArtifact) for item in self.expected_artifacts
        ):
            raise Gate0Error(
                "expected_artifacts must contain only ExpectedArtifact values"
            )
        if not all(
            isinstance(item, ExpectedReceipt) for item in self.expected_receipts
        ):
            raise Gate0Error(
                "expected_receipts must contain only ExpectedReceipt values"
            )
        if not all(
            isinstance(permission, str)
            for permission in self.authorized_permissions
        ):
            raise Gate0Error("authorized_permissions must contain only strings")
        if not self.required_gates:
            raise Gate0Error("at least one Gate 0 requirement is needed")
        if len(set(self.required_gates)) != len(self.required_gates):
            raise Gate0Error("required_gates contains duplicates")
        if set(self.required_gates) != set(GateName):
            missing = sorted(
                gate.value for gate in set(GateName).difference(self.required_gates)
            )
            extra = sorted(
                str(gate) for gate in set(self.required_gates).difference(GateName)
            )
            raise Gate0Error(
                "required_gates must contain the complete locked Gate 0 set; "
                f"missing={missing}, extra={extra}"
            )
        if len(set(self.independent_audit_gates)) != len(
            self.independent_audit_gates
        ):
            raise Gate0Error("independent_audit_gates contains duplicates")
        if set(self.independent_audit_gates) != _REQUIRED_INDEPENDENT_AUDIT_GATES:
            missing = sorted(
                gate.value
                for gate in _REQUIRED_INDEPENDENT_AUDIT_GATES.difference(
                    self.independent_audit_gates
                )
            )
            extra = sorted(
                gate.value
                for gate in set(self.independent_audit_gates).difference(
                    _REQUIRED_INDEPENDENT_AUDIT_GATES
                )
            )
            raise Gate0Error(
                "independent_audit_gates must match the locked audit set; "
                f"missing={missing}, extra={extra}"
            )
        keys = [(item.gate, item.name) for item in self.expected_artifacts]
        if len(set(keys)) != len(keys):
            raise Gate0Error("expected_artifacts contains duplicate gate/name keys")
        artifact_gates = {item.gate for item in self.expected_artifacts}
        if artifact_gates != set(self.required_gates):
            missing = sorted(
                gate.value for gate in set(self.required_gates).difference(
                    artifact_gates
                )
            )
            extra = sorted(
                gate.value for gate in artifact_gates.difference(self.required_gates)
            )
            raise Gate0Error(
                "expected_artifacts must externally anchor at least one artifact "
                f"for every required gate; missing={missing}, extra={extra}"
            )
        receipt_gates = [item.gate for item in self.expected_receipts]
        if len(set(receipt_gates)) != len(receipt_gates):
            raise Gate0Error("expected_receipts contains duplicate gates")
        if set(receipt_gates) != set(self.required_gates):
            missing = sorted(
                gate.value for gate in set(self.required_gates).difference(receipt_gates)
            )
            extra = sorted(
                gate.value for gate in set(receipt_gates).difference(self.required_gates)
            )
            raise Gate0Error(
                "expected_receipts must externally anchor every required gate; "
                f"missing={missing}, extra={extra}"
            )
        if len(set(self.authorized_permissions)) != len(
            self.authorized_permissions
        ):
            raise Gate0Error("authorized_permissions contains duplicates")
        if not self.authorized_permissions:
            raise Gate0Error("Gate 0 must grant at least one scoped permission")
        unknown_permissions = set(self.authorized_permissions).difference(_PERMISSIONS)
        if unknown_permissions:
            raise Gate0Error(
                f"unknown Gate 0 permissions: {sorted(unknown_permissions)}"
            )
        unsafe_permissions = set(self.authorized_permissions).intersection(
            _POST_GATE0_PERMISSIONS
        )
        if unsafe_permissions:
            raise Gate0Error(
                "outer or external evaluation requires a later model-freeze "
                f"authority, not Gate 0: {sorted(unsafe_permissions)}"
            )

    def canonical_payload(self) -> Mapping[str, object]:
        """Return the canonical representation of the frozen policy."""

        return {
            "schema_version": "pairformer_gate0_policy_v1",
            "protocol_sha256": self.protocol_sha256,
            "required_gates": sorted(gate.value for gate in self.required_gates),
            "independent_audit_gates": sorted(
                gate.value for gate in self.independent_audit_gates
            ),
            "expected_artifacts": [
                {
                    "gate": item.gate.value,
                    "name": item.name,
                    "sha256": item.sha256,
                }
                for item in sorted(self.expected_artifacts)
            ],
            "expected_receipts": [
                {"gate": item.gate.value, "sha256": item.sha256}
                for item in sorted(self.expected_receipts)
            ],
            "authorized_permissions": sorted(self.authorized_permissions),
        }

    @property
    def policy_sha256(self) -> str:
        """Return the SHA-256 digest of the canonical policy."""

        return _canonical_sha256(self.canonical_payload())


@dataclass(frozen=True)
class Gate0Decision:
    """Deterministic, machine-verifiable Gate 0 decision."""

    schema_version: str
    policy_sha256: str
    evaluated_at_utc: str
    authorized: bool
    contact_training_authorized: bool
    model_selection_authorized: bool
    outer_evaluation_authorized: bool
    external_test_inference_authorized: bool
    receipt_sha256_by_gate: tuple[tuple[str, str], ...]
    blocking_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_sha256(self.policy_sha256, "decision policy_sha256")
        _validate_utc_timestamp(self.evaluated_at_utc)
        if not isinstance(self.receipt_sha256_by_gate, tuple):
            raise Gate0Error("decision receipt bindings must be an immutable tuple")
        if not isinstance(self.blocking_reasons, tuple):
            raise Gate0Error("decision blocking reasons must be an immutable tuple")
        if self.schema_version != "pairformer_gate0_decision_v1":
            raise Gate0Error("unsupported Gate 0 decision schema")
        for binding in self.receipt_sha256_by_gate:
            if (
                not isinstance(binding, tuple)
                or len(binding) != 2
                or not all(isinstance(item, str) for item in binding)
            ):
                raise Gate0Error(
                    "each decision receipt binding must be an immutable "
                    "(gate, SHA-256) string tuple"
                )
        receipt_gates = [name for name, _ in self.receipt_sha256_by_gate]
        if receipt_gates != sorted(receipt_gates):
            raise Gate0Error("decision receipt bindings must be sorted")
        if len(set(receipt_gates)) != len(receipt_gates):
            raise Gate0Error("decision receipt bindings contain duplicate gates")
        for gate, digest in self.receipt_sha256_by_gate:
            if gate not in {item.value for item in GateName}:
                raise Gate0Error(f"unsupported decision receipt gate {gate!r}")
            _validate_sha256(digest, f"receipt digest for {gate}")
        if not all(
            isinstance(reason, str)
            and reason
            and reason.strip() == reason
            and not _contains_control_character(reason)
            for reason in self.blocking_reasons
        ):
            raise Gate0Error(
                "decision blocking reasons must be normalized nonempty strings"
            )
        if tuple(sorted(set(self.blocking_reasons))) != self.blocking_reasons:
            raise Gate0Error("blocking reasons must be unique and sorted")
        boolean_fields = (
            ("authorized", self.authorized),
            ("contact_training_authorized", self.contact_training_authorized),
            ("model_selection_authorized", self.model_selection_authorized),
            ("outer_evaluation_authorized", self.outer_evaluation_authorized),
            (
                "external_test_inference_authorized",
                self.external_test_inference_authorized,
            ),
        )
        for name, value in boolean_fields:
            if not isinstance(value, bool):
                raise Gate0Error(f"decision {name} must be an explicit boolean")
        permissions = (
            self.contact_training_authorized,
            self.model_selection_authorized,
            self.outer_evaluation_authorized,
            self.external_test_inference_authorized,
        )
        if self.authorized and self.blocking_reasons:
            raise Gate0Error("an authorized decision cannot have blocking reasons")
        if not self.authorized and any(permissions):
            raise Gate0Error("a blocked decision cannot grant any authority")
        if self.authorized and not any(permissions):
            raise Gate0Error("a passed Gate 0 decision must grant a scoped permission")
        if self.authorized and set(receipt_gates) != {
            gate.value for gate in GateName
        }:
            raise Gate0Error(
                "an authorized decision must bind every locked Gate 0 receipt"
            )
        if self.outer_evaluation_authorized or self.external_test_inference_authorized:
            raise Gate0Error(
                "Gate 0 cannot grant outer or external evaluation authority"
            )

    def canonical_payload(self) -> Mapping[str, object]:
        """Return a canonical representation suitable for launch verification."""

        return {
            "schema_version": self.schema_version,
            "policy_sha256": self.policy_sha256,
            "evaluated_at_utc": self.evaluated_at_utc,
            "authorized": self.authorized,
            "permissions": {
                "contact_training": self.contact_training_authorized,
                "model_selection": self.model_selection_authorized,
                "outer_evaluation": self.outer_evaluation_authorized,
                "external_test_inference": self.external_test_inference_authorized,
            },
            "receipt_sha256_by_gate": [
                {"gate": gate, "sha256": digest}
                for gate, digest in self.receipt_sha256_by_gate
            ],
            "blocking_reasons": list(self.blocking_reasons),
        }

    @property
    def decision_sha256(self) -> str:
        """Return the SHA-256 digest of the complete decision."""

        return _canonical_sha256(self.canonical_payload())


def decide_gate0(
    receipts: Iterable[GateReceipt],
    *,
    policy: Gate0Policy,
    evaluated_at_utc: str,
) -> Gate0Decision:
    """Combine receipts into a fail-closed authorization decision.

    Missing, pending, failed, mismatched, non-independent, or test-exposed
    evidence blocks every downstream permission. This function never grants
    partial authority.
    """

    if not isinstance(policy, Gate0Policy):
        raise Gate0Error("policy must be a validated Gate0Policy")
    _validate_utc_timestamp(evaluated_at_utc)
    by_gate: dict[GateName, GateReceipt] = {}
    reasons: list[str] = []
    evaluated_at = _parse_utc_timestamp(evaluated_at_utc)

    for receipt in receipts:
        if not isinstance(receipt, GateReceipt):
            raise Gate0Error("receipts must contain only GateReceipt values")
        if receipt.gate in by_gate:
            raise Gate0Error(f"duplicate receipt for gate {receipt.gate.value}")
        by_gate[receipt.gate] = receipt

    expected_by_gate: dict[GateName, dict[str, str]] = {}
    for expected in policy.expected_artifacts:
        expected_by_gate.setdefault(expected.gate, {})[expected.name] = expected.sha256
    expected_receipts = {
        expected.gate: expected.sha256 for expected in policy.expected_receipts
    }
    producer_ids = {receipt.producer_id for receipt in by_gate.values()}

    for gate in policy.required_gates:
        receipt = by_gate.get(gate)
        if receipt is None:
            reasons.append(f"{gate.value}:missing_receipt")
            continue
        if receipt.receipt_sha256 != expected_receipts[gate]:
            reasons.append(f"{gate.value}:receipt_digest_mismatch")
        if receipt.protocol_sha256 != policy.protocol_sha256:
            reasons.append(f"{gate.value}:protocol_digest_mismatch")
        if receipt.status is not ReceiptStatus.PASS:
            reasons.append(f"{gate.value}:status_{receipt.status.value}")
        if receipt.sealed_or_test_data_accessed:
            reasons.append(f"{gate.value}:sealed_or_test_access")
        if _parse_utc_timestamp(receipt.issued_at_utc) > evaluated_at:
            reasons.append(f"{gate.value}:issued_after_evaluation")
        if not receipt.artifacts:
            reasons.append(f"{gate.value}:no_artifact_digest")
        if gate in policy.independent_audit_gates:
            if receipt.auditor_id is None:
                reasons.append(f"{gate.value}:independent_auditor_missing")
            elif receipt.auditor_id == receipt.producer_id:
                reasons.append(f"{gate.value}:producer_auditor_not_independent")
            elif receipt.auditor_id in producer_ids:
                reasons.append(f"{gate.value}:auditor_also_receipt_producer")

        actual = {item.name: item.sha256 for item in receipt.artifacts}
        expected = expected_by_gate[gate]
        for name, expected_sha256 in expected.items():
            actual_sha256 = actual.get(name)
            if actual_sha256 is None:
                reasons.append(f"{gate.value}:expected_artifact_missing:{name}")
            elif actual_sha256 != expected_sha256:
                reasons.append(f"{gate.value}:artifact_digest_mismatch:{name}")
        for name in sorted(set(actual).difference(expected)):
            reasons.append(f"{gate.value}:unexpected_artifact:{name}")

    unexpected = sorted(set(by_gate).difference(policy.required_gates), key=lambda x: x.value)
    for gate in unexpected:
        reasons.append(f"{gate.value}:unexpected_receipt")

    blocking_reasons = tuple(sorted(set(reasons)))
    authorized = not blocking_reasons
    granted = set(policy.authorized_permissions) if authorized else set()
    receipt_bindings = tuple(
        sorted(
            (gate.value, receipt.receipt_sha256)
            for gate, receipt in by_gate.items()
        )
    )
    return Gate0Decision(
        schema_version="pairformer_gate0_decision_v1",
        policy_sha256=policy.policy_sha256,
        evaluated_at_utc=evaluated_at_utc,
        authorized=authorized,
        contact_training_authorized="contact_training" in granted,
        model_selection_authorized="model_selection" in granted,
        outer_evaluation_authorized=False,
        external_test_inference_authorized=False,
        receipt_sha256_by_gate=receipt_bindings,
        blocking_reasons=blocking_reasons,
    )


def verify_gate0_decision(
    decision: Gate0Decision,
    *,
    expected_policy_sha256: str,
    expected_decision_sha256: str,
    required_permission: str,
) -> None:
    """Fail closed unless a frozen decision grants the requested permission."""

    if not isinstance(decision, Gate0Decision):
        raise Gate0Error("decision must be a validated Gate0Decision")
    _validate_sha256(expected_policy_sha256, "expected policy digest")
    _validate_sha256(expected_decision_sha256, "expected decision digest")
    if decision.policy_sha256 != expected_policy_sha256:
        raise Gate0Error("Gate 0 policy digest does not match the launcher")
    if decision.decision_sha256 != expected_decision_sha256:
        raise Gate0Error("Gate 0 decision digest does not match the launcher")

    permissions = {
        "contact_training": decision.contact_training_authorized,
        "model_selection": decision.model_selection_authorized,
        "outer_evaluation": decision.outer_evaluation_authorized,
        "external_test_inference": decision.external_test_inference_authorized,
    }
    if required_permission not in permissions:
        raise Gate0Error(f"unknown Gate 0 permission {required_permission!r}")
    if not decision.authorized or not permissions[required_permission]:
        raise Gate0Error(
            f"Gate 0 does not authorize {required_permission}: "
            f"{', '.join(decision.blocking_reasons) or 'blocked'}"
        )


def gate0_decision_from_json(value: str | bytes | bytearray) -> Gate0Decision:
    """Parse untrusted JSON while rejecting duplicate object keys.

    Callers should use this entry point for raw JSON. Parsing JSON elsewhere
    can erase duplicate keys before :func:`gate0_decision_from_payload` sees
    them, making producer-dependent "first key" versus "last key" semantics
    impossible to audit.
    """

    if not isinstance(value, (str, bytes, bytearray)):
        raise Gate0Error("decision JSON must be text or UTF-8 bytes")
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Gate0Error("decision JSON bytes must use UTF-8") from exc
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Gate0Error("decision JSON must be valid strict UTF-8 JSON") from exc
    return gate0_decision_from_payload(payload)


def gate0_decision_from_payload(payload: Mapping[str, object]) -> Gate0Decision:
    """Strictly parse an already JSON-decoded Gate 0 decision.

    This function cannot detect duplicate keys erased by another JSON parser.
    Raw untrusted JSON must be passed to :func:`gate0_decision_from_json`.
    """

    value = _exact_mapping(
        payload,
        {
            "schema_version",
            "policy_sha256",
            "evaluated_at_utc",
            "authorized",
            "permissions",
            "receipt_sha256_by_gate",
            "blocking_reasons",
        },
        "decision",
    )
    permissions = _exact_mapping(
        value["permissions"],
        {
            "contact_training",
            "model_selection",
            "outer_evaluation",
            "external_test_inference",
        },
        "decision.permissions",
    )
    for name, permission in permissions.items():
        if not isinstance(permission, bool):
            raise Gate0Error(f"decision.permissions.{name} must be a boolean")

    raw_bindings = value["receipt_sha256_by_gate"]
    if not isinstance(raw_bindings, list):
        raise Gate0Error("decision.receipt_sha256_by_gate must be an array")
    bindings: list[tuple[str, str]] = []
    for index, raw_binding in enumerate(raw_bindings):
        binding = _exact_mapping(
            raw_binding,
            {"gate", "sha256"},
            f"decision.receipt_sha256_by_gate[{index}]",
        )
        gate = binding["gate"]
        if not isinstance(gate, str) or gate not in {item.value for item in GateName}:
            raise Gate0Error(
                f"decision.receipt_sha256_by_gate[{index}].gate is unsupported"
            )
        digest = binding["sha256"]
        _validate_sha256(
            digest,
            f"decision.receipt_sha256_by_gate[{index}].sha256",
        )
        bindings.append((gate, digest))

    raw_reasons = value["blocking_reasons"]
    if not isinstance(raw_reasons, list) or not all(
        isinstance(reason, str) and reason for reason in raw_reasons
    ):
        raise Gate0Error("decision.blocking_reasons must be an array of strings")
    if not isinstance(value["authorized"], bool):
        raise Gate0Error("decision.authorized must be a boolean")
    if not isinstance(value["schema_version"], str):
        raise Gate0Error("decision.schema_version must be a string")
    if not isinstance(value["policy_sha256"], str):
        raise Gate0Error("decision.policy_sha256 must be a string")
    if not isinstance(value["evaluated_at_utc"], str):
        raise Gate0Error("decision.evaluated_at_utc must be a string")

    return Gate0Decision(
        schema_version=value["schema_version"],
        policy_sha256=value["policy_sha256"],
        evaluated_at_utc=value["evaluated_at_utc"],
        authorized=value["authorized"],
        contact_training_authorized=permissions["contact_training"],
        model_selection_authorized=permissions["model_selection"],
        outer_evaluation_authorized=permissions["outer_evaluation"],
        external_test_inference_authorized=permissions["external_test_inference"],
        receipt_sha256_by_gate=tuple(bindings),
        blocking_reasons=tuple(raw_reasons),
    )


def _validate_artifact_tuple(artifacts: Sequence[DigestBinding]) -> None:
    if not isinstance(artifacts, tuple):
        raise Gate0Error("receipt artifacts must be an immutable tuple")
    if not all(isinstance(item, DigestBinding) for item in artifacts):
        raise Gate0Error("receipt artifacts must contain only DigestBinding values")
    names = [item.name for item in artifacts]
    if len(set(names)) != len(names):
        raise Gate0Error("receipt artifacts contain duplicate logical names")


def _validate_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Gate0Error(f"{field} must be a lowercase 64-character SHA-256")


def _validate_utc_timestamp(value: str) -> None:
    _parse_utc_timestamp(value)


def _parse_utc_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise Gate0Error(
            "timestamps must use canonical UTC YYYY-MM-DDTHH:MM:SSZ"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Gate0Error(
            "timestamps must be valid canonical UTC calendar times"
        ) from exc
    if parsed.tzinfo != timezone.utc:
        raise Gate0Error("timestamps must use the UTC timezone")
    return parsed


def _validate_identity(value: object, field: str) -> None:
    if not isinstance(value, str) or _IDENTITY_RE.fullmatch(value) is None:
        raise Gate0Error(
            f"{field} must be a canonical lowercase ASCII authority identifier"
        )


def _validate_logical_name(value: object, field: str) -> None:
    if not isinstance(value, str) or _LOGICAL_NAME_RE.fullmatch(value) is None:
        raise Gate0Error(
            f"{field} must be a canonical lowercase ASCII logical identifier"
        )


def _validate_assertion(value: object) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or _contains_control_character(value)
    ):
        raise Gate0Error(
            "receipt assertion must be a normalized nonempty text string"
        )


def _contains_control_character(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_mapping(
    value: object,
    expected: set[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise Gate0Error(f"{name} must be a JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise Gate0Error(
            f"{name} fields do not match the locked schema; "
            f"missing={missing}, extra={extra}"
        )
    return value


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Gate0Error(f"decision JSON contains duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise Gate0Error(f"decision JSON contains unsupported constant {value!r}")
