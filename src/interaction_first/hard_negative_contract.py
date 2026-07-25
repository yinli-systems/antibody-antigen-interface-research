"""Fail-closed semantics for train-only partner controls and hard negatives.

This module deliberately does not read data, generate controls, or train a
model. It validates immutable metadata prepared by an upstream, independently
audited data pipeline. In particular, an unmeasured decoy is never converted
into a supervised non-binder label. Serialized documents are authorized only
against an expected manifest digest obtained from an external immutable ledger;
their embedded self-hash is never treated as an authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
import unicodedata
from typing import Any, Mapping, Optional, Sequence


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MUTABLE_REVISIONS = {
    "current",
    "head",
    "latest",
    "main",
    "master",
    "tip",
}
_MISSING_LICENSES = {
    "none",
    "unknown",
    "unlicensed",
    "unspecified",
}
_SCHEMA_VERSION = "hard_negative_control_manifest_v2"
_MASK_ENCODING = "row_major_bool_bytes_v1"
_HASH_DOMAIN_PREFIX = b"interaction_first:hard_negative_contract:v2\x00"
_MAX_SAFE_JSON_INTEGER = (1 << 53) - 1


class BindingEvidence(str, Enum):
    """Experimental evidence attached to the represented antibody-antigen pair."""

    MEASURED_BINDER = "measured_binder"
    MEASURED_NON_BINDER = "measured_non_binder"
    UNKNOWN_UNMEASURED = "unknown_unmeasured"


class ControlFamily(str, Enum):
    """Semantically distinct cognate and control construction families."""

    COGNATE = "cognate"
    MEASURED_NON_BINDER = "measured_non_binder"
    UNKNOWN_UNMEASURED_DECOY = "unknown_unmeasured_decoy"
    SAME_ANTIGEN_WRONG_ANTIBODY = "same_antigen_wrong_antibody"
    SAME_ANTIBODY_WRONG_ANTIGEN = "same_antibody_wrong_antigen"
    MUTATION_FLIP = "mutation_flip"
    WRONG_PATCH = "wrong_patch"
    WRONG_POSE = "wrong_pose"


class PartnerAxis(str, Enum):
    """Partner axis changed by a measured mutation flip."""

    ANTIBODY = "antibody"
    ANTIGEN = "antigen"


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty and exactly trimmed")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{field_name} must use NFC-normalized Unicode")
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        for character in value
    ):
        raise ValueError(
            f"{field_name} cannot contain control, format, or surrogate characters"
        )
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    value = _require_text(value, field_name)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_optional_sha256(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    return _require_sha256(value, field_name)


def _require_nonnegative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    if value > _MAX_SAFE_JSON_INTEGER:
        raise ValueError(
            f"{field_name} exceeds the interoperable JSON integer range"
        )
    return value


def _require_positive_integer(value: Any, field_name: str) -> int:
    value = _require_nonnegative_integer(value, field_name)
    if value == 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _canonical_identifier_tuple(
    values: Sequence[str],
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of identifiers")
    normalized = tuple(_require_text(value, field_name) for value in values)
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} cannot contain duplicate identifiers")
    return tuple(sorted(normalized))


def _canonical_sha256_tuple(
    values: Sequence[str],
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of SHA-256 digests")
    normalized = tuple(_require_sha256(value, field_name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} cannot contain duplicate digests")
    return tuple(sorted(normalized))


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _payload_sha256(payload: Mapping[str, Any], *, domain: str) -> str:
    domain = _require_text(domain, "hash_domain")
    preimage = (
        _HASH_DOMAIN_PREFIX
        + domain.encode("ascii")
        + b"\x00"
        + _canonical_json_bytes(payload)
    )
    return hashlib.sha256(preimage).hexdigest()


@dataclass(frozen=True)
class SourceBinding:
    """Immutable dataset, license, and provenance binding."""

    source_id: str
    revision: str
    license_id: str
    license_receipt_sha256: str
    provenance_sha256: str
    source_content_sha256: str
    dataset_admission_receipt_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.source_id, "source_id")
        revision = _require_text(self.revision, "revision")
        lowered_revision = revision.casefold()
        if (
            lowered_revision in _MUTABLE_REVISIONS
            or lowered_revision.startswith("refs/heads/")
        ):
            raise ValueError("revision must identify an immutable source snapshot")
        license_id = _require_text(self.license_id, "license_id")
        if license_id.casefold() in _MISSING_LICENSES:
            raise ValueError("license_id must identify an explicit license")
        _require_sha256(self.license_receipt_sha256, "license_receipt_sha256")
        _require_sha256(self.provenance_sha256, "provenance_sha256")
        _require_sha256(self.source_content_sha256, "source_content_sha256")
        _require_sha256(
            self.dataset_admission_receipt_sha256,
            "dataset_admission_receipt_sha256",
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "dataset_admission_receipt_sha256": (
                self.dataset_admission_receipt_sha256
            ),
            "license_id": self.license_id,
            "license_receipt_sha256": self.license_receipt_sha256,
            "provenance_sha256": self.provenance_sha256,
            "revision": self.revision,
            "source_content_sha256": self.source_content_sha256,
            "source_id": self.source_id,
        }

    @property
    def binding_sha256(self) -> str:
        return _payload_sha256(self.to_payload(), domain="source_binding")


@dataclass(frozen=True)
class TrainOnlySplitBinding:
    """Exact nested-fold role proven to be train-only."""

    split_manifest_sha256: str
    train_membership_receipt_sha256: str
    outer_fold: int
    inner_fold: int
    outer_role: str = "train"
    inner_role: str = "train"

    def __post_init__(self) -> None:
        _require_sha256(self.split_manifest_sha256, "split_manifest_sha256")
        _require_sha256(
            self.train_membership_receipt_sha256,
            "train_membership_receipt_sha256",
        )
        _require_nonnegative_integer(self.outer_fold, "outer_fold")
        _require_nonnegative_integer(self.inner_fold, "inner_fold")
        if self.outer_role != "train" or self.inner_role != "train":
            raise ValueError(
                "Hard-negative construction is restricted to outer-train and "
                "inner-train records"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "inner_fold": self.inner_fold,
            "inner_role": self.inner_role,
            "outer_fold": self.outer_fold,
            "outer_role": self.outer_role,
            "split_manifest_sha256": self.split_manifest_sha256,
            "train_membership_receipt_sha256": (
                self.train_membership_receipt_sha256
            ),
        }

    @property
    def binding_sha256(self) -> str:
        return _payload_sha256(self.to_payload(), domain="train_split_binding")


@dataclass(frozen=True)
class GenerationBinding:
    """Deterministic implementation and configuration for one control run."""

    strategy_id: str
    strategy_version: str
    implementation_sha256: str
    configuration_sha256: str
    generation_run_sha256: str
    deterministic_seed: int

    def __post_init__(self) -> None:
        _require_text(self.strategy_id, "strategy_id")
        _require_text(self.strategy_version, "strategy_version")
        _require_sha256(self.implementation_sha256, "implementation_sha256")
        _require_sha256(self.configuration_sha256, "configuration_sha256")
        _require_sha256(self.generation_run_sha256, "generation_run_sha256")
        _require_nonnegative_integer(self.deterministic_seed, "deterministic_seed")

    def to_payload(self) -> dict[str, Any]:
        return {
            "configuration_sha256": self.configuration_sha256,
            "deterministic_seed": self.deterministic_seed,
            "generation_run_sha256": self.generation_run_sha256,
            "implementation_sha256": self.implementation_sha256,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
        }

    @property
    def binding_sha256(self) -> str:
        return _payload_sha256(self.to_payload(), domain="generation_binding")


@dataclass(frozen=True)
class TrainOnlyAllowlist:
    """Exact antibody and antigen identifiers admitted to one nested train role."""

    allowlist_id: str
    split_manifest_sha256: str
    train_membership_receipt_sha256: str
    outer_fold: int
    inner_fold: int
    antibody_entity_ids: tuple[str, ...]
    antigen_entity_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.allowlist_id, "allowlist_id")
        _require_sha256(self.split_manifest_sha256, "split_manifest_sha256")
        _require_sha256(
            self.train_membership_receipt_sha256,
            "train_membership_receipt_sha256",
        )
        _require_nonnegative_integer(self.outer_fold, "outer_fold")
        _require_nonnegative_integer(self.inner_fold, "inner_fold")
        object.__setattr__(
            self,
            "antibody_entity_ids",
            _canonical_identifier_tuple(
                self.antibody_entity_ids,
                "antibody_entity_ids",
            ),
        )
        object.__setattr__(
            self,
            "antigen_entity_ids",
            _canonical_identifier_tuple(
                self.antigen_entity_ids,
                "antigen_entity_ids",
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "allowlist_id": self.allowlist_id,
            "antibody_entity_ids": list(self.antibody_entity_ids),
            "antigen_entity_ids": list(self.antigen_entity_ids),
            "inner_fold": self.inner_fold,
            "outer_fold": self.outer_fold,
            "split_manifest_sha256": self.split_manifest_sha256,
            "train_membership_receipt_sha256": (
                self.train_membership_receipt_sha256
            ),
        }

    @property
    def allowlist_sha256(self) -> str:
        return _payload_sha256(self.to_payload(), domain="train_allowlist")

    def validate_split(self, split: TrainOnlySplitBinding) -> None:
        if (
            split.split_manifest_sha256 != self.split_manifest_sha256
            or split.train_membership_receipt_sha256
            != self.train_membership_receipt_sha256
            or split.outer_fold != self.outer_fold
            or split.inner_fold != self.inner_fold
        ):
            raise ValueError("Record split does not match the train-only allowlist")


@dataclass(frozen=True)
class ResidueDomainSupport:
    """Residue identity, domain, and boolean-mask support for one scored pair.

    Mask digests must hash canonical row-major boolean bytes only. Residue
    identity digests are separate, so support parity can be checked without
    pretending that two different partners contain the same residues.
    """

    antibody_domain_size: int
    antigen_domain_size: int
    antibody_valid_count: int
    antigen_valid_count: int
    candidate_pair_count: int
    antibody_mask_sha256: str
    antigen_mask_sha256: str
    pair_mask_sha256: str
    antibody_residue_ids_sha256: str
    antigen_residue_ids_sha256: str
    canonical_domain_manifest_sha256: str
    domain_selection_policy_sha256: str
    candidate_mask_policy_sha256: str

    def __post_init__(self) -> None:
        antibody_domain_size = _require_positive_integer(
            self.antibody_domain_size,
            "antibody_domain_size",
        )
        antigen_domain_size = _require_positive_integer(
            self.antigen_domain_size,
            "antigen_domain_size",
        )
        antibody_valid_count = _require_positive_integer(
            self.antibody_valid_count,
            "antibody_valid_count",
        )
        antigen_valid_count = _require_positive_integer(
            self.antigen_valid_count,
            "antigen_valid_count",
        )
        candidate_pair_count = _require_positive_integer(
            self.candidate_pair_count,
            "candidate_pair_count",
        )
        if antibody_valid_count > antibody_domain_size:
            raise ValueError("antibody_valid_count exceeds antibody_domain_size")
        if antigen_valid_count > antigen_domain_size:
            raise ValueError("antigen_valid_count exceeds antigen_domain_size")
        if candidate_pair_count > antibody_valid_count * antigen_valid_count:
            raise ValueError("candidate_pair_count exceeds valid Cartesian support")
        _require_sha256(self.antibody_mask_sha256, "antibody_mask_sha256")
        _require_sha256(self.antigen_mask_sha256, "antigen_mask_sha256")
        _require_sha256(self.pair_mask_sha256, "pair_mask_sha256")
        _require_sha256(
            self.antibody_residue_ids_sha256,
            "antibody_residue_ids_sha256",
        )
        _require_sha256(
            self.antigen_residue_ids_sha256,
            "antigen_residue_ids_sha256",
        )
        _require_sha256(
            self.canonical_domain_manifest_sha256,
            "canonical_domain_manifest_sha256",
        )
        _require_sha256(
            self.domain_selection_policy_sha256,
            "domain_selection_policy_sha256",
        )
        _require_sha256(
            self.candidate_mask_policy_sha256,
            "candidate_mask_policy_sha256",
        )

    @property
    def support_signature(self) -> tuple[Any, ...]:
        """Return the exact shape and mask support used for matched comparison."""
        return (
            self.antibody_domain_size,
            self.antigen_domain_size,
            self.antibody_valid_count,
            self.antigen_valid_count,
            self.candidate_pair_count,
            self.antibody_mask_sha256,
            self.antigen_mask_sha256,
            self.pair_mask_sha256,
            self.domain_selection_policy_sha256,
            self.candidate_mask_policy_sha256,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "antibody_domain_size": self.antibody_domain_size,
            "antibody_mask_sha256": self.antibody_mask_sha256,
            "antibody_residue_ids_sha256": (
                self.antibody_residue_ids_sha256
            ),
            "antibody_valid_count": self.antibody_valid_count,
            "antigen_domain_size": self.antigen_domain_size,
            "antigen_mask_sha256": self.antigen_mask_sha256,
            "antigen_residue_ids_sha256": self.antigen_residue_ids_sha256,
            "antigen_valid_count": self.antigen_valid_count,
            "candidate_pair_count": self.candidate_pair_count,
            "canonical_domain_manifest_sha256": (
                self.canonical_domain_manifest_sha256
            ),
            "candidate_mask_policy_sha256": (
                self.candidate_mask_policy_sha256
            ),
            "domain_selection_policy_sha256": (
                self.domain_selection_policy_sha256
            ),
            "mask_encoding": _MASK_ENCODING,
            "pair_mask_sha256": self.pair_mask_sha256,
        }

    @property
    def support_sha256(self) -> str:
        """Hash the exact domain, residue, policy, and candidate-mask support."""
        return _payload_sha256(self.to_payload(), domain="residue_domain_support")


@dataclass(frozen=True)
class TrainOnlyEvidenceRegistry:
    """Externally audited evidence bindings admitted to one nested train role.

    The authority receipt and final manifest digest must be anchored outside the
    document. Embedded digests alone do not establish experimental authenticity.
    """

    split_manifest_sha256: str
    train_membership_receipt_sha256: str
    outer_fold: int
    inner_fold: int
    train_allowlist_sha256: str
    authority_receipt_sha256: str
    measured_evidence_binding_sha256s: tuple[str, ...]
    mutation_relation_binding_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_sha256(self.split_manifest_sha256, "split_manifest_sha256")
        _require_sha256(
            self.train_membership_receipt_sha256,
            "train_membership_receipt_sha256",
        )
        _require_nonnegative_integer(self.outer_fold, "outer_fold")
        _require_nonnegative_integer(self.inner_fold, "inner_fold")
        _require_sha256(
            self.train_allowlist_sha256,
            "train_allowlist_sha256",
        )
        _require_sha256(
            self.authority_receipt_sha256,
            "authority_receipt_sha256",
        )
        object.__setattr__(
            self,
            "measured_evidence_binding_sha256s",
            _canonical_sha256_tuple(
                self.measured_evidence_binding_sha256s,
                "measured_evidence_binding_sha256s",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "mutation_relation_binding_sha256s",
            _canonical_sha256_tuple(
                self.mutation_relation_binding_sha256s,
                "mutation_relation_binding_sha256s",
                allow_empty=True,
            ),
        )

    def validate_split(self, split: TrainOnlySplitBinding) -> None:
        if (
            split.split_manifest_sha256 != self.split_manifest_sha256
            or split.train_membership_receipt_sha256
            != self.train_membership_receipt_sha256
            or split.outer_fold != self.outer_fold
            or split.inner_fold != self.inner_fold
        ):
            raise ValueError(
                "Evidence registry does not match the exact train-only split"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "authority_receipt_sha256": self.authority_receipt_sha256,
            "inner_fold": self.inner_fold,
            "measured_evidence_binding_sha256s": list(
                self.measured_evidence_binding_sha256s
            ),
            "mutation_relation_binding_sha256s": list(
                self.mutation_relation_binding_sha256s
            ),
            "outer_fold": self.outer_fold,
            "split_manifest_sha256": self.split_manifest_sha256,
            "train_allowlist_sha256": self.train_allowlist_sha256,
            "train_membership_receipt_sha256": (
                self.train_membership_receipt_sha256
            ),
        }

    @property
    def registry_sha256(self) -> str:
        return _payload_sha256(
            self.to_payload(),
            domain="train_evidence_registry",
        )


@dataclass(frozen=True)
class ControlRecord:
    """One cognate or matched control with explicit supervision semantics."""

    record_id: str
    match_group_id: str
    parent_cognate_id: Optional[str]
    control_family: ControlFamily
    binding_evidence: BindingEvidence
    antibody_entity_id: str
    antigen_entity_id: str
    patch_id: str
    pose_id: str
    patch_definition_sha256: str
    pose_transform_sha256: str
    source: SourceBinding
    split: TrainOnlySplitBinding
    generation: GenerationBinding
    train_allowlist_sha256: str
    sample_content_sha256: str
    construction_provenance_sha256: str
    support: ResidueDomainSupport
    label_provenance_sha256: Optional[str] = None
    assay_protocol_sha256: Optional[str] = None
    measurement_receipt_sha256: Optional[str] = None
    mutation_axis: Optional[PartnerAxis] = None
    mutation_spec_sha256: Optional[str] = None
    paired_assay_receipt_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        _require_text(self.record_id, "record_id")
        _require_text(self.match_group_id, "match_group_id")
        if self.parent_cognate_id is not None:
            _require_text(self.parent_cognate_id, "parent_cognate_id")
        if not isinstance(self.control_family, ControlFamily):
            raise TypeError("control_family must be a ControlFamily")
        if not isinstance(self.binding_evidence, BindingEvidence):
            raise TypeError("binding_evidence must be a BindingEvidence")
        _require_text(self.antibody_entity_id, "antibody_entity_id")
        _require_text(self.antigen_entity_id, "antigen_entity_id")
        _require_text(self.patch_id, "patch_id")
        _require_text(self.pose_id, "pose_id")
        _require_sha256(
            self.patch_definition_sha256,
            "patch_definition_sha256",
        )
        _require_sha256(
            self.pose_transform_sha256,
            "pose_transform_sha256",
        )
        if not isinstance(self.source, SourceBinding):
            raise TypeError("source must be a SourceBinding")
        if not isinstance(self.split, TrainOnlySplitBinding):
            raise TypeError("split must be a TrainOnlySplitBinding")
        if not isinstance(self.generation, GenerationBinding):
            raise TypeError("generation must be a GenerationBinding")
        if not isinstance(self.support, ResidueDomainSupport):
            raise TypeError("support must be a ResidueDomainSupport")
        _require_sha256(self.train_allowlist_sha256, "train_allowlist_sha256")
        _require_sha256(self.sample_content_sha256, "sample_content_sha256")
        _require_sha256(
            self.construction_provenance_sha256,
            "construction_provenance_sha256",
        )
        label_provenance = _require_optional_sha256(
            self.label_provenance_sha256,
            "label_provenance_sha256",
        )
        assay_protocol = _require_optional_sha256(
            self.assay_protocol_sha256,
            "assay_protocol_sha256",
        )
        measurement_receipt = _require_optional_sha256(
            self.measurement_receipt_sha256,
            "measurement_receipt_sha256",
        )
        mutation_spec = _require_optional_sha256(
            self.mutation_spec_sha256,
            "mutation_spec_sha256",
        )
        paired_assay_receipt = _require_optional_sha256(
            self.paired_assay_receipt_sha256,
            "paired_assay_receipt_sha256",
        )

        measurement_fields = (
            label_provenance,
            assay_protocol,
            measurement_receipt,
        )
        if self.binding_evidence is BindingEvidence.UNKNOWN_UNMEASURED:
            if any(value is not None for value in measurement_fields):
                raise ValueError(
                    "Unknown/unmeasured controls cannot carry measured-label fields"
                )
        elif any(value is None for value in measurement_fields):
            raise ValueError(
                "Measured binding evidence requires assay, label-provenance, "
                "and measurement-receipt digests"
            )

        family = self.control_family
        evidence = self.binding_evidence
        if family is ControlFamily.COGNATE:
            if evidence is not BindingEvidence.MEASURED_BINDER:
                raise ValueError("A cognate record must be a measured binder")
            if self.parent_cognate_id is not None:
                raise ValueError("A cognate record cannot have a parent cognate")
        else:
            if self.parent_cognate_id is None:
                raise ValueError("Every control must identify its parent cognate")

        if family is ControlFamily.MEASURED_NON_BINDER:
            if evidence is not BindingEvidence.MEASURED_NON_BINDER:
                raise ValueError(
                    "The measured-non-binder family requires measured "
                    "non-binder evidence"
                )
        elif family is ControlFamily.UNKNOWN_UNMEASURED_DECOY:
            if evidence is not BindingEvidence.UNKNOWN_UNMEASURED:
                raise ValueError(
                    "An unmeasured decoy must remain unknown/unmeasured"
                )
        elif family in {
            ControlFamily.SAME_ANTIGEN_WRONG_ANTIBODY,
            ControlFamily.SAME_ANTIBODY_WRONG_ANTIGEN,
        }:
            if evidence is BindingEvidence.MEASURED_BINDER:
                raise ValueError(
                    "A measured binder cannot be called a wrong partner"
                )
        elif family is ControlFamily.MUTATION_FLIP:
            if evidence is not BindingEvidence.MEASURED_NON_BINDER:
                raise ValueError(
                    "A mutation flip requires a measured non-binder mutant"
                )
            if not isinstance(self.mutation_axis, PartnerAxis):
                raise ValueError(
                    "A mutation flip requires an explicit mutation_axis"
                )
            if mutation_spec is None or paired_assay_receipt is None:
                raise ValueError(
                    "A mutation flip requires mutation and paired-assay receipts"
                )
        elif family in {ControlFamily.WRONG_PATCH, ControlFamily.WRONG_POSE}:
            if evidence is not BindingEvidence.UNKNOWN_UNMEASURED:
                raise ValueError(
                    "Constructed wrong-patch and wrong-pose controls are "
                    "unmeasured ranking controls"
                )

        if family is not ControlFamily.MUTATION_FLIP:
            if (
                self.mutation_axis is not None
                or mutation_spec is not None
                or paired_assay_receipt is not None
            ):
                raise ValueError(
                    "Mutation metadata is only valid for mutation-flip controls"
                )

    @property
    def binary_target(self) -> Optional[int]:
        """Return a supervised label only when the contract permits one."""
        if self.control_family is ControlFamily.COGNATE:
            return 1
        if self.binding_evidence is BindingEvidence.MEASURED_NON_BINDER:
            return 0
        return None

    @property
    def supervision_mode(self) -> str:
        target = self.binary_target
        if target == 1:
            return "binary_positive"
        if target == 0:
            return "binary_negative"
        return "ranking_control_only"

    @property
    def measured_evidence_binding_sha256(self) -> Optional[str]:
        """Bind measured evidence to the exact represented sample and support."""
        if self.binding_evidence is BindingEvidence.UNKNOWN_UNMEASURED:
            return None
        payload = {
            "antibody_entity_id": self.antibody_entity_id,
            "antigen_entity_id": self.antigen_entity_id,
            "assay_protocol_sha256": self.assay_protocol_sha256,
            "binding_evidence": self.binding_evidence.value,
            "label_provenance_sha256": self.label_provenance_sha256,
            "measurement_receipt_sha256": self.measurement_receipt_sha256,
            "patch_definition_sha256": self.patch_definition_sha256,
            "patch_id": self.patch_id,
            "pose_id": self.pose_id,
            "pose_transform_sha256": self.pose_transform_sha256,
            "sample_content_sha256": self.sample_content_sha256,
            "source_binding_sha256": self.source.binding_sha256,
            "split_binding_sha256": self.split.binding_sha256,
            "support_sha256": self.support.support_sha256,
        }
        return _payload_sha256(payload, domain="measured_evidence_binding")

    def identity_key(self) -> tuple[str, ...]:
        return (
            self.antibody_entity_id,
            self.antigen_entity_id,
            self.patch_id,
            self.patch_definition_sha256,
            self.pose_id,
            self.pose_transform_sha256,
            self.support.canonical_domain_manifest_sha256,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "antibody_entity_id": self.antibody_entity_id,
            "antigen_entity_id": self.antigen_entity_id,
            "assay_protocol_sha256": self.assay_protocol_sha256,
            "binary_target": self.binary_target,
            "binding_evidence": self.binding_evidence.value,
            "construction_provenance_sha256": (
                self.construction_provenance_sha256
            ),
            "control_family": self.control_family.value,
            "generation": self.generation.to_payload(),
            "label_provenance_sha256": self.label_provenance_sha256,
            "match_group_id": self.match_group_id,
            "measured_evidence_binding_sha256": (
                self.measured_evidence_binding_sha256
            ),
            "measurement_receipt_sha256": self.measurement_receipt_sha256,
            "mutation_axis": (
                self.mutation_axis.value
                if self.mutation_axis is not None
                else None
            ),
            "mutation_spec_sha256": self.mutation_spec_sha256,
            "paired_assay_receipt_sha256": (
                self.paired_assay_receipt_sha256
            ),
            "parent_cognate_id": self.parent_cognate_id,
            "patch_definition_sha256": self.patch_definition_sha256,
            "patch_id": self.patch_id,
            "pose_transform_sha256": self.pose_transform_sha256,
            "pose_id": self.pose_id,
            "record_id": self.record_id,
            "sample_content_sha256": self.sample_content_sha256,
            "source": self.source.to_payload(),
            "split": self.split.to_payload(),
            "supervision_mode": self.supervision_mode,
            "support": self.support.to_payload(),
            "train_allowlist_sha256": self.train_allowlist_sha256,
        }


def _assert_support_parity(
    cognate: ControlRecord,
    control: ControlRecord,
) -> None:
    if control.support.support_signature != cognate.support.support_signature:
        raise ValueError(
            f"Control {control.record_id!r} has different residue-domain "
            "mask/support from its cognate"
        )


def _assert_same_pair_domain_identity(
    cognate: ControlRecord,
    control: ControlRecord,
) -> None:
    if (
        control.support.antibody_residue_ids_sha256
        != cognate.support.antibody_residue_ids_sha256
        or control.support.antigen_residue_ids_sha256
        != cognate.support.antigen_residue_ids_sha256
        or control.support.canonical_domain_manifest_sha256
        != cognate.support.canonical_domain_manifest_sha256
    ):
        raise ValueError(
            f"Control {control.record_id!r} changes residue identities for "
            "a same-pair control"
        )


def _validate_control_against_cognate(
    cognate: ControlRecord,
    control: ControlRecord,
) -> None:
    _assert_support_parity(cognate, control)
    family = control.control_family

    if family is ControlFamily.SAME_ANTIGEN_WRONG_ANTIBODY:
        if (
            control.antigen_entity_id != cognate.antigen_entity_id
            or control.antibody_entity_id == cognate.antibody_entity_id
        ):
            raise ValueError(
                "same-antigen/wrong-antibody controls must preserve only "
                "the antigen entity"
            )
        if (
            control.support.antigen_residue_ids_sha256
            != cognate.support.antigen_residue_ids_sha256
        ):
            raise ValueError(
                "same-antigen controls must preserve antigen residue identities"
            )
    elif family is ControlFamily.SAME_ANTIBODY_WRONG_ANTIGEN:
        if (
            control.antibody_entity_id != cognate.antibody_entity_id
            or control.antigen_entity_id == cognate.antigen_entity_id
        ):
            raise ValueError(
                "same-antibody/wrong-antigen controls must preserve only "
                "the antibody entity"
            )
        if (
            control.support.antibody_residue_ids_sha256
            != cognate.support.antibody_residue_ids_sha256
        ):
            raise ValueError(
                "same-antibody controls must preserve antibody residue identities"
            )
    elif family is ControlFamily.MUTATION_FLIP:
        if control.assay_protocol_sha256 != cognate.assay_protocol_sha256:
            raise ValueError(
                "Mutation flips must use the cognate assay protocol"
            )
        if (
            control.patch_id != cognate.patch_id
            or control.pose_id != cognate.pose_id
            or control.patch_definition_sha256
            != cognate.patch_definition_sha256
            or control.pose_transform_sha256 != cognate.pose_transform_sha256
        ):
            raise ValueError("Mutation flips cannot change patch or pose identity")
        if control.mutation_axis is PartnerAxis.ANTIBODY:
            if (
                control.antibody_entity_id == cognate.antibody_entity_id
                or control.antigen_entity_id != cognate.antigen_entity_id
                or control.support.antibody_residue_ids_sha256
                == cognate.support.antibody_residue_ids_sha256
                or control.support.antigen_residue_ids_sha256
                != cognate.support.antigen_residue_ids_sha256
            ):
                raise ValueError(
                    "An antibody mutation flip must change only the antibody"
                )
        else:
            if (
                control.antigen_entity_id == cognate.antigen_entity_id
                or control.antibody_entity_id != cognate.antibody_entity_id
                or control.support.antigen_residue_ids_sha256
                == cognate.support.antigen_residue_ids_sha256
                or control.support.antibody_residue_ids_sha256
                != cognate.support.antibody_residue_ids_sha256
            ):
                raise ValueError(
                    "An antigen mutation flip must change only the antigen"
                )
    elif family is ControlFamily.WRONG_PATCH:
        if (
            control.antibody_entity_id != cognate.antibody_entity_id
            or control.antigen_entity_id != cognate.antigen_entity_id
            or control.patch_id == cognate.patch_id
            or control.pose_id != cognate.pose_id
            or control.patch_definition_sha256
            == cognate.patch_definition_sha256
            or control.pose_transform_sha256
            != cognate.pose_transform_sha256
        ):
            raise ValueError(
                "A wrong-patch control must change only the patch definition"
            )
        _assert_same_pair_domain_identity(cognate, control)
    elif family is ControlFamily.WRONG_POSE:
        if (
            control.antibody_entity_id != cognate.antibody_entity_id
            or control.antigen_entity_id != cognate.antigen_entity_id
            or control.patch_id != cognate.patch_id
            or control.pose_id == cognate.pose_id
            or control.patch_definition_sha256
            != cognate.patch_definition_sha256
            or control.pose_transform_sha256
            == cognate.pose_transform_sha256
        ):
            raise ValueError(
                "A wrong-pose control must change only the pose transform"
            )
        _assert_same_pair_domain_identity(cognate, control)
    elif family in {
        ControlFamily.MEASURED_NON_BINDER,
        ControlFamily.UNKNOWN_UNMEASURED_DECOY,
    }:
        if (
            control.antibody_entity_id == cognate.antibody_entity_id
            and control.antigen_entity_id == cognate.antigen_entity_id
        ):
            raise ValueError(
                "A generic pair control must change at least one partner; "
                "use wrong-patch or wrong-pose semantics for geometry decoys"
            )


def mutation_relation_binding_sha256(
    cognate: ControlRecord,
    mutation: ControlRecord,
) -> str:
    """Hash the exact, validated parent-mutant relationship and paired assay."""
    if not isinstance(cognate, ControlRecord) or not isinstance(
        mutation,
        ControlRecord,
    ):
        raise TypeError("cognate and mutation must be ControlRecord values")
    if cognate.control_family is not ControlFamily.COGNATE:
        raise ValueError("cognate must use the cognate control family")
    if mutation.control_family is not ControlFamily.MUTATION_FLIP:
        raise ValueError("mutation must use the mutation-flip control family")
    if mutation.parent_cognate_id != cognate.record_id:
        raise ValueError("mutation does not reference the supplied cognate")
    if mutation.match_group_id != cognate.match_group_id:
        raise ValueError("mutation and cognate must share a match group")
    if (
        mutation.source != cognate.source
        or mutation.split != cognate.split
        or mutation.generation != cognate.generation
    ):
        raise ValueError(
            "mutation relation cannot cross source, split, or generation"
        )
    _validate_control_against_cognate(cognate, mutation)
    payload = {
        "assay_protocol_sha256": mutation.assay_protocol_sha256,
        "cognate_evidence_binding_sha256": (
            cognate.measured_evidence_binding_sha256
        ),
        "generation_binding_sha256": mutation.generation.binding_sha256,
        "mutation_axis": mutation.mutation_axis.value,
        "mutation_evidence_binding_sha256": (
            mutation.measured_evidence_binding_sha256
        ),
        "mutation_spec_sha256": mutation.mutation_spec_sha256,
        "paired_assay_receipt_sha256": mutation.paired_assay_receipt_sha256,
        "source_binding_sha256": mutation.source.binding_sha256,
        "split_binding_sha256": mutation.split.binding_sha256,
    }
    return _payload_sha256(payload, domain="mutation_relation_binding")


@dataclass(frozen=True)
class HardNegativeManifest:
    """Deterministic, immutable collection of train-only matched controls."""

    allowlist: TrainOnlyAllowlist
    records: tuple[ControlRecord, ...]
    evidence_registry: TrainOnlyEvidenceRegistry

    def __post_init__(self) -> None:
        if not isinstance(self.allowlist, TrainOnlyAllowlist):
            raise TypeError("allowlist must be a TrainOnlyAllowlist")
        if not isinstance(self.evidence_registry, TrainOnlyEvidenceRegistry):
            raise TypeError(
                "evidence_registry must be a TrainOnlyEvidenceRegistry"
            )
        if (
            self.evidence_registry.train_allowlist_sha256
            != self.allowlist.allowlist_sha256
        ):
            raise ValueError(
                "Evidence registry is bound to a different train-only allowlist"
            )
        if isinstance(self.records, (str, bytes)) or not isinstance(
            self.records,
            Sequence,
        ):
            raise TypeError("records must be a sequence of ControlRecord values")
        records = tuple(self.records)
        if len(records) < 2:
            raise ValueError("A matched-control manifest needs at least two records")
        if any(not isinstance(record, ControlRecord) for record in records):
            raise TypeError("records must contain only ControlRecord values")
        if len({record.record_id for record in records}) != len(records):
            raise ValueError("record_id values must be unique")
        object.__setattr__(
            self,
            "records",
            tuple(sorted(records, key=lambda record: record.record_id)),
        )

        expected_allowlist_sha256 = self.allowlist.allowlist_sha256
        first = self.records[0]
        context = (first.source, first.split, first.generation)
        self.evidence_registry.validate_split(first.split)
        groups: dict[str, list[ControlRecord]] = {}
        observed_evidence_bindings: list[str] = []
        observed_measurement_receipts: set[str] = set()
        global_identities: set[tuple[str, ...]] = set()
        for record in self.records:
            self.allowlist.validate_split(record.split)
            self.evidence_registry.validate_split(record.split)
            if record.train_allowlist_sha256 != expected_allowlist_sha256:
                raise ValueError(
                    f"Record {record.record_id!r} is not bound to this allowlist"
                )
            if record.antibody_entity_id not in self.allowlist.antibody_entity_ids:
                raise ValueError(
                    f"Antibody {record.antibody_entity_id!r} is outside the "
                    "train-only allowlist"
                )
            if record.antigen_entity_id not in self.allowlist.antigen_entity_ids:
                raise ValueError(
                    f"Antigen {record.antigen_entity_id!r} is outside the "
                    "train-only allowlist"
                )
            if (record.source, record.split, record.generation) != context:
                raise ValueError(
                    "Matched controls cannot cross source, split, or generation "
                    "strategy boundaries"
                )
            identity = record.identity_key()
            if identity in global_identities:
                raise ValueError(
                    f"Record {record.record_id!r} duplicates a constructed "
                    "identity across the manifest"
                )
            global_identities.add(identity)
            evidence_binding = record.measured_evidence_binding_sha256
            if evidence_binding is not None:
                observed_evidence_bindings.append(evidence_binding)
                measurement_receipt = record.measurement_receipt_sha256
                if measurement_receipt in observed_measurement_receipts:
                    raise ValueError(
                        "A measurement receipt cannot authorize multiple records"
                    )
                observed_measurement_receipts.add(measurement_receipt)
            groups.setdefault(record.match_group_id, []).append(record)

        observed_mutation_relations: list[str] = []
        for group_id, group in sorted(groups.items()):
            cognates = [
                record
                for record in group
                if record.control_family is ControlFamily.COGNATE
            ]
            if len(cognates) != 1:
                raise ValueError(
                    f"Match group {group_id!r} must contain exactly one cognate"
                )
            if len(group) < 2:
                raise ValueError(
                    f"Match group {group_id!r} must contain at least one control"
                )
            cognate = cognates[0]
            seen_identities = {cognate.identity_key()}
            for control in group:
                if control is cognate:
                    continue
                if control.parent_cognate_id != cognate.record_id:
                    raise ValueError(
                        f"Control {control.record_id!r} does not reference the "
                        "group cognate"
                    )
                if control.identity_key() in seen_identities:
                    raise ValueError(
                        f"Control {control.record_id!r} duplicates a constructed "
                        "identity in its match group"
                    )
                seen_identities.add(control.identity_key())
                _validate_control_against_cognate(cognate, control)
                if control.control_family is ControlFamily.MUTATION_FLIP:
                    observed_mutation_relations.append(
                        mutation_relation_binding_sha256(cognate, control)
                    )

        if tuple(sorted(observed_evidence_bindings)) != (
            self.evidence_registry.measured_evidence_binding_sha256s
        ):
            raise ValueError(
                "Measured evidence does not exactly match the externally "
                "audited train-only evidence registry"
            )
        if tuple(sorted(observed_mutation_relations)) != (
            self.evidence_registry.mutation_relation_binding_sha256s
        ):
            raise ValueError(
                "Mutation relations do not exactly match the externally "
                "audited train-only evidence registry"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "allowlist": self.allowlist.to_payload(),
            "evidence_registry": self.evidence_registry.to_payload(),
            "manifest_type": "train_only_hard_negative_controls",
            "records": [record.to_payload() for record in self.records],
            "schema_version": _SCHEMA_VERSION,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.payload())

    @property
    def manifest_sha256(self) -> str:
        return _payload_sha256(self.payload(), domain="manifest")

    def to_document(self) -> dict[str, Any]:
        document = self.payload()
        document["manifest_sha256"] = self.manifest_sha256
        return document


def _require_exact_mapping(
    value: Any,
    expected_keys: set[str],
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(repr(key) for key in actual_keys - expected_keys)
        raise ValueError(
            f"{field_name} has invalid fields; missing={missing}, extra={extra}"
        )
    return value


def _source_from_payload(value: Any) -> SourceBinding:
    payload = _require_exact_mapping(
        value,
        {
            "dataset_admission_receipt_sha256",
            "license_id",
            "license_receipt_sha256",
            "provenance_sha256",
            "revision",
            "source_content_sha256",
            "source_id",
        },
        "source",
    )
    return SourceBinding(**payload)


def _split_from_payload(value: Any) -> TrainOnlySplitBinding:
    payload = _require_exact_mapping(
        value,
        {
            "inner_fold",
            "inner_role",
            "outer_fold",
            "outer_role",
            "split_manifest_sha256",
            "train_membership_receipt_sha256",
        },
        "split",
    )
    return TrainOnlySplitBinding(**payload)


def _generation_from_payload(value: Any) -> GenerationBinding:
    payload = _require_exact_mapping(
        value,
        {
            "configuration_sha256",
            "deterministic_seed",
            "generation_run_sha256",
            "implementation_sha256",
            "strategy_id",
            "strategy_version",
        },
        "generation",
    )
    return GenerationBinding(**payload)


def _allowlist_from_payload(value: Any) -> TrainOnlyAllowlist:
    payload = _require_exact_mapping(
        value,
        {
            "allowlist_id",
            "antibody_entity_ids",
            "antigen_entity_ids",
            "inner_fold",
            "outer_fold",
            "split_manifest_sha256",
            "train_membership_receipt_sha256",
        },
        "allowlist",
    )
    return TrainOnlyAllowlist(
        allowlist_id=payload["allowlist_id"],
        split_manifest_sha256=payload["split_manifest_sha256"],
        train_membership_receipt_sha256=(
            payload["train_membership_receipt_sha256"]
        ),
        outer_fold=payload["outer_fold"],
        inner_fold=payload["inner_fold"],
        antibody_entity_ids=payload["antibody_entity_ids"],
        antigen_entity_ids=payload["antigen_entity_ids"],
    )


def _evidence_registry_from_payload(value: Any) -> TrainOnlyEvidenceRegistry:
    payload = _require_exact_mapping(
        value,
        {
            "authority_receipt_sha256",
            "inner_fold",
            "measured_evidence_binding_sha256s",
            "mutation_relation_binding_sha256s",
            "outer_fold",
            "split_manifest_sha256",
            "train_allowlist_sha256",
            "train_membership_receipt_sha256",
        },
        "evidence_registry",
    )
    return TrainOnlyEvidenceRegistry(
        split_manifest_sha256=payload["split_manifest_sha256"],
        train_membership_receipt_sha256=(
            payload["train_membership_receipt_sha256"]
        ),
        outer_fold=payload["outer_fold"],
        inner_fold=payload["inner_fold"],
        train_allowlist_sha256=payload["train_allowlist_sha256"],
        authority_receipt_sha256=payload["authority_receipt_sha256"],
        measured_evidence_binding_sha256s=payload[
            "measured_evidence_binding_sha256s"
        ],
        mutation_relation_binding_sha256s=payload[
            "mutation_relation_binding_sha256s"
        ],
    )


def _support_from_payload(value: Any) -> ResidueDomainSupport:
    payload = _require_exact_mapping(
        value,
        {
            "antibody_domain_size",
            "antibody_mask_sha256",
            "antibody_residue_ids_sha256",
            "antibody_valid_count",
            "antigen_domain_size",
            "antigen_mask_sha256",
            "antigen_residue_ids_sha256",
            "antigen_valid_count",
            "candidate_pair_count",
            "candidate_mask_policy_sha256",
            "canonical_domain_manifest_sha256",
            "domain_selection_policy_sha256",
            "mask_encoding",
            "pair_mask_sha256",
        },
        "support",
    )
    if payload["mask_encoding"] != _MASK_ENCODING:
        raise ValueError("support uses an unsupported mask encoding")
    return ResidueDomainSupport(
        antibody_domain_size=payload["antibody_domain_size"],
        antigen_domain_size=payload["antigen_domain_size"],
        antibody_valid_count=payload["antibody_valid_count"],
        antigen_valid_count=payload["antigen_valid_count"],
        candidate_pair_count=payload["candidate_pair_count"],
        antibody_mask_sha256=payload["antibody_mask_sha256"],
        antigen_mask_sha256=payload["antigen_mask_sha256"],
        pair_mask_sha256=payload["pair_mask_sha256"],
        antibody_residue_ids_sha256=payload[
            "antibody_residue_ids_sha256"
        ],
        antigen_residue_ids_sha256=payload["antigen_residue_ids_sha256"],
        canonical_domain_manifest_sha256=payload[
            "canonical_domain_manifest_sha256"
        ],
        domain_selection_policy_sha256=payload[
            "domain_selection_policy_sha256"
        ],
        candidate_mask_policy_sha256=payload[
            "candidate_mask_policy_sha256"
        ],
    )


def _control_from_payload(value: Any) -> ControlRecord:
    payload = _require_exact_mapping(
        value,
        {
            "antibody_entity_id",
            "antigen_entity_id",
            "assay_protocol_sha256",
            "binary_target",
            "binding_evidence",
            "construction_provenance_sha256",
            "control_family",
            "generation",
            "label_provenance_sha256",
            "match_group_id",
            "measured_evidence_binding_sha256",
            "measurement_receipt_sha256",
            "mutation_axis",
            "mutation_spec_sha256",
            "paired_assay_receipt_sha256",
            "parent_cognate_id",
            "patch_definition_sha256",
            "patch_id",
            "pose_transform_sha256",
            "pose_id",
            "record_id",
            "sample_content_sha256",
            "source",
            "split",
            "supervision_mode",
            "support",
            "train_allowlist_sha256",
        },
        "record",
    )
    try:
        family = ControlFamily(payload["control_family"])
    except (TypeError, ValueError) as error:
        raise ValueError("record has an invalid control_family") from error
    try:
        evidence = BindingEvidence(payload["binding_evidence"])
    except (TypeError, ValueError) as error:
        raise ValueError("record has invalid binding_evidence") from error
    raw_mutation_axis = payload["mutation_axis"]
    if raw_mutation_axis is None:
        mutation_axis = None
    else:
        try:
            mutation_axis = PartnerAxis(raw_mutation_axis)
        except (TypeError, ValueError) as error:
            raise ValueError("record has an invalid mutation_axis") from error
    return ControlRecord(
        record_id=payload["record_id"],
        match_group_id=payload["match_group_id"],
        parent_cognate_id=payload["parent_cognate_id"],
        control_family=family,
        binding_evidence=evidence,
        antibody_entity_id=payload["antibody_entity_id"],
        antigen_entity_id=payload["antigen_entity_id"],
        patch_id=payload["patch_id"],
        pose_id=payload["pose_id"],
        patch_definition_sha256=payload["patch_definition_sha256"],
        pose_transform_sha256=payload["pose_transform_sha256"],
        source=_source_from_payload(payload["source"]),
        split=_split_from_payload(payload["split"]),
        generation=_generation_from_payload(payload["generation"]),
        train_allowlist_sha256=payload["train_allowlist_sha256"],
        sample_content_sha256=payload["sample_content_sha256"],
        construction_provenance_sha256=payload[
            "construction_provenance_sha256"
        ],
        support=_support_from_payload(payload["support"]),
        label_provenance_sha256=payload["label_provenance_sha256"],
        assay_protocol_sha256=payload["assay_protocol_sha256"],
        measurement_receipt_sha256=payload["measurement_receipt_sha256"],
        mutation_axis=mutation_axis,
        mutation_spec_sha256=payload["mutation_spec_sha256"],
        paired_assay_receipt_sha256=payload[
            "paired_assay_receipt_sha256"
        ],
    )


def manifest_from_document(
    document: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
) -> HardNegativeManifest:
    """Verify an authority-anchored document and reconstruct typed records.

    ``expected_manifest_sha256`` must come from an immutable external authority,
    not from the untrusted document. A self-consistent embedded digest cannot
    authorize experimental evidence.
    """
    top_level = _require_exact_mapping(
        document,
        {
            "allowlist",
            "evidence_registry",
            "manifest_sha256",
            "manifest_type",
            "records",
            "schema_version",
        },
        "document",
    )
    expected = _require_sha256(
        top_level["manifest_sha256"],
        "manifest_sha256",
    )
    external_expected = _require_sha256(
        expected_manifest_sha256,
        "expected_manifest_sha256",
    )
    if expected != external_expected:
        raise ValueError(
            "Embedded manifest digest does not match the external authority"
        )
    payload = {
        key: value
        for key, value in top_level.items()
        if key != "manifest_sha256"
    }
    observed = _payload_sha256(payload, domain="manifest")
    if observed != expected:
        raise ValueError("Manifest SHA-256 verification failed")
    if top_level["manifest_type"] != "train_only_hard_negative_controls":
        raise ValueError("Unsupported manifest_type")
    if top_level["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("Unsupported hard-negative schema_version")
    raw_records = top_level["records"]
    if not isinstance(raw_records, list):
        raise TypeError("records must be a canonical JSON array")
    manifest = HardNegativeManifest(
        allowlist=_allowlist_from_payload(top_level["allowlist"]),
        records=tuple(_control_from_payload(record) for record in raw_records),
        evidence_registry=_evidence_registry_from_payload(
            top_level["evidence_registry"]
        ),
    )
    if manifest.canonical_bytes != _canonical_json_bytes(payload):
        raise ValueError(
            "Serialized manifest is not the canonical semantic representation"
        )
    if manifest.manifest_sha256 != expected:
        raise ValueError("Typed manifest SHA-256 does not match the document")
    return manifest


def verify_manifest_document(
    document: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
) -> str:
    """Verify an externally anchored, already parsed mapping.

    Use :func:`manifest_from_json_bytes` for untrusted serialized input so
    duplicate keys and noncanonical byte encodings cannot be hidden by a parser.
    """
    return manifest_from_document(
        document,
        expected_manifest_sha256=expected_manifest_sha256,
    ).manifest_sha256


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key {key!r}")
        result[key] = value
    return result


def manifest_from_json_bytes(
    document_bytes: bytes,
    *,
    expected_manifest_sha256: str,
) -> HardNegativeManifest:
    """Parse one byte-exact canonical JSON document without key ambiguity."""
    if type(document_bytes) is not bytes:
        raise TypeError("document_bytes must be immutable bytes")
    try:
        text = document_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Manifest is not valid UTF-8") from error

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r} is forbidden")

    try:
        document = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("Manifest is not valid JSON") from error
    if not isinstance(document, Mapping):
        raise TypeError("Manifest JSON root must be an object")
    manifest = manifest_from_document(
        document,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if document_bytes != _canonical_json_bytes(manifest.to_document()):
        raise ValueError("Manifest JSON bytes are not the canonical serialization")
    return manifest


__all__ = [
    "BindingEvidence",
    "ControlFamily",
    "ControlRecord",
    "GenerationBinding",
    "HardNegativeManifest",
    "PartnerAxis",
    "ResidueDomainSupport",
    "SourceBinding",
    "TrainOnlyAllowlist",
    "TrainOnlyEvidenceRegistry",
    "TrainOnlySplitBinding",
    "manifest_from_document",
    "manifest_from_json_bytes",
    "mutation_relation_binding_sha256",
    "verify_manifest_document",
]
