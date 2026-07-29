"""Fail-closed, train-only execution for prospective pair-geometry models.

This module deliberately separates model execution from distributed reduction.
It never initializes a process group and never reads a dataset. Callers must:

1. construct and seal an explicit train-only batch;
2. obtain an externally frozen batch authority bound to Gate 0 and provenance;
3. retain every microbatch graph in one nonempty accumulation window;
4. SUM-all-reduce the detached statistics/rank-presence packet externally;
5. obtain an externally frozen collective authority for all rank packets;
6. compose the DDP-correct local backward scalar with the reduced denominator;
7. perform exactly one optimizer/scaler step at the valid window boundary;
8. bind persisted checkpoint bytes to the post-step state receipt.

The local backward scalar multiplies each local numerator by ``world_size``.
Standard DistributedDataParallel averages parameter gradients, so its average
then equals the gradient of the global equal-per-complex objective. Averaging
rank-local means is never an accepted operation.

This reference implementation retains every microbatch autograd graph until
the complete window has been reduced and composed. Its activation memory
therefore grows with ``accumulation_steps``. It is an auditable mathematical
implementation, not the final high-scale systems path. Exact feature, model,
optimizer, scaler, and RNG hashing may also copy state to CPU and is unsuitable
for every hot-path microbatch in a large run. Production should
precompute denominators from sealed masks, SUM-all-reduce those denominators,
and then use DDP ``no_sync`` for streaming per-microbatch backward, with scaler
step/update only at the final boundary. This module does not initialize or
attest a process group, and its tests simulate collectives rather than launching
multi-process jobs. Rank-presence evidence detects reduction mistakes but is not
a security boundary against a caller that fabricates an entire reduced tensor.
The collective digest closes that audit gap only when its issuer and expected
digest are controlled outside the worker. Empty local-rank windows are rejected;
a production sampler must guarantee one microbatch per rank or use a separately
proven DDP-join protocol. Gate 0 hashes are bound here, but this module does not
authenticate their issuers.
Autograd ancestry proves that decoy scores touch the current model parameters;
it cannot by itself prove that an arbitrary custom scorer is semantically
identical to the matched-partner scorer.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterator, Mapping, Sequence

from .pair_geometry import ProspectivePairGeometryModel
from .pair_geometry_losses import (
    PairGeometryLossWeights,
    pair_geometry_objective,
)
from .pair_geometry_schema import PairGeometrySchema
from .hard_negative_contract import (
    BindingEvidence,
    ControlFamily,
    ControlRecord,
    HardNegativeManifest,
)


TRAINING_STEP_FORMAT_VERSION = "pairformer_train_only_step_v1"
SEED_RECEIPT_FORMAT_VERSION = "pairformer_deterministic_seed_receipt_v1"
SOURCE_RECEIPT_FORMAT_VERSION = "pairformer_source_receipt_v1"
OPTIMIZER_RECEIPT_FORMAT_VERSION = "pairformer_optimizer_step_receipt_v3"
REDUCED_PACKET_FORMAT_VERSION = "pairformer_sum_packet_v2"
REDUCTION_AUTHORITY_FORMAT_VERSION = "pairformer_sum_authority_v1"
BATCH_AUTHORITY_FORMAT_VERSION = "pairformer_train_batch_authority_v1"
CHECKPOINT_RECEIPT_FORMAT_VERSION = "pairformer_checkpoint_receipt_v1"
LOSS_TERM_ORDER = (
    "contact_bce",
    "distance_ce",
    "orientation_ce",
    "epitope_consistency",
    "paratope_consistency",
    "matched_decoy_ranking",
    "contact_brier",
)
ALLOWED_DECOY_CONTROL_FAMILIES = frozenset(
    family for family in ControlFamily if family is not ControlFamily.COGNATE
)
PADDING_DECOY_SEMANTIC_TYPE = "padding"
EXPERIMENTAL_LABEL_SEMANTIC_TYPE = "experimentally_resolved_pair_geometry"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
_HASH_DOMAIN_PREFIX = b"interaction_first:pairformer_training_step:v2\x00"


def _require_torch() -> Any:
    """Import PyTorch lazily so receipt inspection remains lightweight."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for PAIRFormer training") from error
    return torch


def _validate_canonical_json_value(value: Any, path: str = "$") -> None:
    """Reject values with ambiguous cross-runtime JSON representations."""
    value_type = type(value)
    if value is None or value_type is bool:
        return
    if value_type is int:
        if abs(value) > _MAX_SAFE_JSON_INTEGER:
            raise ValueError(f"{path} exceeds the interoperable JSON integer range")
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        if value == 0.0 and math.copysign(1.0, value) < 0.0:
            raise ValueError(f"{path} cannot contain negative zero")
        return
    if value_type is str:
        if unicodedata.normalize("NFC", value) != value:
            raise ValueError(f"{path} must use NFC-normalized Unicode")
        if any(
            unicodedata.category(character) in {"Cc", "Cf", "Cs"}
            for character in value
        ):
            raise ValueError(
                f"{path} cannot contain control, format, or surrogate characters"
            )
        return
    if value_type is list:
        for index, item in enumerate(value):
            _validate_canonical_json_value(item, f"{path}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be exact strings")
            _validate_canonical_json_value(key, f"{path}.<key>")
            _validate_canonical_json_value(item, f"{path}.{key}")
        return
    raise TypeError(
        f"{path} contains unsupported or subclassed JSON value {value_type.__name__}"
    )


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize strict finite metadata with a stable byte representation."""
    _validate_canonical_json_value(value)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Receipt metadata must be finite and JSON-compatible"
        ) from error
    return (encoded + "\n").encode("utf-8")


def _json_sha256(value: Any, *, domain: str = "generic") -> str:
    """Return a domain-separated digest for strict canonical JSON metadata."""
    if (
        type(domain) is not str
        or not domain
        or not domain.isascii()
        or domain.strip() != domain
    ):
        raise ValueError("hash domain must be a trimmed nonempty ASCII string")
    preimage = (
        _HASH_DOMAIN_PREFIX
        + domain.encode("ascii")
        + b"\x00"
        + _canonical_json_bytes(value)
    )
    return hashlib.sha256(preimage).hexdigest()


def _validate_sha256(value: Any, name: str) -> str:
    """Require one canonical lower-case SHA-256 digest."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    """Return a nonnegative non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    if value > _MAX_SAFE_JSON_INTEGER:
        raise ValueError(f"{name} exceeds the interoperable JSON integer range")
    return int(value)


def _positive_integer(value: Any, name: str) -> int:
    """Return a positive non-boolean integer."""
    converted = _nonnegative_integer(value, name)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _finite_nonnegative(value: Any, name: str) -> float:
    """Return a finite nonnegative scalar."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite nonnegative scalar") from error
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return converted


def _module_sha256(module_name: str) -> str:
    """Hash the exact imported module source used by one training process."""
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise RuntimeError(f"Cannot locate source for {module_name}")
    path = Path(module_file).resolve()
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"Cannot read source for {module_name}") from error
    return hashlib.sha256(payload).hexdigest()


def _tensor_descriptor(tensor: Any, name: str) -> dict[str, Any]:
    """Describe a tensor without serializing its potentially large payload."""
    torch = _require_torch()
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a PyTorch tensor")
    return {
        "dtype": str(tensor.dtype),
        "shape": [int(value) for value in tensor.shape],
    }


def tensor_sha256(tensor: Any, name: str = "tensor") -> str:
    """Hash one tensor's dtype, shape, and exact contiguous CPU bytes."""
    torch = _require_torch()
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a PyTorch tensor")
    detached = tensor.detach().cpu().contiguous()
    header = _canonical_json_bytes(_tensor_descriptor(detached, name))
    payload = detached.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(
        _HASH_DOMAIN_PREFIX + b"tensor\x00" + header + payload
    ).hexdigest()


@dataclass(frozen=True)
class TrainOnlyLabelProvenance:
    """Immutable evidence binding one training example to observed labels."""

    example_id: str
    source_role: str
    outer_role: str
    inner_role: str
    outer_fold: int
    inner_fold: int
    split_manifest_sha256: str
    train_membership_receipt_sha256: str
    source_semantic_type: str
    structure_artifact_sha256: str
    label_artifact_sha256: str
    label_builder_sha256: str
    geometry_schema_sha256: str
    example_train_membership_binding_sha256: str
    dataset_admission_receipt_sha256: str

    def validate(self) -> None:
        """Reject predicted labels, non-training roles, and malformed hashes."""
        if (
            not isinstance(self.example_id, str)
            or not self.example_id
            or self.example_id.strip() != self.example_id
        ):
            raise ValueError("Label provenance example_id must be a trimmed string")
        if (
            self.source_role != "train"
            or self.outer_role != "train"
            or self.inner_role != "train"
        ):
            raise ValueError(
                "Label provenance source_role/outer_role/inner_role must all "
                "equal 'train'"
            )
        _nonnegative_integer(self.outer_fold, "outer_fold")
        _nonnegative_integer(self.inner_fold, "inner_fold")
        _validate_sha256(self.split_manifest_sha256, "split_manifest_sha256")
        _validate_sha256(
            self.train_membership_receipt_sha256,
            "train_membership_receipt_sha256",
        )
        if self.source_semantic_type != EXPERIMENTAL_LABEL_SEMANTIC_TYPE:
            raise ValueError(
                "Label provenance must identify experimentally resolved geometry"
            )
        for name in (
            "structure_artifact_sha256",
            "label_artifact_sha256",
            "label_builder_sha256",
            "geometry_schema_sha256",
            "example_train_membership_binding_sha256",
            "dataset_admission_receipt_sha256",
        ):
            _validate_sha256(getattr(self, name), name)

    def canonical_metadata(self) -> dict[str, Any]:
        """Return the exact JSON-compatible provenance representation."""
        self.validate()
        return {
            "example_id": self.example_id,
            "source_role": self.source_role,
            "outer_role": self.outer_role,
            "inner_role": self.inner_role,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "split_manifest_sha256": self.split_manifest_sha256,
            "train_membership_receipt_sha256": (
                self.train_membership_receipt_sha256
            ),
            "source_semantic_type": self.source_semantic_type,
            "structure_artifact_sha256": self.structure_artifact_sha256,
            "label_artifact_sha256": self.label_artifact_sha256,
            "label_builder_sha256": self.label_builder_sha256,
            "geometry_schema_sha256": self.geometry_schema_sha256,
            "example_train_membership_binding_sha256": (
                self.example_train_membership_binding_sha256
            ),
            "dataset_admission_receipt_sha256": (
                self.dataset_admission_receipt_sha256
            ),
        }


@dataclass(frozen=True)
class TrainOnlyDecoyProvenance:
    """One manifest-bound hard-negative control used only for training."""

    semantic_type: ControlFamily
    binding_evidence: BindingEvidence
    source_role: str
    record_id: str
    match_group_id: str
    parent_cognate_id: str
    outer_fold: int
    inner_fold: int
    split_manifest_sha256: str
    train_membership_receipt_sha256: str
    train_allowlist_sha256: str
    hard_negative_manifest_sha256: str
    evidence_registry_sha256: str
    evidence_authority_receipt_sha256: str
    control_record_sha256: str
    source_artifact_sha256: str
    control_support_sha256: str
    measured_evidence_binding_sha256: str | None

    @classmethod
    def from_control_record(
        cls,
        record: ControlRecord,
        *,
        hard_negative_manifest: HardNegativeManifest,
        expected_manifest_sha256: str,
        expected_evidence_authority_receipt_sha256: str,
    ) -> "TrainOnlyDecoyProvenance":
        """Adapt a record only after checking externally frozen manifest evidence."""
        if not isinstance(record, ControlRecord):
            raise TypeError("record must be a canonical ControlRecord")
        if not isinstance(hard_negative_manifest, HardNegativeManifest):
            raise TypeError(
                "hard_negative_manifest must be a canonical HardNegativeManifest"
            )
        if record.control_family is ControlFamily.COGNATE:
            raise ValueError("A cognate record cannot be adapted as a decoy")
        expected_manifest_sha256 = _validate_sha256(
            expected_manifest_sha256,
            "expected_manifest_sha256",
        )
        expected_evidence_authority_receipt_sha256 = _validate_sha256(
            expected_evidence_authority_receipt_sha256,
            "expected_evidence_authority_receipt_sha256",
        )
        if hard_negative_manifest.manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "Hard-negative manifest does not match the external expected digest"
            )
        if (
            hard_negative_manifest.evidence_registry.authority_receipt_sha256
            != expected_evidence_authority_receipt_sha256
        ):
            raise ValueError(
                "Evidence registry does not match the external authority receipt"
            )
        manifest_records = tuple(
            value
            for value in hard_negative_manifest.records
            if value.record_id == record.record_id
        )
        if len(manifest_records) != 1 or manifest_records[0] != record:
            raise ValueError("Control record is not an exact member of the manifest")
        return cls(
            semantic_type=record.control_family,
            binding_evidence=record.binding_evidence,
            source_role="train",
            record_id=record.record_id,
            match_group_id=record.match_group_id,
            parent_cognate_id=str(record.parent_cognate_id),
            outer_fold=record.split.outer_fold,
            inner_fold=record.split.inner_fold,
            split_manifest_sha256=record.split.split_manifest_sha256,
            train_membership_receipt_sha256=(
                record.split.train_membership_receipt_sha256
            ),
            train_allowlist_sha256=record.train_allowlist_sha256,
            hard_negative_manifest_sha256=(
                hard_negative_manifest.manifest_sha256
            ),
            evidence_registry_sha256=(
                hard_negative_manifest.evidence_registry.registry_sha256
            ),
            evidence_authority_receipt_sha256=(
                hard_negative_manifest.evidence_registry.authority_receipt_sha256
            ),
            control_record_sha256=_json_sha256(
                record.to_payload(),
                domain="hard_negative_control_record",
            ),
            source_artifact_sha256=record.sample_content_sha256,
            control_support_sha256=record.support.support_sha256,
            measured_evidence_binding_sha256=(
                record.measured_evidence_binding_sha256
            ),
        )

    def validate(self) -> None:
        """Reject untyped, unmanifested, or non-training negative controls."""
        if (
            not isinstance(self.semantic_type, ControlFamily)
            or self.semantic_type not in ALLOWED_DECOY_CONTROL_FAMILIES
        ):
            raise ValueError("Decoy provenance semantic_type is not train-only")
        if not isinstance(self.binding_evidence, BindingEvidence):
            raise ValueError(
                "Decoy provenance binding_evidence must use the canonical enum"
            )
        if self.source_role != "train":
            raise ValueError("Decoy provenance source_role must equal 'train'")
        for name in ("record_id", "match_group_id", "parent_cognate_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or value.strip() != value
            ):
                raise ValueError(f"Decoy provenance {name} must be a trimmed string")
        _nonnegative_integer(self.outer_fold, "outer_fold")
        _nonnegative_integer(self.inner_fold, "inner_fold")
        for name in (
            "split_manifest_sha256",
            "train_membership_receipt_sha256",
            "train_allowlist_sha256",
            "hard_negative_manifest_sha256",
            "evidence_registry_sha256",
            "evidence_authority_receipt_sha256",
            "control_record_sha256",
            "source_artifact_sha256",
            "control_support_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if self.measured_evidence_binding_sha256 is not None:
            _validate_sha256(
                self.measured_evidence_binding_sha256,
                "measured_evidence_binding_sha256",
            )
        if self.binding_evidence is BindingEvidence.UNKNOWN_UNMEASURED:
            if self.measured_evidence_binding_sha256 is not None:
                raise ValueError(
                    "Unknown/unmeasured decoys cannot carry measured evidence"
                )
        else:
            if self.measured_evidence_binding_sha256 is None:
                raise ValueError(
                    "Measured decoys require an exact evidence binding"
                )
        if (
            self.semantic_type is ControlFamily.MEASURED_NON_BINDER
            or self.semantic_type is ControlFamily.MUTATION_FLIP
        ) and self.binding_evidence is not BindingEvidence.MEASURED_NON_BINDER:
            raise ValueError(
                "Measured-negative and mutation-flip controls require "
                "measured non-binder evidence"
            )
        if (
            self.semantic_type is ControlFamily.UNKNOWN_UNMEASURED_DECOY
            and self.binding_evidence is not BindingEvidence.UNKNOWN_UNMEASURED
        ):
            raise ValueError(
                "Unknown/unmeasured controls must remain unknown/unmeasured"
            )
        if (
            self.semantic_type
            in {
                ControlFamily.SAME_ANTIGEN_WRONG_ANTIBODY,
                ControlFamily.SAME_ANTIBODY_WRONG_ANTIGEN,
            }
            and self.binding_evidence is BindingEvidence.MEASURED_BINDER
        ):
            raise ValueError("A measured binder cannot be a wrong-partner control")
        if (
            self.semantic_type
            in {ControlFamily.WRONG_PATCH, ControlFamily.WRONG_POSE}
            and self.binding_evidence is not BindingEvidence.UNKNOWN_UNMEASURED
        ):
            raise ValueError(
                "Constructed patch/pose controls must remain unmeasured"
            )

    def canonical_metadata(self) -> dict[str, Any]:
        """Return exact decoy lineage for a training-step receipt."""
        self.validate()
        return {
            "semantic_type": self.semantic_type.value,
            "binding_evidence": self.binding_evidence.value,
            "source_role": self.source_role,
            "record_id": self.record_id,
            "match_group_id": self.match_group_id,
            "parent_cognate_id": self.parent_cognate_id,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "split_manifest_sha256": self.split_manifest_sha256,
            "train_membership_receipt_sha256": (
                self.train_membership_receipt_sha256
            ),
            "train_allowlist_sha256": self.train_allowlist_sha256,
            "hard_negative_manifest_sha256": (
                self.hard_negative_manifest_sha256
            ),
            "evidence_registry_sha256": self.evidence_registry_sha256,
            "evidence_authority_receipt_sha256": (
                self.evidence_authority_receipt_sha256
            ),
            "control_record_sha256": self.control_record_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "control_support_sha256": self.control_support_sha256,
            "measured_evidence_binding_sha256": (
                self.measured_evidence_binding_sha256
            ),
        }


@dataclass(frozen=True)
class PairFormerTrainBatch:
    """One explicit, manifest-bound PAIRFormer training microbatch."""

    role: str
    outer_fold: int
    inner_fold: int
    split_manifest_sha256: str
    batch_manifest_sha256: str
    feature_provenance_sha256: str
    geometry_schema_sha256: str
    candidate_mask_sha256: str
    batch_contract_sha256: str
    example_ids: tuple[str, ...]
    label_provenance: tuple[TrainOnlyLabelProvenance, ...]
    decoy_semantic_types: tuple[tuple[str, ...], ...]
    decoy_provenance: tuple[
        tuple[TrainOnlyDecoyProvenance | None, ...],
        ...,
    ]
    decoy_scorer_source_sha256: str | None
    antibody_features: Any
    antigen_features: Any
    antibody_mask: Any
    antigen_mask: Any
    candidate_pair_mask: Any
    contact_targets: Any | None = None
    contact_mask: Any | None = None
    distance_bin_targets: Any | None = None
    distance_mask: Any | None = None
    orientation_bin_targets: Any | None = None
    orientation_mask: Any | None = None
    epitope_node_probabilities: Any | None = None
    epitope_consistency_mask: Any | None = None
    paratope_node_probabilities: Any | None = None
    paratope_consistency_mask: Any | None = None
    decoy_scores: Any | None = None
    decoy_mask: Any | None = None


@dataclass(frozen=True)
class TrainOnlyBatchAuthority:
    """Externally frozen authority for one exact sealed training microbatch.

    The authority digest must be supplied by a launcher-controlled immutable
    ledger. Constructing this object from an untrusted batch and then accepting
    its self-computed digest does not establish authorization.
    """

    gate0_policy_sha256: str
    gate0_decision_sha256: str
    launcher_receipt_sha256: str
    batch_contract_sha256: str
    split_manifest_sha256: str
    train_membership_receipt_sha256s: tuple[str, ...]
    geometry_schema_sha256: str
    label_artifact_sha256s: tuple[str, ...]
    example_train_membership_binding_sha256s: tuple[str, ...]
    dataset_admission_receipt_sha256s: tuple[str, ...]
    hard_negative_manifest_sha256s: tuple[str, ...]
    hard_negative_evidence_registry_sha256s: tuple[str, ...]
    hard_negative_evidence_authority_sha256s: tuple[str, ...]
    format_version: str = BATCH_AUTHORITY_FORMAT_VERSION

    def canonical_metadata(self) -> dict[str, Any]:
        """Return the exact authority contract for external digest freezing."""
        if self.format_version != BATCH_AUTHORITY_FORMAT_VERSION:
            raise ValueError("Batch authority format version changed")
        result: dict[str, Any] = {"format_version": self.format_version}
        for name in (
            "gate0_policy_sha256",
            "gate0_decision_sha256",
            "launcher_receipt_sha256",
            "batch_contract_sha256",
            "split_manifest_sha256",
            "geometry_schema_sha256",
        ):
            result[name] = _validate_sha256(getattr(self, name), name)
        for name in (
            "train_membership_receipt_sha256s",
            "label_artifact_sha256s",
            "example_train_membership_binding_sha256s",
            "dataset_admission_receipt_sha256s",
            "hard_negative_manifest_sha256s",
            "hard_negative_evidence_registry_sha256s",
            "hard_negative_evidence_authority_sha256s",
        ):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or tuple(sorted(set(values))) != values
            ):
                raise ValueError(f"{name} must be a sorted unique tuple")
            for value in values:
                _validate_sha256(value, name)
            result[name] = list(values)
        if not self.train_membership_receipt_sha256s:
            raise ValueError("Batch authority needs train-membership evidence")
        if not self.label_artifact_sha256s:
            raise ValueError("Batch authority needs label-artifact evidence")
        if not self.example_train_membership_binding_sha256s:
            raise ValueError(
                "Batch authority needs exact example-membership evidence"
            )
        if not self.dataset_admission_receipt_sha256s:
            raise ValueError("Batch authority needs dataset-admission evidence")
        negative_groups = (
            self.hard_negative_manifest_sha256s,
            self.hard_negative_evidence_registry_sha256s,
            self.hard_negative_evidence_authority_sha256s,
        )
        if any(negative_groups) and not all(negative_groups):
            raise ValueError(
                "Hard-negative manifest, registry, and authority evidence "
                "must be jointly present"
            )
        return result

    @property
    def sha256(self) -> str:
        """Return the externally freezeable batch-authority digest."""
        return _json_sha256(
            self.canonical_metadata(),
            domain="train_only_batch_authority",
        )


def build_train_only_batch_authority(
    batch: PairFormerTrainBatch,
    schema: PairGeometrySchema,
    *,
    gate0_policy_sha256: str,
    gate0_decision_sha256: str,
    launcher_receipt_sha256: str,
) -> TrainOnlyBatchAuthority:
    """Build authority metadata for independent freezing after batch sealing."""
    validate_train_only_batch(batch, schema)
    label_values = batch.label_provenance
    decoys = tuple(
        value
        for row in batch.decoy_provenance
        for value in row
        if value is not None
    )
    authority = TrainOnlyBatchAuthority(
        gate0_policy_sha256=_validate_sha256(
            gate0_policy_sha256,
            "gate0_policy_sha256",
        ),
        gate0_decision_sha256=_validate_sha256(
            gate0_decision_sha256,
            "gate0_decision_sha256",
        ),
        launcher_receipt_sha256=_validate_sha256(
            launcher_receipt_sha256,
            "launcher_receipt_sha256",
        ),
        batch_contract_sha256=batch.batch_contract_sha256,
        split_manifest_sha256=batch.split_manifest_sha256,
        train_membership_receipt_sha256s=tuple(
            sorted(
                {
                    value.train_membership_receipt_sha256
                    for value in label_values
                }
                | {
                    value.train_membership_receipt_sha256
                    for value in decoys
                }
            )
        ),
        geometry_schema_sha256=batch.geometry_schema_sha256,
        label_artifact_sha256s=tuple(
            sorted({value.label_artifact_sha256 for value in label_values})
        ),
        example_train_membership_binding_sha256s=tuple(
            sorted(
                {
                    value.example_train_membership_binding_sha256
                    for value in label_values
                }
            )
        ),
        dataset_admission_receipt_sha256s=tuple(
            sorted(
                {
                    value.dataset_admission_receipt_sha256
                    for value in label_values
                }
            )
        ),
        hard_negative_manifest_sha256s=tuple(
            sorted({value.hard_negative_manifest_sha256 for value in decoys})
        ),
        hard_negative_evidence_registry_sha256s=tuple(
            sorted({value.evidence_registry_sha256 for value in decoys})
        ),
        hard_negative_evidence_authority_sha256s=tuple(
            sorted(
                {
                    value.evidence_authority_receipt_sha256
                    for value in decoys
                }
            )
        ),
    )
    authority.canonical_metadata()
    return authority


def validate_train_only_batch_authority(
    batch: PairFormerTrainBatch,
    schema: PairGeometrySchema,
    authority: TrainOnlyBatchAuthority,
    *,
    expected_authority_sha256: str,
) -> None:
    """Require an externally expected authority for the exact sealed batch."""
    if not isinstance(authority, TrainOnlyBatchAuthority):
        raise TypeError("authority must be TrainOnlyBatchAuthority")
    observed = build_train_only_batch_authority(
        batch,
        schema,
        gate0_policy_sha256=authority.gate0_policy_sha256,
        gate0_decision_sha256=authority.gate0_decision_sha256,
        launcher_receipt_sha256=authority.launcher_receipt_sha256,
    )
    expected_authority_sha256 = _validate_sha256(
        expected_authority_sha256,
        "expected_authority_sha256",
    )
    if authority != observed:
        raise ValueError("Batch authority does not match the exact sealed batch")
    if authority.sha256 != expected_authority_sha256:
        raise ValueError(
            "Batch authority does not match the external expected digest"
        )


def _batch_size(batch: PairFormerTrainBatch) -> int:
    """Return the explicit identifier count for one batch."""
    return len(batch.example_ids)


def _validate_optional_tensor_pair(
    left: Any | None,
    right: Any | None,
    left_name: str,
    right_name: str,
) -> None:
    """Require optional target/mask values to be jointly present or absent."""
    torch = _require_torch()
    if (left is None) != (right is None):
        raise ValueError(f"{left_name} and {right_name} must be jointly present")
    for value, name in ((left, left_name), (right, right_name)):
        if value is not None and not torch.is_tensor(value):
            raise TypeError(f"{name} must be a PyTorch tensor")


def _validate_batch_metadata(
    batch: PairFormerTrainBatch,
    schema: PairGeometrySchema,
) -> None:
    """Validate role, fold, provenance, schema, and decoy semantics."""
    torch = _require_torch()
    if not isinstance(batch, PairFormerTrainBatch):
        raise TypeError("batch must be PairFormerTrainBatch")
    if batch.role != "train":
        raise ValueError("PAIRFormer execution accepts role='train' only")
    _nonnegative_integer(batch.outer_fold, "outer_fold")
    _nonnegative_integer(batch.inner_fold, "inner_fold")
    for name in (
        "split_manifest_sha256",
        "batch_manifest_sha256",
        "feature_provenance_sha256",
        "geometry_schema_sha256",
    ):
        _validate_sha256(getattr(batch, name), name)
    if batch.geometry_schema_sha256 != schema.sha256:
        raise ValueError("Batch geometry schema does not match the model schema")
    batch_size = _batch_size(batch)
    if batch_size <= 0:
        raise ValueError("example_ids must be nonempty")
    if len(set(batch.example_ids)) != batch_size:
        raise ValueError("example_ids must be unique within a microbatch")
    if any(
        not isinstance(value, str) or not value or value.strip() != value
        for value in batch.example_ids
    ):
        raise ValueError("example_ids must contain trimmed nonempty strings")
    if len(batch.label_provenance) != batch_size:
        raise ValueError("label_provenance must align with example_ids")
    for index, provenance in enumerate(batch.label_provenance):
        if not isinstance(provenance, TrainOnlyLabelProvenance):
            raise TypeError("label_provenance entries must use the locked receipt")
        provenance.validate()
        if provenance.example_id != batch.example_ids[index]:
            raise ValueError("label_provenance order must match example_ids")
        if (
            provenance.outer_fold != batch.outer_fold
            or provenance.inner_fold != batch.inner_fold
            or provenance.split_manifest_sha256 != batch.split_manifest_sha256
        ):
            raise ValueError(
                "Label provenance must match the exact batch split and folds"
            )
        if provenance.geometry_schema_sha256 != batch.geometry_schema_sha256:
            raise ValueError(
                "Label provenance must bind the exact batch geometry schema"
            )
    if len(batch.decoy_semantic_types) != batch_size:
        raise ValueError("decoy_semantic_types must align with example_ids")
    if len(batch.decoy_provenance) != batch_size:
        raise ValueError("decoy_provenance must align with example_ids")
    if any(
        not isinstance(row, tuple) or any(not isinstance(value, str) for value in row)
        for row in batch.decoy_semantic_types
    ):
        raise TypeError("decoy_semantic_types must be immutable string tuples")
    if any(not isinstance(row, tuple) for row in batch.decoy_provenance):
        raise TypeError("decoy_provenance rows must be immutable tuples")

    for name in (
        "antibody_features",
        "antigen_features",
        "antibody_mask",
        "antigen_mask",
        "candidate_pair_mask",
    ):
        if not torch.is_tensor(getattr(batch, name)):
            raise TypeError(f"{name} must be a PyTorch tensor")
    if batch.antibody_features.ndim != 3 or batch.antigen_features.ndim != 3:
        raise ValueError("Feature tensors must be rank-three batch tensors")
    if (
        not torch.is_floating_point(batch.antibody_features)
        or not torch.is_floating_point(batch.antigen_features)
        or batch.antibody_features.dtype != batch.antigen_features.dtype
    ):
        raise ValueError("Feature tensors must share one floating dtype")
    if (
        not bool(torch.isfinite(batch.antibody_features).all().item())
        or not bool(torch.isfinite(batch.antigen_features).all().item())
    ):
        raise ValueError("Feature tensors must contain only finite values")
    if int(batch.antibody_features.shape[0]) != batch_size or int(
        batch.antigen_features.shape[0]
    ) != batch_size:
        raise ValueError("Feature batch axes must align with example_ids")
    if tuple(batch.antibody_mask.shape) != tuple(batch.antibody_features.shape[:2]):
        raise ValueError("antibody_mask must align with antibody_features")
    if tuple(batch.antigen_mask.shape) != tuple(batch.antigen_features.shape[:2]):
        raise ValueError("antigen_mask must align with antigen_features")
    common_device = batch.antibody_features.device
    if (
        batch.antigen_features.device != common_device
        or batch.antibody_mask.device != common_device
        or batch.antigen_mask.device != common_device
        or batch.antibody_mask.dtype != torch.bool
        or batch.antigen_mask.dtype != torch.bool
    ):
        raise ValueError("Features and boolean residue masks must share one device")
    expected_candidate_shape = (
        batch_size,
        int(batch.antibody_features.shape[1]),
        int(batch.antigen_features.shape[1]),
    )
    if (
        batch.candidate_pair_mask.dtype != torch.bool
        or tuple(batch.candidate_pair_mask.shape) != expected_candidate_shape
        or batch.candidate_pair_mask.device != common_device
    ):
        raise ValueError(
            "candidate_pair_mask must be boolean [batch, antibody, antigen]"
        )
    residue_cartesian_support = batch.antibody_mask.unsqueeze(2) & (
        batch.antigen_mask.unsqueeze(1)
    )
    if bool((batch.candidate_pair_mask & ~residue_cartesian_support).any().item()):
        raise ValueError(
            "candidate_pair_mask cannot include padded or unsupported residues"
        )

    for left, right, left_name, right_name in (
        (
            batch.contact_targets,
            batch.contact_mask,
            "contact_targets",
            "contact_mask",
        ),
        (
            batch.distance_bin_targets,
            batch.distance_mask,
            "distance_bin_targets",
            "distance_mask",
        ),
        (
            batch.orientation_bin_targets,
            batch.orientation_mask,
            "orientation_bin_targets",
            "orientation_mask",
        ),
        (
            batch.epitope_node_probabilities,
            batch.epitope_consistency_mask,
            "epitope_node_probabilities",
            "epitope_consistency_mask",
        ),
        (
            batch.paratope_node_probabilities,
            batch.paratope_consistency_mask,
            "paratope_node_probabilities",
            "paratope_consistency_mask",
        ),
        (
            batch.decoy_scores,
            batch.decoy_mask,
            "decoy_scores",
            "decoy_mask",
        ),
    ):
        _validate_optional_tensor_pair(left, right, left_name, right_name)

    if batch.decoy_scores is None:
        if any(row for row in batch.decoy_semantic_types):
            raise ValueError("Decoy semantics require aligned decoy tensors")
        if any(row for row in batch.decoy_provenance):
            raise ValueError("Decoy provenance requires aligned decoy tensors")
        if batch.decoy_scorer_source_sha256 is not None:
            raise ValueError("A decoy scorer receipt requires aligned decoy tensors")
    else:
        _validate_sha256(
            batch.decoy_scorer_source_sha256,
            "decoy_scorer_source_sha256",
        )
        if (
            not torch.is_floating_point(batch.decoy_scores)
            or not bool(torch.isfinite(batch.decoy_scores).all().item())
        ):
            raise ValueError("decoy_scores must be finite and floating")
        if batch.decoy_scores.ndim != 2 or tuple(batch.decoy_scores.shape) != tuple(
            batch.decoy_mask.shape
        ):
            raise ValueError("decoy_scores and decoy_mask must be aligned matrices")
        if (
            batch.decoy_mask.dtype != torch.bool
            or int(batch.decoy_scores.shape[0]) != batch_size
        ):
            raise ValueError("decoy_mask must be boolean and batch-aligned")
        width = int(batch.decoy_scores.shape[1])
        if width <= 0:
            raise ValueError("Decoy tensors must contain at least one slot")
        if any(len(row) != width for row in batch.decoy_semantic_types):
            raise ValueError("Decoy semantic rows must align with decoy width")
        if any(len(row) != width for row in batch.decoy_provenance):
            raise ValueError("Decoy provenance rows must align with decoy width")
        batch_manifest_sha256s: set[str] = set()
        batch_allowlist_sha256s: set[str] = set()
        batch_membership_sha256s: set[str] = set()
        batch_evidence_registry_sha256s: set[str] = set()
        batch_evidence_authority_sha256s: set[str] = set()
        active_record_ids: set[str] = set()
        for row_index, row in enumerate(batch.decoy_semantic_types):
            active_group_ids: set[str] = set()
            active_manifest_sha256s: set[str] = set()
            for column_index, semantic_type in enumerate(row):
                active = bool(batch.decoy_mask[row_index, column_index].item())
                provenance = batch.decoy_provenance[row_index][column_index]
                if active:
                    try:
                        control_family = ControlFamily(semantic_type)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            "Active decoys require one canonical ControlFamily"
                        ) from error
                    if control_family not in ALLOWED_DECOY_CONTROL_FAMILIES:
                        raise ValueError(
                            "Cognate records cannot occupy active decoy slots"
                        )
                if not active and semantic_type != PADDING_DECOY_SEMANTIC_TYPE:
                    raise ValueError("Inactive decoy slots must use padding semantics")
                if active:
                    if not isinstance(provenance, TrainOnlyDecoyProvenance):
                        raise ValueError(
                            "Every active decoy requires manifest-bound provenance"
                        )
                    provenance.validate()
                    if provenance.semantic_type is not control_family:
                        raise ValueError(
                            "Decoy provenance must match canonical ControlFamily "
                            "and order"
                        )
                    if (
                        provenance.outer_fold != batch.outer_fold
                        or provenance.inner_fold != batch.inner_fold
                        or provenance.split_manifest_sha256
                        != batch.split_manifest_sha256
                    ):
                        raise ValueError(
                            "Decoy provenance must match the exact batch split "
                            "and folds"
                        )
                    if provenance.parent_cognate_id != batch.example_ids[row_index]:
                        raise ValueError(
                            "Every active decoy must identify its aligned batch "
                            "example as parent cognate"
                        )
                    active_group_ids.add(provenance.match_group_id)
                    active_manifest_sha256s.add(
                        provenance.hard_negative_manifest_sha256
                    )
                    if provenance.record_id in active_record_ids:
                        raise ValueError(
                            "A hard-negative record cannot be replayed in one batch"
                        )
                    active_record_ids.add(provenance.record_id)
                    batch_manifest_sha256s.add(
                        provenance.hard_negative_manifest_sha256
                    )
                    batch_allowlist_sha256s.add(
                        provenance.train_allowlist_sha256
                    )
                    batch_membership_sha256s.add(
                        provenance.train_membership_receipt_sha256
                    )
                    batch_evidence_registry_sha256s.add(
                        provenance.evidence_registry_sha256
                    )
                    batch_evidence_authority_sha256s.add(
                        provenance.evidence_authority_receipt_sha256
                    )
                elif provenance is not None:
                    raise ValueError("Padding decoy slots cannot carry provenance")
            if len(active_group_ids) > 1:
                raise ValueError(
                    "All active controls for one cognate must share a match group"
                )
            if len(active_manifest_sha256s) > 1:
                raise ValueError(
                    "All active controls for one cognate must share one manifest"
                )
        if len(batch_manifest_sha256s) > 1:
            raise ValueError("One training batch cannot mix hard-negative manifests")
        if len(batch_allowlist_sha256s) > 1:
            raise ValueError("One training batch cannot mix train-only allowlists")
        if len(batch_membership_sha256s) > 1:
            raise ValueError(
                "One training batch cannot mix train-membership receipts"
            )
        if len(batch_evidence_registry_sha256s) > 1:
            raise ValueError(
                "One training batch cannot mix hard-negative evidence registries"
            )
        if len(batch_evidence_authority_sha256s) > 1:
            raise ValueError(
                "One training batch cannot mix evidence-authority receipts"
            )


def _contract_tensor_hashes(batch: PairFormerTrainBatch) -> dict[str, Any]:
    """Hash every consumed tensor, including potentially large feature payloads."""
    hashes: dict[str, Any] = {}
    for name in (
        "antibody_features",
        "antigen_features",
        "antibody_mask",
        "antigen_mask",
        "candidate_pair_mask",
        "contact_targets",
        "contact_mask",
        "distance_bin_targets",
        "distance_mask",
        "orientation_bin_targets",
        "orientation_mask",
        "epitope_node_probabilities",
        "epitope_consistency_mask",
        "paratope_node_probabilities",
        "paratope_consistency_mask",
        "decoy_scores",
        "decoy_mask",
    ):
        value = getattr(batch, name)
        hashes[name] = None if value is None else tensor_sha256(value, name)
    return hashes


def canonical_train_batch_contract(
    batch: PairFormerTrainBatch,
    schema: PairGeometrySchema,
    *,
    candidate_mask_sha256: str | None = None,
) -> dict[str, Any]:
    """Return the immutable metadata contract for one prepared microbatch."""
    _validate_batch_metadata(batch, schema)
    candidate_digest = (
        tensor_sha256(batch.candidate_pair_mask, "candidate_pair_mask")
        if candidate_mask_sha256 is None
        else _validate_sha256(candidate_mask_sha256, "candidate_mask_sha256")
    )
    return {
        "format_version": TRAINING_STEP_FORMAT_VERSION,
        "role": batch.role,
        "outer_fold": batch.outer_fold,
        "inner_fold": batch.inner_fold,
        "split_manifest_sha256": batch.split_manifest_sha256,
        "batch_manifest_sha256": batch.batch_manifest_sha256,
        "feature_provenance_sha256": batch.feature_provenance_sha256,
        "geometry_schema_sha256": batch.geometry_schema_sha256,
        "candidate_mask_sha256": candidate_digest,
        "example_ids": list(batch.example_ids),
        "label_provenance": [
            value.canonical_metadata() for value in batch.label_provenance
        ],
        "decoy_semantic_types": [
            list(values) for values in batch.decoy_semantic_types
        ],
        "decoy_provenance": [
            [
                None if value is None else value.canonical_metadata()
                for value in row
            ]
            for row in batch.decoy_provenance
        ],
        "decoy_scorer_source_sha256": batch.decoy_scorer_source_sha256,
        "tensor_descriptors": {
            name: _tensor_descriptor(getattr(batch, name), name)
            for name in (
                "antibody_features",
                "antigen_features",
                "antibody_mask",
                "antigen_mask",
                "candidate_pair_mask",
            )
        },
        "supervision_tensor_sha256": _contract_tensor_hashes(batch),
    }


def seal_train_only_batch(
    batch: PairFormerTrainBatch,
    schema: PairGeometrySchema,
) -> PairFormerTrainBatch:
    """Return a copy carrying hashes computed from the supplied train tensors."""
    if not isinstance(schema, PairGeometrySchema):
        raise TypeError("schema must be PairGeometrySchema")
    _validate_batch_metadata(batch, schema)
    candidate_digest = tensor_sha256(
        batch.candidate_pair_mask,
        "candidate_pair_mask",
    )
    provisional = replace(batch, candidate_mask_sha256=candidate_digest)
    contract = canonical_train_batch_contract(
        provisional,
        schema,
        candidate_mask_sha256=candidate_digest,
    )
    return replace(provisional, batch_contract_sha256=_json_sha256(contract))


def validate_train_only_batch(
    batch: PairFormerTrainBatch,
    schema: PairGeometrySchema,
) -> None:
    """Fail unless all metadata and consumed tensor bytes match the seal.

    This intentionally re-hashes feature tensors and may copy them to CPU. A
    production loader should seal/validate before device transfer and retain
    the immutable batch object; silently trusting a caller-supplied feature
    digest is not an accepted integrity mode.
    """
    if not isinstance(schema, PairGeometrySchema):
        raise TypeError("schema must be PairGeometrySchema")
    _validate_batch_metadata(batch, schema)
    _validate_sha256(batch.candidate_mask_sha256, "candidate_mask_sha256")
    _validate_sha256(batch.batch_contract_sha256, "batch_contract_sha256")
    observed_candidate_digest = tensor_sha256(
        batch.candidate_pair_mask,
        "candidate_pair_mask",
    )
    if observed_candidate_digest != batch.candidate_mask_sha256:
        raise ValueError("candidate_pair_mask does not match its sealed digest")
    observed_contract = _json_sha256(
        canonical_train_batch_contract(
            batch,
            schema,
            candidate_mask_sha256=observed_candidate_digest,
        )
    )
    if observed_contract != batch.batch_contract_sha256:
        raise ValueError("Training batch content does not match its sealed contract")


@dataclass(frozen=True)
class AutocastPolicy:
    """Explicit autocast configuration; gradient scaling remains separate."""

    enabled: bool = False
    device_type: str = "cpu"
    dtype_name: str = "bfloat16"
    cache_enabled: bool = True

    def validate(self, tensor_device_type: str) -> None:
        """Reject implicit devices, unsupported dtypes, and mismatched inputs."""
        if not isinstance(self.enabled, bool) or not isinstance(
            self.cache_enabled,
            bool,
        ):
            raise ValueError("Autocast enabled/cache_enabled must be boolean")
        if self.device_type not in {"cpu", "cuda"}:
            raise ValueError("Autocast device_type must be 'cpu' or 'cuda'")
        if self.dtype_name not in {"bfloat16", "float16"}:
            raise ValueError("Autocast dtype_name must be 'bfloat16' or 'float16'")
        if self.device_type != tensor_device_type:
            raise ValueError("Autocast device_type must match model input tensors")

    def context(self, tensor_device_type: str) -> Any:
        """Return an autocast context without constructing a scaler."""
        torch = _require_torch()
        self.validate(tensor_device_type)
        if not self.enabled:
            return nullcontext()
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype_name]
        return torch.autocast(
            device_type=self.device_type,
            dtype=dtype,
            enabled=True,
            cache_enabled=self.cache_enabled,
        )


@dataclass(frozen=True)
class DeterministicSeedReceipt:
    """Hash-bound seed used for exactly one microbatch forward graph."""

    base_seed: int
    optimizer_step: int
    accumulation_index: int
    rank: int
    batch_contract_sha256: str
    run_manifest_sha256: str
    checkpoint_parent_sha256: str
    derived_seed: int
    format_version: str = SEED_RECEIPT_FORMAT_VERSION

    def canonical_metadata(self) -> dict[str, Any]:
        """Return validated metadata suitable for an immutable ledger."""
        base_seed = _nonnegative_integer(self.base_seed, "base_seed")
        optimizer_step = _nonnegative_integer(
            self.optimizer_step,
            "optimizer_step",
        )
        accumulation_index = _nonnegative_integer(
            self.accumulation_index,
            "accumulation_index",
        )
        rank = _nonnegative_integer(self.rank, "rank")
        batch_contract_sha256 = _validate_sha256(
            self.batch_contract_sha256,
            "batch_contract_sha256",
        )
        run_manifest_sha256 = _validate_sha256(
            self.run_manifest_sha256,
            "run_manifest_sha256",
        )
        checkpoint_parent_sha256 = _validate_sha256(
            self.checkpoint_parent_sha256,
            "checkpoint_parent_sha256",
        )
        if (
            isinstance(self.derived_seed, bool)
            or not isinstance(self.derived_seed, int)
            or self.derived_seed < 0
        ):
            raise ValueError("derived_seed must be a nonnegative integer")
        derived_seed = int(self.derived_seed)
        if derived_seed >= 2**63:
            raise ValueError("derived_seed must fit in a signed 63-bit range")
        if self.format_version != SEED_RECEIPT_FORMAT_VERSION:
            raise ValueError("Seed receipt format version changed")
        metadata = {
            "format_version": self.format_version,
            "base_seed": base_seed,
            "optimizer_step": optimizer_step,
            "accumulation_index": accumulation_index,
            "rank": rank,
            "batch_contract_sha256": batch_contract_sha256,
            "run_manifest_sha256": run_manifest_sha256,
            "checkpoint_parent_sha256": checkpoint_parent_sha256,
        }
        expected_seed = int.from_bytes(
            hashlib.sha256(_canonical_json_bytes(metadata)).digest()[:8],
            "big",
        ) % (2**63)
        if derived_seed != expected_seed:
            raise ValueError("derived_seed does not match the receipt coordinates")
        metadata["derived_seed_decimal"] = str(derived_seed)
        return metadata

    @property
    def sha256(self) -> str:
        """Return the receipt's canonical digest."""
        return _json_sha256(self.canonical_metadata())


def derive_seed_receipt(
    *,
    base_seed: int,
    optimizer_step: int,
    accumulation_index: int,
    rank: int,
    batch_contract_sha256: str,
    run_manifest_sha256: str,
    checkpoint_parent_sha256: str,
) -> DeterministicSeedReceipt:
    """Derive a platform-stable seed from immutable training coordinates."""
    metadata = {
        "format_version": SEED_RECEIPT_FORMAT_VERSION,
        "base_seed": _nonnegative_integer(base_seed, "base_seed"),
        "optimizer_step": _nonnegative_integer(optimizer_step, "optimizer_step"),
        "accumulation_index": _nonnegative_integer(
            accumulation_index,
            "accumulation_index",
        ),
        "rank": _nonnegative_integer(rank, "rank"),
        "batch_contract_sha256": _validate_sha256(
            batch_contract_sha256,
            "batch_contract_sha256",
        ),
        "run_manifest_sha256": _validate_sha256(
            run_manifest_sha256,
            "run_manifest_sha256",
        ),
        "checkpoint_parent_sha256": _validate_sha256(
            checkpoint_parent_sha256,
            "checkpoint_parent_sha256",
        ),
    }
    derived = int.from_bytes(
        hashlib.sha256(_canonical_json_bytes(metadata)).digest()[:8],
        "big",
    ) % (2**63)
    return DeterministicSeedReceipt(
        base_seed=metadata["base_seed"],
        optimizer_step=metadata["optimizer_step"],
        accumulation_index=metadata["accumulation_index"],
        rank=metadata["rank"],
        batch_contract_sha256=metadata["batch_contract_sha256"],
        run_manifest_sha256=metadata["run_manifest_sha256"],
        checkpoint_parent_sha256=metadata["checkpoint_parent_sha256"],
        derived_seed=derived,
    )


@contextmanager
def deterministic_seed_scope(
    receipt: DeterministicSeedReceipt,
    *,
    device: Any,
) -> Iterator[None]:
    """Execute one forward graph without leaking RNG changes to its caller."""
    torch = _require_torch()
    receipt.canonical_metadata()
    device = torch.device(device)
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device() if device.index is None else int(device.index)
        ]
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(receipt.derived_seed)
        yield


@dataclass(frozen=True)
class SourceCodeReceipt:
    """Exact source hashes for the schema/model/loss/training implementation."""

    schema_source_sha256: str
    model_source_sha256: str
    objective_source_sha256: str
    hard_negative_contract_source_sha256: str
    training_step_source_sha256: str
    format_version: str = SOURCE_RECEIPT_FORMAT_VERSION

    @classmethod
    def current(cls) -> "SourceCodeReceipt":
        """Hash the imported implementation directly from disk."""
        return cls(
            schema_source_sha256=_module_sha256(
                "interaction_first.pair_geometry_schema"
            ),
            model_source_sha256=_module_sha256("interaction_first.pair_geometry"),
            objective_source_sha256=_module_sha256(
                "interaction_first.pair_geometry_losses"
            ),
            hard_negative_contract_source_sha256=_module_sha256(
                "interaction_first.hard_negative_contract"
            ),
            training_step_source_sha256=_module_sha256(__name__),
        )

    def canonical_metadata(self) -> dict[str, Any]:
        """Return validated immutable source metadata."""
        if self.format_version != SOURCE_RECEIPT_FORMAT_VERSION:
            raise ValueError("Source receipt format version changed")
        result = {"format_version": self.format_version}
        for name in (
            "schema_source_sha256",
            "model_source_sha256",
            "objective_source_sha256",
            "hard_negative_contract_source_sha256",
            "training_step_source_sha256",
        ):
            result[name] = _validate_sha256(getattr(self, name), name)
        return result

    @property
    def sha256(self) -> str:
        """Return the canonical source-receipt digest."""
        return _json_sha256(self.canonical_metadata())


def _weights_tuple(weights: PairGeometryLossWeights) -> tuple[float, ...]:
    """Return loss weights in the fixed distributed packet order."""
    if not isinstance(weights, PairGeometryLossWeights):
        raise TypeError("weights must be PairGeometryLossWeights")
    weights.validate()
    return tuple(float(getattr(weights, name)) for name in LOSS_TERM_ORDER)


def _valid_complex_count(mask: Any, name: str, batch_size: int) -> int:
    """Count complexes with at least one supported supervised entry."""
    torch = _require_torch()
    if not torch.is_tensor(mask) or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor")
    if mask.ndim < 2 or int(mask.shape[0]) != batch_size:
        raise ValueError(f"{name} must have an aligned batch axis")
    return int((mask.reshape(batch_size, -1).sum(dim=1) > 0).sum().item())


def _pairformer_core(model: Any) -> ProspectivePairGeometryModel:
    """Return the PAIRFormer core from a direct model or one DDP-like wrapper."""
    if isinstance(model, ProspectivePairGeometryModel):
        return model
    wrapped = getattr(model, "module", None)
    if isinstance(wrapped, ProspectivePairGeometryModel):
        return wrapped
    raise TypeError(
        "model must be ProspectivePairGeometryModel or a wrapper exposing it "
        "as .module"
    )


def _model_parameter_binding(
    model: Any,
) -> tuple[tuple[int, ...], str]:
    """Bind an in-process parameter identity set and a portable layout digest."""
    parameters: list[int] = []
    layout: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameters.append(id(parameter))
            layout.append(
                {
                    "name": name,
                    "dtype": str(parameter.dtype),
                    "shape": [int(value) for value in parameter.shape],
                }
            )
    if not parameters or len(set(parameters)) != len(parameters):
        raise ValueError("Model must expose unique trainable parameters")
    return tuple(sorted(parameters)), _json_sha256(layout)


def _state_value_contract(value: Any, path: str) -> Any:
    """Convert a tensor state tree into strict canonical hash metadata."""
    torch = _require_torch()
    if torch.is_tensor(value):
        return {
            "kind": "tensor",
            "sha256": tensor_sha256(value, path),
            "descriptor": _tensor_descriptor(value, path),
        }
    if value is None or type(value) in {bool, int, float, str}:
        _validate_canonical_json_value(value, path)
        return {"kind": "scalar", "value": value}
    if type(value) in {list, tuple}:
        return {
            "kind": "sequence",
            "sequence_type": "tuple" if type(value) is tuple else "list",
            "items": [
                _state_value_contract(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if type(value) is dict:
        items: list[dict[str, Any]] = []
        for key, item in value.items():
            if type(key) not in {str, int}:
                raise TypeError(
                    f"{path} has unsupported state key type {type(key).__name__}"
                )
            if type(key) is int:
                _nonnegative_integer(key, f"{path} key")
                key_contract = {"type": "int", "value": key}
            else:
                _validate_canonical_json_value(key, f"{path} key")
                key_contract = {"type": "str", "value": key}
            items.append(
                {
                    "key": key_contract,
                    "value": _state_value_contract(item, f"{path}[{key!r}]"),
                }
            )
        items.sort(
            key=lambda item: _canonical_json_bytes(item["key"])
        )
        return {"kind": "mapping", "items": items}
    raise TypeError(
        f"{path} contains unsupported state value {type(value).__name__}"
    )


def _state_tree_sha256(value: Any, *, domain: str, path: str) -> str:
    """Hash a strict tensor/scalar state tree without pickle serialization."""
    return _json_sha256(
        _state_value_contract(value, path),
        domain=domain,
    )


def _model_state_sha256(model: Any) -> str:
    """Hash the exact named parameter and persistent-buffer state."""
    state = model.state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("model.state_dict() must return a mapping")
    if any(type(name) is not str for name in state):
        raise TypeError("Model state keys must be exact strings")
    normalized = dict(state)
    return _state_tree_sha256(
        normalized,
        domain="model_state",
        path="model_state",
    )


def _optimizer_state_sha256(optimizer: Any) -> str:
    """Hash the exact portable optimizer state_dict tree."""
    state_dict = getattr(optimizer, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("optimizer must implement state_dict()")
    return _state_tree_sha256(
        state_dict(),
        domain="optimizer_state",
        path="optimizer_state",
    )


def _scaler_state_sha256(grad_scaler: Any | None) -> str | None:
    """Hash a GradScaler state tree, or return null when scaling is absent."""
    if grad_scaler is None:
        return None
    state_dict = getattr(grad_scaler, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("grad_scaler must implement state_dict() for receipts")
    return _state_tree_sha256(
        state_dict(),
        domain="grad_scaler_state",
        path="grad_scaler_state",
    )


def _rng_state_sha256(parameters: Sequence[Any]) -> str:
    """Hash CPU and every parameter-owning CUDA device RNG state."""
    torch = _require_torch()
    payload: dict[str, Any] = {
        "cpu": tensor_sha256(torch.random.get_rng_state(), "cpu_rng_state"),
        "cuda": [],
    }
    cuda_indices = sorted(
        {
            (
                torch.cuda.current_device()
                if parameter.device.index is None
                else int(parameter.device.index)
            )
            for parameter in parameters
            if parameter.device.type == "cuda"
        }
    )
    for index in cuda_indices:
        payload["cuda"].append(
            {
                "device_index": index,
                "sha256": tensor_sha256(
                    torch.cuda.get_rng_state(index),
                    f"cuda_rng_state_{index}",
                ),
            }
        )
    return _json_sha256(payload, domain="rng_state")


def _autograd_parameter_ids(tensor: Any) -> frozenset[int]:
    """Return leaf parameter identities reachable from one autograd output."""
    torch = _require_torch()
    if not torch.is_tensor(tensor) or tensor.grad_fn is None:
        return frozenset()
    result: set[int] = set()
    seen: set[int] = set()
    pending = [tensor.grad_fn]
    while pending:
        function = pending.pop()
        if function is None or id(function) in seen:
            continue
        seen.add(id(function))
        variable = getattr(function, "variable", None)
        if (
            torch.is_tensor(variable)
            and variable.requires_grad
            and bool(getattr(variable, "is_leaf", False))
        ):
            result.add(id(variable))
        pending.extend(
            next_function
            for next_function, _ in getattr(function, "next_functions", ())
            if next_function is not None
        )
    return frozenset(result)


@dataclass(frozen=True)
class PairFormerMicrobatchStatistics:
    """Autograd numerators plus integer denominators for one microbatch."""

    term_numerators: tuple[Any, ...]
    term_denominators: tuple[int, ...]
    term_weights: tuple[float, ...]
    schema_sha256: str
    source_receipt: SourceCodeReceipt
    model_parameter_ids: tuple[int, ...]
    model_parameter_manifest_sha256: str
    model_state_sha256: str
    batch_contract_sha256: str
    batch_authority_sha256: str
    seed_receipt: DeterministicSeedReceipt
    autocast_policy: AutocastPolicy

    def validate(self) -> None:
        """Reject reordered, detached, nonfinite, or inconsistent statistics."""
        torch = _require_torch()
        if (
            len(self.term_numerators) != len(LOSS_TERM_ORDER)
            or len(self.term_denominators) != len(LOSS_TERM_ORDER)
            or len(self.term_weights) != len(LOSS_TERM_ORDER)
        ):
            raise ValueError("Loss statistics must follow LOSS_TERM_ORDER")
        devices = set()
        for index, numerator in enumerate(self.term_numerators):
            if not torch.is_tensor(numerator) or numerator.ndim != 0:
                raise ValueError("Every loss numerator must be a scalar tensor")
            if not torch.is_floating_point(numerator) or not bool(
                torch.isfinite(numerator).item()
            ):
                raise ValueError("Every loss numerator must be finite and floating")
            if self.term_weights[index] > 0.0 and not numerator.requires_grad:
                raise ValueError(
                    "Every active loss numerator must retain its autograd graph"
                )
            devices.add(numerator.device)
            _nonnegative_integer(
                self.term_denominators[index],
                f"{LOSS_TERM_ORDER[index]} denominator",
            )
            _finite_nonnegative(
                self.term_weights[index],
                f"{LOSS_TERM_ORDER[index]} weight",
            )
        if len(devices) != 1:
            raise ValueError("All loss numerators must share one device")
        self.autocast_policy.validate(next(iter(devices)).type)
        if not any(weight > 0.0 for weight in self.term_weights):
            raise ValueError("At least one distributed loss term must be active")
        if (
            not isinstance(self.model_parameter_ids, tuple)
            or not self.model_parameter_ids
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.model_parameter_ids
            )
            or tuple(sorted(set(self.model_parameter_ids)))
            != self.model_parameter_ids
        ):
            raise ValueError("model_parameter_ids must be a sorted unique tuple")
        for index, denominator in enumerate(self.term_denominators):
            if denominator == 0 and float(
                self.term_numerators[index].detach().item()
            ) != 0.0:
                raise ValueError(
                    "A zero-support loss term must have an exact zero numerator"
                )
        _validate_sha256(self.schema_sha256, "schema_sha256")
        self.source_receipt.canonical_metadata()
        _validate_sha256(
            self.model_parameter_manifest_sha256,
            "model_parameter_manifest_sha256",
        )
        _validate_sha256(self.model_state_sha256, "model_state_sha256")
        _validate_sha256(self.batch_contract_sha256, "batch_contract_sha256")
        _validate_sha256(self.batch_authority_sha256, "batch_authority_sha256")
        self.seed_receipt.canonical_metadata()
        if self.seed_receipt.batch_contract_sha256 != self.batch_contract_sha256:
            raise ValueError("Seed receipt must bind the same training batch")

    @property
    def device(self) -> Any:
        """Return the common numerator device."""
        self.validate()
        return self.term_numerators[0].device


@dataclass(frozen=True)
class PairFormerForwardResult:
    """Model outputs and distributed statistics for one synthetic/train batch."""

    outputs: Mapping[str, Any]
    local_objective: Any
    components: Mapping[str, Any]
    statistics: PairFormerMicrobatchStatistics


def _enabled_heads(weights: PairGeometryLossWeights) -> frozenset[str]:
    """Execute only decoder heads required by positive-weight objectives."""
    result: set[str] = set()
    if (
        weights.contact_bce > 0.0
        or weights.contact_brier > 0.0
        or weights.epitope_consistency > 0.0
        or weights.paratope_consistency > 0.0
    ):
        result.update({"contact", "distance"})
    if weights.distance_ce > 0.0:
        result.add("distance")
    if weights.orientation_ce > 0.0:
        result.add("orientation")
    if weights.matched_decoy_ranking > 0.0:
        result.add("interaction")
    if not result:
        raise ValueError("At least one model head must be active")
    return frozenset(result)


def build_pairformer_microbatch_statistics(
    model: Any,
    batch: PairFormerTrainBatch,
    *,
    batch_authority: TrainOnlyBatchAuthority,
    expected_batch_authority_sha256: str,
    weights: PairGeometryLossWeights = PairGeometryLossWeights(),
    contact_positive_weight: float = 1.0,
    ranking_margin: float = 1.0,
    seed_receipt: DeterministicSeedReceipt,
    autocast_policy: AutocastPolicy = AutocastPolicy(),
) -> PairFormerForwardResult:
    """Run one sealed train microbatch and expose exact macro numerators."""
    torch = _require_torch()
    core_model = _pairformer_core(model)
    if not isinstance(core_model.geometry_schema, PairGeometrySchema):
        raise TypeError("model must carry an explicit PairGeometrySchema")
    weights_tuple = _weights_tuple(weights)
    validate_train_only_batch(batch, core_model.geometry_schema)
    validate_train_only_batch_authority(
        batch,
        core_model.geometry_schema,
        batch_authority,
        expected_authority_sha256=expected_batch_authority_sha256,
    )
    model_parameter_ids, model_parameter_manifest_sha256 = (
        _model_parameter_binding(model)
    )
    model_state_sha256 = _model_state_sha256(model)
    if weights.matched_decoy_ranking > 0.0:
        if batch.decoy_scores is None:
            raise ValueError("Matched-decoy ranking requires decoy scores")
        if (
            not batch.decoy_scores.requires_grad
            or batch.decoy_scores.grad_fn is None
        ):
            raise ValueError(
                "Ranking decoy_scores must be non-leaf values connected to the "
                "current autograd graph, not detached or constant batch inputs"
            )
        decoy_parameter_ids = _autograd_parameter_ids(batch.decoy_scores)
        if not decoy_parameter_ids.intersection(model_parameter_ids):
            raise ValueError(
                "Ranking decoy_scores must depend on current model parameters"
            )
    if not isinstance(seed_receipt, DeterministicSeedReceipt):
        raise TypeError("seed_receipt must be DeterministicSeedReceipt")
    seed_receipt.canonical_metadata()
    if seed_receipt.batch_contract_sha256 != batch.batch_contract_sha256:
        raise ValueError("seed_receipt does not bind the supplied batch")
    if not isinstance(autocast_policy, AutocastPolicy):
        raise TypeError("autocast_policy must be AutocastPolicy")
    device = batch.antibody_features.device
    if batch.antigen_features.device != device:
        raise ValueError("Antibody and antigen feature tensors must share a device")
    autocast_policy.validate(device.type)

    with deterministic_seed_scope(seed_receipt, device=device):
        with autocast_policy.context(device.type):
            outputs = model(
                batch.antibody_features,
                batch.antigen_features,
                batch.antibody_mask,
                batch.antigen_mask,
                batch.candidate_pair_mask,
                return_pair_representations=False,
                enabled_heads=_enabled_heads(weights),
            )
            local_objective, components = pair_geometry_objective(
                outputs,
                contact_targets=batch.contact_targets,
                contact_mask=batch.contact_mask,
                distance_bin_targets=batch.distance_bin_targets,
                distance_mask=batch.distance_mask,
                orientation_bin_targets=batch.orientation_bin_targets,
                orientation_mask=batch.orientation_mask,
                target_schema=core_model.geometry_schema,
                weights=weights,
                contact_positive_weight=contact_positive_weight,
                epitope_node_probabilities=batch.epitope_node_probabilities,
                epitope_consistency_mask=batch.epitope_consistency_mask,
                paratope_node_probabilities=batch.paratope_node_probabilities,
                paratope_consistency_mask=batch.paratope_consistency_mask,
                decoy_scores=batch.decoy_scores,
                decoy_mask=batch.decoy_mask,
                ranking_margin=ranking_margin,
            )

    batch_size = _batch_size(batch)
    reported = components["valid_complex_counts"]
    denominators = {
        "contact_bce": int(reported["contact_bce"]),
        "distance_ce": int(reported["distance_ce"]),
        "orientation_ce": int(reported["orientation_ce"]),
        "epitope_consistency": (
            0
            if batch.epitope_consistency_mask is None
            else _valid_complex_count(
                batch.epitope_consistency_mask,
                "epitope_consistency_mask",
                batch_size,
            )
        ),
        "paratope_consistency": (
            0
            if batch.paratope_consistency_mask is None
            else _valid_complex_count(
                batch.paratope_consistency_mask,
                "paratope_consistency_mask",
                batch_size,
            )
        ),
        "matched_decoy_ranking": (
            0
            if batch.decoy_mask is None
            else _valid_complex_count(batch.decoy_mask, "decoy_mask", batch_size)
        ),
        "contact_brier": int(reported["contact_brier"]),
    }
    numerators = tuple(
        components[name] * denominators[name] for name in LOSS_TERM_ORDER
    )
    statistics = PairFormerMicrobatchStatistics(
        term_numerators=numerators,
        term_denominators=tuple(denominators[name] for name in LOSS_TERM_ORDER),
        term_weights=weights_tuple,
        schema_sha256=core_model.geometry_schema.sha256,
        source_receipt=SourceCodeReceipt.current(),
        model_parameter_ids=model_parameter_ids,
        model_parameter_manifest_sha256=model_parameter_manifest_sha256,
        model_state_sha256=model_state_sha256,
        batch_contract_sha256=batch.batch_contract_sha256,
        batch_authority_sha256=batch_authority.sha256,
        seed_receipt=seed_receipt,
        autocast_policy=autocast_policy,
    )
    statistics.validate()
    if local_objective.ndim != 0 or not bool(torch.isfinite(local_objective).item()):
        raise RuntimeError("Local pair-geometry objective must be finite and scalar")
    return PairFormerForwardResult(
        outputs=outputs,
        local_objective=local_objective,
        components=components,
        statistics=statistics,
    )


def _validate_statistics_window(
    statistics: Sequence[PairFormerMicrobatchStatistics],
) -> PairFormerMicrobatchStatistics:
    """Require one nonempty, schema/source/weight-consistent window."""
    if isinstance(statistics, (str, bytes)) or not isinstance(
        statistics,
        Sequence,
    ):
        raise TypeError("statistics must be a sequence")
    if not statistics:
        raise ValueError("An accumulation window must contain a microbatch")
    first = statistics[0]
    first.validate()
    expected = (
        first.term_weights,
        first.schema_sha256,
        first.source_receipt,
        first.model_parameter_ids,
        first.model_parameter_manifest_sha256,
        first.model_state_sha256,
        first.seed_receipt.optimizer_step,
        first.seed_receipt.rank,
        first.seed_receipt.base_seed,
        first.seed_receipt.run_manifest_sha256,
        first.seed_receipt.checkpoint_parent_sha256,
        first.autocast_policy,
    )
    seen_accumulation_indices: set[int] = set()
    seen_batch_hashes: set[str] = set()
    seen_batch_authorities: set[str] = set()
    for value in statistics:
        if not isinstance(value, PairFormerMicrobatchStatistics):
            raise TypeError("Window entries must be PairFormerMicrobatchStatistics")
        value.validate()
        observed = (
            value.term_weights,
            value.schema_sha256,
            value.source_receipt,
            value.model_parameter_ids,
            value.model_parameter_manifest_sha256,
            value.model_state_sha256,
            value.seed_receipt.optimizer_step,
            value.seed_receipt.rank,
            value.seed_receipt.base_seed,
            value.seed_receipt.run_manifest_sha256,
            value.seed_receipt.checkpoint_parent_sha256,
            value.autocast_policy,
        )
        if observed != expected:
            raise ValueError(
                "All accumulation entries must share model, weights, schema, "
                "source, seed base, step, rank, and autocast policy"
            )
        if value.batch_contract_sha256 in seen_batch_hashes:
            raise ValueError("A batch contract cannot be replayed within one window")
        if value.batch_authority_sha256 in seen_batch_authorities:
            raise ValueError("A batch authority cannot be replayed within one window")
        accumulation_index = value.seed_receipt.accumulation_index
        if accumulation_index in seen_accumulation_indices:
            raise ValueError("Accumulation indices must be unique within a window")
        seen_accumulation_indices.add(accumulation_index)
        seen_batch_hashes.add(value.batch_contract_sha256)
        seen_batch_authorities.add(value.batch_authority_sha256)
    if seen_accumulation_indices != set(range(len(statistics))):
        raise ValueError("Accumulation indices must be contiguous from zero")
    return first


def pack_detached_loss_statistics(
    statistics: Sequence[PairFormerMicrobatchStatistics],
    *,
    world_size: int,
) -> Any:
    """Pack one rank's SUM-reducible statistics and one-hot rank evidence.

    The returned detached float64 tensor has three rows: term numerators, term
    denominators, and a one-hot rank-presence vector. External code must
    SUM-all-reduce the complete tensor. The exact all-ones reduced presence
    vector rejects mean reduction, missing ranks, duplicated ranks, and a
    mismatched ``world_size`` before backward.
    """
    torch = _require_torch()
    first = _validate_statistics_window(statistics)
    world_size = _positive_integer(world_size, "world_size")
    rank = first.seed_receipt.rank
    if rank >= world_size:
        raise ValueError("Seed receipt rank must be smaller than world_size")
    local_numerators = torch.stack(
        [
            sum(
                (item.term_numerators[index] for item in statistics),
                start=first.term_numerators[index].new_zeros(()),
            )
            for index in range(len(LOSS_TERM_ORDER))
        ]
    )
    local_denominators = torch.tensor(
        [
            sum(item.term_denominators[index] for item in statistics)
            for index in range(len(LOSS_TERM_ORDER))
        ],
        dtype=torch.float64,
        device=first.device,
    )
    width = max(len(LOSS_TERM_ORDER), world_size)
    packet = torch.zeros(
        (3, width),
        dtype=torch.float64,
        device=first.device,
    )
    packet[0, : len(LOSS_TERM_ORDER)] = local_numerators.detach().to(
        dtype=torch.float64
    )
    packet[1, : len(LOSS_TERM_ORDER)] = local_denominators
    packet[2, rank] = 1.0
    return packet


def _shared_reduction_context(
    statistics: Sequence[PairFormerMicrobatchStatistics],
    *,
    world_size: int,
) -> dict[str, Any]:
    """Return metadata that an authority must prove equal across every rank."""
    first = _validate_statistics_window(statistics)
    return {
        "format_version": REDUCED_PACKET_FORMAT_VERSION,
        "world_size": _positive_integer(world_size, "world_size"),
        "optimizer_step": first.seed_receipt.optimizer_step,
        "base_seed": first.seed_receipt.base_seed,
        "run_manifest_sha256": first.seed_receipt.run_manifest_sha256,
        "checkpoint_parent_sha256": (
            first.seed_receipt.checkpoint_parent_sha256
        ),
        "schema_sha256": first.schema_sha256,
        "source_receipt_sha256": first.source_receipt.sha256,
        "model_parameter_manifest_sha256": (
            first.model_parameter_manifest_sha256
        ),
        "model_state_sha256": first.model_state_sha256,
        "term_weights": list(first.term_weights),
        "autocast_policy": {
            "enabled": first.autocast_policy.enabled,
            "device_type": first.autocast_policy.device_type,
            "dtype_name": first.autocast_policy.dtype_name,
            "cache_enabled": first.autocast_policy.cache_enabled,
        },
    }


def local_reduction_contract_sha256(
    statistics: Sequence[PairFormerMicrobatchStatistics],
    *,
    world_size: int,
) -> str:
    """Bind one rank's exact accumulation window to its detached packet."""
    first = _validate_statistics_window(statistics)
    packet = pack_detached_loss_statistics(statistics, world_size=world_size)
    payload = {
        "shared_context_sha256": _json_sha256(
            _shared_reduction_context(statistics, world_size=world_size),
            domain="shared_reduction_context",
        ),
        "rank": first.seed_receipt.rank,
        "accumulation_steps": len(statistics),
        "batch_contract_sha256": [
            item.batch_contract_sha256 for item in statistics
        ],
        "batch_authority_sha256": [
            item.batch_authority_sha256 for item in statistics
        ],
        "seed_receipt_sha256": [
            item.seed_receipt.sha256 for item in statistics
        ],
        "term_denominators": [
            sum(item.term_denominators[index] for item in statistics)
            for index in range(len(LOSS_TERM_ORDER))
        ],
        "packet_sha256": tensor_sha256(packet, "local_reduction_packet"),
    }
    return _json_sha256(payload, domain="local_reduction_contract")


@dataclass(frozen=True)
class SumReductionAuthority:
    """Externally attested SUM collective for one optimizer boundary."""

    collective_id_sha256: str
    process_group_manifest_sha256: str
    issuer_receipt_sha256: str
    backend: str
    world_size: int
    shared_context_sha256: str
    local_contract_sha256_by_rank: tuple[tuple[int, str], ...]
    reduced_packet_sha256: str
    reduction_operation: str = "sum"
    ddp_gradient_reduction: str = "mean_over_initial_world_size"
    divide_by_initial_world_size: bool = True
    custom_communication_hook_sha256: str | None = None
    format_version: str = REDUCTION_AUTHORITY_FORMAT_VERSION

    def canonical_metadata(self) -> dict[str, Any]:
        """Return validated collective evidence for external freezing."""
        if self.format_version != REDUCTION_AUTHORITY_FORMAT_VERSION:
            raise ValueError("Reduction authority format version changed")
        for name in (
            "collective_id_sha256",
            "process_group_manifest_sha256",
            "issuer_receipt_sha256",
            "shared_context_sha256",
            "reduced_packet_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        world_size = _positive_integer(self.world_size, "world_size")
        if (
            type(self.backend) is not str
            or not self.backend
            or self.backend.strip() != self.backend
            or not self.backend.isascii()
        ):
            raise ValueError("backend must be a trimmed nonempty ASCII string")
        if self.reduction_operation != "sum":
            raise ValueError("Only SUM packet reduction is authorized")
        if self.ddp_gradient_reduction != "mean_over_initial_world_size":
            raise ValueError("The loss scaling assumes standard DDP mean reduction")
        if self.divide_by_initial_world_size is not True:
            raise ValueError("DDP join must divide by the initial world size")
        if self.custom_communication_hook_sha256 is not None:
            raise ValueError(
                "Custom DDP communication hooks are outside this proof boundary"
            )
        if not isinstance(self.local_contract_sha256_by_rank, tuple):
            raise TypeError("Rank contracts must be an immutable tuple")
        expected_ranks = tuple(range(world_size))
        observed_ranks = tuple(
            rank for rank, _ in self.local_contract_sha256_by_rank
        )
        if observed_ranks != expected_ranks:
            raise ValueError(
                "Reduction authority must bind every rank exactly once in order"
            )
        for rank, digest in self.local_contract_sha256_by_rank:
            _nonnegative_integer(rank, "rank")
            _validate_sha256(digest, f"local contract for rank {rank}")
        return {
            "format_version": self.format_version,
            "collective_id_sha256": self.collective_id_sha256,
            "process_group_manifest_sha256": (
                self.process_group_manifest_sha256
            ),
            "issuer_receipt_sha256": self.issuer_receipt_sha256,
            "backend": self.backend,
            "world_size": world_size,
            "shared_context_sha256": self.shared_context_sha256,
            "local_contract_sha256_by_rank": [
                {"rank": rank, "sha256": digest}
                for rank, digest in self.local_contract_sha256_by_rank
            ],
            "reduced_packet_sha256": self.reduced_packet_sha256,
            "reduction_operation": self.reduction_operation,
            "ddp_gradient_reduction": self.ddp_gradient_reduction,
            "divide_by_initial_world_size": (
                self.divide_by_initial_world_size
            ),
            "custom_communication_hook_sha256": (
                self.custom_communication_hook_sha256
            ),
        }

    @property
    def sha256(self) -> str:
        """Return the externally expected collective-authority digest."""
        return _json_sha256(
            self.canonical_metadata(),
            domain="sum_reduction_authority",
        )


def build_sum_reduction_authority(
    statistics_by_rank: Mapping[
        int,
        Sequence[PairFormerMicrobatchStatistics],
    ],
    reduced_packet: Any,
    *,
    world_size: int,
    collective_id_sha256: str,
    process_group_manifest_sha256: str,
    issuer_receipt_sha256: str,
    backend: str,
) -> SumReductionAuthority:
    """Build authority metadata after independently observing every rank packet.

    Production must freeze the returned digest outside the worker process.
    This helper verifies the arithmetic SUM but does not itself establish the
    identity or independence of the collective authority.
    """
    torch = _require_torch()
    if not isinstance(statistics_by_rank, Mapping):
        raise TypeError("statistics_by_rank must be a mapping")
    world_size = _positive_integer(world_size, "world_size")
    if tuple(sorted(statistics_by_rank)) != tuple(range(world_size)):
        raise ValueError("statistics_by_rank must contain every rank exactly once")
    contexts: list[dict[str, Any]] = []
    contracts: list[tuple[int, str]] = []
    packets: list[Any] = []
    for rank in range(world_size):
        statistics = statistics_by_rank[rank]
        first = _validate_statistics_window(statistics)
        if first.seed_receipt.rank != rank:
            raise ValueError("Rank statistics are stored under the wrong rank")
        contexts.append(
            _shared_reduction_context(statistics, world_size=world_size)
        )
        contracts.append(
            (
                rank,
                local_reduction_contract_sha256(
                    statistics,
                    world_size=world_size,
                ),
            )
        )
        packets.append(
            pack_detached_loss_statistics(
                statistics,
                world_size=world_size,
            )
        )
    first_context = contexts[0]
    if any(context != first_context for context in contexts[1:]):
        raise ValueError(
            "Every rank must share the exact model, source, seed, and DDP context"
        )
    expected_sum = torch.stack(packets).sum(dim=0)
    if (
        not torch.is_tensor(reduced_packet)
        or reduced_packet.dtype != torch.float64
        or reduced_packet.device != expected_sum.device
        or not torch.equal(reduced_packet, expected_sum)
    ):
        raise ValueError("reduced_packet is not the exact arithmetic SUM")
    authority = SumReductionAuthority(
        collective_id_sha256=_validate_sha256(
            collective_id_sha256,
            "collective_id_sha256",
        ),
        process_group_manifest_sha256=_validate_sha256(
            process_group_manifest_sha256,
            "process_group_manifest_sha256",
        ),
        issuer_receipt_sha256=_validate_sha256(
            issuer_receipt_sha256,
            "issuer_receipt_sha256",
        ),
        backend=backend,
        world_size=world_size,
        shared_context_sha256=_json_sha256(
            first_context,
            domain="shared_reduction_context",
        ),
        local_contract_sha256_by_rank=tuple(contracts),
        reduced_packet_sha256=tensor_sha256(
            reduced_packet,
            "reduced_packet",
        ),
    )
    authority.canonical_metadata()
    return authority


def validate_sum_reduction_authority(
    statistics: Sequence[PairFormerMicrobatchStatistics],
    reduced_packet: Any,
    authority: SumReductionAuthority,
    *,
    world_size: int,
    expected_authority_sha256: str,
) -> None:
    """Validate the local rank against an externally frozen SUM authority."""
    first = _validate_statistics_window(statistics)
    if not isinstance(authority, SumReductionAuthority):
        raise TypeError("authority must be SumReductionAuthority")
    authority.canonical_metadata()
    world_size = _positive_integer(world_size, "world_size")
    if authority.world_size != world_size:
        raise ValueError("Reduction authority world_size does not match the step")
    expected_authority_sha256 = _validate_sha256(
        expected_authority_sha256,
        "expected_authority_sha256",
    )
    if authority.sha256 != expected_authority_sha256:
        raise ValueError(
            "Reduction authority does not match the external expected digest"
        )
    observed_context_sha256 = _json_sha256(
        _shared_reduction_context(statistics, world_size=world_size),
        domain="shared_reduction_context",
    )
    if observed_context_sha256 != authority.shared_context_sha256:
        raise ValueError("Local statistics do not match the collective context")
    observed_local_contract = local_reduction_contract_sha256(
        statistics,
        world_size=world_size,
    )
    expected_local_contract = dict(
        authority.local_contract_sha256_by_rank
    )[first.seed_receipt.rank]
    if observed_local_contract != expected_local_contract:
        raise ValueError("Local accumulation window is not in the authority")
    observed_packet_sha256 = tensor_sha256(
        reduced_packet,
        "reduced_packet",
    )
    if observed_packet_sha256 != authority.reduced_packet_sha256:
        raise ValueError("Reduced packet does not match the collective authority")


def _validate_reduced_packet(
    reduced_packet: Any,
    term_weights: tuple[float, ...],
    *,
    world_size: int,
) -> tuple[Any, Any]:
    """Validate an externally SUM-reduced packet and exact rank participation."""
    torch = _require_torch()
    world_size = _positive_integer(world_size, "world_size")
    width = max(len(LOSS_TERM_ORDER), world_size)
    if (
        not torch.is_tensor(reduced_packet)
        or reduced_packet.ndim != 2
        or tuple(reduced_packet.shape) != (3, width)
        or reduced_packet.dtype != torch.float64
        or not torch.is_floating_point(reduced_packet)
    ):
        raise ValueError(
            "reduced_packet violates the float64 SUM-packet shape contract"
        )
    if not bool(torch.isfinite(reduced_packet).all().item()):
        raise ValueError("reduced_packet must contain finite values")
    term_count = len(LOSS_TERM_ORDER)
    numerators = reduced_packet[0, :term_count]
    denominators = reduced_packet[1, :term_count]
    if bool((numerators < 0.0).any().item()):
        raise ValueError("Reduced loss numerators must be nonnegative")
    if width > term_count and not bool(
        torch.equal(
            reduced_packet[:2, term_count:],
            torch.zeros_like(reduced_packet[:2, term_count:]),
        )
    ):
        raise ValueError("Reduced packet padding must remain exact zero")
    expected_rank_presence = torch.zeros_like(reduced_packet[2])
    expected_rank_presence[:world_size] = 1.0
    if not bool(torch.equal(reduced_packet[2], expected_rank_presence)):
        raise ValueError(
            "Reduced packet must be a SUM with every rank present exactly once"
        )
    if bool((denominators < 0.0).any().item()) or not bool(
        torch.equal(denominators, denominators.round())
    ):
        raise ValueError("Reduced denominators must be nonnegative integers")
    if bool((denominators > float(2**53)).any().item()):
        raise ValueError("Reduced denominators exceed exact float64 integer range")
    for index, weight in enumerate(term_weights):
        if weight > 0.0 and float(denominators[index].item()) <= 0.0:
            raise ValueError(
                f"Global support is empty for active term {LOSS_TERM_ORDER[index]}"
            )
    return numerators, denominators


def recompute_reduced_equal_complex_objective(
    reduced_packet: Any,
    term_weights: tuple[float, ...],
    *,
    world_size: int,
) -> Any:
    """Recompute the exact global objective after an external SUM all-reduce."""
    torch = _require_torch()
    if (
        not isinstance(term_weights, tuple)
        or len(term_weights) != len(LOSS_TERM_ORDER)
    ):
        raise ValueError("term_weights must follow LOSS_TERM_ORDER")
    validated_weights = tuple(
        _finite_nonnegative(value, f"{LOSS_TERM_ORDER[index]} weight")
        for index, value in enumerate(term_weights)
    )
    if not any(value > 0.0 for value in validated_weights):
        raise ValueError("At least one reduced loss term must be active")
    numerators, denominators = _validate_reduced_packet(
        reduced_packet,
        validated_weights,
        world_size=world_size,
    )
    result = torch.zeros(
        (),
        dtype=reduced_packet.dtype,
        device=reduced_packet.device,
    )
    for index, weight in enumerate(validated_weights):
        if weight > 0.0:
            result = result + weight * numerators[index] / denominators[index]
    if not bool(torch.isfinite(result).item()):
        raise RuntimeError("Reduced equal-complex objective is nonfinite")
    return result


def compose_ddp_accumulation_loss(
    statistics: Sequence[PairFormerMicrobatchStatistics],
    reduced_packet: Any,
    *,
    world_size: int,
    reduction_authority: SumReductionAuthority,
    expected_reduction_authority_sha256: str,
) -> Any:
    """Compose a window-level scalar whose DDP-averaged gradient is exact.

    ``reduced_packet`` must be the external SUM of every rank's
    :func:`pack_detached_loss_statistics` result for the complete accumulation
    window. Standard DDP divides parameter gradients by ``world_size``. The
    explicit factor here cancels that division and yields the gradient of
    ``global_numerator / global_valid_complex_count`` for every active term.
    """
    torch = _require_torch()
    first = _validate_statistics_window(statistics)
    world_size = _positive_integer(world_size, "world_size")
    if first.seed_receipt.rank >= world_size:
        raise ValueError("Seed receipt rank must be smaller than world_size")
    if reduced_packet.device != first.device:
        raise ValueError(
            "Reduced packet and local autograd numerators must share a device"
        )
    validate_sum_reduction_authority(
        statistics,
        reduced_packet,
        reduction_authority,
        world_size=world_size,
        expected_authority_sha256=expected_reduction_authority_sha256,
    )
    _, global_denominators = _validate_reduced_packet(
        reduced_packet,
        first.term_weights,
        world_size=world_size,
    )
    local_numerators = [
        sum(
            (item.term_numerators[index] for item in statistics),
            start=first.term_numerators[index].new_zeros(()),
        )
        for index in range(len(LOSS_TERM_ORDER))
    ]
    loss = first.term_numerators[0].new_zeros(())
    for index, weight in enumerate(first.term_weights):
        if weight > 0.0:
            denominator = global_denominators[index].to(
                device=loss.device,
                dtype=loss.dtype,
            )
            loss = (
                loss
                + world_size * weight * local_numerators[index] / denominator
            )
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise RuntimeError("DDP accumulation loss must be finite and scalar")
    return loss


@dataclass(frozen=True)
class OptimizerStepReceipt:
    """Immutable record of one applied or scaler-skipped optimizer boundary."""

    optimizer_step_before: int
    optimizer_step_after: int
    status: str
    optimizer_class: str
    accumulation_steps: int
    world_size: int
    schema_sha256: str
    source_receipt_sha256: str
    model_parameter_manifest_sha256: str
    model_state_sha256_before: str
    model_state_sha256_after: str
    optimizer_state_sha256_before: str
    optimizer_state_sha256_after: str
    rng_state_sha256_before: str
    rng_state_sha256_after: str
    grad_scaler_state_sha256_before: str | None
    grad_scaler_state_sha256_after: str | None
    checkpoint_parent_sha256: str
    reduced_packet_sha256: str
    reduction_authority_sha256: str
    reduction_format_version: str
    reduction_operation: str
    participating_ranks: tuple[int, ...]
    batch_contract_sha256: tuple[str, ...]
    batch_authority_sha256: tuple[str, ...]
    seed_receipt_sha256: tuple[str, ...]
    global_term_numerators: tuple[float, ...]
    global_term_denominators: tuple[int, ...]
    term_weights: tuple[float, ...]
    global_equal_complex_objective: float
    autocast_enabled: bool
    grad_scaler_used: bool
    scaler_scale_before: float | None
    scaler_scale_after: float | None
    max_grad_norm: float | None
    preclip_gradient_norm: float | None
    gradients_finite: bool
    format_version: str = OPTIMIZER_RECEIPT_FORMAT_VERSION

    def canonical_metadata(self) -> dict[str, Any]:
        """Return validated JSON metadata for durable experiment ledgers."""
        _nonnegative_integer(self.optimizer_step_before, "optimizer_step_before")
        _nonnegative_integer(self.optimizer_step_after, "optimizer_step_after")
        if self.status not in {"applied", "skipped_by_grad_scaler"}:
            raise ValueError("Optimizer receipt status is invalid")
        expected_after = self.optimizer_step_before + (
            1 if self.status == "applied" else 0
        )
        if self.optimizer_step_after != expected_after:
            raise ValueError("Optimizer step transition does not match status")
        if (
            not isinstance(self.optimizer_class, str)
            or not self.optimizer_class
            or self.optimizer_class.strip() != self.optimizer_class
        ):
            raise ValueError("optimizer_class must be a trimmed string")
        _positive_integer(self.accumulation_steps, "accumulation_steps")
        _positive_integer(self.world_size, "world_size")
        _validate_sha256(self.schema_sha256, "schema_sha256")
        _validate_sha256(self.source_receipt_sha256, "source_receipt_sha256")
        _validate_sha256(
            self.model_parameter_manifest_sha256,
            "model_parameter_manifest_sha256",
        )
        for name in (
            "model_state_sha256_before",
            "model_state_sha256_after",
            "optimizer_state_sha256_before",
            "optimizer_state_sha256_after",
            "rng_state_sha256_before",
            "rng_state_sha256_after",
            "checkpoint_parent_sha256",
            "reduction_authority_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        for name in (
            "grad_scaler_state_sha256_before",
            "grad_scaler_state_sha256_after",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_sha256(value, name)
        _validate_sha256(self.reduced_packet_sha256, "reduced_packet_sha256")
        if self.reduction_format_version != REDUCED_PACKET_FORMAT_VERSION:
            raise ValueError("Reduction packet format version changed")
        if self.reduction_operation != "sum":
            raise ValueError("Only SUM reduction is permitted")
        if self.participating_ranks != tuple(range(self.world_size)):
            raise ValueError("Receipt must contain every rank exactly once")
        if (
            len(self.batch_contract_sha256) != self.accumulation_steps
            or len(self.batch_authority_sha256) != self.accumulation_steps
            or len(self.seed_receipt_sha256) != self.accumulation_steps
        ):
            raise ValueError("Optimizer receipts must cover every accumulation entry")
        for name, values in (
            ("batch_contract_sha256", self.batch_contract_sha256),
            ("batch_authority_sha256", self.batch_authority_sha256),
            ("seed_receipt_sha256", self.seed_receipt_sha256),
        ):
            for value in values:
                _validate_sha256(value, name)
        if (
            len(self.global_term_numerators) != len(LOSS_TERM_ORDER)
            or len(self.global_term_denominators) != len(LOSS_TERM_ORDER)
            or len(self.term_weights) != len(LOSS_TERM_ORDER)
        ):
            raise ValueError("Global term statistics must follow LOSS_TERM_ORDER")
        for index, value in enumerate(self.global_term_numerators):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"Global numerator for {LOSS_TERM_ORDER[index]} must be "
                    "finite and nonnegative"
                )
        for index, value in enumerate(self.global_term_denominators):
            _nonnegative_integer(
                value,
                f"Global denominator for {LOSS_TERM_ORDER[index]}",
            )
        validated_weights = tuple(
            _finite_nonnegative(
                value,
                f"Weight for {LOSS_TERM_ORDER[index]}",
            )
            for index, value in enumerate(self.term_weights)
        )
        if not any(value > 0.0 for value in validated_weights):
            raise ValueError("At least one receipt loss term must be active")
        expected_objective = 0.0
        for index, weight in enumerate(validated_weights):
            denominator = self.global_term_denominators[index]
            if weight > 0.0:
                if denominator <= 0:
                    raise ValueError(
                        f"Active receipt term {LOSS_TERM_ORDER[index]} "
                        "has no support"
                    )
                expected_objective += (
                    weight
                    * float(self.global_term_numerators[index])
                    / denominator
                )
        if not math.isfinite(float(self.global_equal_complex_objective)):
            raise ValueError("Global equal-complex objective must be finite")
        if not math.isclose(
            float(self.global_equal_complex_objective),
            expected_objective,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Global equal-complex objective disagrees with receipt statistics"
            )
        if not isinstance(self.autocast_enabled, bool) or not isinstance(
            self.grad_scaler_used,
            bool,
        ):
            raise ValueError("AMP receipt fields must be boolean")
        if not isinstance(self.gradients_finite, bool):
            raise ValueError("gradients_finite must be boolean")
        if self.status == "applied" and not self.gradients_finite:
            raise ValueError("An applied optimizer step requires finite gradients")
        if self.status == "skipped_by_grad_scaler" and self.gradients_finite:
            raise ValueError("A scaler-skipped step requires nonfinite gradients")
        for name in ("scaler_scale_before", "scaler_scale_after"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if self.grad_scaler_used != (self.scaler_scale_before is not None):
            raise ValueError("Scaler presence and scale receipt disagree")
        if self.grad_scaler_used != (self.scaler_scale_after is not None):
            raise ValueError("Scaler presence and updated scale receipt disagree")
        if self.grad_scaler_used != (
            self.grad_scaler_state_sha256_before is not None
        ):
            raise ValueError("Scaler presence and pre-step state receipt disagree")
        if self.grad_scaler_used != (
            self.grad_scaler_state_sha256_after is not None
        ):
            raise ValueError("Scaler presence and post-step state receipt disagree")
        if self.status == "skipped_by_grad_scaler" and not self.grad_scaler_used:
            raise ValueError("Only GradScaler may authorize a skipped step")
        if self.grad_scaler_used and not self.autocast_enabled:
            raise ValueError("GradScaler receipt requires enabled autocast")
        if self.grad_scaler_used:
            if (
                self.status == "skipped_by_grad_scaler"
                and not float(self.scaler_scale_after)
                < float(self.scaler_scale_before)
            ):
                raise ValueError("A skipped GradScaler step must reduce its scale")
            if (
                self.status == "applied"
                and float(self.scaler_scale_after)
                < float(self.scaler_scale_before)
            ):
                raise ValueError("An applied GradScaler step cannot reduce its scale")
        if self.max_grad_norm is not None and (
            not math.isfinite(float(self.max_grad_norm))
            or float(self.max_grad_norm) <= 0.0
        ):
            raise ValueError("max_grad_norm must be positive and finite")
        if self.preclip_gradient_norm is not None and (
            not math.isfinite(float(self.preclip_gradient_norm))
            or float(self.preclip_gradient_norm) < 0.0
        ):
            raise ValueError("preclip_gradient_norm must be finite and nonnegative")
        if self.format_version != OPTIMIZER_RECEIPT_FORMAT_VERSION:
            raise ValueError("Optimizer receipt format version changed")
        return {
            "format_version": self.format_version,
            "optimizer_step_before": self.optimizer_step_before,
            "optimizer_step_after": self.optimizer_step_after,
            "status": self.status,
            "optimizer_class": self.optimizer_class,
            "accumulation_steps": self.accumulation_steps,
            "world_size": self.world_size,
            "schema_sha256": self.schema_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "model_parameter_manifest_sha256": (
                self.model_parameter_manifest_sha256
            ),
            "model_state_sha256_before": self.model_state_sha256_before,
            "model_state_sha256_after": self.model_state_sha256_after,
            "optimizer_state_sha256_before": (
                self.optimizer_state_sha256_before
            ),
            "optimizer_state_sha256_after": self.optimizer_state_sha256_after,
            "rng_state_sha256_before": self.rng_state_sha256_before,
            "rng_state_sha256_after": self.rng_state_sha256_after,
            "grad_scaler_state_sha256_before": (
                self.grad_scaler_state_sha256_before
            ),
            "grad_scaler_state_sha256_after": (
                self.grad_scaler_state_sha256_after
            ),
            "checkpoint_parent_sha256": self.checkpoint_parent_sha256,
            "reduced_packet_sha256": self.reduced_packet_sha256,
            "reduction_authority_sha256": self.reduction_authority_sha256,
            "reduction_format_version": self.reduction_format_version,
            "reduction_operation": self.reduction_operation,
            "participating_ranks": list(self.participating_ranks),
            "batch_contract_sha256": list(self.batch_contract_sha256),
            "batch_authority_sha256": list(self.batch_authority_sha256),
            "seed_receipt_sha256": list(self.seed_receipt_sha256),
            "global_term_numerators": list(self.global_term_numerators),
            "global_term_denominators": list(self.global_term_denominators),
            "term_weights": list(self.term_weights),
            "global_equal_complex_objective": self.global_equal_complex_objective,
            "autocast_enabled": self.autocast_enabled,
            "grad_scaler_used": self.grad_scaler_used,
            "scaler_scale_before": self.scaler_scale_before,
            "scaler_scale_after": self.scaler_scale_after,
            "max_grad_norm": self.max_grad_norm,
            "preclip_gradient_norm": self.preclip_gradient_norm,
            "gradients_finite": self.gradients_finite,
        }

    @property
    def sha256(self) -> str:
        """Return the canonical optimizer-boundary digest."""
        return _json_sha256(self.canonical_metadata())


def _trainable_parameters(optimizer: Any) -> tuple[Any, ...]:
    """Return unique trainable parameters owned by an optimizer."""
    torch = _require_torch()
    parameters: list[Any] = []
    seen: set[int] = set()
    for group in getattr(optimizer, "param_groups", ()):
        for parameter in group.get("params", ()):
            if (
                torch.is_tensor(parameter)
                and parameter.requires_grad
                and id(parameter) not in seen
            ):
                seen.add(id(parameter))
                parameters.append(parameter)
    if not parameters:
        raise ValueError("Optimizer must own at least one trainable parameter")
    return tuple(parameters)


def _gradients_are_finite(parameters: Sequence[Any]) -> bool:
    """Return whether every present gradient is finite and at least one exists."""
    torch = _require_torch()
    present = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not present:
        return False
    return all(bool(torch.isfinite(gradient).all().item()) for gradient in present)


def _validate_ddp_runtime(
    model: Any,
    *,
    world_size: int,
    rank: int,
    reduction_authority: SumReductionAuthority,
) -> None:
    """Require a live standard DDP reducer whenever world_size exceeds one."""
    torch = _require_torch()
    if world_size == 1:
        return
    ddp_class = torch.nn.parallel.DistributedDataParallel
    if not isinstance(model, ddp_class):
        raise ValueError(
            "world_size > 1 requires a live DistributedDataParallel wrapper"
        )
    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        raise RuntimeError("The DDP process group is not initialized")
    process_group = model.process_group
    observed_world_size = int(distributed.get_world_size(process_group))
    observed_rank = int(distributed.get_rank(process_group))
    if observed_world_size != world_size or observed_rank != rank:
        raise ValueError(
            "Live DDP rank/world_size differs from the training receipts"
        )
    observed_backend = str(distributed.get_backend(process_group))
    if reduction_authority.backend != observed_backend:
        raise ValueError("Live DDP backend differs from the reduction authority")
    if getattr(model, "require_backward_grad_sync", None) is not True:
        raise ValueError(
            "Final accumulation backward must enable DDP gradient synchronization"
        )


def perform_optimizer_step(
    loss: Any,
    model: Any,
    optimizer: Any,
    statistics: Sequence[PairFormerMicrobatchStatistics],
    reduced_packet: Any,
    *,
    world_size: int,
    optimizer_step: int,
    reduction_authority: SumReductionAuthority,
    expected_reduction_authority_sha256: str,
    grad_scaler: Any | None = None,
    max_grad_norm: float | None = None,
    set_to_none: bool = True,
) -> OptimizerStepReceipt:
    """Apply one valid accumulation boundary and return an immutable receipt.

    The function owns ``zero_grad``, ``backward``, scaler unscale/step/update,
    optional clipping, and the optimizer boundary. A scaler update is never
    performed before the reduced global-support contract has been validated.
    """
    torch = _require_torch()
    first = _validate_statistics_window(statistics)
    world_size = _positive_integer(world_size, "world_size")
    optimizer_step = _nonnegative_integer(optimizer_step, "optimizer_step")
    _validate_ddp_runtime(
        model,
        world_size=world_size,
        rank=first.seed_receipt.rank,
        reduction_authority=reduction_authority,
    )
    if first.seed_receipt.optimizer_step != optimizer_step:
        raise ValueError("Seed receipts do not match optimizer_step")
    if first.source_receipt != SourceCodeReceipt.current():
        raise ValueError(
            "Training source files changed after the forward statistics were built"
        )
    if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.is_floating_point(
        loss
    ):
        raise ValueError("loss must be a floating scalar tensor")
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("loss must be finite before backward")
    expected_loss = compose_ddp_accumulation_loss(
        statistics,
        reduced_packet,
        world_size=world_size,
        reduction_authority=reduction_authority,
        expected_reduction_authority_sha256=(
            expected_reduction_authority_sha256
        ),
    )
    if not torch.allclose(loss.detach(), expected_loss.detach(), rtol=1e-6, atol=1e-7):
        raise ValueError("loss does not match the locked DDP accumulation formula")
    global_objective_tensor = recompute_reduced_equal_complex_objective(
        reduced_packet,
        first.term_weights,
        world_size=world_size,
    )
    global_numerators, global_denominators = _validate_reduced_packet(
        reduced_packet,
        first.term_weights,
        world_size=world_size,
    )
    if not isinstance(set_to_none, bool):
        raise ValueError("set_to_none must be boolean")
    parameters = _trainable_parameters(optimizer)
    model_parameter_ids, model_parameter_manifest_sha256 = (
        _model_parameter_binding(model)
    )
    if model_parameter_manifest_sha256 != first.model_parameter_manifest_sha256:
        raise ValueError("Model parameter layout changed after the forward window")
    optimizer_parameter_ids = tuple(sorted(id(value) for value in parameters))
    if (
        optimizer_parameter_ids != first.model_parameter_ids
        or model_parameter_ids != first.model_parameter_ids
    ):
        raise ValueError(
            "Optimizer parameters do not exactly match the bound model parameters"
        )
    model_state_before = _model_state_sha256(model)
    if model_state_before != first.model_state_sha256:
        raise ValueError("Model state changed during the accumulation window")
    if grad_scaler is not None and (
        not first.autocast_policy.enabled
        or first.autocast_policy.dtype_name != "float16"
    ):
        raise ValueError(
            "GradScaler is permitted only for enabled float16 autocast"
        )
    optimizer.zero_grad(set_to_none=set_to_none)
    versions_before = tuple(int(parameter._version) for parameter in parameters)
    optimizer_state_before = _optimizer_state_sha256(optimizer)
    rng_state_before = _rng_state_sha256(parameters)
    scaler_state_before = _scaler_state_sha256(grad_scaler)

    scaler_before = None
    scaler_after = None
    preclip_norm = None
    gradients_finite = False
    if max_grad_norm is not None:
        max_grad_norm = float(_finite_nonnegative(max_grad_norm, "max_grad_norm"))
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
    if grad_scaler is None:
        expected_loss.backward()
        gradients_finite = _gradients_are_finite(parameters)
        if not gradients_finite:
            raise RuntimeError(
                "Unscaled gradients must be present and finite before optimizer.step"
            )
        if max_grad_norm is not None:
            preclip_norm_tensor = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_grad_norm,
                error_if_nonfinite=True,
            )
            preclip_norm = float(preclip_norm_tensor.item())
        optimizer.step()
    else:
        for name in ("scale", "unscale_", "step", "update", "get_scale"):
            if not callable(getattr(grad_scaler, name, None)):
                raise TypeError("grad_scaler does not implement the GradScaler API")
        scaler_before = float(grad_scaler.get_scale())
        if not math.isfinite(scaler_before) or scaler_before <= 0.0:
            raise ValueError("GradScaler scale must be positive and finite")
        grad_scaler.scale(expected_loss).backward()
        grad_scaler.unscale_(optimizer)
        gradients_finite = _gradients_are_finite(parameters)
        if max_grad_norm is not None and gradients_finite:
            preclip_norm_tensor = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_grad_norm,
                error_if_nonfinite=True,
            )
            preclip_norm = float(preclip_norm_tensor.item())
        grad_scaler.step(optimizer)
        grad_scaler.update()
        scaler_after = float(grad_scaler.get_scale())
        if not math.isfinite(scaler_after) or scaler_after <= 0.0:
            raise RuntimeError("Updated GradScaler scale must be positive and finite")

    model_state_after = _model_state_sha256(model)
    optimizer_state_after = _optimizer_state_sha256(optimizer)
    rng_state_after = _rng_state_sha256(parameters)
    scaler_state_after = _scaler_state_sha256(grad_scaler)
    versions_after = tuple(int(parameter._version) for parameter in parameters)
    applied = any(
        after > before
        for before, after in zip(versions_before, versions_after)
    )
    if grad_scaler is None and not applied:
        raise RuntimeError("Optimizer did not update any trainable parameter version")
    status = "applied" if applied else "skipped_by_grad_scaler"
    if grad_scaler is not None:
        if applied != gradients_finite:
            raise RuntimeError(
                "GradScaler optimizer application disagrees with gradient finiteness"
            )
        if status == "skipped_by_grad_scaler" and not scaler_after < scaler_before:
            raise RuntimeError("A skipped GradScaler step must reduce its scale")
        if status == "applied" and scaler_after < scaler_before:
            raise RuntimeError("An applied GradScaler step cannot reduce its scale")
    if status == "skipped_by_grad_scaler":
        if model_state_after != model_state_before:
            raise RuntimeError("A scaler-skipped step changed the model state")
        if optimizer_state_after != optimizer_state_before:
            raise RuntimeError("A scaler-skipped step changed the optimizer state")
    receipt = OptimizerStepReceipt(
        optimizer_step_before=optimizer_step,
        optimizer_step_after=optimizer_step + (1 if applied else 0),
        status=status,
        optimizer_class=(
            f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}"
        ),
        accumulation_steps=len(statistics),
        world_size=world_size,
        schema_sha256=first.schema_sha256,
        source_receipt_sha256=first.source_receipt.sha256,
        model_parameter_manifest_sha256=(
            first.model_parameter_manifest_sha256
        ),
        model_state_sha256_before=model_state_before,
        model_state_sha256_after=model_state_after,
        optimizer_state_sha256_before=optimizer_state_before,
        optimizer_state_sha256_after=optimizer_state_after,
        rng_state_sha256_before=rng_state_before,
        rng_state_sha256_after=rng_state_after,
        grad_scaler_state_sha256_before=scaler_state_before,
        grad_scaler_state_sha256_after=scaler_state_after,
        checkpoint_parent_sha256=(
            first.seed_receipt.checkpoint_parent_sha256
        ),
        reduced_packet_sha256=tensor_sha256(
            reduced_packet,
            "reduced_packet",
        ),
        reduction_authority_sha256=reduction_authority.sha256,
        reduction_format_version=REDUCED_PACKET_FORMAT_VERSION,
        reduction_operation="sum",
        participating_ranks=tuple(range(world_size)),
        batch_contract_sha256=tuple(
            value.batch_contract_sha256 for value in statistics
        ),
        batch_authority_sha256=tuple(
            value.batch_authority_sha256 for value in statistics
        ),
        seed_receipt_sha256=tuple(
            value.seed_receipt.sha256 for value in statistics
        ),
        global_term_numerators=tuple(
            float(value) for value in global_numerators.detach().cpu().tolist()
        ),
        global_term_denominators=tuple(
            int(round(value))
            for value in global_denominators.detach().cpu().tolist()
        ),
        term_weights=first.term_weights,
        global_equal_complex_objective=float(
            global_objective_tensor.detach().cpu().item()
        ),
        autocast_enabled=first.autocast_policy.enabled,
        grad_scaler_used=grad_scaler is not None,
        scaler_scale_before=scaler_before,
        scaler_scale_after=scaler_after,
        max_grad_norm=max_grad_norm,
        preclip_gradient_norm=preclip_norm,
        gradients_finite=gradients_finite,
    )
    receipt.canonical_metadata()
    return receipt


@dataclass(frozen=True)
class CheckpointContinuationReceipt:
    """Bind checkpoint bytes to one fully recorded optimizer boundary."""

    checkpoint_artifact_sha256: str
    checkpoint_writer_source_sha256: str
    immutable_storage_receipt_sha256: str
    optimizer_step_receipt_sha256: str
    checkpoint_parent_sha256: str
    optimizer_step_after: int
    model_state_sha256: str
    optimizer_state_sha256: str
    rng_state_sha256: str
    grad_scaler_state_sha256: str | None
    format_version: str = CHECKPOINT_RECEIPT_FORMAT_VERSION

    def canonical_metadata(self) -> dict[str, Any]:
        """Return validated checkpoint lineage without loading checkpoint bytes."""
        if self.format_version != CHECKPOINT_RECEIPT_FORMAT_VERSION:
            raise ValueError("Checkpoint receipt format version changed")
        for name in (
            "checkpoint_artifact_sha256",
            "checkpoint_writer_source_sha256",
            "immutable_storage_receipt_sha256",
            "optimizer_step_receipt_sha256",
            "checkpoint_parent_sha256",
            "model_state_sha256",
            "optimizer_state_sha256",
            "rng_state_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if self.grad_scaler_state_sha256 is not None:
            _validate_sha256(
                self.grad_scaler_state_sha256,
                "grad_scaler_state_sha256",
            )
        _nonnegative_integer(self.optimizer_step_after, "optimizer_step_after")
        return {
            "format_version": self.format_version,
            "checkpoint_artifact_sha256": self.checkpoint_artifact_sha256,
            "checkpoint_writer_source_sha256": (
                self.checkpoint_writer_source_sha256
            ),
            "immutable_storage_receipt_sha256": (
                self.immutable_storage_receipt_sha256
            ),
            "optimizer_step_receipt_sha256": (
                self.optimizer_step_receipt_sha256
            ),
            "checkpoint_parent_sha256": self.checkpoint_parent_sha256,
            "optimizer_step_after": self.optimizer_step_after,
            "model_state_sha256": self.model_state_sha256,
            "optimizer_state_sha256": self.optimizer_state_sha256,
            "rng_state_sha256": self.rng_state_sha256,
            "grad_scaler_state_sha256": self.grad_scaler_state_sha256,
        }

    @property
    def sha256(self) -> str:
        """Return the externally freezeable checkpoint receipt digest."""
        return _json_sha256(
            self.canonical_metadata(),
            domain="checkpoint_continuation_receipt",
        )


def build_checkpoint_continuation_receipt(
    optimizer_receipt: OptimizerStepReceipt,
    *,
    checkpoint_artifact_sha256: str,
    checkpoint_writer_source_sha256: str,
    immutable_storage_receipt_sha256: str,
) -> CheckpointContinuationReceipt:
    """Bind persisted checkpoint bytes to the exact post-step state hashes."""
    if not isinstance(optimizer_receipt, OptimizerStepReceipt):
        raise TypeError("optimizer_receipt must be OptimizerStepReceipt")
    optimizer_receipt.canonical_metadata()
    receipt = CheckpointContinuationReceipt(
        checkpoint_artifact_sha256=_validate_sha256(
            checkpoint_artifact_sha256,
            "checkpoint_artifact_sha256",
        ),
        checkpoint_writer_source_sha256=_validate_sha256(
            checkpoint_writer_source_sha256,
            "checkpoint_writer_source_sha256",
        ),
        immutable_storage_receipt_sha256=_validate_sha256(
            immutable_storage_receipt_sha256,
            "immutable_storage_receipt_sha256",
        ),
        optimizer_step_receipt_sha256=optimizer_receipt.sha256,
        checkpoint_parent_sha256=optimizer_receipt.checkpoint_parent_sha256,
        optimizer_step_after=optimizer_receipt.optimizer_step_after,
        model_state_sha256=optimizer_receipt.model_state_sha256_after,
        optimizer_state_sha256=optimizer_receipt.optimizer_state_sha256_after,
        rng_state_sha256=optimizer_receipt.rng_state_sha256_after,
        grad_scaler_state_sha256=(
            optimizer_receipt.grad_scaler_state_sha256_after
        ),
    )
    receipt.canonical_metadata()
    return receipt


def verify_checkpoint_continuation_receipt(
    receipt: CheckpointContinuationReceipt,
    *,
    expected_receipt_sha256: str,
    expected_checkpoint_artifact_sha256: str,
) -> None:
    """Fail closed unless external ledgers freeze receipt and checkpoint bytes."""
    if not isinstance(receipt, CheckpointContinuationReceipt):
        raise TypeError("receipt must be CheckpointContinuationReceipt")
    receipt.canonical_metadata()
    expected_receipt_sha256 = _validate_sha256(
        expected_receipt_sha256,
        "expected_receipt_sha256",
    )
    expected_checkpoint_artifact_sha256 = _validate_sha256(
        expected_checkpoint_artifact_sha256,
        "expected_checkpoint_artifact_sha256",
    )
    if receipt.sha256 != expected_receipt_sha256:
        raise ValueError("Checkpoint receipt does not match the external digest")
    if (
        receipt.checkpoint_artifact_sha256
        != expected_checkpoint_artifact_sha256
    ):
        raise ValueError("Checkpoint artifact does not match the external digest")
