"""External-freeze authority for one-shot R2 outer and test evaluation.

This module authenticates frozen metadata only. It cannot create scientific
authority by itself, mount a test set, load a checkpoint, or run inference.
Expected registry and controller-receipt digests must arrive out of band.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


OUTER_PHASE = "outer_oof"
EXTERNAL_TEST_PHASE = "provided_external_test"
PHASES = frozenset({OUTER_PHASE, EXTERNAL_TEST_PHASE})
FROZEN_SEEDS = (17, 42, 73, 101, 137)
CANDIDATE_FINAL_EPOCH = {"D0": 58, "R1": 40, "R2": 40}
EXTERNAL_RESULT_LABEL = (
    "provided_external_test_split_with_pre_scoring_schema_access_"
    "not_pristine_blind"
)
REGISTRY_FORMAT = "sabdab2_partner_geometry_r2_evaluation_registry_v1"
FREEZE_FORMAT = "sabdab2_partner_geometry_r2_external_freeze_receipt_v1"
ATTEMPT_FORMAT = "sabdab2_partner_geometry_r2_one_shot_attempt_v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(
    r"^[a-z0-9](?:[a-z0-9._:@/-]{0,126}[a-z0-9])?$"
)
_UTC_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)


class R2EvaluationFreezeError(ValueError):
    """Raised when one evaluation-freeze artifact fails closed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode finite ASCII JSON deterministically."""
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise R2EvaluationFreezeError(
            "evaluation-freeze metadata must be finite ASCII JSON"
        ) from error


def canonical_sha256(value: Any) -> str:
    """Hash one canonical evaluation-freeze payload."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise R2EvaluationFreezeError(
            f"{name} must be a lowercase 64-character SHA-256"
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or _IDENTIFIER_PATTERN.fullmatch(value) is None
    ):
        raise R2EvaluationFreezeError(
            f"{name} must be a canonical lowercase ASCII identifier"
        )
    return value


def _timestamp(value: Any, name: str) -> str:
    if not isinstance(value, str) or _UTC_PATTERN.fullmatch(value) is None:
        raise R2EvaluationFreezeError(
            f"{name} must use UTC YYYY-MM-DDTHH:MM:SSZ"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise R2EvaluationFreezeError(
            f"{name} must be a valid UTC calendar timestamp"
        ) from error
    if parsed.tzinfo != timezone.utc:
        raise R2EvaluationFreezeError(f"{name} must use UTC")
    return value


@dataclass(frozen=True, order=True)
class CheckpointBinding:
    """Bind one fixed seed and final epoch to immutable checkpoint bytes."""

    seed: int
    final_epoch: int
    checkpoint_sha256: str
    checkpoint_receipt_sha256: str
    source_run_id: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed not in FROZEN_SEEDS
        ):
            raise R2EvaluationFreezeError("checkpoint seed is not frozen")
        if (
            isinstance(self.final_epoch, bool)
            or not isinstance(self.final_epoch, int)
            or self.final_epoch <= 0
        ):
            raise R2EvaluationFreezeError(
                "checkpoint final_epoch must be positive"
            )
        _sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _sha256(
            self.checkpoint_receipt_sha256,
            "checkpoint_receipt_sha256",
        )
        _identifier(self.source_run_id, "source_run_id")

    def canonical_payload(self) -> dict[str, Any]:
        """Return the immutable checkpoint representation."""
        return {
            "seed": self.seed,
            "final_epoch": self.final_epoch,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "source_run_id": self.source_run_id,
        }


@dataclass(frozen=True)
class FrozenEvaluationRegistry:
    """Freeze every choice that may affect one outer or external evaluation."""

    phase: str
    protocol_file_sha256: str
    execution_plan_sha256: str
    gate0_decision_sha256: str
    split_assignment_sha256: str
    inner_selection_receipt_sha256: str
    source_manifest_sha256: str
    runtime_manifest_sha256: str
    candidate_domain_sha256: str
    metric_registry_sha256: str
    calibration_registry_sha256: str
    threshold_registry_sha256: str
    analysis_registry_sha256: str
    candidate_id: str
    checkpoint_rule: str
    checkpoints: tuple[CheckpointBinding, ...]
    frozen_at_utc: str
    outer_fold: int | None = None
    completed_outer_analysis_receipt_sha256: str | None = None
    prior_external_test_schema_access: bool = False
    result_label: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise R2EvaluationFreezeError("evaluation phase is unsupported")
        for name in (
            "protocol_file_sha256",
            "execution_plan_sha256",
            "gate0_decision_sha256",
            "split_assignment_sha256",
            "inner_selection_receipt_sha256",
            "source_manifest_sha256",
            "runtime_manifest_sha256",
            "candidate_domain_sha256",
            "metric_registry_sha256",
            "calibration_registry_sha256",
            "threshold_registry_sha256",
            "analysis_registry_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.candidate_id not in CANDIDATE_FINAL_EPOCH:
            raise R2EvaluationFreezeError("candidate_id is not preregistered")
        expected_epoch = CANDIDATE_FINAL_EPOCH[self.candidate_id]
        expected_rule = f"fixed_final_epoch_{expected_epoch}"
        if self.checkpoint_rule != expected_rule:
            raise R2EvaluationFreezeError("checkpoint rule differs")
        if (
            not isinstance(self.checkpoints, tuple)
            or len(self.checkpoints) != len(FROZEN_SEEDS)
            or any(
                not isinstance(item, CheckpointBinding)
                for item in self.checkpoints
            )
        ):
            raise R2EvaluationFreezeError(
                "exactly five immutable checkpoint bindings are required"
            )
        ordered = tuple(sorted(self.checkpoints))
        if ordered != self.checkpoints:
            raise R2EvaluationFreezeError(
                "checkpoint bindings must be sorted by seed"
            )
        if tuple(item.seed for item in self.checkpoints) != FROZEN_SEEDS:
            raise R2EvaluationFreezeError(
                "checkpoint bindings must cover every frozen seed"
            )
        if any(item.final_epoch != expected_epoch for item in self.checkpoints):
            raise R2EvaluationFreezeError("checkpoint epoch differs")
        if len(
            {item.checkpoint_sha256 for item in self.checkpoints}
        ) != len(FROZEN_SEEDS):
            raise R2EvaluationFreezeError(
                "each seed must bind distinct checkpoint bytes"
            )
        _timestamp(self.frozen_at_utc, "frozen_at_utc")
        if not isinstance(self.prior_external_test_schema_access, bool):
            raise R2EvaluationFreezeError(
                "prior_external_test_schema_access must be boolean"
            )

        if self.phase == OUTER_PHASE:
            if (
                isinstance(self.outer_fold, bool)
                or not isinstance(self.outer_fold, int)
                or self.outer_fold not in range(3)
            ):
                raise R2EvaluationFreezeError(
                    "outer phase requires outer_fold 0, 1, or 2"
                )
            if self.completed_outer_analysis_receipt_sha256 is not None:
                raise R2EvaluationFreezeError(
                    "outer phase cannot consume completed outer results"
                )
            if self.prior_external_test_schema_access:
                raise R2EvaluationFreezeError(
                    "outer registry cannot request external test access"
                )
            if self.result_label is not None:
                raise R2EvaluationFreezeError(
                    "outer registry cannot carry an external result label"
                )
        else:
            if self.outer_fold is not None:
                raise R2EvaluationFreezeError(
                    "external phase cannot name one outer fold"
                )
            _sha256(
                self.completed_outer_analysis_receipt_sha256,
                "completed_outer_analysis_receipt_sha256",
            )
            if not self.prior_external_test_schema_access:
                raise R2EvaluationFreezeError(
                    "known pre-scoring schema access must be disclosed"
                )
            if self.result_label != EXTERNAL_RESULT_LABEL:
                raise R2EvaluationFreezeError(
                    "external result label must disclose prior schema access"
                )

    def canonical_payload(self) -> dict[str, Any]:
        """Return the complete frozen registry representation."""
        return {
            "format_version": REGISTRY_FORMAT,
            "phase": self.phase,
            "protocol_file_sha256": self.protocol_file_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "gate0_decision_sha256": self.gate0_decision_sha256,
            "split_assignment_sha256": self.split_assignment_sha256,
            "inner_selection_receipt_sha256": (
                self.inner_selection_receipt_sha256
            ),
            "source_manifest_sha256": self.source_manifest_sha256,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "candidate_domain_sha256": self.candidate_domain_sha256,
            "metric_registry_sha256": self.metric_registry_sha256,
            "calibration_registry_sha256": self.calibration_registry_sha256,
            "threshold_registry_sha256": self.threshold_registry_sha256,
            "analysis_registry_sha256": self.analysis_registry_sha256,
            "candidate_id": self.candidate_id,
            "checkpoint_rule": self.checkpoint_rule,
            "checkpoints": [
                item.canonical_payload() for item in self.checkpoints
            ],
            "frozen_at_utc": self.frozen_at_utc,
            "outer_fold": self.outer_fold,
            "completed_outer_analysis_receipt_sha256": (
                self.completed_outer_analysis_receipt_sha256
            ),
            "prior_external_test_schema_access": (
                self.prior_external_test_schema_access
            ),
            "result_label": self.result_label,
            "checkpoint_selection_from_outer_or_test": False,
            "threshold_selection_from_outer_or_test": False,
            "calibration_selection_from_outer_or_test": False,
            "attempt_limit": 1,
        }

    @property
    def registry_sha256(self) -> str:
        """Return the complete registry digest."""
        return canonical_sha256(self.canonical_payload())


@dataclass(frozen=True)
class ExternalFreezeReceipt:
    """Record an out-of-worker controller freeze for one exact registry."""

    phase: str
    registry_sha256: str
    controller_id: str
    frozen_at_utc: str
    create_only: bool = True
    replacement_authorized: bool = False
    attempt_limit: int = 1

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise R2EvaluationFreezeError("freeze phase is unsupported")
        _sha256(self.registry_sha256, "registry_sha256")
        _identifier(self.controller_id, "controller_id")
        _timestamp(self.frozen_at_utc, "frozen_at_utc")
        if self.create_only is not True:
            raise R2EvaluationFreezeError("freeze receipt must be create-only")
        if self.replacement_authorized is not False:
            raise R2EvaluationFreezeError(
                "freeze receipt cannot authorize replacement"
            )
        if self.attempt_limit != 1:
            raise R2EvaluationFreezeError(
                "freeze receipt must permit exactly one attempt"
            )

    def canonical_payload(self) -> dict[str, Any]:
        """Return the external controller receipt representation."""
        return {
            "format_version": FREEZE_FORMAT,
            "phase": self.phase,
            "registry_sha256": self.registry_sha256,
            "controller_id": self.controller_id,
            "frozen_at_utc": self.frozen_at_utc,
            "create_only": self.create_only,
            "replacement_authorized": self.replacement_authorized,
            "attempt_limit": self.attempt_limit,
        }

    @property
    def receipt_sha256(self) -> str:
        """Return the external controller receipt digest."""
        return canonical_sha256(self.canonical_payload())


def verify_external_freeze(
    registry: FrozenEvaluationRegistry,
    receipt: ExternalFreezeReceipt,
    *,
    expected_registry_sha256: str,
    expected_receipt_sha256: str,
    required_phase: str,
) -> None:
    """Verify out-of-band digests before any evaluation data can be read."""
    if not isinstance(registry, FrozenEvaluationRegistry):
        raise R2EvaluationFreezeError(
            "registry must be a FrozenEvaluationRegistry"
        )
    if not isinstance(receipt, ExternalFreezeReceipt):
        raise R2EvaluationFreezeError(
            "receipt must be an ExternalFreezeReceipt"
        )
    _sha256(expected_registry_sha256, "expected_registry_sha256")
    _sha256(expected_receipt_sha256, "expected_receipt_sha256")
    if required_phase not in PHASES:
        raise R2EvaluationFreezeError("required phase is unsupported")
    if registry.registry_sha256 != expected_registry_sha256:
        raise R2EvaluationFreezeError("evaluation registry digest differs")
    if receipt.receipt_sha256 != expected_receipt_sha256:
        raise R2EvaluationFreezeError("external freeze receipt digest differs")
    if registry.phase != required_phase or receipt.phase != required_phase:
        raise R2EvaluationFreezeError("evaluation phase differs")
    if receipt.registry_sha256 != registry.registry_sha256:
        raise R2EvaluationFreezeError(
            "external freeze does not bind the evaluation registry"
        )


def create_one_shot_attempt_ledger(
    path: Path,
    *,
    registry: FrozenEvaluationRegistry,
    receipt: ExternalFreezeReceipt,
    expected_registry_sha256: str,
    expected_receipt_sha256: str,
    started_at_utc: str,
) -> dict[str, Any]:
    """Create the only allowed attempt ledger before reading evaluation data."""
    verify_external_freeze(
        registry,
        receipt,
        expected_registry_sha256=expected_registry_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
        required_phase=registry.phase,
    )
    _timestamp(started_at_utc, "started_at_utc")
    target = Path(os.path.abspath(os.path.expanduser(str(path))))
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"one-shot ledger already exists: {target}")
    parent = target.parent
    if not parent.is_dir() or parent.is_symlink():
        raise R2EvaluationFreezeError(
            "one-shot ledger parent must be a regular directory"
        )
    payload = {
        "format_version": ATTEMPT_FORMAT,
        "phase": registry.phase,
        "registry_sha256": registry.registry_sha256,
        "freeze_receipt_sha256": receipt.receipt_sha256,
        "attempt_number": 1,
        "status": "started_before_evaluation_data_read",
        "started_at_utc": started_at_utc,
        "evaluation_data_read": False,
        "inference_started": False,
        "performance_scored": False,
        "replacement_authorized": False,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o444)
    try:
        encoded = canonical_json_bytes(payload)
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    parent_descriptor = os.open(parent, directory_flags)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return payload


__all__ = [
    "ATTEMPT_FORMAT",
    "CANDIDATE_FINAL_EPOCH",
    "CheckpointBinding",
    "EXTERNAL_RESULT_LABEL",
    "EXTERNAL_TEST_PHASE",
    "ExternalFreezeReceipt",
    "FROZEN_SEEDS",
    "FrozenEvaluationRegistry",
    "OUTER_PHASE",
    "R2EvaluationFreezeError",
    "canonical_json_bytes",
    "canonical_sha256",
    "create_one_shot_attempt_ledger",
    "verify_external_freeze",
]
