"""Fail-closed helpers for the PAIRScale E1/E2 posthoc contact audit."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

from .development_authority import canonical_json_bytes
from .pairscale_e1_e2_evaluation import EVALUATION_FORMAT
from .pairscale_e1_e2_training import REPORT_FORMAT


POSTHOC_AUDIT_FORMAT = "pairscale_e1_e2_posthoc_contact_audit_v2"
V1_EVALUATION_FORMAT = "pairscale_e1_e2_development_evaluation_v1"
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_V2_TOP_LEVEL_FIELDS = frozenset(
    {
        "macro_interaction_residual_ap_lift",
        "macro_interaction_residual_precision_at_true_contact_count",
        "macro_unary_additive_ap_lift",
        "macro_unary_additive_precision_at_true_contact_count",
    }
)
_V2_PER_COMPLEX_FIELDS = frozenset(
    {
        "interaction_residual_ap_lift",
        "interaction_residual_precision_at_true_contact_count",
        "interaction_residual_precision_at_true_contact_count_tie_policy",
        "unary_additive_ap_lift",
        "unary_additive_precision_at_true_contact_count",
        "unary_additive_precision_at_true_contact_count_tie_policy",
    }
)


class PairscaleE1E2PosthocAuditError(ValueError):
    """Raised when a posthoc checkpoint or evaluation fails closed."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise PairscaleE1E2PosthocAuditError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def validate_frozen_report(
    report: Mapping[str, Any],
    *,
    seed: int,
    expected_holdout_count: int,
) -> Mapping[str, Any]:
    """Validate that a report describes one fixed epoch-260 development run."""
    required = {
        "format_version": REPORT_FORMAT,
        "screening_seed": seed,
        "completed_epoch": 260,
        "fixed_epoch_rule_satisfied": True,
        "early_stopping_used": False,
        "development_holdout_evaluation_performed": True,
        "development_holdout_is_untouched_final_test": False,
        "new_final_test_accessed": False,
        "model_selection_authorized": False,
        "checkpoint_promotion_authorized": False,
        "confirmatory_claim_authorized": False,
        "scaling_authorized": False,
    }
    for name, expected in required.items():
        if report.get(name) != expected:
            raise PairscaleE1E2PosthocAuditError(
                f"Frozen report field {name} differs"
            )
    checkpoint = report.get("frozen_checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("epoch") != 260:
        raise PairscaleE1E2PosthocAuditError(
            "Frozen checkpoint must be the fixed epoch-260 generation"
        )
    for name in (
        "checkpoint_sha256",
        "receipt_file_sha256",
    ):
        _digest(checkpoint.get(name), f"frozen_checkpoint.{name}")
    generation = checkpoint.get("generation")
    if not isinstance(generation, str) or not generation.startswith("epoch-0260-"):
        raise PairscaleE1E2PosthocAuditError(
            "Frozen checkpoint generation is not canonical epoch 260"
        )
    provenance = checkpoint.get("provenance_hashes")
    if not isinstance(provenance, Mapping) or not provenance:
        raise PairscaleE1E2PosthocAuditError(
            "Frozen checkpoint provenance is unavailable"
        )
    for name, digest in provenance.items():
        if not isinstance(name, str) or not name:
            raise PairscaleE1E2PosthocAuditError(
                "Frozen checkpoint provenance name is invalid"
            )
        _digest(digest, f"frozen_checkpoint.provenance_hashes.{name}")
    evaluation = report.get("development_holdout_evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("format_version") != (
        V1_EVALUATION_FORMAT
    ):
        raise PairscaleE1E2PosthocAuditError(
            "Frozen report development evaluation differs"
        )
    rows = evaluation.get("per_complex")
    if not isinstance(rows, list) or len(rows) != expected_holdout_count:
        raise PairscaleE1E2PosthocAuditError(
            "Frozen report holdout inventory differs"
        )
    identifiers = [
        row.get("instance_id") for row in rows if isinstance(row, Mapping)
    ]
    if len(identifiers) != len(rows) or len(set(identifiers)) != len(rows):
        raise PairscaleE1E2PosthocAuditError(
            "Frozen report holdout identifiers are invalid"
        )
    return checkpoint


def validate_posthoc_source_boundary(
    frozen_provenance: Mapping[str, Any],
    current_source_hashes: Mapping[str, str],
) -> Mapping[str, Any]:
    """Allow only the preregistered evaluator implementation to differ."""
    allowed_difference = "pairscale_e1_e2_evaluation.py"
    observed_differences: list[str] = []
    for name, current_digest in sorted(current_source_hashes.items()):
        _digest(current_digest, f"current_source_hashes.{name}")
        frozen_digest = frozen_provenance.get(f"source_{name}")
        _digest(frozen_digest, f"frozen_provenance.source_{name}")
        if frozen_digest != current_digest:
            observed_differences.append(name)
    if observed_differences != [allowed_difference]:
        raise PairscaleE1E2PosthocAuditError(
            "Posthoc source boundary differs outside the evaluation-only change"
        )
    return {
        "allowed_difference": allowed_difference,
        "frozen_sha256": frozen_provenance[f"source_{allowed_difference}"],
        "posthoc_sha256": current_source_hashes[allowed_difference],
        "all_other_executed_sources_exact": True,
    }


def verify_v2_reproduces_v1(
    v1_evaluation: Mapping[str, Any],
    v2_evaluation: Mapping[str, Any],
) -> Mapping[str, str]:
    """Require the enhanced evaluator to reproduce every pre-existing field."""
    if v1_evaluation.get("format_version") != V1_EVALUATION_FORMAT:
        raise PairscaleE1E2PosthocAuditError("Reference evaluation is not V1")
    if v2_evaluation.get("format_version") != EVALUATION_FORMAT:
        raise PairscaleE1E2PosthocAuditError("Posthoc evaluation is not V2")
    normalized = copy.deepcopy(dict(v2_evaluation))
    normalized["format_version"] = V1_EVALUATION_FORMAT
    for name in _V2_TOP_LEVEL_FIELDS:
        if name not in normalized:
            raise PairscaleE1E2PosthocAuditError(
                f"Posthoc evaluation is missing {name}"
            )
        normalized.pop(name)
    rows = normalized.get("per_complex")
    if not isinstance(rows, list):
        raise PairscaleE1E2PosthocAuditError(
            "Posthoc per-complex inventory is unavailable"
        )
    for row in rows:
        if not isinstance(row, dict):
            raise PairscaleE1E2PosthocAuditError(
                "Posthoc per-complex row is invalid"
            )
        for name in _V2_PER_COMPLEX_FIELDS:
            if name not in row:
                raise PairscaleE1E2PosthocAuditError(
                    f"Posthoc per-complex row is missing {name}"
                )
            row.pop(name)
    if normalized != dict(v1_evaluation):
        raise PairscaleE1E2PosthocAuditError(
            "Posthoc evaluation does not exactly reproduce the V1 payload"
        )
    return {
        "reference_v1_canonical_sha256": hashlib.sha256(
            canonical_json_bytes(dict(v1_evaluation))
        ).hexdigest(),
        "reproduced_v1_canonical_sha256": hashlib.sha256(
            canonical_json_bytes(normalized)
        ).hexdigest(),
        "posthoc_v2_canonical_sha256": hashlib.sha256(
            canonical_json_bytes(dict(v2_evaluation))
        ).hexdigest(),
    }


__all__ = [
    "POSTHOC_AUDIT_FORMAT",
    "PairscaleE1E2PosthocAuditError",
    "validate_frozen_report",
    "validate_posthoc_source_boundary",
    "verify_v2_reproduces_v1",
]
