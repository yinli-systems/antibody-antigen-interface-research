"""Authenticated paired statistics for the PAIRScale centering ablation."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Mapping

from .development_holdout import connected_components, parse_component_holdout_assignment
from .pairscale_holdout_statistics import (
    PairscaleHoldoutStatisticsError,
    _file_sha256,
    _finite,
    _sha256,
    _strict_json,
    _utc,
    _validate_evaluation,
)
from .pairscale_unary_statistics import (
    PairscaleUnaryStatisticsError,
    _authenticated_bytes,
    _bootstrap,
    _digest,
)


RESULT_FORMAT = "pairscale_centering_ablation_cluster_statistics_v1"
_NO_CENTERING_EXTRA_FIELDS = {"scaling_authorized", "training_partner_score"}
_NO_CENTERING_MISSING_BASE_FIELDS = {"interpretation"}
_IMMUTABLE_ROW_FIELDS = {
    "count",
    "instance_id",
    "label_marginal_oracle_auprc",
    "positive_count",
    "prevalence",
    "precision_at_true_contact_count_tie_policy",
}


class PairscaleCenteringStatisticsError(ValueError):
    """Raised when the centering comparison cannot be authenticated."""


def _validate_no_centering_evaluation(
    evaluation: Mapping[str, Any],
    *,
    expected_instances: set[str],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    """Validate the exact no-centering output schema and shared metrics schema."""

    reference_shape = set(evaluation).difference(_NO_CENTERING_EXTRA_FIELDS)
    reference_shape.update(_NO_CENTERING_MISSING_BASE_FIELDS)
    expected_shape = reference_shape.union(_NO_CENTERING_EXTRA_FIELDS).difference(
        _NO_CENTERING_MISSING_BASE_FIELDS
    )
    if set(evaluation) != expected_shape:
        raise PairscaleCenteringStatisticsError("No-centering fields differ")
    if evaluation["training_partner_score"] != "raw_topk_only_no_centering":
        raise PairscaleCenteringStatisticsError("No-centering treatment differs")
    if evaluation["scaling_authorized"] is not False:
        raise PairscaleCenteringStatisticsError("No-centering output authorizes scaling")
    if evaluation["scope"] != (
        "authority_train_only_development_double_cold_no_centering"
    ):
        raise PairscaleCenteringStatisticsError("No-centering scope differs")
    if evaluation["evaluation_role"] != "single_fixed_checkpoint_development_holdout":
        raise PairscaleCenteringStatisticsError("No-centering evaluation role differs")
    shared = {
        key: value
        for key, value in evaluation.items()
        if key not in _NO_CENTERING_EXTRA_FIELDS
    }
    shared["interpretation"] = "schema_adapter_only_not_source_evidence"
    try:
        return _validate_evaluation(shared, expected_instances=expected_instances)
    except PairscaleHoldoutStatisticsError as error:
        raise PairscaleCenteringStatisticsError(
            "No-centering evaluation schema authentication failed"
        ) from error


def _metric_map(
    rows: list[Mapping[str, Any]],
    metric: str,
) -> dict[str, float]:
    return {
        str(row["instance_id"]): _finite(row[metric], metric)
        for row in rows
    }


def _partner_delta_map(
    rows: list[Mapping[str, Any]],
    partner: Mapping[str, Any],
    *,
    matched_field: str,
    decoy_field: str,
) -> dict[str, float]:
    identifiers = [str(row["instance_id"]) for row in rows]
    matched = partner[matched_field]
    decoy = partner[decoy_field]
    if len(matched) != len(identifiers) or len(decoy) != len(identifiers):
        raise PairscaleCenteringStatisticsError("Partner score inventory differs")
    return {
        identifier: _finite(matched_value, matched_field)
        - _finite(decoy_value, decoy_field)
        for identifier, matched_value, decoy_value in zip(
            identifiers,
            matched,
            decoy,
        )
    }


def _close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)


def _verify_internal_summaries(
    evaluation: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    partner: Mapping[str, Any],
    *,
    name: str,
) -> None:
    """Recompute every available macro and partner summary from atomic rows."""

    summary_fields = {
        "macro_label_marginal_oracle_auprc": "label_marginal_oracle_auprc",
        "macro_model_marginal_outer_auprc": "model_marginal_outer_auprc",
        "macro_pair_auprc": "auprc",
        "macro_precision_at_true_contact_count": "precision_at_true_contact_count",
    }
    for summary_field, row_field in summary_fields.items():
        expected = math.fsum(_finite(row[row_field], row_field) for row in rows) / len(rows)
        observed = _finite(evaluation[summary_field], summary_field)
        if not _close(observed, expected):
            raise PairscaleCenteringStatisticsError(f"{name} {summary_field} differs")
    for prefix, matched_field, decoy_field in (
        ("matched_partner_topk", "matched_topk_scores", "decoy_topk_scores"),
        (
            "double_centered_partner",
            "matched_double_centered_scores",
            "decoy_double_centered_scores",
        ),
    ):
        deltas = [
            _finite(matched, matched_field) - _finite(decoy, decoy_field)
            for matched, decoy in zip(
                partner[matched_field],
                partner[decoy_field],
            )
        ]
        expected_delta = math.fsum(deltas) / len(deltas)
        expected_win_rate = math.fsum(float(value > 0.0) for value in deltas) / len(
            deltas
        )
        if not _close(
            _finite(partner[f"{prefix}_mean_delta"], f"{prefix}_mean_delta"),
            expected_delta,
        ):
            raise PairscaleCenteringStatisticsError(f"{name} {prefix} mean differs")
        if not _close(
            _finite(partner[f"{prefix}_win_rate"], f"{prefix}_win_rate"),
            expected_win_rate,
        ):
            raise PairscaleCenteringStatisticsError(
                f"{name} {prefix} win rate differs"
            )


def _verify_shared_label_inventory(
    reference_rows: list[Mapping[str, Any]],
    no_centering_rows: list[Mapping[str, Any]],
) -> None:
    reference = {str(row["instance_id"]): row for row in reference_rows}
    treatment = {str(row["instance_id"]): row for row in no_centering_rows}
    if set(reference) != set(treatment):
        raise PairscaleCenteringStatisticsError("Evaluation instance inventories differ")
    for identifier in sorted(reference):
        for field in _IMMUTABLE_ROW_FIELDS:
            if reference[identifier][field] != treatment[identifier][field]:
                raise PairscaleCenteringStatisticsError(
                    f"Immutable holdout field differs: {field}"
                )


def compare_pairscale_centering_ablation(
    *,
    assignment_path: Path,
    assignment_receipt_path: Path,
    reference_evaluation_path: Path,
    no_centering_evaluation_path: Path,
    expected_assignment_file_sha256: str,
    expected_assignment_content_sha256: str,
    expected_assignment_receipt_sha256: str,
    expected_reference_evaluation_sha256: str,
    expected_no_centering_evaluation_sha256: str,
    iterations: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Compare the frozen centered reference with the raw-top-only treatment."""

    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise PairscaleCenteringStatisticsError("iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PairscaleCenteringStatisticsError("seed must be an integer")
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    try:
        _utc(timestamp)
    except PairscaleHoldoutStatisticsError as error:
        raise PairscaleCenteringStatisticsError("created_at_utc is invalid") from error

    try:
        assignment_bytes = _authenticated_bytes(
            assignment_path, expected_assignment_file_sha256, "assignment"
        )
        receipt_bytes = _authenticated_bytes(
            assignment_receipt_path,
            expected_assignment_receipt_sha256,
            "assignment_receipt",
        )
        reference_bytes = _authenticated_bytes(
            reference_evaluation_path,
            expected_reference_evaluation_sha256,
            "reference_evaluation",
        )
        no_centering_bytes = _authenticated_bytes(
            no_centering_evaluation_path,
            expected_no_centering_evaluation_sha256,
            "no_centering_evaluation",
        )
    except PairscaleUnaryStatisticsError as error:
        raise PairscaleCenteringStatisticsError("Artifact authentication failed") from error

    try:
        raw_assignment = _strict_json(assignment_bytes, "assignment")
        assignment = parse_component_holdout_assignment(
            assignment_bytes,
            receipt_bytes,
            expected_assignment_file_sha256=expected_assignment_file_sha256,
            expected_assignment_content_sha256=_digest(
                expected_assignment_content_sha256,
                "expected_assignment_content_sha256",
            ),
            expected_receipt_file_sha256=expected_assignment_receipt_sha256,
            expected_source_subset_sha256=raw_assignment["source_subset_sha256"],
            expected_policy_sha256=raw_assignment["policy_sha256"],
        )
    except (KeyError, TypeError, ValueError, PairscaleHoldoutStatisticsError) as error:
        raise PairscaleCenteringStatisticsError("Assignment authentication failed") from error
    components = connected_components(assignment.holdout_instances)
    if sorted(value.component_sha256 for value in components) != raw_assignment[
        "holdout"
    ]["component_sha256"]:
        raise PairscaleCenteringStatisticsError("Holdout component identities differ")
    clusters = {
        instance.instance_id: component.component_sha256
        for component in components
        for instance in component.instances
    }
    expected_instances = set(clusters)

    try:
        reference_evaluation = _strict_json(reference_bytes, "reference evaluation")
        reference_rows, reference_partner = _validate_evaluation(
            reference_evaluation,
            expected_instances=expected_instances,
        )
    except PairscaleHoldoutStatisticsError as error:
        raise PairscaleCenteringStatisticsError(
            "Reference evaluation authentication failed"
        ) from error
    if reference_evaluation["scope"] != (
        "authority_train_only_development_double_cold_holdout"
    ) or reference_evaluation["evaluation_role"] != (
        "single_frozen_development_holdout_evaluation"
    ):
        raise PairscaleCenteringStatisticsError("Reference evaluation role differs")
    no_centering_evaluation = _strict_json(
        no_centering_bytes,
        "no-centering evaluation",
    )
    no_centering_rows, no_centering_partner = _validate_no_centering_evaluation(
        no_centering_evaluation,
        expected_instances=expected_instances,
    )
    _verify_internal_summaries(
        reference_evaluation,
        reference_rows,
        reference_partner,
        name="reference",
    )
    _verify_internal_summaries(
        no_centering_evaluation,
        no_centering_rows,
        no_centering_partner,
        name="no-centering",
    )
    _verify_shared_label_inventory(reference_rows, no_centering_rows)

    bootstrap_arguments = {
        "clusters": clusters,
        "iterations": iterations,
        "seed": seed,
        "confidence": confidence,
    }
    reference_raw_delta = _partner_delta_map(
        reference_rows,
        reference_partner,
        matched_field="matched_topk_scores",
        decoy_field="decoy_topk_scores",
    )
    treatment_raw_delta = _partner_delta_map(
        no_centering_rows,
        no_centering_partner,
        matched_field="matched_topk_scores",
        decoy_field="decoy_topk_scores",
    )
    reference_centered_delta = _partner_delta_map(
        reference_rows,
        reference_partner,
        matched_field="matched_double_centered_scores",
        decoy_field="decoy_double_centered_scores",
    )
    treatment_centered_delta = _partner_delta_map(
        no_centering_rows,
        no_centering_partner,
        matched_field="matched_double_centered_scores",
        decoy_field="decoy_double_centered_scores",
    )
    zero = {identifier: 0.0 for identifier in expected_instances}
    comparisons = {
        "reference_minus_no_centering_macro_pair_auprc": _bootstrap(
            _metric_map(reference_rows, "auprc"),
            _metric_map(no_centering_rows, "auprc"),
            **bootstrap_arguments,
        ),
        "reference_minus_no_centering_precision_at_true_contact_count": _bootstrap(
            _metric_map(reference_rows, "precision_at_true_contact_count"),
            _metric_map(no_centering_rows, "precision_at_true_contact_count"),
            **bootstrap_arguments,
        ),
        "reference_minus_no_centering_raw_partner_delta": _bootstrap(
            reference_raw_delta,
            treatment_raw_delta,
            **bootstrap_arguments,
        ),
        "reference_minus_no_centering_double_centered_partner_delta": _bootstrap(
            reference_centered_delta,
            treatment_centered_delta,
            **bootstrap_arguments,
        ),
        "reference_matched_minus_decoy_raw_partner_score": _bootstrap(
            reference_raw_delta,
            zero,
            **bootstrap_arguments,
        ),
        "reference_matched_minus_decoy_double_centered_partner_score": _bootstrap(
            reference_centered_delta,
            zero,
            **bootstrap_arguments,
        ),
        "no_centering_matched_minus_decoy_raw_partner_score": _bootstrap(
            treatment_raw_delta,
            zero,
            **bootstrap_arguments,
        ),
        "no_centering_matched_minus_decoy_double_centered_partner_score": _bootstrap(
            treatment_centered_delta,
            zero,
            **bootstrap_arguments,
        ),
    }
    centering_checks = {
        "reference_pair_auprc_exceeds_no_centering_lower_bound_above_zero": comparisons[
            "reference_minus_no_centering_macro_pair_auprc"
        ]["lower"]
        > 0.0,
        "reference_double_centered_partner_delta_exceeds_no_centering_lower_bound_above_zero": comparisons[
            "reference_minus_no_centering_double_centered_partner_delta"
        ]["lower"]
        > 0.0,
    }
    gate_l1_checks = {
        "reference_raw_partner_decline_lower_bound_above_zero": comparisons[
            "reference_matched_minus_decoy_raw_partner_score"
        ]["lower"]
        > 0.0,
        "reference_double_centered_partner_decline_lower_bound_above_zero": comparisons[
            "reference_matched_minus_decoy_double_centered_partner_score"
        ]["lower"]
        > 0.0,
    }
    return {
        "artifacts": {
            "assignment_content_sha256": assignment.assignment_content_sha256,
            "assignment_file_sha256": _sha256(assignment_bytes),
            "assignment_receipt_file_sha256": _sha256(receipt_bytes),
            "no_centering_evaluation_sha256": _sha256(no_centering_bytes),
            "reference_evaluation_sha256": _sha256(reference_bytes),
        },
        "bootstrap": {
            "comparison_count": len(comparisons),
            "confidence": confidence,
            "independent_component_count": len(components),
            "instance_count": len(expected_instances),
            "iterations": iterations,
            "seed": seed,
            "unit": "connected_endpoint_component",
        },
        "centering_ablation": {
            "checks": centering_checks,
            "completed": True,
            "interpretation_policy": (
                "Effects are descriptive development-only evidence; the ablation "
                "does not independently authorize scaling."
            ),
        },
        "claim_boundary": (
            "Development-only endpoint-disjoint centering ablation. No validation, "
            "test, confirmatory, checkpoint-promotion, scaling, biological-"
            "generalization, comparative-superiority, or state-of-the-art claim is "
            "authorized."
        ),
        "comparisons": comparisons,
        "created_at_utc": timestamp,
        "format_version": RESULT_FORMAT,
        "gate_l1": {
            "checks": gate_l1_checks,
            "failed_checks": sorted(
                name for name, passed in gate_l1_checks.items() if not passed
            ),
            "passed": False,
            "scaling_authorized": False,
            "status": (
                "failed_required_partner_specificity_checks"
                if not all(gate_l1_checks.values())
                else "requires_combined_unary_and_retrieval_gate_review"
            ),
            "unresolved_required_controls": [],
        },
        "source_implementation_sha256": {
            "development_holdout.py": _file_sha256(
                Path(__file__).with_name("development_holdout.py")
            ),
            "metrics.py": _file_sha256(Path(__file__).with_name("metrics.py")),
            "pairscale_centering_ablation_statistics.py": _file_sha256(Path(__file__)),
            "pairscale_holdout_statistics.py": _file_sha256(
                Path(__file__).with_name("pairscale_holdout_statistics.py")
            ),
            "pairscale_unary_statistics.py": _file_sha256(
                Path(__file__).with_name("pairscale_unary_statistics.py")
            ),
        },
    }


__all__ = [
    "PairscaleCenteringStatisticsError",
    "compare_pairscale_centering_ablation",
]
