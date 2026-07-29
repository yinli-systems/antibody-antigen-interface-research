"""Authenticated paired statistics for PAIRScale unary-control comparisons."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .development_holdout import (
    connected_components,
    parse_component_holdout_assignment,
)
from .metrics import paired_cluster_bootstrap_mean_difference
from .pairscale_holdout_statistics import (
    PairscaleHoldoutStatisticsError,
    _file_sha256,
    _finite,
    _regular_file,
    _sha256,
    _strict_json,
    _utc,
    _validate_evaluation,
)


RESULT_FORMAT = "pairscale_unary_control_cluster_statistics_v1"
_UNARY_EXTRA_FIELDS = {
    "architecture_reads_partner_feature_values",
    "control_side",
    "scaling_authorized",
}


class PairscaleUnaryStatisticsError(ValueError):
    """Raised when a unary-control comparison fails closed."""


def _digest(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PairscaleUnaryStatisticsError(f"{name} must be a lowercase SHA-256")
    return value


def _authenticated_bytes(path: Path, expected_sha256: str, name: str) -> bytes:
    try:
        raw = _regular_file(path, name).read_bytes()
    except (OSError, PairscaleHoldoutStatisticsError) as error:
        raise PairscaleUnaryStatisticsError(f"Cannot read {name}") from error
    if _sha256(raw) != _digest(expected_sha256, f"expected_{name}_sha256"):
        raise PairscaleUnaryStatisticsError(f"{name} SHA-256 differs")
    return raw


def _validate_unary_evaluation(
    evaluation: Mapping[str, Any],
    *,
    expected_instances: set[str],
    expected_control_side: str,
) -> list[Mapping[str, Any]]:
    if set(evaluation) - _UNARY_EXTRA_FIELDS == set(evaluation):
        raise PairscaleUnaryStatisticsError("Unary-control fields are missing")
    stripped = {
        key: value
        for key, value in evaluation.items()
        if key not in _UNARY_EXTRA_FIELDS
    }
    if set(evaluation) != set(stripped).union(_UNARY_EXTRA_FIELDS):
        raise PairscaleUnaryStatisticsError("Unary-control fields differ")
    if evaluation["control_side"] != expected_control_side:
        raise PairscaleUnaryStatisticsError("Unary-control side differs")
    if evaluation["architecture_reads_partner_feature_values"] is not False:
        raise PairscaleUnaryStatisticsError("Unary control reads partner features")
    if evaluation["scaling_authorized"] is not False:
        raise PairscaleUnaryStatisticsError("Unary evaluation authorizes scaling")
    if evaluation["scope"] != (
        "authority_train_only_development_double_cold_unary_control"
    ):
        raise PairscaleUnaryStatisticsError("Unary evaluation scope differs")
    if evaluation["evaluation_role"] != (
        "single_fixed_checkpoint_development_holdout"
    ):
        raise PairscaleUnaryStatisticsError("Unary evaluation role differs")
    try:
        rows, _ = _validate_evaluation(
            stripped,
            expected_instances=expected_instances,
        )
    except PairscaleHoldoutStatisticsError as error:
        raise PairscaleUnaryStatisticsError(
            "Unary evaluation schema authentication failed"
        ) from error
    return rows


def _metric_map(
    rows: list[Mapping[str, Any]],
    metric: str,
) -> dict[str, float]:
    return {
        str(row["instance_id"]): _finite(row[metric], metric)
        for row in rows
    }


def _bootstrap(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    clusters: Mapping[str, str],
    *,
    iterations: int,
    seed: int,
    confidence: float,
) -> Mapping[str, Any]:
    try:
        return paired_cluster_bootstrap_mean_difference(
            candidate,
            reference,
            clusters,
            iterations=iterations,
            seed=seed,
            confidence=confidence,
        )
    except ValueError as error:
        raise PairscaleUnaryStatisticsError(
            "Paired connected-component bootstrap failed"
        ) from error


def compare_pairscale_unary_controls(
    *,
    assignment_path: Path,
    assignment_receipt_path: Path,
    pair_evaluation_path: Path,
    antibody_only_evaluation_path: Path,
    antigen_only_evaluation_path: Path,
    expected_assignment_file_sha256: str,
    expected_assignment_content_sha256: str,
    expected_assignment_receipt_sha256: str,
    expected_pair_evaluation_sha256: str,
    expected_antibody_only_evaluation_sha256: str,
    expected_antigen_only_evaluation_sha256: str,
    iterations: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Compare a frozen pair model with two independently trained unary controls."""

    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise PairscaleUnaryStatisticsError("iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PairscaleUnaryStatisticsError("seed must be an integer")
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    try:
        _utc(timestamp)
    except PairscaleHoldoutStatisticsError as error:
        raise PairscaleUnaryStatisticsError("created_at_utc is invalid") from error

    assignment_bytes = _authenticated_bytes(
        assignment_path,
        expected_assignment_file_sha256,
        "assignment",
    )
    receipt_bytes = _authenticated_bytes(
        assignment_receipt_path,
        expected_assignment_receipt_sha256,
        "assignment_receipt",
    )
    pair_bytes = _authenticated_bytes(
        pair_evaluation_path,
        expected_pair_evaluation_sha256,
        "pair_evaluation",
    )
    antibody_bytes = _authenticated_bytes(
        antibody_only_evaluation_path,
        expected_antibody_only_evaluation_sha256,
        "antibody_only_evaluation",
    )
    antigen_bytes = _authenticated_bytes(
        antigen_only_evaluation_path,
        expected_antigen_only_evaluation_sha256,
        "antigen_only_evaluation",
    )
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
        raise PairscaleUnaryStatisticsError(
            "Assignment authentication failed"
        ) from error
    components = connected_components(assignment.holdout_instances)
    observed_component_ids = sorted(value.component_sha256 for value in components)
    if observed_component_ids != raw_assignment["holdout"]["component_sha256"]:
        raise PairscaleUnaryStatisticsError("Holdout component identities differ")
    cluster_by_instance = {
        instance.instance_id: component.component_sha256
        for component in components
        for instance in component.instances
    }
    expected_instances = set(cluster_by_instance)
    try:
        pair_evaluation = _strict_json(pair_bytes, "pair evaluation")
        pair_rows, pair_partner = _validate_evaluation(
            pair_evaluation,
            expected_instances=expected_instances,
        )
        antibody_rows = _validate_unary_evaluation(
            _strict_json(antibody_bytes, "antibody-only evaluation"),
            expected_instances=expected_instances,
            expected_control_side="antibody_only",
        )
        antigen_rows = _validate_unary_evaluation(
            _strict_json(antigen_bytes, "antigen-only evaluation"),
            expected_instances=expected_instances,
            expected_control_side="antigen_only",
        )
    except PairscaleHoldoutStatisticsError as error:
        raise PairscaleUnaryStatisticsError("Evaluation authentication failed") from error

    pair_auprc = _metric_map(pair_rows, "auprc")
    antibody_auprc = _metric_map(antibody_rows, "auprc")
    antigen_auprc = _metric_map(antigen_rows, "auprc")
    model_marginal_auprc = _metric_map(pair_rows, "model_marginal_outer_auprc")
    label_oracle_auprc = _metric_map(pair_rows, "label_marginal_oracle_auprc")
    strongest_separable = {
        identifier: max(
            antibody_auprc[identifier],
            antigen_auprc[identifier],
            model_marginal_auprc[identifier],
        )
        for identifier in pair_auprc
    }
    bootstrap_arguments = {
        "clusters": cluster_by_instance,
        "iterations": iterations,
        "seed": seed,
        "confidence": confidence,
    }
    comparisons = {
        "pair_minus_antibody_only_macro_auprc": _bootstrap(
            pair_auprc,
            antibody_auprc,
            **bootstrap_arguments,
        ),
        "pair_minus_antigen_only_macro_auprc": _bootstrap(
            pair_auprc,
            antigen_auprc,
            **bootstrap_arguments,
        ),
        "pair_minus_model_marginal_outer_macro_auprc": _bootstrap(
            pair_auprc,
            model_marginal_auprc,
            **bootstrap_arguments,
        ),
        "pair_minus_label_marginal_oracle_macro_auprc": _bootstrap(
            pair_auprc,
            label_oracle_auprc,
            **bootstrap_arguments,
        ),
        "pair_minus_per_instance_separable_upper_envelope_macro_auprc": _bootstrap(
            pair_auprc,
            strongest_separable,
            **bootstrap_arguments,
        ),
    }
    ordered_ids = [str(row["instance_id"]) for row in pair_rows]
    for name, matched_name, decoy_name in (
        (
            "matched_minus_decoy_topk_score",
            "matched_topk_scores",
            "decoy_topk_scores",
        ),
        (
            "matched_minus_decoy_double_centered_score",
            "matched_double_centered_scores",
            "decoy_double_centered_scores",
        ),
    ):
        if len(ordered_ids) != len(pair_partner[matched_name]):
            raise PairscaleUnaryStatisticsError("Matched partner inventory differs")
        if len(ordered_ids) != len(pair_partner[decoy_name]):
            raise PairscaleUnaryStatisticsError("Decoy partner inventory differs")
        matched = {
            identifier: _finite(value, matched_name)
            for identifier, value in zip(
                ordered_ids,
                pair_partner[matched_name],
            )
        }
        decoy = {
            identifier: _finite(value, decoy_name)
            for identifier, value in zip(
                ordered_ids,
                pair_partner[decoy_name],
            )
        }
        comparisons[name] = _bootstrap(
            matched,
            decoy,
            **bootstrap_arguments,
        )

    checks = {
        "pair_exceeds_antibody_only_lower_bound_above_zero": comparisons[
            "pair_minus_antibody_only_macro_auprc"
        ]["lower"]
        > 0.0,
        "pair_exceeds_antigen_only_lower_bound_above_zero": comparisons[
            "pair_minus_antigen_only_macro_auprc"
        ]["lower"]
        > 0.0,
        "pair_exceeds_separable_upper_envelope_lower_bound_above_zero": comparisons[
            "pair_minus_per_instance_separable_upper_envelope_macro_auprc"
        ]["lower"]
        > 0.0,
        "partner_topk_decline_lower_bound_above_zero": comparisons[
            "matched_minus_decoy_topk_score"
        ]["lower"]
        > 0.0,
        "partner_double_centered_decline_lower_bound_above_zero": comparisons[
            "matched_minus_decoy_double_centered_score"
        ]["lower"]
        > 0.0,
    }
    unresolved = ["no_centering_ablation"]
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "artifacts": {
            "assignment_content_sha256": assignment.assignment_content_sha256,
            "assignment_file_sha256": _sha256(assignment_bytes),
            "assignment_receipt_file_sha256": _sha256(receipt_bytes),
            "pair_evaluation_sha256": _sha256(pair_bytes),
            "antibody_only_evaluation_sha256": _sha256(antibody_bytes),
            "antigen_only_evaluation_sha256": _sha256(antigen_bytes),
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
        "claim_boundary": (
            "Development-only endpoint-disjoint comparison. No validation, test, "
            "confirmatory, checkpoint-promotion, scaling, biological-generalization, "
            "or state-of-the-art claim is authorized."
        ),
        "comparisons": comparisons,
        "created_at_utc": timestamp,
        "format_version": RESULT_FORMAT,
        "gate_l1": {
            "checks": checks,
            "failed_checks": failed_checks,
            "passed": not failed_checks and not unresolved,
            "scaling_authorized": False,
            "status": (
                "pending_no_centering_ablation"
                if not failed_checks
                else "failed_required_partner_specificity_checks"
            ),
            "unresolved_required_controls": unresolved,
        },
        "separable_upper_envelope_semantics": (
            "Per-complex maximum of antibody-only, antigen-only, and the pair "
            "model's separable marginal. This label-evaluated envelope is a "
            "conservative diagnostic upper envelope, not a deployable selector."
        ),
        "source_implementation_sha256": {
            "development_holdout.py": _file_sha256(
                Path(__file__).with_name("development_holdout.py")
            ),
            "metrics.py": _file_sha256(Path(__file__).with_name("metrics.py")),
            "pairscale_holdout_statistics.py": _file_sha256(
                Path(__file__).with_name("pairscale_holdout_statistics.py")
            ),
            "pairscale_unary_statistics.py": _file_sha256(Path(__file__)),
        },
    }


__all__ = [
    "PairscaleUnaryStatisticsError",
    "compare_pairscale_unary_controls",
]
