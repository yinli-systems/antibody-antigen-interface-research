"""Authenticated statistics for the PAIRScale centered multi-decoy repair."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .development_holdout import (
    connected_components,
    parse_component_holdout_assignment,
)
from .pairscale_centered_multidecoy import DECOY_COUNT, DECOY_POLICY, PARTNER_OBJECTIVE
from .pairscale_centering_ablation_statistics import (
    PairscaleCenteringStatisticsError,
    _metric_map,
    _partner_delta_map,
    _validate_no_centering_evaluation,
    _verify_internal_summaries,
    _verify_shared_label_inventory,
)
from .pairscale_holdout_statistics import (
    PairscaleHoldoutStatisticsError,
    _file_sha256,
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
    _validate_unary_evaluation,
)


RESULT_FORMAT = "pairscale_centered_multidecoy_cluster_statistics_v1"
_REPAIR_EXTRA_FIELDS = {
    "partner_objective",
    "scaling_authorized",
    "training_decoy_count",
    "training_decoy_policy",
}


class PairscaleMultidecoyStatisticsError(ValueError):
    """Raised when multi-decoy repair evidence fails closed."""


def _validate_repair_evaluation(
    evaluation: Mapping[str, Any],
    *,
    expected_instances: set[str],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    """Validate the repair envelope before delegating shared schema checks."""

    shared = {
        key: value
        for key, value in evaluation.items()
        if key not in _REPAIR_EXTRA_FIELDS
    }
    if set(evaluation) != set(shared).union(_REPAIR_EXTRA_FIELDS):
        raise PairscaleMultidecoyStatisticsError("Repair evaluation fields differ")
    if evaluation["partner_objective"] != PARTNER_OBJECTIVE:
        raise PairscaleMultidecoyStatisticsError("Repair partner objective differs")
    if evaluation["training_decoy_policy"] != DECOY_POLICY:
        raise PairscaleMultidecoyStatisticsError("Repair decoy policy differs")
    if evaluation["training_decoy_count"] != DECOY_COUNT:
        raise PairscaleMultidecoyStatisticsError("Repair decoy count differs")
    if evaluation["scaling_authorized"] is not False:
        raise PairscaleMultidecoyStatisticsError("Repair output authorizes scaling")
    if evaluation["scope"] != (
        "authority_train_only_development_double_cold_centered_multidecoy"
    ):
        raise PairscaleMultidecoyStatisticsError("Repair evaluation scope differs")
    if evaluation["evaluation_role"] != (
        "single_fixed_checkpoint_development_holdout"
    ):
        raise PairscaleMultidecoyStatisticsError("Repair evaluation role differs")
    try:
        return _validate_evaluation(shared, expected_instances=expected_instances)
    except PairscaleHoldoutStatisticsError as error:
        raise PairscaleMultidecoyStatisticsError(
            "Repair evaluation schema authentication failed"
        ) from error


def _load_assignment_and_clusters(
    assignment_bytes: bytes,
    receipt_bytes: bytes,
    *,
    expected_assignment_file_sha256: str,
    expected_assignment_content_sha256: str,
    expected_assignment_receipt_sha256: str,
) -> tuple[Any, Mapping[str, Any], dict[str, str]]:
    """Authenticate the split and derive independent-component membership."""

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
        raise PairscaleMultidecoyStatisticsError(
            "Assignment authentication failed"
        ) from error
    components = connected_components(assignment.holdout_instances)
    if sorted(value.component_sha256 for value in components) != raw_assignment[
        "holdout"
    ]["component_sha256"]:
        raise PairscaleMultidecoyStatisticsError(
            "Holdout component identities differ"
        )
    clusters = {
        instance.instance_id: component.component_sha256
        for component in components
        for instance in component.instances
    }
    return assignment, raw_assignment, clusters


def compare_pairscale_multidecoy_repair(
    *,
    assignment_path: Path,
    assignment_receipt_path: Path,
    repair_evaluation_path: Path,
    reference_evaluation_path: Path,
    no_centering_evaluation_path: Path,
    antibody_only_evaluation_path: Path,
    antigen_only_evaluation_path: Path,
    expected_assignment_file_sha256: str,
    expected_assignment_content_sha256: str,
    expected_assignment_receipt_sha256: str,
    expected_repair_evaluation_sha256: str,
    expected_reference_evaluation_sha256: str,
    expected_no_centering_evaluation_sha256: str,
    expected_antibody_only_evaluation_sha256: str,
    expected_antigen_only_evaluation_sha256: str,
    iterations: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Audit the repair against every frozen Gate-L1 development control."""

    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        raise PairscaleMultidecoyStatisticsError(
            "iterations must be a positive integer"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PairscaleMultidecoyStatisticsError("seed must be an integer")
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    try:
        _utc(timestamp)
    except PairscaleHoldoutStatisticsError as error:
        raise PairscaleMultidecoyStatisticsError("created_at_utc is invalid") from error

    artifact_specs = {
        "assignment": (assignment_path, expected_assignment_file_sha256),
        "assignment_receipt": (
            assignment_receipt_path,
            expected_assignment_receipt_sha256,
        ),
        "repair_evaluation": (
            repair_evaluation_path,
            expected_repair_evaluation_sha256,
        ),
        "reference_evaluation": (
            reference_evaluation_path,
            expected_reference_evaluation_sha256,
        ),
        "no_centering_evaluation": (
            no_centering_evaluation_path,
            expected_no_centering_evaluation_sha256,
        ),
        "antibody_only_evaluation": (
            antibody_only_evaluation_path,
            expected_antibody_only_evaluation_sha256,
        ),
        "antigen_only_evaluation": (
            antigen_only_evaluation_path,
            expected_antigen_only_evaluation_sha256,
        ),
    }
    try:
        artifacts = {
            name: _authenticated_bytes(path, digest, name)
            for name, (path, digest) in artifact_specs.items()
        }
    except PairscaleUnaryStatisticsError as error:
        raise PairscaleMultidecoyStatisticsError(
            "Artifact authentication failed"
        ) from error

    assignment, _, clusters = _load_assignment_and_clusters(
        artifacts["assignment"],
        artifacts["assignment_receipt"],
        expected_assignment_file_sha256=expected_assignment_file_sha256,
        expected_assignment_content_sha256=expected_assignment_content_sha256,
        expected_assignment_receipt_sha256=expected_assignment_receipt_sha256,
    )
    expected_instances = set(clusters)
    repair_evaluation = _strict_json(
        artifacts["repair_evaluation"], "repair evaluation"
    )
    repair_rows, repair_partner = _validate_repair_evaluation(
        repair_evaluation,
        expected_instances=expected_instances,
    )
    try:
        reference_evaluation = _strict_json(
            artifacts["reference_evaluation"], "reference evaluation"
        )
        reference_rows, reference_partner = _validate_evaluation(
            reference_evaluation,
            expected_instances=expected_instances,
        )
        no_centering_evaluation = _strict_json(
            artifacts["no_centering_evaluation"], "no-centering evaluation"
        )
        no_centering_rows, no_centering_partner = _validate_no_centering_evaluation(
            no_centering_evaluation,
            expected_instances=expected_instances,
        )
        antibody_rows = _validate_unary_evaluation(
            _strict_json(
                artifacts["antibody_only_evaluation"],
                "antibody-only evaluation",
            ),
            expected_instances=expected_instances,
            expected_control_side="antibody_only",
        )
        antigen_rows = _validate_unary_evaluation(
            _strict_json(
                artifacts["antigen_only_evaluation"],
                "antigen-only evaluation",
            ),
            expected_instances=expected_instances,
            expected_control_side="antigen_only",
        )
    except (
        PairscaleCenteringStatisticsError,
        PairscaleHoldoutStatisticsError,
        PairscaleUnaryStatisticsError,
    ) as error:
        raise PairscaleMultidecoyStatisticsError(
            "Control evaluation authentication failed"
        ) from error
    if reference_evaluation["scope"] != (
        "authority_train_only_development_double_cold_holdout"
    ) or reference_evaluation["evaluation_role"] != (
        "single_frozen_development_holdout_evaluation"
    ):
        raise PairscaleMultidecoyStatisticsError(
            "Reference evaluation role differs"
        )

    try:
        _verify_internal_summaries(
            repair_evaluation,
            repair_rows,
            repair_partner,
            name="repair",
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
        for rows in (
            reference_rows,
            no_centering_rows,
            antibody_rows,
            antigen_rows,
        ):
            _verify_shared_label_inventory(repair_rows, rows)
    except PairscaleCenteringStatisticsError as error:
        raise PairscaleMultidecoyStatisticsError(
            f"Evaluation consistency check failed: {error}"
        ) from error

    repair_auprc = _metric_map(repair_rows, "auprc")
    antibody_auprc = _metric_map(antibody_rows, "auprc")
    antigen_auprc = _metric_map(antigen_rows, "auprc")
    model_marginal_auprc = _metric_map(
        repair_rows, "model_marginal_outer_auprc"
    )
    strongest_separable = {
        identifier: max(
            antibody_auprc[identifier],
            antigen_auprc[identifier],
            model_marginal_auprc[identifier],
        )
        for identifier in repair_auprc
    }
    repair_raw_delta = _partner_delta_map(
        repair_rows,
        repair_partner,
        matched_field="matched_topk_scores",
        decoy_field="decoy_topk_scores",
    )
    repair_centered_delta = _partner_delta_map(
        repair_rows,
        repair_partner,
        matched_field="matched_double_centered_scores",
        decoy_field="decoy_double_centered_scores",
    )
    reference_raw_delta = _partner_delta_map(
        reference_rows,
        reference_partner,
        matched_field="matched_topk_scores",
        decoy_field="decoy_topk_scores",
    )
    reference_centered_delta = _partner_delta_map(
        reference_rows,
        reference_partner,
        matched_field="matched_double_centered_scores",
        decoy_field="decoy_double_centered_scores",
    )
    no_centering_raw_delta = _partner_delta_map(
        no_centering_rows,
        no_centering_partner,
        matched_field="matched_topk_scores",
        decoy_field="decoy_topk_scores",
    )
    no_centering_centered_delta = _partner_delta_map(
        no_centering_rows,
        no_centering_partner,
        matched_field="matched_double_centered_scores",
        decoy_field="decoy_double_centered_scores",
    )
    zero = {identifier: 0.0 for identifier in repair_auprc}
    bootstrap_arguments = {
        "clusters": clusters,
        "iterations": iterations,
        "seed": seed,
        "confidence": confidence,
    }
    comparisons = {
        "repair_minus_reference_macro_pair_auprc": _bootstrap(
            repair_auprc,
            _metric_map(reference_rows, "auprc"),
            **bootstrap_arguments,
        ),
        "repair_minus_no_centering_macro_pair_auprc": _bootstrap(
            repair_auprc,
            _metric_map(no_centering_rows, "auprc"),
            **bootstrap_arguments,
        ),
        "repair_minus_reference_precision_at_true_contact_count": _bootstrap(
            _metric_map(repair_rows, "precision_at_true_contact_count"),
            _metric_map(reference_rows, "precision_at_true_contact_count"),
            **bootstrap_arguments,
        ),
        "repair_minus_antibody_only_macro_pair_auprc": _bootstrap(
            repair_auprc,
            antibody_auprc,
            **bootstrap_arguments,
        ),
        "repair_minus_antigen_only_macro_pair_auprc": _bootstrap(
            repair_auprc,
            antigen_auprc,
            **bootstrap_arguments,
        ),
        "repair_minus_model_marginal_macro_pair_auprc": _bootstrap(
            repair_auprc,
            model_marginal_auprc,
            **bootstrap_arguments,
        ),
        "repair_minus_separable_upper_envelope_macro_pair_auprc": _bootstrap(
            repair_auprc,
            strongest_separable,
            **bootstrap_arguments,
        ),
        "repair_matched_minus_decoy_raw_partner_score": _bootstrap(
            repair_raw_delta,
            zero,
            **bootstrap_arguments,
        ),
        "repair_matched_minus_decoy_double_centered_partner_score": _bootstrap(
            repair_centered_delta,
            zero,
            **bootstrap_arguments,
        ),
        "repair_minus_reference_raw_partner_delta": _bootstrap(
            repair_raw_delta,
            reference_raw_delta,
            **bootstrap_arguments,
        ),
        "repair_minus_reference_double_centered_partner_delta": _bootstrap(
            repair_centered_delta,
            reference_centered_delta,
            **bootstrap_arguments,
        ),
        "repair_minus_no_centering_raw_partner_delta": _bootstrap(
            repair_raw_delta,
            no_centering_raw_delta,
            **bootstrap_arguments,
        ),
        "repair_minus_no_centering_double_centered_partner_delta": _bootstrap(
            repair_centered_delta,
            no_centering_centered_delta,
            **bootstrap_arguments,
        ),
    }
    checks = {
        "repair_exceeds_antibody_only_lower_bound_above_zero": comparisons[
            "repair_minus_antibody_only_macro_pair_auprc"
        ]["lower"]
        > 0.0,
        "repair_exceeds_antigen_only_lower_bound_above_zero": comparisons[
            "repair_minus_antigen_only_macro_pair_auprc"
        ]["lower"]
        > 0.0,
        "repair_exceeds_separable_upper_envelope_lower_bound_above_zero": comparisons[
            "repair_minus_separable_upper_envelope_macro_pair_auprc"
        ]["lower"]
        > 0.0,
        "repair_raw_partner_decline_lower_bound_above_zero": comparisons[
            "repair_matched_minus_decoy_raw_partner_score"
        ]["lower"]
        > 0.0,
        "repair_double_centered_partner_decline_lower_bound_above_zero": comparisons[
            "repair_matched_minus_decoy_double_centered_partner_score"
        ]["lower"]
        > 0.0,
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "artifacts": {
            "assignment_content_sha256": assignment.assignment_content_sha256,
            **{
                f"{name}_sha256": _sha256(raw)
                for name, raw in sorted(artifacts.items())
            },
        },
        "bootstrap": {
            "comparison_count": len(comparisons),
            "confidence": confidence,
            "independent_component_count": len(set(clusters.values())),
            "instance_count": len(expected_instances),
            "iterations": iterations,
            "seed": seed,
            "unit": "connected_endpoint_component",
        },
        "claim_boundary": (
            "Development-only endpoint-disjoint repair evidence. Statistical gate "
            "passage, if observed, requires independent review before any scaling. "
            "No validation, test, confirmatory, checkpoint-promotion, biological-"
            "generalization, comparative-superiority, or state-of-the-art claim is "
            "authorized."
        ),
        "comparisons": comparisons,
        "created_at_utc": timestamp,
        "format_version": RESULT_FORMAT,
        "gate_l1": {
            "checks": checks,
            "failed_checks": failed_checks,
            "passed_required_statistical_checks": not failed_checks,
            "scaling_authorized": False,
            "status": (
                "passed_required_statistics_requires_independent_review"
                if not failed_checks
                else "failed_required_partner_specificity_checks"
            ),
            "unresolved_required_controls": [],
        },
        "separable_upper_envelope_semantics": (
            "Per-complex maximum of antibody-only, antigen-only, and the repair "
            "model marginal. It is a conservative label-evaluated diagnostic "
            "envelope, not a deployable selector."
        ),
        "source_implementation_sha256": {
            "development_holdout.py": _file_sha256(
                Path(__file__).with_name("development_holdout.py")
            ),
            "metrics.py": _file_sha256(Path(__file__).with_name("metrics.py")),
            "pairscale_centering_ablation_statistics.py": _file_sha256(
                Path(__file__).with_name("pairscale_centering_ablation_statistics.py")
            ),
            "pairscale_holdout_statistics.py": _file_sha256(
                Path(__file__).with_name("pairscale_holdout_statistics.py")
            ),
            "pairscale_multidecoy_repair_statistics.py": _file_sha256(
                Path(__file__)
            ),
            "pairscale_unary_statistics.py": _file_sha256(
                Path(__file__).with_name("pairscale_unary_statistics.py")
            ),
        },
    }


__all__ = [
    "PairscaleMultidecoyStatisticsError",
    "compare_pairscale_multidecoy_repair",
]
