"""Prospective full-matrix evaluation for an untrained PAIRFormer candidate.

This module defines an API-level evaluation contract only. It does not load
data, select checkpoints, authorize held-out evaluation, or establish any
performance claim. Every contact-map evaluation is performed on the complete
Cartesian product of the declared valid antibody and antigen residues.
Negative-pair subsampling is intentionally unsupported.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .metrics import (
    VALIDATED_MINIMUM_INDEPENDENT_CLUSTERS,
    average_precision,
    binary_metrics,
    calibration_metrics,
    contact_map_metrics,
    optional_mean,
    paired_cluster_bootstrap_mean_difference,
    ranking_at_k,
    risk_coverage_metrics,
    roc_auc,
)
from .partner_controls import (
    partner_specificity_control_metrics,
    validate_partner_score_support,
)


def _numpy() -> Any:
    """Import NumPy only when prospective evaluation is requested."""
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError("NumPy is required for PAIRFormer evaluation") from error
    return np


@dataclass(frozen=True)
class PairComplexEvaluationInput:
    """One padded complex bound to a frozen, content-addressed residue domain."""

    complex_id: str
    contact_labels: Any
    contact_probabilities: Any
    antibody_residue_mask: Any
    antigen_residue_mask: Any
    pair_evaluation_mask: Any
    antibody_residue_ids: Sequence[str | None]
    antigen_residue_ids: Sequence[str | None]
    frozen_domain_manifest: Mapping[str, Any]
    frozen_domain_manifest_sha256: str
    pair_evaluation_mask_sha256: str


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _nonempty_identifier(value: Any, name: str) -> str:
    """Return a normalized nonempty identifier."""
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    return normalized


def _strict_bool_array(value: Any, *, ndim: int, name: str) -> Any:
    """Require an exact boolean array with the requested rank."""
    np = _numpy()
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional boolean array")
    if any(int(size) == 0 for size in array.shape):
        raise ValueError(f"{name} cannot contain an empty dimension")
    return array


def _validated_sha256(value: Any, name: str) -> str:
    """Require a canonical lowercase SHA-256 digest."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return value


def _canonical_json_sha256(value: Any, name: str) -> str:
    """Hash one JSON-safe value with a deterministic canonical encoding."""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be canonical JSON data") from error
    return hashlib.sha256(payload).hexdigest()


def _validated_residue_axis(
    residue_ids: Sequence[str | None],
    residue_mask: Any,
    *,
    name: str,
) -> tuple[str | None, ...]:
    """Bind every valid axis position to one unique canonical residue ID."""
    if isinstance(residue_ids, (str, bytes)):
        raise ValueError(f"{name} must be a residue-ID sequence")
    try:
        residue_axis = tuple(residue_ids)
    except TypeError as error:
        raise ValueError(f"{name} must be a residue-ID sequence") from error
    if len(residue_axis) != int(residue_mask.size):
        raise ValueError(f"{name} must align with its residue mask")
    normalized: list[str | None] = []
    valid_ids: set[str] = set()
    for axis_index, (raw_identifier, is_valid) in enumerate(
        zip(residue_axis, residue_mask)
    ):
        if bool(is_valid):
            if not isinstance(raw_identifier, str):
                raise ValueError(
                    f"{name}[{axis_index}] must identify every valid residue"
                )
            identifier = raw_identifier.strip()
            if not identifier or identifier != raw_identifier:
                raise ValueError(
                    f"{name} valid residue IDs must be nonempty and normalized"
                )
            if identifier in valid_ids:
                raise ValueError(f"{name} valid residue IDs must be unique")
            valid_ids.add(identifier)
            normalized.append(identifier)
        else:
            if raw_identifier is not None:
                raise ValueError(
                    f"{name}[{axis_index}] must be None for a padded position"
                )
            normalized.append(None)
    return tuple(normalized)


def build_pair_domain_manifest(
    *,
    complex_id: str,
    antibody_residue_ids: Sequence[str | None],
    antigen_residue_ids: Sequence[str | None],
    antibody_residue_mask: Any,
    antigen_residue_mask: Any,
) -> dict[str, Any]:
    """Build the exact canonical manifest required by pair evaluation."""
    complex_id = _nonempty_identifier(complex_id, "complex_id")
    antibody_mask = _strict_bool_array(
        antibody_residue_mask,
        ndim=1,
        name="antibody_residue_mask",
    )
    antigen_mask = _strict_bool_array(
        antigen_residue_mask,
        ndim=1,
        name="antigen_residue_mask",
    )
    antibody_ids = _validated_residue_axis(
        antibody_residue_ids,
        antibody_mask,
        name="antibody_residue_ids",
    )
    antigen_ids = _validated_residue_axis(
        antigen_residue_ids,
        antigen_mask,
        name="antigen_residue_ids",
    )
    return {
        "schema_version": "pair_complex_domain_v1",
        "complex_id": complex_id,
        "pair_orientation": "antibody_by_antigen",
        "candidate_domain_definition": (
            "complete_cartesian_product_of_frozen_valid_residue_ids"
        ),
        "antibody_axis": [
            {
                "axis_index": axis_index,
                "residue_id": residue_id,
                "valid": residue_id is not None,
            }
            for axis_index, residue_id in enumerate(antibody_ids)
        ],
        "antigen_axis": [
            {
                "axis_index": axis_index,
                "residue_id": residue_id,
                "valid": residue_id is not None,
            }
            for axis_index, residue_id in enumerate(antigen_ids)
        ],
    }


def pair_domain_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the content address of a frozen pair-domain manifest."""
    if not isinstance(manifest, Mapping):
        raise ValueError("frozen_domain_manifest must be a mapping")
    return _canonical_json_sha256(
        dict(manifest),
        "frozen_domain_manifest",
    )


def pair_evaluation_mask_sha256(pair_evaluation_mask: Any) -> str:
    """Return a shape-aware content address for one exact boolean pair mask."""
    mask = _strict_bool_array(
        pair_evaluation_mask,
        ndim=2,
        name="pair_evaluation_mask",
    )
    payload = {
        "schema_version": "pair_evaluation_mask_v1",
        "shape": [int(size) for size in mask.shape],
        "rows": mask.astype("uint8").tolist(),
    }
    return _canonical_json_sha256(payload, "pair_evaluation_mask")


def _strict_binary_matrix(value: Any, name: str) -> Any:
    """Require a finite two-dimensional matrix containing only zero and one."""
    np = _numpy()
    raw = np.asarray(value)
    if raw.ndim != 2 or any(int(size) == 0 for size in raw.shape):
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric binary values") from error
    if not bool(np.isfinite(array).all()) or not bool(np.isin(array, (0.0, 1.0)).all()):
        raise ValueError(f"{name} must contain only finite binary values")
    return array.astype(np.uint8)


def _strict_probability_matrix(value: Any, name: str) -> Any:
    """Require a finite two-dimensional probability matrix."""
    np = _numpy()
    raw = np.asarray(value)
    if raw.ndim != 2 or any(int(size) == 0 for size in raw.shape):
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric probabilities") from error
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must contain only finite probabilities")
    if bool(np.logical_or(array < 0.0, array > 1.0).any()):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return array


def _probability_threshold(value: Any) -> float:
    """Validate a fixed binary decision threshold."""
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("threshold must be a finite probability") from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite probability in [0, 1]")
    return threshold


def _positive_unique_cutoffs(values: Sequence[int]) -> tuple[int, ...]:
    """Require an explicit finite sequence of unique positive integer cutoffs."""
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Ranking cutoffs must be positive integers")
        normalized.append(value)
    if not normalized:
        raise ValueError("At least one ranking cutoff is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Ranking cutoffs must be unique")
    return tuple(normalized)


def _validate_complete_pair_support(
    record: PairComplexEvaluationInput,
) -> tuple[str, Any, Any, Any, Any, Any]:
    """Validate shapes and prove that no valid residue pair was removed."""
    np = _numpy()
    if not isinstance(record, PairComplexEvaluationInput):
        raise TypeError("Every record must be a PairComplexEvaluationInput")
    complex_id = _nonempty_identifier(record.complex_id, "complex_id")
    labels = _strict_binary_matrix(record.contact_labels, "contact_labels")
    probabilities = _strict_probability_matrix(
        record.contact_probabilities,
        "contact_probabilities",
    )
    antibody_mask = _strict_bool_array(
        record.antibody_residue_mask,
        ndim=1,
        name="antibody_residue_mask",
    )
    antigen_mask = _strict_bool_array(
        record.antigen_residue_mask,
        ndim=1,
        name="antigen_residue_mask",
    )
    pair_mask = _strict_bool_array(
        record.pair_evaluation_mask,
        ndim=2,
        name="pair_evaluation_mask",
    )
    expected_shape = (int(antibody_mask.size), int(antigen_mask.size))
    if labels.shape != expected_shape or probabilities.shape != expected_shape:
        raise ValueError(
            "Contact labels and probabilities must have shape "
            "[antibody_residue, antigen_residue]"
        )
    if pair_mask.shape != expected_shape:
        raise ValueError("pair_evaluation_mask must align with contact matrices")
    if not bool(antibody_mask.any()) or not bool(antigen_mask.any()):
        raise ValueError("Each complex must contain valid residues on both partners")

    expected_domain_manifest = build_pair_domain_manifest(
        complex_id=complex_id,
        antibody_residue_ids=record.antibody_residue_ids,
        antigen_residue_ids=record.antigen_residue_ids,
        antibody_residue_mask=antibody_mask,
        antigen_residue_mask=antigen_mask,
    )
    frozen_domain_manifest_sha256 = _validated_sha256(
        record.frozen_domain_manifest_sha256,
        "frozen_domain_manifest_sha256",
    )
    actual_domain_manifest_sha256 = pair_domain_manifest_sha256(
        record.frozen_domain_manifest
    )
    if actual_domain_manifest_sha256 != frozen_domain_manifest_sha256:
        raise ValueError(
            "frozen_domain_manifest does not match its declared SHA-256 digest"
        )
    if dict(record.frozen_domain_manifest) != expected_domain_manifest:
        raise ValueError(
            "Frozen domain manifest does not exactly match the complete residue axes"
        )
    declared_pair_mask_sha256 = _validated_sha256(
        record.pair_evaluation_mask_sha256,
        "pair_evaluation_mask_sha256",
    )
    actual_pair_mask_sha256 = pair_evaluation_mask_sha256(pair_mask)
    if actual_pair_mask_sha256 != declared_pair_mask_sha256:
        raise ValueError(
            "pair_evaluation_mask does not match its declared SHA-256 digest"
        )

    complete_cartesian_mask = antibody_mask[:, None] & antigen_mask[None, :]
    if not bool(np.array_equal(pair_mask, complete_cartesian_mask)):
        raise ValueError(
            "pair_evaluation_mask must equal the complete Cartesian product of "
            "valid antibody and antigen residues; negative subsampling is forbidden"
        )
    if bool((labels[~pair_mask] != 0).any()):
        raise ValueError("Padded contact labels must be exactly zero")
    if bool((probabilities[~pair_mask] != 0.0).any()):
        raise ValueError("Padded contact probabilities must be exactly zero")

    evaluated_labels = labels[pair_mask]
    positive_count = int(evaluated_labels.sum())
    if positive_count == 0:
        raise ValueError("Each evaluated complex must contain at least one contact")
    if positive_count == int(evaluated_labels.size):
        raise ValueError(
            "Each evaluated complex must contain at least one non-contact pair"
        )
    expected_count = int(antibody_mask.sum()) * int(antigen_mask.sum())
    if int(pair_mask.sum()) != expected_count:
        raise AssertionError("Complete pair-support validation is inconsistent")
    return complex_id, labels, probabilities, antibody_mask, antigen_mask, pair_mask


def _conditional_ranking(
    labels: Any,
    probabilities: Any,
    *,
    cutoffs: Sequence[int],
    unit_name: str,
) -> dict[str, Any]:
    """Rank the opposite partner separately for every residue with a contact."""
    np = _numpy()
    if labels.ndim != 2 or probabilities.shape != labels.shape:
        raise ValueError("Conditional-ranking matrices must be aligned and two-dimensional")
    if labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError("Conditional ranking requires nonempty matrices")

    average_precisions: list[float] = []
    aucs: list[float] = []
    reciprocal_ranks: list[float] = []
    precision_by_cutoff: dict[int, list[float]] = {int(k): [] for k in cutoffs}
    recall_by_cutoff: dict[int, list[float]] = {int(k): [] for k in cutoffs}
    no_positive_count = 0
    no_negative_among_positive_count = 0
    for unit_labels, unit_probabilities in zip(labels, probabilities):
        positive_count = int(unit_labels.sum())
        if positive_count == 0:
            no_positive_count += 1
            continue
        ap = average_precision(unit_labels, unit_probabilities)
        if ap is None:
            raise AssertionError("A positive conditional unit must have defined AP")
        average_precisions.append(float(ap))
        auc = roc_auc(unit_labels, unit_probabilities)
        if auc is None:
            no_negative_among_positive_count += 1
        else:
            aucs.append(float(auc))

        best_positive = float(unit_probabilities[unit_labels.astype(bool)].max())
        negative_scores = unit_probabilities[~unit_labels.astype(bool)]
        conservative_rank = 1 + int((negative_scores >= best_positive).sum())
        reciprocal_ranks.append(1.0 / conservative_rank)
        ranking = ranking_at_k(unit_labels, unit_probabilities, cutoffs=cutoffs)
        for cutoff in precision_by_cutoff:
            precision_by_cutoff[cutoff].append(
                float(ranking[f"precision_at_{cutoff}"])
            )
            recall = ranking[f"recall_at_{cutoff}"]
            if recall is None:
                raise AssertionError("A positive conditional unit must have defined recall")
            recall_by_cutoff[cutoff].append(float(recall))

    valid_unit_count = int(labels.shape[0])
    eligible_count = len(average_precisions)
    if eligible_count == 0:
        raise ValueError(f"{unit_name} conditional ranking has no positive-eligible unit")
    result: dict[str, Any] = {
        "unit_name": unit_name,
        "valid_unit_count": valid_unit_count,
        "positive_eligible_unit_count": eligible_count,
        "ineligible_no_positive_unit_count": no_positive_count,
        "auroc_eligible_unit_count": len(aucs),
        "ineligible_no_negative_among_positive_unit_count": (
            no_negative_among_positive_count
        ),
        "mean_average_precision": float(np.mean(average_precisions)),
        "mean_auroc": float(np.mean(aucs)) if aucs else None,
        "mean_reciprocal_rank_first_positive": float(np.mean(reciprocal_ranks)),
        "tie_policy": "conservative_negative_score_greater_than_or_equal_to_best_positive",
    }
    for cutoff in precision_by_cutoff:
        result[f"mean_precision_at_{cutoff}"] = float(
            np.mean(precision_by_cutoff[cutoff])
        )
        result[f"mean_recall_at_{cutoff}"] = float(
            np.mean(recall_by_cutoff[cutoff])
        )
    return result


def _pair_derived_residue_metrics(
    labels: Any,
    probabilities: Any,
    *,
    threshold: float,
    cutoffs: Sequence[int],
    residue_role: str,
) -> dict[str, Any]:
    """Evaluate residue labels derived only from the contact matrix."""
    residue_labels = labels.any(axis=0).astype("uint8")
    residue_probabilities = probabilities.max(axis=0)
    result = binary_metrics(residue_labels, residue_probabilities, threshold=threshold)
    result["dice"] = result["f1"]
    result.update(ranking_at_k(residue_labels, residue_probabilities, cutoffs=cutoffs))
    result["residue_role"] = residue_role
    result["score_derivation"] = "maximum_pair_contact_probability"
    return result


def evaluate_pair_complex(
    record: PairComplexEvaluationInput,
    *,
    threshold: float = 0.5,
    ranking_cutoffs: Sequence[int] = (1, 5, 10),
    calibration_bins: int = 15,
    risk_coverages: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    """Evaluate one complex on every valid antibody-antigen residue pair."""
    threshold = _probability_threshold(threshold)
    ranking_cutoffs = _positive_unique_cutoffs(ranking_cutoffs)
    (
        complex_id,
        labels,
        probabilities,
        antibody_mask,
        antigen_mask,
        pair_mask,
    ) = _validate_complete_pair_support(record)
    evaluated_labels = labels[pair_mask]
    evaluated_probabilities = probabilities[pair_mask]

    pair_metrics = contact_map_metrics(
        evaluated_labels,
        evaluated_probabilities,
        threshold=threshold,
    )
    pair_metrics["dice"] = pair_metrics["f1"]
    pair_metrics.update(
        ranking_at_k(
            evaluated_labels,
            evaluated_probabilities,
            cutoffs=ranking_cutoffs,
        )
    )
    pair_calibration = calibration_metrics(
        evaluated_labels,
        evaluated_probabilities,
        bins=calibration_bins,
        strategy="adaptive",
    )
    pair_risk_coverage = risk_coverage_metrics(
        evaluated_labels,
        evaluated_probabilities,
        coverages=risk_coverages,
        decision_threshold=threshold,
    )

    dense_labels = labels[antibody_mask][:, antigen_mask]
    dense_probabilities = probabilities[antibody_mask][:, antigen_mask]
    row_conditional = _conditional_ranking(
        dense_labels,
        dense_probabilities,
        cutoffs=ranking_cutoffs,
        unit_name="antibody_residue_row_ranking_antigen_residues",
    )
    column_conditional = _conditional_ranking(
        dense_labels.T,
        dense_probabilities.T,
        cutoffs=ranking_cutoffs,
        unit_name="antigen_residue_column_ranking_antibody_residues",
    )
    epitope_metrics = _pair_derived_residue_metrics(
        dense_labels,
        dense_probabilities,
        threshold=threshold,
        cutoffs=ranking_cutoffs,
        residue_role="epitope",
    )
    paratope_metrics = _pair_derived_residue_metrics(
        dense_labels.T,
        dense_probabilities.T,
        threshold=threshold,
        cutoffs=ranking_cutoffs,
        residue_role="paratope",
    )
    return {
        "complex_id": complex_id,
        "evaluation_contract": "prospective_pairformer_full_matrix_v2",
        "pair_orientation": "antibody_by_antigen",
        "negative_subsampling_performed": False,
        "support": {
            "antibody_residue_count": int(antibody_mask.sum()),
            "antigen_residue_count": int(antigen_mask.sum()),
            "evaluated_pair_count": int(pair_mask.sum()),
            "expected_complete_cartesian_pair_count": (
                int(antibody_mask.sum()) * int(antigen_mask.sum())
            ),
            "complete_cartesian_support": True,
            "frozen_residue_domain_bound": True,
            "frozen_domain_manifest_sha256": (
                record.frozen_domain_manifest_sha256
            ),
            "pair_evaluation_mask_sha256": (
                record.pair_evaluation_mask_sha256
            ),
        },
        "pair": pair_metrics,
        "pair_calibration": pair_calibration,
        "pair_risk_coverage": pair_risk_coverage,
        "conditional_ranking": {
            "row_conditional": row_conditional,
            "column_conditional": column_conditional,
        },
        "pair_derived_marginals": {
            "epitope": epitope_metrics,
            "paratope": paratope_metrics,
        },
    }


def _macro_metrics(
    records: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> dict[str, Any]:
    """Compute a complex-level macro mean and report definition counts."""
    result: dict[str, Any] = {}
    for key in keys:
        values = [record[key] for record in records]
        result[key] = optional_mean(values)
        result[f"{key}_defined_complex_count"] = sum(
            value is not None for value in values
        )
    return result


def _aggregate_conditional(
    records: Sequence[Mapping[str, Any]],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    """Aggregate conditional rankings without hiding eligibility."""
    scalar_keys = [
        "mean_average_precision",
        "mean_auroc",
        "mean_reciprocal_rank_first_positive",
    ]
    for cutoff in cutoffs:
        scalar_keys.extend(
            (f"mean_precision_at_{int(cutoff)}", f"mean_recall_at_{int(cutoff)}")
        )
    result = _macro_metrics(records, scalar_keys)
    for key in (
        "valid_unit_count",
        "positive_eligible_unit_count",
        "ineligible_no_positive_unit_count",
        "auroc_eligible_unit_count",
        "ineligible_no_negative_among_positive_unit_count",
    ):
        result[f"total_{key}"] = sum(int(record[key]) for record in records)
    result["complex_count"] = len(records)
    result["aggregation"] = "macro_over_complex_level_conditional_means"
    return result


def evaluate_pair_dataset(
    records: Sequence[PairComplexEvaluationInput],
    *,
    threshold: float = 0.5,
    ranking_cutoffs: Sequence[int] = (1, 5, 10),
    calibration_bins: int = 15,
    risk_coverages: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    """Evaluate a dataset with per-complex full-matrix macro AUPRC primary."""
    if not records:
        raise ValueError("Dataset evaluation requires at least one complex")
    ranking_cutoffs = _positive_unique_cutoffs(ranking_cutoffs)
    per_complex = [
        evaluate_pair_complex(
            record,
            threshold=threshold,
            ranking_cutoffs=ranking_cutoffs,
            calibration_bins=calibration_bins,
            risk_coverages=risk_coverages,
        )
        for record in records
    ]
    identifiers = [record["complex_id"] for record in per_complex]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Dataset complex identifiers must be unique")

    pair_keys = [
        "auprc",
        "auroc",
        "mcc",
        "f1",
        "dice",
        "precision",
        "recall",
        "contact_iou",
        "precision_at_true_contact_count",
    ]
    for cutoff in ranking_cutoffs:
        pair_keys.extend(
            (f"precision_at_{int(cutoff)}", f"recall_at_{int(cutoff)}")
        )
    pair_records = [record["pair"] for record in per_complex]
    pair_macro = _macro_metrics(pair_records, pair_keys)
    primary_value = pair_macro["auprc"]
    if primary_value is None:
        raise AssertionError("Validated complexes must have defined pair AUPRC")

    calibration_records = [record["pair_calibration"] for record in per_complex]
    calibration_macro = _macro_metrics(
        calibration_records,
        (
            "brier_score",
            "negative_log_likelihood",
            "expected_calibration_error",
            "maximum_calibration_error",
        ),
    )
    calibration_macro["strategy"] = "adaptive_tie_preserving"
    calibration_macro["bin_count"] = int(calibration_bins)

    risk_points: list[dict[str, Any]] = []
    for point_index, requested_coverage in enumerate(risk_coverages):
        points = [
            record["pair_risk_coverage"]["points"][point_index]
            for record in per_complex
        ]
        risk_points.append(
            {
                "requested_coverage": float(requested_coverage),
                "macro_realized_coverage": optional_mean(
                    point["realized_coverage"] for point in points
                ),
                "macro_classification_risk": optional_mean(
                    point["classification_risk"] for point in points
                ),
                "macro_accuracy": optional_mean(point["accuracy"] for point in points),
                "complex_count": len(points),
            }
        )

    row_records = [
        record["conditional_ranking"]["row_conditional"] for record in per_complex
    ]
    column_records = [
        record["conditional_ranking"]["column_conditional"]
        for record in per_complex
    ]
    epitope_records = [
        record["pair_derived_marginals"]["epitope"] for record in per_complex
    ]
    paratope_records = [
        record["pair_derived_marginals"]["paratope"] for record in per_complex
    ]
    marginal_keys = [
        "auprc",
        "auroc",
        "mcc",
        "f1",
        "dice",
        "precision",
        "recall",
    ]
    for cutoff in ranking_cutoffs:
        marginal_keys.extend(
            (f"precision_at_{int(cutoff)}", f"recall_at_{int(cutoff)}")
        )
    return {
        "evaluation_contract": "prospective_pairformer_full_matrix_v2",
        "candidate_status": "untrained_api_level_evaluator",
        "complex_count": len(per_complex),
        "negative_subsampling_performed": False,
        "primary_metric": {
            "name": "macro_per_complex_full_matrix_pair_auprc",
            "value": float(primary_value),
            "defined_complex_count": pair_macro["auprc_defined_complex_count"],
            "aggregation": "unweighted_mean_over_complexes",
        },
        "macro": {
            "pair": pair_macro,
            "pair_calibration": calibration_macro,
            "pair_risk_coverage": {
                "selection_rule": (
                    "include_all_examples_with_absolute_decision_margin_at_or_"
                    "above_the_requested_coverage_boundary"
                ),
                "selection_tie_policy": (
                    "include_all_boundary_confidence_ties"
                ),
                "classification_decision_threshold": float(
                    _probability_threshold(threshold)
                ),
                "classification_decision_rule": (
                    "probability_greater_than_or_equal_to_threshold_predicts_"
                    "positive"
                ),
                "points": risk_points,
            },
            "row_conditional": _aggregate_conditional(
                row_records,
                ranking_cutoffs,
            ),
            "column_conditional": _aggregate_conditional(
                column_records,
                ranking_cutoffs,
            ),
            "pair_derived_epitope": _macro_metrics(
                epitope_records,
                marginal_keys,
            ),
            "pair_derived_paratope": _macro_metrics(
                paratope_records,
                marginal_keys,
            ),
        },
        "per_complex": per_complex,
    }


def _directional_cognate_retrieval(
    scores: Any,
    positive_mask: Any,
    *,
    query_name: str,
    candidate_name: str,
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    """Evaluate one exhaustive direction with separate retrieval and pairwise rules."""
    np = _numpy()
    ranks: list[int] = []
    normalized_ranks: list[float] = []
    hardest_negative_strict_successes: list[float] = []
    hardest_negative_ties: list[float] = []
    query_pairwise_half_tie_win_rates: list[float] = []
    query_pairwise_strict_win_rates: list[float] = []
    query_pairwise_tie_rates: list[float] = []
    pairwise_comparison_counts: list[int] = []
    total_pairwise_strict_wins = 0
    total_pairwise_ties = 0
    total_pairwise_comparisons = 0
    positive_counts: list[int] = []
    for query_scores, query_positives in zip(scores, positive_mask):
        positive_count = int(query_positives.sum())
        negative_count = int(query_positives.size - positive_count)
        if positive_count == 0:
            raise ValueError(f"Every {query_name} query must have a cognate candidate")
        if negative_count == 0:
            raise ValueError(
                f"Every {query_name} query must have a noncognate candidate"
            )
        best_positive = float(query_scores[query_positives].max())
        negative_scores = query_scores[~query_positives]
        hardest_negative = float(negative_scores.max())
        rank = 1 + int((negative_scores >= best_positive).sum())
        ranks.append(rank)
        normalized_ranks.append((rank - 1) / negative_count)
        hardest_negative_strict_successes.append(
            float(best_positive > hardest_negative)
        )
        hardest_negative_ties.append(float(best_positive == hardest_negative))

        positive_scores = query_scores[query_positives]
        pairwise_strict_wins = int(
            (positive_scores[:, None] > negative_scores[None, :]).sum()
        )
        pairwise_ties = int(
            (positive_scores[:, None] == negative_scores[None, :]).sum()
        )
        pairwise_comparison_count = positive_count * negative_count
        query_pairwise_strict_win_rates.append(
            pairwise_strict_wins / pairwise_comparison_count
        )
        query_pairwise_tie_rates.append(
            pairwise_ties / pairwise_comparison_count
        )
        query_pairwise_half_tie_win_rates.append(
            (pairwise_strict_wins + 0.5 * pairwise_ties)
            / pairwise_comparison_count
        )
        pairwise_comparison_counts.append(pairwise_comparison_count)
        total_pairwise_strict_wins += pairwise_strict_wins
        total_pairwise_ties += pairwise_ties
        total_pairwise_comparisons += pairwise_comparison_count
        positive_counts.append(positive_count)

    rank_array = np.asarray(ranks, dtype=np.int64)
    result: dict[str, Any] = {
        "query_name": query_name,
        "candidate_name": candidate_name,
        "query_count": int(rank_array.size),
        "candidate_count_per_query": int(scores.shape[1]),
        "minimum_cognate_count_per_query": min(positive_counts),
        "maximum_cognate_count_per_query": max(positive_counts),
        "mean_reciprocal_rank": float((1.0 / rank_array).mean()),
        "mean_rank": float(rank_array.mean()),
        "median_rank": float(np.median(rank_array)),
        "mean_normalized_rank": float(np.mean(normalized_ranks)),
        "hardest_negative_strict_top1_success_rate": float(
            np.mean(hardest_negative_strict_successes)
        ),
        "hardest_negative_top1_tie_rate": float(
            np.mean(hardest_negative_ties)
        ),
        "hardest_negative_top1_definition": (
            "best_cognate_score_strictly_exceeds_every_noncognate_score"
        ),
        "retrieval_rank_tie_policy": (
            "every_noncognate_tied_with_the_best_cognate_conservatively_"
            "precedes_it"
        ),
        "pairwise_cognate_vs_noncognate_win_rate": float(
            np.mean(query_pairwise_half_tie_win_rates)
        ),
        "pairwise_strict_win_rate": float(
            np.mean(query_pairwise_strict_win_rates)
        ),
        "pairwise_tie_rate": float(np.mean(query_pairwise_tie_rates)),
        "micro_pairwise_cognate_vs_noncognate_win_rate": (
            total_pairwise_strict_wins + 0.5 * total_pairwise_ties
        )
        / total_pairwise_comparisons,
        "total_pairwise_comparison_count": total_pairwise_comparisons,
        "minimum_pairwise_comparison_count_per_query": min(
            pairwise_comparison_counts
        ),
        "maximum_pairwise_comparison_count_per_query": max(
            pairwise_comparison_counts
        ),
        "pairwise_tie_policy": "one_half_credit_per_tied_cognate_noncognate_pair",
        "pairwise_multi_cognate_policy": (
            "all_cognate_by_noncognate_pairs_within_each_query_then_"
            "macro_average_queries"
        ),
    }
    for cutoff in cutoffs:
        result[f"recall_at_{int(cutoff)}"] = float(
            (rank_array <= int(cutoff)).mean()
        )
    return result


def evaluate_bidirectional_cognate_retrieval(
    score_matrix: Any,
    cognate_mask: Any,
    exhaustive_candidate_mask: Any,
    *,
    cutoffs: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """Evaluate exhaustive antibody-to-antigen and antigen-to-antibody retrieval."""
    np = _numpy()
    cutoffs = _positive_unique_cutoffs(cutoffs)
    raw_scores = np.asarray(score_matrix)
    if raw_scores.ndim != 2 or any(int(size) < 2 for size in raw_scores.shape):
        raise ValueError("score_matrix must have shape [antibodies, antigens] with both axes >= 2")
    try:
        scores = np.asarray(raw_scores, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("score_matrix must contain numeric scores") from error
    if not bool(np.isfinite(scores).all()):
        raise ValueError("score_matrix must contain only finite scores")
    positives = _strict_bool_array(cognate_mask, ndim=2, name="cognate_mask")
    candidates = _strict_bool_array(
        exhaustive_candidate_mask,
        ndim=2,
        name="exhaustive_candidate_mask",
    )
    if positives.shape != scores.shape or candidates.shape != scores.shape:
        raise ValueError("Retrieval masks must align with score_matrix")
    if bool((positives & ~candidates).any()):
        raise ValueError("cognate_mask cannot include a noncandidate pair")
    if not bool(candidates.all()):
        raise ValueError(
            "exhaustive_candidate_mask must include the complete antibody-antigen "
            "candidate matrix; candidate subsampling is forbidden"
        )

    antibody_to_antigen = _directional_cognate_retrieval(
        scores,
        positives,
        query_name="antibody",
        candidate_name="antigen",
        cutoffs=cutoffs,
    )
    antigen_to_antibody = _directional_cognate_retrieval(
        scores.T,
        positives.T,
        query_name="antigen",
        candidate_name="antibody",
        cutoffs=cutoffs,
    )
    macro_keys = [
        "mean_reciprocal_rank",
        "mean_normalized_rank",
        "hardest_negative_strict_top1_success_rate",
        "hardest_negative_top1_tie_rate",
        "pairwise_cognate_vs_noncognate_win_rate",
        "pairwise_strict_win_rate",
        "pairwise_tie_rate",
    ]
    for cutoff in cutoffs:
        macro_keys.append(f"recall_at_{int(cutoff)}")
    macro = {
        key: float(
            (
                float(antibody_to_antigen[key])
                + float(antigen_to_antibody[key])
            )
            / 2.0
        )
        for key in macro_keys
    }
    return {
        "evaluation_contract": "prospective_pairformer_exhaustive_retrieval_v2",
        "negative_candidate_subsampling_performed": False,
        "antibody_to_antigen": antibody_to_antigen,
        "antigen_to_antibody": antigen_to_antibody,
        "macro_bidirectional": macro,
    }


def evaluate_named_partner_controls(
    matched_scores: Sequence[float],
    controls_by_name: Mapping[str, Any],
    matched_candidate_pair_counts: Sequence[int],
    control_candidate_pair_counts_by_name: Mapping[str, Any],
    *,
    matched_axis_valid_counts: Any,
    control_axis_valid_counts_by_name: Mapping[str, Any],
    matched_candidate_domain_sha256: Sequence[str],
    control_candidate_domain_sha256_by_name: Mapping[str, Any],
    matched_candidate_mask_sha256: Sequence[str],
    control_candidate_mask_sha256_by_name: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate controls only after exact authenticated support parity.

    Every per-axis count is ordered ``[antibody, antigen]``. Domain and mask
    digests must be canonical SHA-256 values and must exactly equal the frozen
    matched-support digests for every control candidate.
    """
    np = _numpy()
    matched = np.asarray(matched_scores, dtype=np.float64)
    matched_counts = np.asarray(matched_candidate_pair_counts)
    if matched.ndim != 1 or matched.size == 0 or not bool(np.isfinite(matched).all()):
        raise ValueError("matched_scores must be a nonempty finite vector")
    if (
        matched_counts.ndim != 1
        or matched_counts.shape != matched.shape
        or not bool(np.issubdtype(matched_counts.dtype, np.integer))
        or bool((matched_counts <= 0).any())
    ):
        raise ValueError(
            "matched_candidate_pair_counts must be aligned positive integers"
        )
    matched_axis_counts = np.asarray(matched_axis_valid_counts)
    if (
        matched_axis_counts.shape != (matched.size, 2)
        or not bool(np.issubdtype(matched_axis_counts.dtype, np.integer))
        or bool((matched_axis_counts <= 0).any())
    ):
        raise ValueError(
            "matched_axis_valid_counts must have shape [instances, 2] and "
            "contain positive integers"
        )
    if bool((matched_counts > matched_axis_counts.prod(axis=1)).any()):
        raise ValueError(
            "Matched candidate-pair counts cannot exceed the Cartesian axis domain"
        )

    def sha256_array(value: Any, shape: tuple[int, ...], name: str) -> Any:
        try:
            array = np.asarray(value, dtype=object)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must have shape {shape}") from error
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        normalized = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            normalized[index] = _validated_sha256(array[index], name)
        return normalized

    matched_domain_hashes = sha256_array(
        matched_candidate_domain_sha256,
        (matched.size,),
        "matched_candidate_domain_sha256",
    )
    matched_mask_hashes = sha256_array(
        matched_candidate_mask_sha256,
        (matched.size,),
        "matched_candidate_mask_sha256",
    )
    control_mappings = (
        control_candidate_pair_counts_by_name,
        control_axis_valid_counts_by_name,
        control_candidate_domain_sha256_by_name,
        control_candidate_mask_sha256_by_name,
    )
    if any(set(controls_by_name) != set(mapping) for mapping in control_mappings):
        raise ValueError(
            "All control score, count, axis, domain-hash, and mask-hash mappings "
            "must have identical names"
        )
    if not controls_by_name:
        raise ValueError("At least one named partner-control family is required")

    support_by_control: dict[str, Any] = {}
    all_exact = True
    for raw_name, raw_scores in controls_by_name.items():
        name = _nonempty_identifier(raw_name, "partner-control name")
        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.ndim == 1:
            scores = scores[:, None]
        if (
            scores.ndim != 2
            or scores.shape[0] != matched.size
            or scores.shape[1] == 0
        ):
            raise ValueError(
                f"Control {name!r} scores must have shape [instances, candidates]"
            )
        raw_counts = np.asarray(control_candidate_pair_counts_by_name[raw_name])
        if raw_counts.ndim == 1:
            raw_counts = raw_counts[:, None]
        if raw_counts.shape != scores.shape:
            raise ValueError(
                f"Control {name!r} candidate counts must align with its scores"
            )
        if (
            not bool(np.issubdtype(raw_counts.dtype, np.integer))
            or bool((raw_counts <= 0).any())
        ):
            raise ValueError(
                f"Control {name!r} candidate counts must be positive integers"
            )
        raw_axis_counts = np.asarray(
            control_axis_valid_counts_by_name[raw_name]
        )
        if raw_axis_counts.ndim == 2 and scores.shape[1] == 1:
            raw_axis_counts = raw_axis_counts[:, None, :]
        if (
            raw_axis_counts.shape != (matched.size, scores.shape[1], 2)
            or not bool(np.issubdtype(raw_axis_counts.dtype, np.integer))
            or bool((raw_axis_counts <= 0).any())
        ):
            raise ValueError(
                f"Control {name!r} axis counts must have shape "
                "[instances, candidates, 2] and contain positive integers"
            )
        if bool((raw_counts > raw_axis_counts.prod(axis=2)).any()):
            raise ValueError(
                f"Control {name!r} candidate-pair counts cannot exceed its "
                "Cartesian axis domain"
            )
        raw_domain_hashes = np.asarray(
            control_candidate_domain_sha256_by_name[raw_name],
            dtype=object,
        )
        raw_mask_hashes = np.asarray(
            control_candidate_mask_sha256_by_name[raw_name],
            dtype=object,
        )
        if raw_domain_hashes.ndim == 1 and scores.shape[1] == 1:
            raw_domain_hashes = raw_domain_hashes[:, None]
        if raw_mask_hashes.ndim == 1 and scores.shape[1] == 1:
            raw_mask_hashes = raw_mask_hashes[:, None]
        control_shape = (matched.size, scores.shape[1])
        raw_domain_hashes = sha256_array(
            raw_domain_hashes,
            control_shape,
            f"Control {name!r} candidate-domain hashes",
        )
        raw_mask_hashes = sha256_array(
            raw_mask_hashes,
            control_shape,
            f"Control {name!r} candidate-mask hashes",
        )

        repeated_matched = np.repeat(matched_counts, scores.shape[1])
        audit = validate_partner_score_support(
            repeated_matched,
            {name: raw_counts.reshape(-1)},
        )
        family_audit = audit["by_control"][name]
        exact_pair_count_match = bool(family_audit["exact_support_match"])
        repeated_axis_counts = np.repeat(
            matched_axis_counts[:, None, :],
            scores.shape[1],
            axis=1,
        )
        exact_axis_count_match = bool(
            np.array_equal(raw_axis_counts, repeated_axis_counts)
        )
        repeated_domain_hashes = np.repeat(
            matched_domain_hashes[:, None],
            scores.shape[1],
            axis=1,
        )
        repeated_mask_hashes = np.repeat(
            matched_mask_hashes[:, None],
            scores.shape[1],
            axis=1,
        )
        exact_domain_hash_match = bool(
            np.array_equal(raw_domain_hashes, repeated_domain_hashes)
        )
        exact_mask_hash_match = bool(
            np.array_equal(raw_mask_hashes, repeated_mask_hashes)
        )
        exact_support_match = (
            exact_pair_count_match
            and exact_axis_count_match
            and exact_domain_hash_match
            and exact_mask_hash_match
        )
        family_audit["exact_candidate_pair_count_match"] = (
            exact_pair_count_match
        )
        family_audit["exact_per_axis_valid_count_match"] = (
            exact_axis_count_match
        )
        family_audit["exact_candidate_domain_hash_match"] = (
            exact_domain_hash_match
        )
        family_audit["exact_candidate_mask_hash_match"] = (
            exact_mask_hash_match
        )
        family_audit["exact_support_match"] = exact_support_match
        family_audit["control_candidates_per_instance"] = int(scores.shape[1])
        support_by_control[name] = family_audit
        all_exact = all_exact and exact_support_match

    support_audit = {
        "instance_count": int(matched.size),
        "all_controls_exact_support_match": all_exact,
        "parity_scope": (
            "exact_pair_count_per_axis_valid_counts_candidate_domain_sha256_"
            "and_candidate_mask_sha256_per_control_candidate"
        ),
        "hash_algorithm": "sha256",
        "axis_count_order": ["antibody", "antigen"],
        "hash_authentication_scope": (
            "content_address_equality_against_frozen_matched_support"
        ),
        "by_control": {
            name: support_by_control[name] for name in sorted(support_by_control)
        },
    }
    if not all_exact:
        raise ValueError(
            "Named partner controls must have authenticated exact candidate "
            "support parity"
        )
    return {
        "candidate_support_audit": support_audit,
        "metrics": partner_specificity_control_metrics(
            matched_scores,
            controls_by_name,
        ),
    }


def paired_cluster_bootstrap_comparison(
    candidate_by_instance: Mapping[str, float],
    reference_by_instance: Mapping[str, float],
    cluster_by_instance: Mapping[str, str],
    *,
    iterations: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
    minimum_independent_clusters: int = VALIDATED_MINIMUM_INDEPENDENT_CLUSTERS,
    insufficient_cluster_policy: str = "raise",
) -> dict[str, Any]:
    """Run the preregistered paired cluster bootstrap, defaulting to 10,000 draws."""
    return paired_cluster_bootstrap_mean_difference(
        candidate_by_instance,
        reference_by_instance,
        cluster_by_instance,
        iterations=iterations,
        seed=seed,
        confidence=confidence,
        minimum_independent_clusters=minimum_independent_clusters,
        insufficient_cluster_policy=insufficient_cluster_policy,
    )


def holm_bonferroni_correction(
    p_values_by_hypothesis: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Control family-wise error for one finite, explicitly named family."""
    if not p_values_by_hypothesis:
        raise ValueError("Multiplicity correction requires a nonempty finite family")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha must be a finite value in (0, 1)") from error
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be a finite value in (0, 1)")

    normalized: dict[str, float] = {}
    for raw_name, raw_p_value in p_values_by_hypothesis.items():
        name = _nonempty_identifier(raw_name, "hypothesis name")
        if name in normalized:
            raise ValueError(f"Duplicate normalized hypothesis name: {name!r}")
        try:
            p_value = float(raw_p_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Hypothesis {name!r} has an invalid p-value") from error
        if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError(f"Hypothesis {name!r} p-value must be in [0, 1]")
        normalized[name] = p_value

    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    running_adjusted = 0.0
    hypotheses: dict[str, dict[str, Any]] = {}
    for rank_index, (name, p_value) in enumerate(ordered):
        multiplier = family_size - rank_index
        running_adjusted = max(running_adjusted, multiplier * p_value)
        adjusted = min(1.0, running_adjusted)
        hypotheses[name] = {
            "raw_p_value": p_value,
            "adjusted_p_value": adjusted,
            "reject_family_wise_null": adjusted <= alpha,
            "step_down_rank": rank_index + 1,
            "multiplier": multiplier,
        }
    return {
        "method": "holm_bonferroni",
        "family_size": family_size,
        "alpha": alpha,
        "hypotheses": {
            name: hypotheses[name] for name in sorted(hypotheses)
        },
    }


__all__ = [
    "PairComplexEvaluationInput",
    "build_pair_domain_manifest",
    "evaluate_bidirectional_cognate_retrieval",
    "evaluate_named_partner_controls",
    "evaluate_pair_complex",
    "evaluate_pair_dataset",
    "holm_bonferroni_correction",
    "pair_domain_manifest_sha256",
    "pair_evaluation_mask_sha256",
    "paired_cluster_bootstrap_comparison",
]
