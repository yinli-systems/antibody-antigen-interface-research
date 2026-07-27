"""Authenticated cluster-level statistics for PAIRScale development holdouts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from interaction_first.development_holdout import (
    connected_components,
    parse_component_holdout_assignment,
)
from interaction_first.metrics import paired_cluster_bootstrap_mean_difference


EVALUATION_FORMAT = "pairscale_development_holdout_evaluation_v1"
RESULT_FORMAT = "pairscale_development_holdout_cluster_statistics_v1"
_EVALUATION_FIELDS = {
    "checkpoint_promotion_authorized",
    "complete_candidate_domain",
    "confirmatory_claim_authorized",
    "evaluation_role",
    "gate_checks",
    "gate_passed",
    "held_out_evaluation_performed",
    "interpretation",
    "macro_label_marginal_oracle_auprc",
    "macro_model_marginal_outer_auprc",
    "macro_pair_auprc",
    "macro_precision_at_true_contact_count",
    "model_selection_authorized",
    "negative_pair_subsampling",
    "partner_controls",
    "per_complex",
    "pooled_pair_auprc",
    "scope",
    "unknown_cross_pair_semantics",
}
_PER_COMPLEX_FIELDS = {
    "accuracy",
    "auprc",
    "auroc",
    "contact_iou",
    "count",
    "f1",
    "instance_id",
    "label_marginal_oracle_auprc",
    "mcc",
    "model_marginal_outer_auprc",
    "positive_count",
    "precision",
    "precision_at_true_contact_count",
    "precision_at_true_contact_count_tie_policy",
    "prevalence",
    "recall",
    "threshold",
}
_PARTNER_FIELDS = {
    "decoy_double_centered_scores",
    "decoy_policy",
    "decoy_topk_scores",
    "double_centered_partner_mean_delta",
    "double_centered_partner_win_rate",
    "matched_double_centered_scores",
    "matched_partner_topk_mean_delta",
    "matched_partner_topk_win_rate",
    "matched_topk_scores",
}


class PairscaleHoldoutStatisticsError(ValueError):
    """Raised when holdout evidence is unauthenticated or internally inconsistent."""


def analyze_development_holdout_statistics(
    *,
    assignment_path: Path,
    assignment_receipt_path: Path,
    evaluation_path: Path,
    expected_assignment_file_sha256: str,
    expected_assignment_content_sha256: str,
    expected_assignment_receipt_sha256: str,
    expected_evaluation_sha256: str,
    iterations: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Authenticate one development result and run component-paired bootstraps."""

    if type(iterations) is not int or isinstance(iterations, bool) or iterations <= 0:
        raise PairscaleHoldoutStatisticsError("iterations must be a positive integer")
    if type(seed) is not int or isinstance(seed, bool):
        raise PairscaleHoldoutStatisticsError("seed must be an integer")
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    _utc(timestamp)
    assignment_path = _regular_file(assignment_path, "assignment")
    receipt_path = _regular_file(assignment_receipt_path, "assignment receipt")
    evaluation_path = _regular_file(evaluation_path, "evaluation")
    assignment_bytes = assignment_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    evaluation_bytes = evaluation_path.read_bytes()
    if _sha256(evaluation_bytes) != _digest(
        expected_evaluation_sha256,
        "expected_evaluation_sha256",
    ):
        raise PairscaleHoldoutStatisticsError("evaluation SHA-256 mismatch")
    raw_assignment = _strict_json(assignment_bytes, "assignment")
    try:
        assignment = parse_component_holdout_assignment(
            assignment_bytes,
            receipt_bytes,
            expected_assignment_file_sha256=_digest(
                expected_assignment_file_sha256,
                "expected_assignment_file_sha256",
            ),
            expected_assignment_content_sha256=_digest(
                expected_assignment_content_sha256,
                "expected_assignment_content_sha256",
            ),
            expected_receipt_file_sha256=_digest(
                expected_assignment_receipt_sha256,
                "expected_assignment_receipt_sha256",
            ),
            expected_source_subset_sha256=raw_assignment["source_subset_sha256"],
            expected_policy_sha256=raw_assignment["policy_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PairscaleHoldoutStatisticsError(
            "assignment authentication failed"
        ) from exc
    components = connected_components(assignment.holdout_instances)
    observed_component_ids = sorted(value.component_sha256 for value in components)
    try:
        claimed_component_ids = raw_assignment["holdout"]["component_sha256"]
    except (KeyError, TypeError) as exc:
        raise PairscaleHoldoutStatisticsError(
            "assignment component inventory is missing"
        ) from exc
    if observed_component_ids != claimed_component_ids:
        raise PairscaleHoldoutStatisticsError("holdout component identities differ")
    cluster_by_instance = {
        instance.instance_id: component.component_sha256
        for component in components
        for instance in component.instances
    }
    evaluation = _strict_json(evaluation_bytes, "evaluation")
    rows, partner = _validate_evaluation(
        evaluation,
        expected_instances=set(cluster_by_instance),
    )
    pair_scores = {row["instance_id"]: _finite(row["auprc"], "auprc") for row in rows}
    model_marginal = {
        row["instance_id"]: _finite(
            row["model_marginal_outer_auprc"],
            "model_marginal_outer_auprc",
        )
        for row in rows
    }
    label_marginal = {
        row["instance_id"]: _finite(
            row["label_marginal_oracle_auprc"],
            "label_marginal_oracle_auprc",
        )
        for row in rows
    }
    ordered_ids = [row["instance_id"] for row in rows]
    comparisons = {
        "pair_head_minus_label_marginal_oracle_auprc": _bootstrap(
            pair_scores,
            label_marginal,
            cluster_by_instance,
            iterations,
            seed,
            confidence,
        ),
        "pair_head_minus_model_marginal_outer_auprc": _bootstrap(
            pair_scores,
            model_marginal,
            cluster_by_instance,
            iterations,
            seed,
            confidence,
        ),
        "matched_minus_decoy_double_centered_score": _bootstrap(
            _score_map(ordered_ids, partner["matched_double_centered_scores"]),
            _score_map(ordered_ids, partner["decoy_double_centered_scores"]),
            cluster_by_instance,
            iterations,
            seed,
            confidence,
        ),
        "matched_minus_decoy_topk_score": _bootstrap(
            _score_map(ordered_ids, partner["matched_topk_scores"]),
            _score_map(ordered_ids, partner["decoy_topk_scores"]),
            cluster_by_instance,
            iterations,
            seed,
            confidence,
        ),
    }
    checks = {
        "double_centered_partner_lower_bound_above_zero": comparisons[
            "matched_minus_decoy_double_centered_score"
        ]["lower"]
        > 0.0,
        "pair_head_exceeds_label_marginal_oracle_lower_bound_above_zero": comparisons[
            "pair_head_minus_label_marginal_oracle_auprc"
        ]["lower"]
        > 0.0,
        "pair_head_exceeds_model_marginal_lower_bound_above_zero": comparisons[
            "pair_head_minus_model_marginal_outer_auprc"
        ]["lower"]
        > 0.0,
        "topk_partner_lower_bound_above_zero": comparisons[
            "matched_minus_decoy_topk_score"
        ]["lower"]
        > 0.0,
    }
    return {
        "artifacts": {
            "assignment_content_sha256": assignment.assignment_content_sha256,
            "assignment_file_sha256": _sha256(assignment_bytes),
            "assignment_receipt_file_sha256": _sha256(receipt_bytes),
            "evaluation_sha256": _sha256(evaluation_bytes),
        },
        "bootstrap": {
            "comparison_count": len(comparisons),
            "confidence": float(confidence),
            "independent_component_count": len(components),
            "instance_count": len(rows),
            "iterations": iterations,
            "seed": seed,
        },
        "claim_boundary": (
            "Development-only endpoint-disjoint evidence. No validation, test, "
            "confirmatory, checkpoint-promotion, biological-generalization, "
            "comparative, or state-of-the-art claim is authorized."
        ),
        "comparisons": comparisons,
        "created_at_utc": timestamp,
        "format_version": RESULT_FORMAT,
        "gate_l1": {
            "checks": checks,
            "passed": False,
            "scaling_authorized": False,
            "status": "failed_current_partner_specificity_evidence",
            "unresolved_required_controls": [
                "antibody_only_control",
                "antigen_only_control",
                "strongest_separable_marginal_control_independent_replay",
            ],
        },
        "source_implementation_sha256": {
            "development_holdout.py": _file_sha256(
                Path(__file__).with_name("development_holdout.py")
            ),
            "metrics.py": _file_sha256(Path(__file__).with_name("metrics.py")),
            "pairscale_holdout_statistics.py": _file_sha256(Path(__file__)),
        },
    }


def _validate_evaluation(
    evaluation: Mapping[str, Any],
    *,
    expected_instances: set[str],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    if set(evaluation) != _EVALUATION_FIELDS:
        raise PairscaleHoldoutStatisticsError("evaluation fields differ")
    required_false = (
        "checkpoint_promotion_authorized",
        "confirmatory_claim_authorized",
        "model_selection_authorized",
        "negative_pair_subsampling",
    )
    if any(evaluation[name] is not False for name in required_false):
        raise PairscaleHoldoutStatisticsError("evaluation authorization differs")
    if (
        evaluation["held_out_evaluation_performed"] is not True
        or evaluation["complete_candidate_domain"] is not True
        or evaluation["gate_passed"] is not False
        or evaluation["gate_checks"] != {}
    ):
        raise PairscaleHoldoutStatisticsError("evaluation state differs")
    if evaluation["unknown_cross_pair_semantics"] != (
        "ranking_control_only_not_a_measured_nonbinder"
    ):
        raise PairscaleHoldoutStatisticsError("negative semantics differ")
    rows = evaluation["per_complex"]
    if type(rows) is not list or len(rows) != len(expected_instances):
        raise PairscaleHoldoutStatisticsError("per-complex inventory differs")
    observed: set[str] = set()
    for row in rows:
        if type(row) is not dict or set(row) != _PER_COMPLEX_FIELDS:
            raise PairscaleHoldoutStatisticsError("per-complex fields differ")
        identifier = row["instance_id"]
        if type(identifier) is not str or identifier.strip() != identifier:
            raise PairscaleHoldoutStatisticsError("instance identifier is invalid")
        if identifier in observed:
            raise PairscaleHoldoutStatisticsError("instance identifier is duplicated")
        observed.add(identifier)
    if observed != expected_instances:
        raise PairscaleHoldoutStatisticsError("evaluation instances differ")
    partner = evaluation["partner_controls"]
    if type(partner) is not dict or set(partner) != _PARTNER_FIELDS:
        raise PairscaleHoldoutStatisticsError("partner-control fields differ")
    if partner["decoy_policy"] != "deterministic_cyclic_non_cognate_antigen":
        raise PairscaleHoldoutStatisticsError("partner decoy policy differs")
    for name in (
        "decoy_double_centered_scores",
        "decoy_topk_scores",
        "matched_double_centered_scores",
        "matched_topk_scores",
    ):
        values = partner[name]
        if type(values) is not list or len(values) != len(rows):
            raise PairscaleHoldoutStatisticsError("partner score inventory differs")
        for value in values:
            _finite(value, name)
    return rows, partner


def _bootstrap(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    clusters: Mapping[str, str],
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
    except ValueError as exc:
        raise PairscaleHoldoutStatisticsError("paired cluster bootstrap failed") from exc


def _score_map(instance_ids: list[str], values: Any) -> dict[str, float]:
    if len(instance_ids) != len(values):
        raise PairscaleHoldoutStatisticsError("partner score inventory differs")
    return {
        identifier: _finite(value, "partner score")
        for identifier, value in zip(instance_ids, values)
    }


def _strict_json(raw: bytes, name: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise PairscaleHoldoutStatisticsError(f"{name} contains {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PairscaleHoldoutStatisticsError(
                    f"{name} contains duplicate key {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PairscaleHoldoutStatisticsError(f"cannot parse {name}") from exc
    if type(value) is not dict:
        raise PairscaleHoldoutStatisticsError(f"{name} must be an object")
    return value


def _regular_file(path: Path, name: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_file():
        raise PairscaleHoldoutStatisticsError(f"{name} must be a regular file")
    return result


def _digest(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PairscaleHoldoutStatisticsError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PairscaleHoldoutStatisticsError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PairscaleHoldoutStatisticsError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise PairscaleHoldoutStatisticsError(f"{name} must be finite")
    return result


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> None:
    if type(value) is not str or not value.endswith("Z"):
        raise PairscaleHoldoutStatisticsError("created_at_utc must use UTC Z form")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PairscaleHoldoutStatisticsError("created_at_utc is invalid") from exc


__all__ = [
    "PairscaleHoldoutStatisticsError",
    "analyze_development_holdout_statistics",
]
