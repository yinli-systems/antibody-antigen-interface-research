"""Fail-closed cross-seed gate statistics for PAIRScale E1/E2 screens."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .development_holdout import connected_components, parse_component_holdout_assignment
from .metrics import paired_cluster_bootstrap_mean_difference


RESULT_FORMAT = "pairscale_e1_e2_gate_statistics_v1"
PROTOCOL_FORMAT = "pairscale_e1_e2_gate_audit_protocol_v1"
REPORT_FORMAT = "pairscale_e1_e2_screen_report_v1"
EVALUATION_FORMAT = "pairscale_e1_e2_development_evaluation_v1"
SCREENING_SEEDS = (17, 42, 73)
SPECIFICITY_DIRECTIONS = ("antibody_to_antigen", "antigen_to_antibody")
SPECIFICITY_SOURCES = ("raw_residual", "centered_residual")


class PairscaleE1E2GateStatisticsError(ValueError):
    """Raised when E1/E2 gate evidence is incomplete or unauthenticated."""


def analyze_e1_e2_screening(
    *,
    assignment_path: Path,
    assignment_receipt_path: Path,
    expected_assignment_file_sha256: str,
    expected_assignment_content_sha256: str,
    expected_assignment_receipt_sha256: str,
    protocol_path: Path,
    expected_protocol_sha256: str,
    report_paths_by_seed: Mapping[int, Path],
    expected_report_sha256_by_seed: Mapping[int, str],
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Authenticate three fixed-horizon reports and apply the frozen screen."""

    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    _utc(timestamp)
    if tuple(sorted(report_paths_by_seed)) != SCREENING_SEEDS:
        raise PairscaleE1E2GateStatisticsError(
            "report_paths_by_seed must contain exactly the screening seeds"
        )
    if set(expected_report_sha256_by_seed) != set(report_paths_by_seed):
        raise PairscaleE1E2GateStatisticsError("report digest inventory differs")

    protocol_path = _regular_file(protocol_path, "gate protocol")
    protocol_bytes = protocol_path.read_bytes()
    if _sha256(protocol_bytes) != _digest(
        expected_protocol_sha256,
        "expected_protocol_sha256",
    ):
        raise PairscaleE1E2GateStatisticsError("gate protocol SHA-256 mismatch")
    protocol = _strict_json(protocol_bytes, "gate protocol")
    _validate_protocol(protocol)

    assignment_path = _regular_file(assignment_path, "assignment")
    assignment_receipt_path = _regular_file(
        assignment_receipt_path,
        "assignment receipt",
    )
    assignment_bytes = assignment_path.read_bytes()
    assignment_receipt_bytes = assignment_receipt_path.read_bytes()
    raw_assignment = _strict_json(assignment_bytes, "assignment")
    try:
        assignment = parse_component_holdout_assignment(
            assignment_bytes,
            assignment_receipt_bytes,
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
        raise PairscaleE1E2GateStatisticsError(
            "assignment authentication failed"
        ) from exc

    components = connected_components(assignment.holdout_instances)
    cluster_by_instance = {
        instance.instance_id: component.component_sha256
        for component in components
        for instance in component.instances
    }
    expected_instances = set(cluster_by_instance)
    reports: dict[int, Mapping[str, Any]] = {}
    report_hashes: dict[str, str] = {}
    common_anchor: Mapping[str, str] | None = None
    for seed in SCREENING_SEEDS:
        path = _regular_file(report_paths_by_seed[seed], f"seed {seed} report")
        raw = path.read_bytes()
        observed_sha256 = _sha256(raw)
        if observed_sha256 != _digest(
            expected_report_sha256_by_seed[seed],
            f"expected_report_sha256_by_seed[{seed}]",
        ):
            raise PairscaleE1E2GateStatisticsError(
                f"seed {seed} report SHA-256 mismatch"
            )
        report = _strict_json(raw, f"seed {seed} report")
        anchor = _validate_report(
            report,
            seed=seed,
            expected_instances=expected_instances,
        )
        if common_anchor is None:
            common_anchor = anchor
        elif common_anchor != anchor:
            raise PairscaleE1E2GateStatisticsError(
                "source or data anchors differ across seeds"
            )
        reports[seed] = report
        report_hashes[str(seed)] = observed_sha256
    expected_assignment_anchors = {
        "assignment_content_sha256": assignment.assignment_content_sha256,
        "assignment_file_sha256": _sha256(assignment_bytes),
        "assignment_receipt_file_sha256": _sha256(assignment_receipt_bytes),
    }
    if common_anchor is None or any(
        common_anchor.get(name) != value
        for name, value in expected_assignment_anchors.items()
    ):
        raise PairscaleE1E2GateStatisticsError(
            "report provenance does not match the authenticated assignment"
        )

    specificity = _specificity_statistics(
        reports,
        cluster_by_instance=cluster_by_instance,
        protocol=protocol,
    )
    retrieval = _retrieval_statistics(reports, protocol=protocol)
    contact = _contact_statistics(
        reports,
        cluster_by_instance=cluster_by_instance,
        protocol=protocol,
    )
    extension_checks = {
        "all_four_specificity_component_bootstrap_lower_bounds_above_zero": all(
            cell["component_bootstrap"]["inferential_authorized"] is True
            and float(cell["component_bootstrap"]["lower"]) > 0.0
            for cell in specificity["cells"].values()
        ),
        "all_four_specificity_cells_have_minimum_positive_seeds": all(
            int(cell["positive_seed_count"])
            >= int(
                protocol["five_seed_extension"][
                    "minimum_positive_screening_seeds_per_specificity_cell"
                ]
            )
            for cell in specificity["cells"].values()
        ),
        "contact_mean_improvement_meets_screen": (
            float(contact["pair_minus_unary"]["mean_difference"])
            >= float(
                protocol["five_seed_extension"][
                    "contact_mean_improvement_minimum"
                ]
            )
        ),
        "contact_worst_seed_improvement_meets_screen": (
            min(float(value) for value in contact["seed_mean_differences"].values())
            >= float(
                protocol["five_seed_extension"][
                    "contact_worst_seed_improvement_minimum"
                ]
            )
        ),
        "all_four_retrieval_cells_meet_minimum_seed_enrichment": all(
            float(cell["minimum_seed_recall_at_1_random_enrichment"])
            >= float(
                protocol["five_seed_extension"][
                    "exhaustive_recall_at_1_random_enrichment_minimum"
                ]
            )
            for cell in retrieval["cells"].values()
        ),
        "all_reports_fixed_epoch_without_selection": True,
        "all_artifact_and_source_audits_agree": True,
    }
    extension_eligible = all(extension_checks.values())
    return {
        "artifacts": {
            "assignment_content_sha256": assignment.assignment_content_sha256,
            "assignment_file_sha256": _sha256(assignment_bytes),
            "assignment_receipt_file_sha256": _sha256(
                assignment_receipt_bytes
            ),
            "gate_protocol_sha256": _sha256(protocol_bytes),
            "report_sha256_by_seed": report_hashes,
        },
        "claim_boundary": (
            "Development-only fixed-horizon screening. This result cannot "
            "authorize scaling, checkpoint selection, confirmation, comparative "
            "superiority, SOTA, or a world-leading claim."
        ),
        "contact": contact,
        "created_at_utc": timestamp,
        "five_seed_extension": {
            "additional_seeds": list(
                protocol["five_seed_extension"]["additional_seeds"]
            ),
            "checks": extension_checks,
            "eligible": extension_eligible,
            "status": "eligible" if extension_eligible else "not_eligible",
        },
        "format_version": RESULT_FORMAT,
        "gates": {
            "C": {
                "passed": False,
                "scaling_authorized": False,
                "status": "pending_independent_separable_precision_at_k_audit",
            },
            "R": {
                "passed": False,
                "scaling_authorized": False,
                "status": "pending_five_seeds_permutation_and_strata",
            },
            "S": {
                "passed": False,
                "scaling_authorized": False,
                "status": "pending_five_seeds_and_homolog_matched_retrieval",
            },
        },
        "independent_component_count": len(components),
        "retrieval": retrieval,
        "screening_seed_count": len(SCREENING_SEEDS),
        "screening_seeds": list(SCREENING_SEEDS),
        "source_and_data_anchor": dict(common_anchor or {}),
        "source_implementation_sha256": {
            "metrics.py": _file_sha256(Path(__file__).with_name("metrics.py")),
            "pairscale_e1_e2_gate_statistics.py": _file_sha256(Path(__file__)),
        },
        "specificity": specificity,
    }


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("format_version") != PROTOCOL_FORMAT:
        raise PairscaleE1E2GateStatisticsError("gate protocol format differs")
    inference = protocol.get("inference")
    extension = protocol.get("five_seed_extension")
    if not isinstance(inference, dict) or not isinstance(extension, dict):
        raise PairscaleE1E2GateStatisticsError("gate protocol sections are missing")
    if (
        inference.get("iterations") != 10_000
        or inference.get("seed") != 42_017
        or inference.get("confidence") != 0.95
        or inference.get("seed_aggregation")
        != "mean_per_instance_before_component_bootstrap"
    ):
        raise PairscaleE1E2GateStatisticsError("inference contract differs")
    if tuple(extension.get("screening_seeds", ())) != SCREENING_SEEDS:
        raise PairscaleE1E2GateStatisticsError("screening seed contract differs")
    if extension.get("exhaustive_recall_at_1_seed_aggregation") != (
        "minimum_across_screening_seeds"
    ):
        raise PairscaleE1E2GateStatisticsError("retrieval aggregation differs")


def _validate_report(
    report: Mapping[str, Any],
    *,
    seed: int,
    expected_instances: set[str],
) -> Mapping[str, str]:
    required_values = {
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
    for name, expected in required_values.items():
        if report.get(name) != expected:
            raise PairscaleE1E2GateStatisticsError(
                f"seed {seed} report field {name} differs"
            )
    checkpoint = report.get("frozen_checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("epoch") != 260:
        raise PairscaleE1E2GateStatisticsError(
            f"seed {seed} frozen checkpoint differs"
        )
    _digest(checkpoint.get("checkpoint_sha256"), "checkpoint_sha256")
    _digest(checkpoint.get("receipt_file_sha256"), "receipt_file_sha256")
    provenance = checkpoint.get("provenance_hashes")
    if not isinstance(provenance, dict):
        raise PairscaleE1E2GateStatisticsError("checkpoint provenance is missing")
    common_names = {
        "cache_manifest_sha256",
        "cache_receipt_file_sha256",
        "assignment_file_sha256",
        "assignment_content_sha256",
        "assignment_receipt_file_sha256",
        "protocol_file_sha256",
        "controller_sha256",
        "scheduler_script_sha256",
    }
    common_names.update(
        name for name in provenance if str(name).startswith("source_")
    )
    anchor = {
        name: _digest(provenance.get(name), f"provenance.{name}")
        for name in sorted(common_names)
    }
    evaluation = report.get("development_holdout_evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("format_version") != (
        EVALUATION_FORMAT
    ):
        raise PairscaleE1E2GateStatisticsError("development evaluation differs")
    if (
        evaluation.get("model_selection_authorized") is not False
        or evaluation.get("threshold_selection_performed") is not False
        or evaluation.get("unknown_cross_pairs_assigned_binary_labels") is not False
    ):
        raise PairscaleE1E2GateStatisticsError(
            "development evaluation authorization differs"
        )
    rows = evaluation.get("per_complex")
    if not isinstance(rows, list) or len(rows) != len(expected_instances):
        raise PairscaleE1E2GateStatisticsError("per-complex inventory differs")
    identifiers = [row.get("instance_id") for row in rows if isinstance(row, dict)]
    if len(identifiers) != len(rows) or len(set(identifiers)) != len(identifiers):
        raise PairscaleE1E2GateStatisticsError("instance inventory is invalid")
    if set(identifiers) != expected_instances:
        raise PairscaleE1E2GateStatisticsError("holdout instances differ")
    return anchor


def _specificity_statistics(
    reports: Mapping[int, Mapping[str, Any]],
    *,
    cluster_by_instance: Mapping[str, str],
    protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    cells: dict[str, Any] = {}
    iterations = int(protocol["inference"]["iterations"])
    bootstrap_seed = int(protocol["inference"]["seed"])
    confidence = float(protocol["inference"]["confidence"])
    matched_reference_by_seed_source: dict[tuple[int, str], list[Any]] = {}
    for direction in SPECIFICITY_DIRECTIONS:
        for source in SPECIFICITY_SOURCES:
            by_seed: dict[str, Mapping[str, float]] = {}
            seed_means: dict[str, float] = {}
            for seed, report in reports.items():
                evaluation = report["development_holdout_evaluation"]
                rows = evaluation["per_complex"]
                panel = evaluation.get("partner_controls")
                if not isinstance(panel, dict) or panel.get(
                    "unknown_cross_pair_semantics"
                ) != "unlabeled_ranking_control_not_nonbinder":
                    raise PairscaleE1E2GateStatisticsError(
                        "partner-control semantics differ"
                    )
                if panel.get("decoy_count_per_direction") != 4:
                    raise PairscaleE1E2GateStatisticsError(
                        "partner-control decoy count differs"
                    )
                cell = panel.get(direction, {}).get(source)
                if not isinstance(cell, dict):
                    raise PairscaleE1E2GateStatisticsError(
                        f"specificity cell {direction}.{source} is missing"
                    )
                matched = cell.get("matched_scores")
                decoys = cell.get("decoy_scores")
                if not isinstance(matched, list) or not isinstance(decoys, list):
                    raise PairscaleE1E2GateStatisticsError(
                        "specificity score inventory differs"
                    )
                if len(matched) != len(rows) or len(decoys) != len(rows):
                    raise PairscaleE1E2GateStatisticsError(
                        "specificity score inventory differs"
                    )
                reference = matched_reference_by_seed_source.setdefault(
                    (seed, source),
                    matched,
                )
                if matched != reference:
                    raise PairscaleE1E2GateStatisticsError(
                        "matched specificity scores differ across cells"
                    )
                values: dict[str, float] = {}
                for row, matched_value, decoy_values in zip(
                    rows,
                    matched,
                    decoys,
                    strict=True,
                ):
                    if not isinstance(decoy_values, list) or len(decoy_values) != 4:
                        raise PairscaleE1E2GateStatisticsError(
                            "specificity decoy count differs"
                        )
                    identifier = str(row["instance_id"])
                    values[identifier] = _finite(
                        matched_value,
                        "matched score",
                    ) - _mean_finite(decoy_values, "decoy score")
                by_seed[str(seed)] = values
                seed_means[str(seed)] = _mean_finite(
                    list(values.values()),
                    "seed specificity delta",
                )
            aggregate = {
                identifier: _mean_finite(
                    [by_seed[str(seed)][identifier] for seed in SCREENING_SEEDS],
                    "cross-seed specificity delta",
                )
                for identifier in sorted(cluster_by_instance)
            }
            bootstrap = _bootstrap_against_zero(
                aggregate,
                clusters=cluster_by_instance,
                iterations=iterations,
                seed=bootstrap_seed,
                confidence=confidence,
            )
            key = f"{direction}.{source}"
            cells[key] = {
                "component_bootstrap": bootstrap,
                "instance_delta_definition": "matched_minus_mean_of_four_decoys",
                "positive_seed_count": sum(
                    float(value) > 0.0 for value in seed_means.values()
                ),
                "seed_mean_differences": seed_means,
            }
    return {
        "cells": cells,
        "score_sources": list(SPECIFICITY_SOURCES),
        "directions": list(SPECIFICITY_DIRECTIONS),
    }


def _retrieval_statistics(
    reports: Mapping[int, Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    first_rows = reports[SCREENING_SEEDS[0]]["development_holdout_evaluation"][
        "per_complex"
    ]
    antibody_to_antigens: dict[str, set[str]] = {}
    antigen_to_antibodies: dict[str, set[str]] = {}
    for row in first_rows:
        antibody_to_antigens.setdefault(str(row["antibody_id"]), set()).add(
            str(row["antigen_id"])
        )
        antigen_to_antibodies.setdefault(str(row["antigen_id"]), set()).add(
            str(row["antibody_id"])
        )
    random_baseline = {
        "antibody_to_antigen": _mean_finite(
            [
                len(values) / len(antigen_to_antibodies)
                for values in antibody_to_antigens.values()
            ],
            "antibody-to-antigen random baseline",
        ),
        "antigen_to_antibody": _mean_finite(
            [
                len(values) / len(antibody_to_antigens)
                for values in antigen_to_antibodies.values()
            ],
            "antigen-to-antibody random baseline",
        ),
    }
    antibody_ids = sorted(antibody_to_antigens)
    antigen_ids = sorted(antigen_to_antibodies)
    observed_pair_count = len(
        {
            (str(row["antibody_id"]), str(row["antigen_id"]))
            for row in first_rows
        }
    )
    cells: dict[str, Any] = {}
    for direction in SPECIFICITY_DIRECTIONS:
        for source in SPECIFICITY_SOURCES:
            enrichments: dict[str, float] = {}
            recall: dict[str, float] = {}
            for seed, report in reports.items():
                evaluation = report["development_holdout_evaluation"]
                exhaustive = evaluation.get("exhaustive_bidirectional_retrieval")
                if not isinstance(exhaustive, dict):
                    raise PairscaleE1E2GateStatisticsError(
                        "exhaustive retrieval is missing"
                    )
                if (
                    exhaustive.get("antibody_ids") != antibody_ids
                    or exhaustive.get("antigen_ids") != antigen_ids
                    or exhaustive.get("observed_cognate_pair_count")
                    != observed_pair_count
                    or exhaustive.get("candidate_pair_count")
                    != len(antibody_ids) * len(antigen_ids)
                    or exhaustive.get("unknown_cross_pair_semantics")
                    != "unlabeled_retrieval_control_not_nonbinder"
                ):
                    raise PairscaleE1E2GateStatisticsError(
                        "exhaustive retrieval domain differs"
                    )
                source_result = exhaustive.get(source)
                if (
                    not isinstance(source_result, dict)
                    or source_result.get("evaluation_contract")
                    != "prospective_pairformer_exhaustive_retrieval_v2"
                    or source_result.get("negative_candidate_subsampling_performed")
                    is not False
                ):
                    raise PairscaleE1E2GateStatisticsError(
                        "exhaustive retrieval contract differs"
                    )
                result = source_result.get(direction)
                if not isinstance(result, dict):
                    raise PairscaleE1E2GateStatisticsError(
                        f"exhaustive retrieval cell {direction}.{source} is missing"
                    )
                expected_query_count = (
                    len(antibody_ids)
                    if direction == "antibody_to_antigen"
                    else len(antigen_ids)
                )
                expected_candidate_count = (
                    len(antigen_ids)
                    if direction == "antibody_to_antigen"
                    else len(antibody_ids)
                )
                if (
                    result.get("query_count") != expected_query_count
                    or result.get("candidate_count_per_query")
                    != expected_candidate_count
                ):
                    raise PairscaleE1E2GateStatisticsError(
                        "exhaustive retrieval dimensions differ"
                    )
                value = _finite(result.get("recall_at_1"), "recall_at_1")
                if not 0.0 <= value <= 1.0:
                    raise PairscaleE1E2GateStatisticsError(
                        "recall_at_1 must be in [0, 1]"
                    )
                recall[str(seed)] = value
                enrichments[str(seed)] = value / random_baseline[direction]
            key = f"{direction}.{source}"
            cells[key] = {
                "random_recall_at_1_baseline": random_baseline[direction],
                "recall_at_1_by_seed": recall,
                "recall_at_1_random_enrichment_by_seed": enrichments,
                "minimum_seed_recall_at_1_random_enrichment": min(
                    enrichments.values()
                ),
                "pool_role": "exhaustive_not_homolog_matched",
            }
    return {
        "cells": cells,
        "homolog_matched_candidate_pool_available": False,
        "homolog_matched_gate_evidence": "pending",
    }


def _contact_statistics(
    reports: Mapping[int, Mapping[str, Any]],
    *,
    cluster_by_instance: Mapping[str, str],
    protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    pair_by_seed: dict[str, dict[str, float]] = {}
    unary_by_seed: dict[str, dict[str, float]] = {}
    seed_mean_differences: dict[str, float] = {}
    ap_lift_seed_mean_differences: dict[str, float] = {}
    for seed, report in reports.items():
        rows = report["development_holdout_evaluation"]["per_complex"]
        pair: dict[str, float] = {}
        unary: dict[str, float] = {}
        ap_lift_differences: list[float] = []
        for row in rows:
            identifier = str(row["instance_id"])
            pair[identifier] = _finite(row.get("auprc"), "pair auprc")
            unary[identifier] = _finite(
                row.get("unary_additive_auprc"),
                "unary additive auprc",
            )
            prevalence = _finite(row.get("prevalence"), "prevalence")
            if not 0.0 < prevalence < 1.0:
                raise PairscaleE1E2GateStatisticsError(
                    "contact prevalence must be in (0, 1)"
                )
            unary_ap_lift = (unary[identifier] - prevalence) / (1.0 - prevalence)
            ap_lift_differences.append(
                _finite(row.get("ap_lift"), "pair ap lift") - unary_ap_lift
            )
        pair_by_seed[str(seed)] = pair
        unary_by_seed[str(seed)] = unary
        seed_mean_differences[str(seed)] = _mean_finite(
            [pair[item] - unary[item] for item in sorted(pair)],
            "contact seed difference",
        )
        ap_lift_seed_mean_differences[str(seed)] = _mean_finite(
            ap_lift_differences,
            "ap lift seed difference",
        )
    pair_aggregate = {
        identifier: _mean_finite(
            [pair_by_seed[str(seed)][identifier] for seed in SCREENING_SEEDS],
            "cross-seed pair auprc",
        )
        for identifier in sorted(cluster_by_instance)
    }
    unary_aggregate = {
        identifier: _mean_finite(
            [unary_by_seed[str(seed)][identifier] for seed in SCREENING_SEEDS],
            "cross-seed unary auprc",
        )
        for identifier in sorted(cluster_by_instance)
    }
    comparison = paired_cluster_bootstrap_mean_difference(
        pair_aggregate,
        unary_aggregate,
        cluster_by_instance,
        iterations=int(protocol["inference"]["iterations"]),
        seed=int(protocol["inference"]["seed"]),
        confidence=float(protocol["inference"]["confidence"]),
    )
    return {
        "ap_lift_seed_mean_differences": ap_lift_seed_mean_differences,
        "available_separable_baseline": "in_model_additive_unary",
        "pair_minus_unary": comparison,
        "precision_at_true_k_separable_baseline_available": False,
        "seed_mean_differences": seed_mean_differences,
        "strongest_separable_baseline_independently_certified": False,
    }


def _bootstrap_against_zero(
    values: Mapping[str, float],
    *,
    clusters: Mapping[str, str],
    iterations: int,
    seed: int,
    confidence: float,
) -> Mapping[str, Any]:
    return paired_cluster_bootstrap_mean_difference(
        values,
        {identifier: 0.0 for identifier in values},
        clusters,
        iterations=iterations,
        seed=seed,
        confidence=confidence,
    )


def _strict_json(raw: bytes, name: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise PairscaleE1E2GateStatisticsError(f"{name} contains {value}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PairscaleE1E2GateStatisticsError(
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
        raise PairscaleE1E2GateStatisticsError(f"cannot parse {name}") from exc
    if type(value) is not dict:
        raise PairscaleE1E2GateStatisticsError(f"{name} must be an object")
    return value


def _regular_file(path: Path, name: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_file():
        raise PairscaleE1E2GateStatisticsError(
            f"{name} must be a regular non-symlink file"
        )
    return result


def _digest(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PairscaleE1E2GateStatisticsError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PairscaleE1E2GateStatisticsError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PairscaleE1E2GateStatisticsError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise PairscaleE1E2GateStatisticsError(f"{name} must be finite")
    return result


def _mean_finite(values: Sequence[Any], name: str) -> float:
    if not values:
        raise PairscaleE1E2GateStatisticsError(f"{name} cannot be empty")
    finite = [_finite(value, name) for value in values]
    return float(sum(finite) / len(finite))


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
        raise PairscaleE1E2GateStatisticsError("created_at_utc must use UTC Z form")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PairscaleE1E2GateStatisticsError(
            "created_at_utc is invalid"
        ) from exc


__all__ = [
    "PairscaleE1E2GateStatisticsError",
    "analyze_e1_e2_screening",
]
