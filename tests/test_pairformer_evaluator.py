"""Focused tests for the prospective full-matrix PAIRFormer evaluator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from interaction_first.metrics import average_precision
from interaction_first.pairformer_evaluator import (
    PairComplexEvaluationInput,
    build_pair_domain_manifest,
    evaluate_bidirectional_cognate_retrieval,
    evaluate_named_partner_controls,
    evaluate_pair_complex,
    evaluate_pair_dataset,
    holm_bonferroni_correction,
    pair_domain_manifest_sha256,
    pair_evaluation_mask_sha256,
    paired_cluster_bootstrap_comparison,
)


def _record(
    complex_id: str,
    probabilities: np.ndarray | None = None,
) -> PairComplexEvaluationInput:
    labels = np.asarray(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    if probabilities is None:
        probabilities = np.asarray(
            [
                [0.95, 0.10, 0.20, 0.0],
                [0.30, 0.05, 0.90, 0.0],
                [0.00, 0.00, 0.00, 0.0],
            ],
            dtype=np.float64,
        )
    antibody_mask = np.asarray([True, True, False])
    antigen_mask = np.asarray([True, True, True, False])
    pair_mask = antibody_mask[:, None] & antigen_mask[None, :]
    antibody_residue_ids = ("H:1", "H:2", None)
    antigen_residue_ids = ("A:10", "A:11", "A:12", None)
    domain_manifest = build_pair_domain_manifest(
        complex_id=complex_id,
        antibody_residue_ids=antibody_residue_ids,
        antigen_residue_ids=antigen_residue_ids,
        antibody_residue_mask=antibody_mask,
        antigen_residue_mask=antigen_mask,
    )
    return PairComplexEvaluationInput(
        complex_id=complex_id,
        contact_labels=labels,
        contact_probabilities=probabilities,
        antibody_residue_mask=antibody_mask,
        antigen_residue_mask=antigen_mask,
        pair_evaluation_mask=pair_mask,
        antibody_residue_ids=antibody_residue_ids,
        antigen_residue_ids=antigen_residue_ids,
        frozen_domain_manifest=domain_manifest,
        frozen_domain_manifest_sha256=pair_domain_manifest_sha256(
            domain_manifest
        ),
        pair_evaluation_mask_sha256=pair_evaluation_mask_sha256(pair_mask),
    )


def test_complex_evaluation_uses_complete_pair_matrix() -> None:
    record = _record("complex-a")
    result = evaluate_pair_complex(
        record,
        ranking_cutoffs=(1, 2, 6),
        calibration_bins=3,
        risk_coverages=(0.5, 1.0),
    )

    assert result["negative_subsampling_performed"] is False
    assert result["support"]["evaluated_pair_count"] == 6
    assert result["support"]["expected_complete_cartesian_pair_count"] == 6
    assert result["support"]["frozen_residue_domain_bound"] is True
    assert result["support"]["frozen_domain_manifest_sha256"] == (
        record.frozen_domain_manifest_sha256
    )
    assert result["support"]["pair_evaluation_mask_sha256"] == (
        record.pair_evaluation_mask_sha256
    )
    assert result["pair"]["auprc"] == pytest.approx(1.0)
    assert result["pair"]["auroc"] == pytest.approx(1.0)
    assert result["pair"]["mcc"] == pytest.approx(1.0)
    assert result["pair"]["f1"] == pytest.approx(1.0)
    assert result["pair"]["dice"] == pytest.approx(result["pair"]["f1"])
    assert result["pair"]["precision_at_2"] == pytest.approx(1.0)
    assert result["pair"]["recall_at_2"] == pytest.approx(1.0)
    assert result["pair_calibration"]["strategy"] == "adaptive"
    assert math.isfinite(result["pair_calibration"]["brier_score"])
    assert math.isfinite(result["pair_calibration"]["negative_log_likelihood"])
    assert len(result["pair_risk_coverage"]["points"]) == 2


def test_complex_risk_coverage_uses_the_evaluation_decision_threshold() -> None:
    result = evaluate_pair_complex(
        _record("risk-threshold"),
        threshold=0.7,
        ranking_cutoffs=(1, 2),
        risk_coverages=(1.0,),
    )
    risk = result["pair_risk_coverage"]
    assert risk["classification_decision_threshold"] == pytest.approx(0.7)
    assert risk["classification_decision_rule"] == (
        "probability_greater_than_or_equal_to_threshold_predicts_positive"
    )


def test_conditional_ranking_reports_all_eligibility_counts() -> None:
    result = evaluate_pair_complex(
        _record("conditional"),
        ranking_cutoffs=(1, 2),
    )
    row = result["conditional_ranking"]["row_conditional"]
    column = result["conditional_ranking"]["column_conditional"]

    assert row["valid_unit_count"] == 2
    assert row["positive_eligible_unit_count"] == 2
    assert row["ineligible_no_positive_unit_count"] == 0
    assert row["auroc_eligible_unit_count"] == 2
    assert row["mean_average_precision"] == pytest.approx(1.0)

    assert column["valid_unit_count"] == 3
    assert column["positive_eligible_unit_count"] == 2
    assert column["ineligible_no_positive_unit_count"] == 1
    assert column["auroc_eligible_unit_count"] == 2
    assert column["mean_reciprocal_rank_first_positive"] == pytest.approx(1.0)


def test_pair_derived_epitope_and_paratope_metrics_have_correct_axes() -> None:
    result = evaluate_pair_complex(
        _record("marginals"),
        ranking_cutoffs=(1, 2),
    )
    epitope = result["pair_derived_marginals"]["epitope"]
    paratope = result["pair_derived_marginals"]["paratope"]

    assert epitope["count"] == 3
    assert epitope["positive_count"] == 2
    assert epitope["score_derivation"] == "maximum_pair_contact_probability"
    assert epitope["auprc"] == pytest.approx(1.0)
    assert paratope["count"] == 2
    assert paratope["positive_count"] == 2
    assert paratope["auroc"] is None
    assert paratope["dice"] == pytest.approx(paratope["f1"])


def test_pair_evaluation_rejects_any_subsampled_valid_pair() -> None:
    record = _record("subsampled")
    subsampled = np.asarray(record.pair_evaluation_mask).copy()
    subsampled[0, 1] = False
    invalid = PairComplexEvaluationInput(
        **{
            **record.__dict__,
            "pair_evaluation_mask": subsampled,
            "pair_evaluation_mask_sha256": pair_evaluation_mask_sha256(
                subsampled
            ),
        }
    )
    with pytest.raises(ValueError, match="negative subsampling is forbidden"):
        evaluate_pair_complex(invalid)


def test_pair_evaluation_rejects_cropping_against_frozen_residue_domain() -> None:
    record = _record("cropping-attack")
    cropped_pair_mask = np.ones((2, 2), dtype=bool)
    cropped = PairComplexEvaluationInput(
        **{
            **record.__dict__,
            "contact_labels": np.asarray(record.contact_labels)[:2, :2],
            "contact_probabilities": np.asarray(
                record.contact_probabilities
            )[:2, :2],
            "antibody_residue_mask": np.ones(2, dtype=bool),
            "antigen_residue_mask": np.ones(2, dtype=bool),
            "pair_evaluation_mask": cropped_pair_mask,
            "antibody_residue_ids": ("H:1", "H:2"),
            "antigen_residue_ids": ("A:10", "A:11"),
            "pair_evaluation_mask_sha256": pair_evaluation_mask_sha256(
                cropped_pair_mask
            ),
        }
    )
    with pytest.raises(ValueError, match="Frozen domain manifest"):
        evaluate_pair_complex(cropped)


def test_pair_evaluation_rejects_mask_or_manifest_digest_spoofing() -> None:
    record = _record("digest-spoofing")
    with pytest.raises(ValueError, match="pair_evaluation_mask.*SHA-256"):
        evaluate_pair_complex(
            PairComplexEvaluationInput(
                **{
                    **record.__dict__,
                    "pair_evaluation_mask_sha256": "0" * 64,
                }
            )
        )
    with pytest.raises(ValueError, match="frozen_domain_manifest.*SHA-256"):
        evaluate_pair_complex(
            PairComplexEvaluationInput(
                **{
                    **record.__dict__,
                    "frozen_domain_manifest_sha256": "0" * 64,
                }
            )
        )


def test_pair_evaluation_requires_unique_ids_and_null_padding_ids() -> None:
    record = _record("residue-ids")
    with pytest.raises(ValueError, match="must be unique"):
        evaluate_pair_complex(
            PairComplexEvaluationInput(
                **{
                    **record.__dict__,
                    "antibody_residue_ids": ("H:1", "H:1", None),
                }
            )
        )
    with pytest.raises(ValueError, match="must be None"):
        evaluate_pair_complex(
            PairComplexEvaluationInput(
                **{
                    **record.__dict__,
                    "antigen_residue_ids": (
                        "A:10",
                        "A:11",
                        "A:12",
                        "A:PAD",
                    ),
                }
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "contact_labels",
            np.asarray([[0.5, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),
            "binary",
        ),
        (
            "contact_probabilities",
            np.asarray(
                [[float("nan"), 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
            ),
            "finite",
        ),
        (
            "antibody_residue_mask",
            np.asarray([1, 1, 0], dtype=np.uint8),
            "boolean",
        ),
    ),
)
def test_pair_evaluation_fails_closed_on_malformed_inputs(
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    record = _record("malformed")
    values = dict(record.__dict__)
    values[field] = value
    with pytest.raises(ValueError, match=message):
        evaluate_pair_complex(PairComplexEvaluationInput(**values))


def test_pair_evaluation_rejects_nonzero_padding_and_single_class_maps() -> None:
    record = _record("padding")
    padded_probabilities = np.asarray(record.contact_probabilities).copy()
    padded_probabilities[2, 0] = 0.1
    with pytest.raises(ValueError, match="Padded contact probabilities"):
        evaluate_pair_complex(
            PairComplexEvaluationInput(
                **{
                    **record.__dict__,
                    "contact_probabilities": padded_probabilities,
                }
            )
        )

    all_positive = np.asarray(record.contact_labels).copy()
    all_positive[np.asarray(record.pair_evaluation_mask)] = 1
    with pytest.raises(ValueError, match="non-contact"):
        evaluate_pair_complex(
            PairComplexEvaluationInput(
                **{
                    **record.__dict__,
                    "contact_labels": all_positive,
                }
            )
        )


def test_dataset_primary_is_unweighted_macro_full_matrix_auprc() -> None:
    second_probabilities = np.asarray(
        [
            [0.40, 0.90, 0.80, 0.0],
            [0.70, 0.60, 0.50, 0.0],
            [0.00, 0.00, 0.00, 0.0],
        ],
        dtype=np.float64,
    )
    first = _record("first")
    second = _record("second", second_probabilities)
    result = evaluate_pair_dataset(
        [first, second],
        ranking_cutoffs=(1, 2),
        calibration_bins=3,
        risk_coverages=(0.5, 1.0),
    )

    second_labels = np.asarray(second.contact_labels)[
        np.asarray(second.pair_evaluation_mask)
    ]
    second_scores = np.asarray(second.contact_probabilities)[
        np.asarray(second.pair_evaluation_mask)
    ]
    expected = (1.0 + float(average_precision(second_labels, second_scores))) / 2.0
    assert result["primary_metric"]["name"] == (
        "macro_per_complex_full_matrix_pair_auprc"
    )
    assert result["primary_metric"]["value"] == pytest.approx(expected)
    assert result["primary_metric"]["defined_complex_count"] == 2
    assert result["negative_subsampling_performed"] is False
    assert result["macro"]["pair_calibration"]["strategy"] == (
        "adaptive_tie_preserving"
    )
    assert result["macro"]["row_conditional"][
        "total_positive_eligible_unit_count"
    ] == 4


def test_dataset_rejects_duplicate_complex_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        evaluate_pair_dataset([_record("same"), _record("same")])


def test_bidirectional_retrieval_reports_conservative_ties() -> None:
    scores = np.asarray(
        [
            [1.0, 0.5, 0.2],
            [0.4, 0.4, 0.1],
            [0.3, 0.2, 0.3],
        ]
    )
    cognates = np.eye(3, dtype=bool)
    result = evaluate_bidirectional_cognate_retrieval(
        scores,
        cognates,
        np.ones_like(cognates),
    )
    row = result["antibody_to_antigen"]

    assert result["negative_candidate_subsampling_performed"] is False
    assert row["recall_at_1"] == pytest.approx(1.0 / 3.0)
    assert row["recall_at_5"] == pytest.approx(1.0)
    assert row["recall_at_10"] == pytest.approx(1.0)
    assert row["mean_reciprocal_rank"] == pytest.approx(2.0 / 3.0)
    assert row["mean_normalized_rank"] == pytest.approx(1.0 / 3.0)
    assert row["hardest_negative_strict_top1_success_rate"] == pytest.approx(
        1.0 / 3.0
    )
    assert row["hardest_negative_top1_tie_rate"] == pytest.approx(2.0 / 3.0)
    assert row["pairwise_cognate_vs_noncognate_win_rate"] == pytest.approx(
        5.0 / 6.0
    )
    assert row["pairwise_strict_win_rate"] == pytest.approx(2.0 / 3.0)
    assert row["pairwise_tie_rate"] == pytest.approx(1.0 / 3.0)
    assert "antigen_to_antibody" in result
    assert "macro_bidirectional" in result


def test_bidirectional_retrieval_uses_every_cognate_noncognate_pair() -> None:
    scores = np.asarray(
        [
            [1.0, 0.1, 0.5],
            [0.4, 0.4, 0.1],
            [0.3, 0.2, 0.3],
        ]
    )
    cognates = np.asarray(
        [
            [True, True, False],
            [False, True, False],
            [False, False, True],
        ]
    )
    row = evaluate_bidirectional_cognate_retrieval(
        scores,
        cognates,
        np.ones_like(cognates),
    )["antibody_to_antigen"]

    assert row["minimum_cognate_count_per_query"] == 1
    assert row["maximum_cognate_count_per_query"] == 2
    assert row["pairwise_cognate_vs_noncognate_win_rate"] == pytest.approx(
        2.0 / 3.0
    )
    assert row["hardest_negative_strict_top1_success_rate"] == pytest.approx(
        1.0 / 3.0
    )
    assert row["pairwise_tie_policy"] == (
        "one_half_credit_per_tied_cognate_noncognate_pair"
    )
    assert "all_cognate_by_noncognate_pairs" in row[
        "pairwise_multi_cognate_policy"
    ]


def test_bidirectional_retrieval_requires_exhaustive_candidates_and_cognates() -> None:
    scores = np.eye(3)
    cognates = np.eye(3, dtype=bool)
    incomplete = np.ones((3, 3), dtype=bool)
    incomplete[0, 1] = False
    with pytest.raises(ValueError, match="candidate subsampling is forbidden"):
        evaluate_bidirectional_cognate_retrieval(scores, cognates, incomplete)

    missing = cognates.copy()
    missing[0, 0] = False
    with pytest.raises(ValueError, match="must have a cognate"):
        evaluate_bidirectional_cognate_retrieval(
            scores,
            missing,
            np.ones((3, 3), dtype=bool),
        )
    with pytest.raises(ValueError, match="positive integers"):
        evaluate_bidirectional_cognate_retrieval(
            scores,
            cognates,
            np.ones((3, 3), dtype=bool),
            cutoffs=(0,),
        )


def _named_control_arguments() -> dict[str, object]:
    return {
        "matched_scores": [3.0, 2.0],
        "controls_by_name": {
            "partner_shuffle": [[1.0, 2.0], [1.5, 1.0]],
            "wrong_patch": [2.5, 2.5],
        },
        "matched_candidate_pair_counts": [6, 8],
        "control_candidate_pair_counts_by_name": {
            "partner_shuffle": [[6, 6], [8, 8]],
            "wrong_patch": [[6], [8]],
        },
        "matched_axis_valid_counts": [[2, 3], [2, 4]],
        "control_axis_valid_counts_by_name": {
            "partner_shuffle": [
                [[2, 3], [2, 3]],
                [[2, 4], [2, 4]],
            ],
            "wrong_patch": [[[2, 3]], [[2, 4]]],
        },
        "matched_candidate_domain_sha256": ["a" * 64, "b" * 64],
        "control_candidate_domain_sha256_by_name": {
            "partner_shuffle": [
                ["a" * 64, "a" * 64],
                ["b" * 64, "b" * 64],
            ],
            "wrong_patch": [["a" * 64], ["b" * 64]],
        },
        "matched_candidate_mask_sha256": ["c" * 64, "d" * 64],
        "control_candidate_mask_sha256_by_name": {
            "partner_shuffle": [
                ["c" * 64, "c" * 64],
                ["d" * 64, "d" * 64],
            ],
            "wrong_patch": [["c" * 64], ["d" * 64]],
        },
    }


def test_named_partner_controls_require_authenticated_exact_support_parity() -> None:
    result = evaluate_named_partner_controls(**_named_control_arguments())
    assert result["candidate_support_audit"][
        "all_controls_exact_support_match"
    ]
    assert result["candidate_support_audit"]["hash_algorithm"] == "sha256"
    family = result["candidate_support_audit"]["by_control"]["partner_shuffle"]
    assert family["exact_candidate_pair_count_match"]
    assert family["exact_per_axis_valid_count_match"]
    assert family["exact_candidate_domain_hash_match"]
    assert family["exact_candidate_mask_hash_match"]
    assert result["metrics"]["control_names"] == [
        "partner_shuffle",
        "wrong_patch",
    ]
    assert result["metrics"]["all_controls"][
        "control_candidates_per_instance"
    ] == 3


@pytest.mark.parametrize(
    "mismatch",
    ("pair_count", "axis_counts", "domain_hash", "mask_hash"),
)
def test_named_partner_controls_reject_spoofed_equal_total_support(
    mismatch: str,
) -> None:
    arguments = _named_control_arguments()
    if mismatch == "pair_count":
        arguments["control_candidate_pair_counts_by_name"][
            "partner_shuffle"
        ][0][1] = 5
    elif mismatch == "axis_counts":
        arguments["control_axis_valid_counts_by_name"]["partner_shuffle"][
            0
        ][1] = [1, 6]
    elif mismatch == "domain_hash":
        arguments["control_candidate_domain_sha256_by_name"][
            "partner_shuffle"
        ][0][1] = "e" * 64
    else:
        arguments["control_candidate_mask_sha256_by_name"][
            "partner_shuffle"
        ][0][1] = "e" * 64
    with pytest.raises(ValueError, match="authenticated exact candidate support"):
        evaluate_named_partner_controls(**arguments)


def test_named_partner_controls_reject_unauthenticated_hashes() -> None:
    arguments = _named_control_arguments()
    arguments["matched_candidate_mask_sha256"][0] = "not-a-sha256"
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_named_partner_controls(**arguments)


def test_paired_cluster_bootstrap_interface_defaults_to_ten_thousand() -> None:
    result = paired_cluster_bootstrap_comparison(
        candidate_by_instance={"a": 0.8, "b": 0.6, "c": 0.5},
        reference_by_instance={"a": 0.3, "b": 0.4, "c": 0.4},
        cluster_by_instance={"a": "x", "b": "x", "c": "y"},
        seed=7,
        insufficient_cluster_policy="mark_non_authorizing",
    )
    assert result["iterations"] == 10_000
    assert result["inferential_authorized"] is False
    assert result["mean_difference"] == pytest.approx(
        ((0.8 - 0.3) + (0.6 - 0.4) + (0.5 - 0.4)) / 3.0
    )
    assert math.isfinite(float(result["lower"]))
    assert math.isfinite(float(result["upper"]))


def test_paired_cluster_bootstrap_fails_closed_below_cluster_minimum() -> None:
    candidate = {f"i{index}": 0.8 for index in range(9)}
    reference = {f"i{index}": 0.5 for index in range(9)}
    clusters = {f"i{index}": f"c{index}" for index in range(9)}
    with pytest.raises(ValueError, match="fewer independent clusters"):
        paired_cluster_bootstrap_comparison(
            candidate,
            reference,
            clusters,
            iterations=20,
        )

    exploratory = paired_cluster_bootstrap_comparison(
        candidate,
        reference,
        clusters,
        iterations=20,
        insufficient_cluster_policy="mark_non_authorizing",
    )
    assert exploratory["inferential_authorized"] is False
    assert "exploratory_only" in exploratory["authorization_reason"]
    assert exploratory["minimum_independent_cluster_count"] == 10

    authorized_candidate = {**candidate, "i9": 0.8}
    authorized_reference = {**reference, "i9": 0.5}
    authorized_clusters = {**clusters, "i9": "c9"}
    authorized = paired_cluster_bootstrap_comparison(
        authorized_candidate,
        authorized_reference,
        authorized_clusters,
        iterations=20,
    )
    assert authorized["inferential_authorized"] is True
    assert authorized["cluster_count"] == 10


def test_holm_bonferroni_correction_is_monotone_and_finite_family() -> None:
    result = holm_bonferroni_correction(
        {
            "pair_auprc": 0.01,
            "retrieval_mrr": 0.03,
            "calibration": 0.20,
        }
    )
    hypotheses = result["hypotheses"]
    assert result["method"] == "holm_bonferroni"
    assert result["family_size"] == 3
    assert hypotheses["pair_auprc"]["adjusted_p_value"] == pytest.approx(0.03)
    assert hypotheses["retrieval_mrr"]["adjusted_p_value"] == pytest.approx(0.06)
    assert hypotheses["calibration"]["adjusted_p_value"] == pytest.approx(0.20)
    assert hypotheses["pair_auprc"]["reject_family_wise_null"]
    assert not hypotheses["retrieval_mrr"]["reject_family_wise_null"]


def test_holm_bonferroni_rejects_invalid_or_duplicate_normalized_names() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_bonferroni_correction({"invalid": 1.1})
    with pytest.raises(ValueError, match="Duplicate normalized"):
        holm_bonferroni_correction({"test": 0.1, " test ": 0.2})
