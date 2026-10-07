"""Regression tests for non-numeric fields in macro metric aggregation.

``macro_metric_summary`` documents itself as averaging numeric keys, but relied
on a hardcoded exclusion list to achieve that. ``ranking_at_k`` reports a string
``tie_policy``, so aggregating records that include ranking metrics raised
``could not convert string to float`` -- after the evaluation had already
written its predictions, which is how seeds of the margin replication failed
with their results intact.
"""

from __future__ import annotations

import pytest

from interaction_first.metrics import macro_metric_summary


def test_a_string_field_is_skipped_rather_than_averaged() -> None:
    records = [
        {"auprc": 0.5, "tie_policy": "fractional_expected_credit_at_score_boundary"},
        {"auprc": 0.7, "tie_policy": "fractional_expected_credit_at_score_boundary"},
    ]

    summary = macro_metric_summary(records)

    assert "tie_policy" not in summary
    assert summary["auprc"] == pytest.approx(0.6)


def test_boolean_fields_are_not_averaged_as_numbers() -> None:
    records = [{"auprc": 0.4, "improved": True}, {"auprc": 0.6, "improved": False}]

    summary = macro_metric_summary(records)

    assert "improved" not in summary
    assert summary["auprc"] == pytest.approx(0.5)


def test_a_key_numeric_in_every_record_is_kept() -> None:
    records = [{"auprc": 0.4, "recall_at_5": 0.2}, {"auprc": 0.6, "recall_at_5": 0.4}]

    summary = macro_metric_summary(records)

    assert summary["recall_at_5"] == pytest.approx(0.3)


def test_a_key_that_is_numeric_in_only_some_records_is_skipped() -> None:
    """Mixed typing is a defect in the producer, not something to average past."""
    records = [{"auprc": 0.4, "mixed": 1.0}, {"auprc": 0.6, "mixed": "n/a"}]

    summary = macro_metric_summary(records)

    assert "mixed" not in summary


def test_none_values_do_not_disqualify_a_numeric_key() -> None:
    records = [{"auprc": 0.4, "sometimes": None}, {"auprc": 0.6, "sometimes": 0.8}]

    summary = macro_metric_summary(records)

    assert summary["sometimes"] == pytest.approx(0.8)


def test_an_all_none_key_aggregates_to_none() -> None:
    records = [{"auprc": 0.4, "never": None}, {"auprc": 0.6, "never": None}]

    assert macro_metric_summary(records)["never"] is None


def test_ranking_output_can_be_aggregated_end_to_end() -> None:
    """The real shape that triggered the failure."""
    import numpy as np

    from interaction_first.metrics import ranking_at_k

    labels = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
    records = [
        ranking_at_k(labels, np.array([0.9, 0.1, 0.8, 0.2, 0.05]), cutoffs=[1, 2]),
        ranking_at_k(labels, np.array([0.7, 0.3, 0.6, 0.4, 0.10]), cutoffs=[1, 2]),
    ]

    summary = macro_metric_summary(records)

    assert "tie_policy" not in summary
    assert summary["precision_at_1"] == pytest.approx(1.0)
