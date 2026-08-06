from __future__ import annotations

import math

from scripts.audit_baseline_metrics import (
    epiformer_rows,
    recover_micro_confusion,
)


def test_epiformer_receipt_reconstructs_micro_confusion_counts() -> None:
    counts = recover_micro_confusion(
        positive_count=7968,
        residue_count=97252,
        f1=0.33503207099551885,
        mcc=0.2715204001684301,
    )
    assert counts == {
        "true_positives": 3813,
        "false_positives": 10981,
        "false_negatives": 4155,
        "true_negatives": 78303,
    }


def test_epiformer_row_is_frozen_sabdab2_epitope_only() -> None:
    row = epiformer_rows()[0]
    assert row["scope"].startswith("frozen SAbDab2 test")
    assert row["test_instance_count"] == 455
    assert row["mapping_failures"] == 0
    assert row["native_pair_interaction_score"] is False
    assert math.isclose(row["precision"], 0.2577396241719616)
    assert math.isclose(row["recall"], 0.47853915662650603)
