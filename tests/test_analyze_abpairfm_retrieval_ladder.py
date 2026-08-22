from __future__ import annotations

import json

from scripts.analyze_abpairfm_retrieval_ladder import (
    canonical_pair_metrics,
    paired_bootstrap,
)


def _row(direction: str, endpoint: str, cognate: bool, score: float) -> dict:
    return {
        "anchor_instance_id": "anchor",
        "candidate_endpoint_sha256": endpoint,
        "canonical_pair_sha256": "pair",
        "dataset": "asep",
        "direction": direction,
        "is_cognate": cognate,
        "score": score,
    }


def test_reproduces_bidirectional_unique_endpoint_metrics(tmp_path) -> None:
    rows = []
    for direction in (
        "fixed_antibody_wrong_antigen",
        "fixed_antigen_wrong_antibody",
    ):
        rows.extend(
            [
                _row(direction, "cognate", True, 1.0),
                _row(direction, "decoy", False, 0.0),
                _row(direction, "decoy", False, 0.0),
            ]
        )
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert canonical_pair_metrics(path, "asep") == {
        "pair": {"mrr": 1.0, "recall_at_1": 1.0}
    }


def test_paired_bootstrap_reports_reference_minus_comparator() -> None:
    reference = {
        "a": {"mrr": 1.0},
        "b": {"mrr": 0.5},
    }
    comparator = {
        "a": {"mrr": 0.5},
        "b": {"mrr": 0.5},
    }
    result = paired_bootstrap(reference, comparator, "mrr", replicates=100, seed=4)
    assert result["delta_reference_minus_comparator"] == 0.25
    assert result["paired_unit_count"] == 2
    assert 0.0 <= result["probability_delta_gt_zero"] <= 1.0
