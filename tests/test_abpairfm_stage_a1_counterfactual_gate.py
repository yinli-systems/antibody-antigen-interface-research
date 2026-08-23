from __future__ import annotations

from copy import deepcopy

from scripts.materialize_abpairfm_stage_a1_counterfactual_gate import (
    materialize_counterfactuals,
)


def _record(index: int) -> dict:
    return {
        "endpoint_cluster1": f"left-cluster-{index}",
        "endpoint_cluster2": f"right-cluster-{index}",
        "endpoint_exact1_sha256": f"left-exact-{index}",
        "endpoint_exact2_sha256": f"right-exact-{index}",
        "example_id": f"example-{index}",
        "known_positive_group": f"group-{index}",
        "pilot_record_index": index,
        "retrieval_supervision": True,
        "sequence1": "A" * 31,
        "sequence2": "C" * 35,
        "source": "train",
    }


def test_counterfactual_materialization_is_complete_and_deterministic() -> None:
    records = [_record(index) for index in range(8)]
    first = materialize_counterfactuals(records, seed=1701)
    second = materialize_counterfactuals(deepcopy(records), seed=1701)

    assert first == second
    assert first["coverage"] == 1.0
    assert first["counts"] == {
        "mapped_anchors": 8,
        "retrieval_records": 8,
        "unmatched_anchors": 0,
    }
    assert first["admissible_candidates_per_anchor"]["minimum"] == 7
    assert len(first["mapping_sha256"]) == 64


def test_counterfactual_materialization_fails_closed_on_cluster_overlap() -> None:
    records = [_record(index) for index in range(4)]
    for record in records:
        record["endpoint_cluster1"] = "shared-cluster"

    result = materialize_counterfactuals(records, seed=1701)
    assert result["coverage"] == 0.0
    assert result["counts"]["unmatched_anchors"] == 4
    assert result["rejection_counts"]["source_cluster_overlap"] == 12
