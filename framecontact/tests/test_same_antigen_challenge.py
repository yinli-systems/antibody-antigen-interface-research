from __future__ import annotations

from interaction_first.same_antigen_challenge import (
    aggregate_same_antigen_pair_metrics,
    bootstrap_same_antigen_group_metrics,
    build_label_free_same_antigen_challenge_entries,
    build_same_antigen_challenge_entries,
    score_same_antigen_pair,
)


def _record(identifier: str, antigen: str, antibody: str) -> dict[str, str]:
    return {
        "instance_id": identifier,
        "antigen_sequence_sha256": antigen,
        "antibody_sequence_sha256": antibody,
    }


def test_challenge_requires_distinct_antibody_cluster_and_epitope() -> None:
    records = [
        _record("a", "ag1", "ab1"),
        _record("b", "ag1", "ab2"),
        _record("c", "ag1", "ab3"),
        _record("d", "ag2", "ab4"),
    ]
    labels = {
        "a": frozenset({1, 2}),
        "b": frozenset({8, 9}),
        "c": frozenset({1, 2}),
        "d": frozenset({4}),
    }
    clusters = {"a": "x", "b": "y", "c": "z", "d": "w"}
    entries = build_same_antigen_challenge_entries(
        records,
        epitope_indices_by_id=labels,
        antibody_cluster_by_id=clusters,
    )
    by_id = {entry["focal_instance_id"]: entry for entry in entries}
    assert [row["instance_id"] for row in by_id["a"]["alternates"]] == ["b"]
    assert {row["instance_id"] for row in by_id["b"]["alternates"]} == {"a", "c"}
    assert [row["instance_id"] for row in by_id["c"]["alternates"]] == ["b"]


def test_label_free_challenge_uses_only_exact_antigen_and_distinct_antibody() -> None:
    records = [
        {**_record("a", "ag1", "ab1"), "antibody_cluster_id": "x"},
        {**_record("b", "ag1", "ab2"), "antibody_cluster_id": "y"},
        {**_record("c", "ag1", "ab3"), "antibody_cluster_id": "x"},
        {**_record("d", "ag2", "ab4"), "antibody_cluster_id": "z"},
    ]
    entries = build_label_free_same_antigen_challenge_entries(records)
    by_id = {entry["focal_instance_id"]: entry for entry in entries}
    assert [row["instance_id"] for row in by_id["a"]["alternates"]] == ["b"]
    assert {row["instance_id"] for row in by_id["b"]["alternates"]} == {"a", "c"}
    assert [row["instance_id"] for row in by_id["c"]["alternates"]] == ["b"]
    assert "d" not in by_id


def test_group_weighting_prevents_large_family_domination() -> None:
    rows = []
    for group, count, value in (("large", 9, 1.0), ("small", 1, -1.0)):
        for _ in range(count):
            rows.append(
                {
                    "antigen_sequence_sha256": group,
                    "alternate_target_auprc_gain": value,
                    "target_preference_shift": value,
                    "alternate_target_topk_recall_gain": value,
                    "epitope_map_mean_absolute_change": abs(value),
                }
            )
    result = aggregate_same_antigen_pair_metrics(rows)
    assert result["ordered_pair_count"] == 10
    assert result["exact_antigen_group_count"] == 2
    assert result["group_weighted"]["alternate_target_auprc_gain"] == 0.0


def test_directed_assignment_and_group_bootstrap_are_reported() -> None:
    rows = []
    for group, value in (("g1", 0.3), ("g2", 0.1)):
        rows.append(
            {
                "antigen_sequence_sha256": group,
                "alternate_target_auprc_gain": value,
                "target_preference_shift": 2 * value,
                "alternate_target_topk_recall_gain": value,
                "epitope_map_mean_absolute_change": 0.2,
                "cognate_assignment_margin": value,
                "swapped_assignment_margin": value,
                "directed_assignment_margin": value,
                "focal_target_auprc_change": -value,
            }
        )
    result = aggregate_same_antigen_pair_metrics(rows)
    assert result["ordered_pair_directed_assignment_success_rate"] == 1.0
    assert result["group_weighted"]["directed_assignment_margin"] == 0.2
    interval = bootstrap_same_antigen_group_metrics(
        rows, replicates=1000, seed=7
    )
    assert interval["metrics"]["directed_assignment_margin"]["estimate"] == 0.2
    assert (
        interval["metrics"]["directed_assignment_margin"]["ci95_lower"]
        <= 0.2
        <= interval["metrics"]["directed_assignment_margin"]["ci95_upper"]
    )


def test_pair_scorer_requires_directional_migration_not_arbitrary_change() -> None:
    focal = [1, 1, 0, 0]
    alternate = [0, 0, 1, 1]
    result = score_same_antigen_pair(
        [0.9, 0.8, 0.2, 0.1],
        [0.1, 0.2, 0.8, 0.9],
        focal,
        alternate,
    )
    assert result["eligible"] is True
    assert result["epitope_jaccard"] == 0.0
    assert result["cognate_assignment_margin"] > 0
    assert result["swapped_assignment_margin"] > 0
    assert result["directed_assignment_margin"] > 0
    assert result["alternate_target_auprc_gain"] > 0
    assert result["alternate_target_topk_recall_gain"] > 0


def test_pair_scorer_marks_zero_epitope_pair_ineligible() -> None:
    result = score_same_antigen_pair(
        [0.2, 0.1], [0.1, 0.2], [0, 0], [1, 0]
    )
    assert result["eligible"] is False
    assert result["ineligible_reason"].startswith("zero_positive")
