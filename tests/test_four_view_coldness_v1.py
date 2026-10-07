from __future__ import annotations

import pytest

from antibody_pretraining.four_view_coldness import (
    REQUIRED_SEQUENCE_VIEWS,
    SequenceSimilarityRelation,
    audit_four_view_coldness,
)


def digest(character: str) -> str:
    return character * 64


def clean_memberships():
    return {
        view: [(digest("a"), "train"), (digest("b"), "test")]
        for view in REQUIRED_SEQUENCE_VIEWS
    }


def empty_relations():
    return {view: [] for view in REQUIRED_SEQUENCE_VIEWS}


def test_clean_frozen_unopened_four_view_cohort_is_authorized() -> None:
    result = audit_four_view_coldness(
        clean_memberships(),
        empty_relations(),
        labels_unopened=True,
        cohort_frozen_before_audit=True,
    )
    assert result.authorized
    assert result.status == "PASS_FROZEN_UNOPENED_FOUR_VIEW_DOUBLE_COLD_GATE"
    assert result.identity_threshold == 0.4


def test_threshold_edge_in_any_view_fails_closed() -> None:
    relations = empty_relations()
    relations["antigen"] = [
        SequenceSimilarityRelation(digest("a"), digest("b"), 0.40, 0.80, 0.95)
    ]
    result = audit_four_view_coldness(
        clean_memberships(),
        relations,
        labels_unopened=True,
        cohort_frozen_before_audit=True,
    )
    assert not result.authorized
    assert result.status == "FAIL_THRESHOLD_FOUR_VIEW_CROSSING"
    assert result.crossing_count_by_view["antigen"] == 1


def test_exact_digest_reuse_across_splits_fails_closed() -> None:
    memberships = clean_memberships()
    memberships["vh"].append((digest("a"), "test"))
    result = audit_four_view_coldness(
        memberships,
        empty_relations(),
        labels_unopened=True,
        cohort_frozen_before_audit=True,
    )
    assert not result.authorized
    assert result.status == "FAIL_EXACT_FOUR_VIEW_CROSSING"


def test_opened_labels_cannot_be_relabelled_blind() -> None:
    result = audit_four_view_coldness(
        clean_memberships(),
        empty_relations(),
        labels_unopened=False,
        cohort_frozen_before_audit=True,
    )
    assert not result.authorized
    assert result.status == "FAIL_LABELS_ALREADY_OPENED"


def test_unknown_relation_digest_is_rejected() -> None:
    relations = empty_relations()
    relations["vl"] = [
        SequenceSimilarityRelation(digest("a"), digest("c"), 0.9, 0.9, 0.9)
    ]
    with pytest.raises(ValueError, match="unknown digest"):
        audit_four_view_coldness(
            clean_memberships(),
            relations,
            labels_unopened=True,
            cohort_frozen_before_audit=True,
        )
