"""Tests for the provisional S2 fold-assignment readiness diagnostics."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from interaction_first.s2_split_readiness import (
    S2ReadinessError,
    authority_membership_report,
    consistency_report,
    membership_sha256,
    retention_frontier,
    retention_required_for_floor,
)

ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT_PATH = (
    ROOT / "protocols" / "train_only_s2_feasibility" / "s2_assignment_provisional_v1.json"
)


@pytest.fixture(scope="module")
def assignment() -> dict:
    return json.loads(ASSIGNMENT_PATH.read_text(encoding="utf-8"))


# --- membership digest ---------------------------------------------------


def test_membership_digest_is_order_and_duplicate_invariant() -> None:
    assert membership_sha256(["b", "a"]) == membership_sha256(["a", "b", "a"])


def test_membership_digest_uses_lf_framing() -> None:
    import hashlib

    expected = hashlib.sha256(b"a\nb\n").hexdigest()

    assert membership_sha256(["b", "a"]) == expected


def test_membership_digest_rejects_non_ascii_and_empty() -> None:
    with pytest.raises(S2ReadinessError):
        membership_sha256(["ok", "nöt-ascii"])
    with pytest.raises(S2ReadinessError):
        membership_sha256(["ok", ""])


# --- retention frontier --------------------------------------------------


def test_required_retention_satisfies_its_own_definition() -> None:
    for floor in (10, 100, 1098, 1647):
        retained = retention_required_for_floor(floor, 3)
        assert retained - math.ceil(retained / 3) >= floor
        assert (retained - 1) - math.ceil((retained - 1) / 3) < floor


def test_a_floor_of_1098_needs_1647_retained_pairs_across_three_folds() -> None:
    assert retention_required_for_floor(1098, 3) == 1647


def test_more_folds_need_less_retention_for_the_same_floor() -> None:
    assert retention_required_for_floor(1000, 5) < retention_required_for_floor(1000, 3)


def test_frontier_reports_achieved_and_best_possible_floors() -> None:
    frontier = retention_frontier([561, 538, 508])

    assert frontier.retained_pairs == 1607
    assert frontier.achieved_training_floor == 1607 - 561
    assert frontier.best_possible_training_floor == 1607 - math.ceil(1607 / 3)


def test_the_optimal_retention_cannot_support_the_nested_floor() -> None:
    """The counting result: 1,607 optimal pairs cannot give three folds >= 1,098."""
    frontier = retention_frontier([561, 538, 508], required_floor=1098)

    assert frontier.floor_is_reachable is False
    assert frontier.retention_required_for_floor == 1647
    assert frontier.retention_shortfall == 40


def test_a_reachable_floor_is_reported_as_reachable() -> None:
    frontier = retention_frontier([561, 538, 508], required_floor=1000)

    assert frontier.floor_is_reachable is True
    assert frontier.retention_shortfall == 0


def test_an_unbalanced_split_can_miss_a_floor_that_rebalancing_would_reach() -> None:
    """Distinguishes a balance problem from a retention problem."""
    frontier = retention_frontier([900, 400, 307], required_floor=1060)

    assert frontier.achieved_training_floor < 1060
    assert frontier.floor_is_reachable is True


def test_frontier_rejects_degenerate_input() -> None:
    with pytest.raises(S2ReadinessError):
        retention_frontier([100])
    with pytest.raises(S2ReadinessError):
        retention_frontier([100, -1])
    with pytest.raises(S2ReadinessError):
        retention_required_for_floor(10, 1)


# --- the committed artifact ---------------------------------------------


def test_the_committed_assignment_is_internally_consistent(assignment: dict) -> None:
    report = consistency_report(assignment)

    assert report["all_passed"], report["checks"]
    assert report["retained_pairs"] == 1607
    assert report["discarded_pairs"] == 223
    assert report["total_pairs"] == 1830
    assert report["fold_pair_counts"] == [561, 538, 508]


def test_the_committed_assignment_retains_no_cross_fold_interaction(assignment: dict) -> None:
    assert consistency_report(assignment)["checks"]["no_cross_fold_interactions_retained"]


def test_consistency_catches_a_broken_partition(assignment: dict) -> None:
    tampered = copy.deepcopy(assignment)
    tampered["ledger_accounting"]["discarded_unique_exact_pair_count"] += 1

    report = consistency_report(tampered)

    assert report["all_passed"] is False
    assert report["checks"]["pairs_partition_total"] is False


def test_consistency_catches_a_nonzero_mip_gap(assignment: dict) -> None:
    tampered = copy.deepcopy(assignment)
    tampered["solver_result"]["mip_gap"] = 0.01

    assert consistency_report(tampered)["checks"]["solver_reported_optimal"] is False


def test_consistency_requires_a_recognisable_assignment() -> None:
    with pytest.raises(S2ReadinessError):
        consistency_report({"solution_summary": {}})


# --- authority membership -----------------------------------------------


def _splits(train: list[str], test: list[str], validation: list[str]) -> dict:
    return {"train": train, "test": test, "validation": validation}


def _authority(train: list[str], sealed: list[str], counts: dict) -> dict:
    return {
        "authority": {
            "source_pool_split": "train",
            "sealed_splits": ["test", "validation"],
            "authorized_membership_sha256": membership_sha256(train),
            "authorized_instance_count": len(set(train)),
            "sealed_membership_sha256": membership_sha256(sealed),
            "sealed_instance_count": len(set(sealed)),
            "sealed_split_counts": counts,
        }
    }


def test_membership_report_accepts_an_unchanged_split() -> None:
    train, test, validation = ["a", "b"], ["c"], ["d"]
    report = authority_membership_report(
        _authority(train, test + validation, {"test": 1, "validation": 1}),
        _splits(train, test, validation),
    )

    assert report["authorized_membership_matches"]
    assert report["sealed_membership_matches"]
    assert report["sealed_partition_matches"]
    assert report["train_sealed_overlap"] == 0


def test_membership_report_separates_a_moved_partition_from_moved_membership() -> None:
    """The sealed pool can be intact while its internal boundary has moved."""
    train = ["a", "b"]
    report = authority_membership_report(
        _authority(train, ["c", "d"], {"test": 2, "validation": 0}),
        _splits(train, ["c"], ["d"]),
    )

    assert report["authorized_membership_matches"]
    assert report["sealed_membership_matches"]
    assert report["sealed_partition_matches"] is False
    assert report["recorded_sealed_counts"] == {"test": 2, "validation": 0}
    assert report["observed_sealed_counts"] == {"test": 1, "validation": 1}


def test_membership_report_detects_changed_membership() -> None:
    train = ["a", "b"]
    report = authority_membership_report(
        _authority(train, ["c", "d"], {"test": 1, "validation": 1}),
        _splits(train, ["c"], ["e"]),
    )

    assert report["authorized_membership_matches"]
    assert report["sealed_membership_matches"] is False


def test_membership_report_detects_train_contamination() -> None:
    train = ["a", "b", "c"]
    report = authority_membership_report(
        _authority(train, ["c", "d"], {"test": 1, "validation": 1}),
        _splits(train, ["c"], ["d"]),
    )

    assert report["train_sealed_overlap"] == 1


def test_membership_report_requires_the_named_splits() -> None:
    train = ["a"]
    with pytest.raises(S2ReadinessError, match="no 'validation' split"):
        authority_membership_report(
            _authority(train, ["b"], {"test": 1, "validation": 0}),
            {"train": train, "test": ["b"]},
        )
