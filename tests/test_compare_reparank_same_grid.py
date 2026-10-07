from __future__ import annotations

import hashlib

import numpy as np
import pytest

from extensions.frontier_multimodel_v1.compare_reparank_same_grid import (
    ESM2_SENSITIVITY_SEEDS,
    PRIMARY_SEEDS,
    crossed_bootstrap,
    pair_key_digest,
)


def test_pair_key_digest_is_order_invariant() -> None:
    ag = np.array([1, 0, 1], dtype=np.int64)
    ab = np.array([0, 2, 1], dtype=np.int64)
    label = np.array([1, 0, 1], dtype=np.uint8)
    first = pair_key_digest(ag, ab, label)
    order = np.array([2, 0, 1])
    second = pair_key_digest(ag[order], ab[order], label[order])
    assert first == second


def test_pair_key_digest_changes_with_label_or_residue() -> None:
    ag = np.array([0, 1], dtype=np.int64)
    ab = np.array([2, 3], dtype=np.int64)
    label = np.array([0, 1], dtype=np.uint8)
    original = pair_key_digest(ag, ab, label)
    assert pair_key_digest(ag, ab, 1 - label) != original
    assert pair_key_digest(ag + 1, ab, label) != original


def test_pair_key_digest_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        pair_key_digest(np.array([0]), np.array([0, 1]), np.array([1]))


def test_crossed_bootstrap_equal_weights_components_and_seeds() -> None:
    differences = {
        1: {"a1": 0.0, "a2": 0.0, "a3": 0.0, "b1": 1.0},
        2: {"a1": 0.0, "a2": 0.0, "a3": 0.0, "b1": 1.0},
    }
    components = {"a1": "a", "a2": "a", "a3": "a", "b1": "b"}
    result = crossed_bootstrap(differences, components, repeats=1000)
    assert result["mean_difference"] == pytest.approx(0.5)
    assert result["complex_weighted_mean_difference"] == pytest.approx(0.25)
    assert result["component_count"] == 2
    assert result["baseline_seeds"] == [1, 2]


def test_crossed_bootstrap_requires_exact_component_coverage() -> None:
    with pytest.raises(ValueError, match="coverage"):
        crossed_bootstrap({1: {"a": 1.0}}, {"a": "x", "b": "y"}, repeats=100)


def test_seed_sets_are_predeclared_and_disjoint() -> None:
    assert PRIMARY_SEEDS == tuple(range(1, 11))
    assert ESM2_SENSITIVITY_SEEDS == tuple(range(11, 21))
    assert set(PRIMARY_SEEDS).isdisjoint(ESM2_SENSITIVITY_SEEDS)
