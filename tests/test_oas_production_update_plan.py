"""Test bounded deterministic update planning for the sharded OAS corpus."""

from __future__ import annotations

import numpy as np
import pytest

from antibody_pretraining.production_update_plan import build_indexed_update_plan
from antibody_pretraining.training_contract import (
    TrainingContractError,
    build_update_plan,
)


def test_indexed_update_plan_matches_frozen_dense_semantics() -> None:
    lengths = np.asarray([3, 7, 11, 13, 17, 19, 23], dtype=np.uint16)
    calls: list[np.ndarray] = []

    def lookup(indices: np.ndarray) -> np.ndarray:
        calls.append(indices.copy())
        return lengths[indices]

    expected = build_update_plan(
        lengths,
        sample_cursor=5,
        target_residue_tokens=91,
        batch_quantum_sequences=4,
        seed=73,
    )
    observed = build_indexed_update_plan(
        record_count=len(lengths),
        residue_length_lookup=lookup,
        sample_cursor=5,
        target_residue_tokens=91,
        batch_quantum_sequences=4,
        seed=73,
    )
    assert np.array_equal(observed.indices, expected.indices)
    assert observed.next_sample_cursor == expected.next_sample_cursor
    assert observed.residue_token_count == expected.residue_token_count
    assert observed.sample_cursor == expected.sample_cursor
    assert calls
    assert all(call.shape == (4,) for call in calls)


def test_indexed_update_plan_rejects_malformed_length_lookup() -> None:
    with pytest.raises(TrainingContractError, match="Indexed residue lengths"):
        build_indexed_update_plan(
            record_count=7,
            residue_length_lookup=lambda indices: np.zeros(len(indices)),
            sample_cursor=0,
            target_residue_tokens=10,
            batch_quantum_sequences=4,
            seed=17,
        )
