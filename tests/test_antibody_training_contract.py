import json
import math
from pathlib import Path

import numpy as np
import pytest

from antibody_pretraining.training_contract import (
    TrainingContractError,
    build_update_plan,
    learning_rate_at_token,
    load_training_protocol,
    sample_indices,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "antibody_training_v1.json"


def test_training_protocol_is_frozen_and_test_independent() -> None:
    protocol = load_training_protocol(PROTOCOL)
    assert protocol["schedule"]["total_residue_tokens"] == 30_000_000_000
    assert protocol["batching"]["batch_quantum_sequences"] == 512
    assert protocol["test_access"] == {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }
    assert json.loads(PROTOCOL.read_text(encoding="ascii")) == protocol


def test_sampler_is_bijective_per_epoch_and_replayable_across_boundaries() -> None:
    first = sample_indices(record_count=17, sample_cursor=0, sample_count=17, seed=42)
    assert sorted(first.tolist()) == list(range(17))
    combined = sample_indices(
        record_count=17,
        sample_cursor=11,
        sample_count=29,
        seed=42,
    )
    split = np.concatenate(
        (
            sample_indices(record_count=17, sample_cursor=11, sample_count=7, seed=42),
            sample_indices(record_count=17, sample_cursor=18, sample_count=22, seed=42),
        )
    )
    assert np.array_equal(combined, split)


def test_update_membership_is_independent_of_rank_partition() -> None:
    lengths = np.arange(80, 117, dtype=np.int64)
    plan = build_update_plan(
        lengths,
        sample_cursor=91,
        target_residue_tokens=12_000,
        batch_quantum_sequences=32,
        seed=73,
    )
    assert plan.residue_token_count >= 12_000
    assert len(plan.indices) % 32 == 0
    one_rank = np.concatenate(np.array_split(plan.indices, 1))
    eight_ranks = np.concatenate(
        [plan.indices[offset::8] for offset in range(8)]
    )
    assert sorted(one_rank.tolist()) == sorted(eight_ranks.tolist())
    assert plan.next_sample_cursor == plan.sample_cursor + len(plan.indices)


def test_token_schedule_hits_exact_boundaries_and_floor() -> None:
    arguments = {
        "peak_learning_rate": 3e-4,
        "warmup_residue_tokens": 100,
        "total_residue_tokens": 1_000,
        "minimum_learning_rate_ratio": 0.1,
    }
    assert learning_rate_at_token(0, **arguments) == 0.0
    assert learning_rate_at_token(50, **arguments) == pytest.approx(1.5e-4)
    assert learning_rate_at_token(100, **arguments) == pytest.approx(3e-4)
    assert learning_rate_at_token(1_000, **arguments) == pytest.approx(3e-5)
    assert learning_rate_at_token(10_000, **arguments) == pytest.approx(3e-5)
    assert math.isfinite(learning_rate_at_token(777, **arguments))


def test_training_contract_rejects_invalid_inputs() -> None:
    with pytest.raises(TrainingContractError):
        sample_indices(record_count=0, sample_cursor=0, sample_count=1, seed=1)
    with pytest.raises(TrainingContractError):
        build_update_plan(
            np.asarray([10, 0]),
            sample_cursor=0,
            target_residue_tokens=10,
            batch_quantum_sequences=2,
            seed=1,
        )
    with pytest.raises(TrainingContractError):
        learning_rate_at_token(
            -1,
            peak_learning_rate=1e-3,
            warmup_residue_tokens=10,
            total_residue_tokens=100,
            minimum_learning_rate_ratio=0.1,
        )
