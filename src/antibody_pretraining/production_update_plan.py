"""Plan deterministic token-budgeted updates over a sharded production corpus."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .training_contract import TrainingContractError, UpdatePlan, sample_indices


def build_indexed_update_plan(
    *,
    record_count: int,
    residue_length_lookup: Callable[[np.ndarray], np.ndarray],
    sample_cursor: int,
    target_residue_tokens: int,
    batch_quantum_sequences: int,
    seed: int,
) -> UpdatePlan:
    """Plan an update through bounded indexed reads without a global length copy."""
    values = (
        record_count,
        sample_cursor,
        target_residue_tokens,
        batch_quantum_sequences,
        seed,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TrainingContractError("Indexed planner arguments must be integers")
    if (
        record_count <= 0
        or sample_cursor < 0
        or target_residue_tokens <= 0
        or batch_quantum_sequences <= 0
        or seed < 0
    ):
        raise TrainingContractError(
            "Indexed planner arguments are outside their valid range"
        )
    selected: list[np.ndarray] = []
    residue_count = 0
    cursor = sample_cursor
    while residue_count < target_residue_tokens:
        quantum = sample_indices(
            record_count=record_count,
            sample_cursor=cursor,
            sample_count=batch_quantum_sequences,
            seed=seed,
        )
        lengths = np.asarray(residue_length_lookup(quantum))
        if (
            lengths.ndim != 1
            or lengths.shape != quantum.shape
            or lengths.dtype.kind not in {"i", "u"}
            or np.any(lengths <= 0)
        ):
            raise TrainingContractError("Indexed residue lengths are invalid")
        selected.append(quantum)
        residue_count += int(lengths.sum(dtype=np.int64))
        cursor += batch_quantum_sequences
    return UpdatePlan(
        indices=np.concatenate(selected),
        next_sample_cursor=cursor,
        residue_token_count=residue_count,
        sample_cursor=sample_cursor,
    )
