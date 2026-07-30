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


def partition_update_indices(
    indices: np.ndarray,
    *,
    micro_batch_per_rank: int,
    world_size: int,
    rank: int,
) -> tuple[np.ndarray, ...]:
    """Partition one global update into synchronized rank-local microbatches."""
    requested = np.asarray(indices)
    if requested.dtype.kind not in {"i", "u"} or requested.ndim != 1:
        raise TrainingContractError(
            "Update indices must be a one-dimensional integer array"
        )
    dimensions = (micro_batch_per_rank, world_size, rank)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in dimensions
    ):
        raise TrainingContractError("Partition dimensions must be integers")
    if micro_batch_per_rank <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise TrainingContractError(
            "Partition dimensions are outside their valid range"
        )
    global_micro_batch = micro_batch_per_rank * world_size
    if len(requested) % global_micro_batch:
        raise TrainingContractError(
            "Global update is not divisible by the global microbatch"
        )
    return tuple(
        requested[
            start + rank * micro_batch_per_rank :
            start + (rank + 1) * micro_batch_per_rank
        ]
        for start in range(0, len(requested), global_micro_batch)
    )
