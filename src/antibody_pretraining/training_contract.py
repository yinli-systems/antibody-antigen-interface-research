"""World-size-independent sampling and token-based optimization contracts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


UINT64_MASK = (1 << 64) - 1


class TrainingContractError(ValueError):
    """Report a malformed training protocol or impossible update plan."""


@dataclass(frozen=True)
class UpdatePlan:
    """Bind one optimizer update to canonical global sequence indices."""

    indices: np.ndarray
    next_sample_cursor: int
    residue_token_count: int
    sample_cursor: int


def canonical_json_bytes(payload: object) -> bytes:
    """Encode finite sorted ASCII JSON with one terminal line feed."""
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def load_training_protocol(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and validate the frozen training protocol."""
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if expected_file_sha256 is not None and observed != expected_file_sha256:
        raise TrainingContractError("Training protocol file digest differs")
    try:
        protocol = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingContractError("Training protocol must be ASCII JSON") from error
    if protocol.get("format_version") != "antibody_training_v1":
        raise TrainingContractError("Training protocol version differs")
    batching = protocol["batching"]
    schedule = protocol["schedule"]
    positive_integers = (
        batching["batch_quantum_sequences"],
        batching["global_residue_token_target_per_update"],
        schedule["total_residue_tokens"],
        schedule["warmup_residue_tokens"],
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in positive_integers):
        raise TrainingContractError("Token and batching values must be positive integers")
    if schedule["warmup_residue_tokens"] >= schedule["total_residue_tokens"]:
        raise TrainingContractError("Warmup must end before the total token budget")
    if batching["partial_final_windows"] != "reject":
        raise TrainingContractError("Partial final windows must be rejected")
    if batching["sampler"] != "counter_affine_bijection_v1":
        raise TrainingContractError("Sampler contract differs")
    return protocol


def _splitmix64(value: int) -> int:
    """Permute one unsigned 64-bit integer with SplitMix64."""
    value = (value + 0x9E3779B97F4A7C15) & UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return value ^ (value >> 31)


def _affine_coefficients(record_count: int, epoch: int, seed: int) -> tuple[int, int]:
    """Derive one bijective affine permutation for a corpus epoch."""
    if record_count == 1:
        return 0, 0
    stream = _splitmix64(seed ^ _splitmix64(epoch))
    multiplier = int(stream % record_count) or 1
    while math.gcd(multiplier, record_count) != 1:
        multiplier = (multiplier + 1) % record_count or 1
    offset = int(_splitmix64(stream) % record_count)
    return multiplier, offset


def sample_indices(
    *,
    record_count: int,
    sample_cursor: int,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Map global sample ordinals to a deterministic epoch-wise permutation."""
    values = (record_count, sample_cursor, sample_count, seed)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TrainingContractError("Sampler arguments must be integers")
    if record_count <= 0 or sample_cursor < 0 or sample_count <= 0 or seed < 0:
        raise TrainingContractError("Sampler arguments are outside their valid range")
    output = np.empty(sample_count, dtype=np.int64)
    written = 0
    cursor = sample_cursor
    while written < sample_count:
        epoch, position = divmod(cursor, record_count)
        take = min(sample_count - written, record_count - position)
        multiplier, offset = _affine_coefficients(record_count, epoch, seed)
        positions = np.arange(position, position + take, dtype=np.int64)
        if record_count == 1:
            output[written : written + take] = 0
        else:
            output[written : written + take] = (
                multiplier * positions + offset
            ) % record_count
        written += take
        cursor += take
    return output


def build_update_plan(
    residue_lengths: np.ndarray,
    *,
    sample_cursor: int,
    target_residue_tokens: int,
    batch_quantum_sequences: int,
    seed: int,
) -> UpdatePlan:
    """Select whole canonical quanta until the global token threshold is met."""
    lengths = np.asarray(residue_lengths)
    if lengths.ndim != 1 or not lengths.size or lengths.dtype.kind not in {"i", "u"}:
        raise TrainingContractError("residue_lengths must be a nonempty integer vector")
    if np.any(lengths <= 0):
        raise TrainingContractError("Every admitted sequence must contain residues")
    if target_residue_tokens <= 0 or batch_quantum_sequences <= 0:
        raise TrainingContractError("Update target and batch quantum must be positive")
    selected: list[np.ndarray] = []
    residue_count = 0
    cursor = sample_cursor
    while residue_count < target_residue_tokens:
        quantum = sample_indices(
            record_count=len(lengths),
            sample_cursor=cursor,
            sample_count=batch_quantum_sequences,
            seed=seed,
        )
        selected.append(quantum)
        residue_count += int(lengths[quantum].sum(dtype=np.int64))
        cursor += batch_quantum_sequences
    indices = np.concatenate(selected)
    return UpdatePlan(
        indices=indices,
        next_sample_cursor=cursor,
        residue_token_count=residue_count,
        sample_cursor=sample_cursor,
    )


def learning_rate_at_token(
    consumed_residue_tokens: int,
    *,
    peak_learning_rate: float,
    warmup_residue_tokens: int,
    total_residue_tokens: int,
    minimum_learning_rate_ratio: float,
) -> float:
    """Evaluate linear warmup followed by token-coordinate cosine decay."""
    if consumed_residue_tokens < 0:
        raise TrainingContractError("Consumed residue tokens cannot be negative")
    if peak_learning_rate <= 0.0 or warmup_residue_tokens <= 0:
        raise TrainingContractError("Learning rate and warmup must be positive")
    if total_residue_tokens <= warmup_residue_tokens:
        raise TrainingContractError("Total token budget must exceed warmup")
    if not 0.0 <= minimum_learning_rate_ratio <= 1.0:
        raise TrainingContractError("Minimum learning-rate ratio is invalid")
    if consumed_residue_tokens < warmup_residue_tokens:
        return peak_learning_rate * (
            consumed_residue_tokens / warmup_residue_tokens
        )
    progress = min(
        1.0,
        (consumed_residue_tokens - warmup_residue_tokens)
        / (total_residue_tokens - warmup_residue_tokens),
    )
    multiplier = minimum_learning_rate_ratio + (
        1.0 - minimum_learning_rate_ratio
    ) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_learning_rate * multiplier
