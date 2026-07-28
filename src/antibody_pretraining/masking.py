"""Counter-derived masked-language-model corruption with NumPy uint64 streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tokenizer import (
    CLS_ID,
    FIRST_RESIDUE_ID,
    MASK_ID,
    PAD_ID,
    SEP_ID,
    STANDARD_RESIDUE_COUNT,
)


UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
POSITION_KEY = np.uint64(0xD2B74407B1CE6E93)
STEP_KEY = np.uint64(0x9E3779B97F4A7C15)
STREAM_KEY = np.uint64(0x94D049BB133111EB)


class MaskingError(ValueError):
    """Report an invalid deterministic MLM masking request."""


@dataclass(frozen=True)
class MaskedBatch:
    """Hold corrupted token IDs, labels, and selected-target count."""

    input_ids: np.ndarray
    labels: np.ndarray
    selected_target_count: int


def _splitmix64(value: np.ndarray) -> np.ndarray:
    """Apply SplitMix64 as a deterministic counter permutation."""
    value = (value + STEP_KEY) & UINT64_MASK
    value = ((value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & UINT64_MASK
    value = ((value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & UINT64_MASK
    return value ^ (value >> np.uint64(31))


def apply_uniform_mlm(
    input_ids: np.ndarray,
    sample_keys: np.ndarray,
    *,
    update: int,
    seed: int,
    mask_fraction: float = 0.15,
) -> MaskedBatch:
    """Apply deterministic 80/10/10 MLM corruption to residue positions."""
    tokens = np.asarray(input_ids)
    keys = np.asarray(sample_keys, dtype=np.uint64)
    if tokens.ndim != 2 or keys.ndim != 1 or keys.shape[0] != tokens.shape[0]:
        raise MaskingError("input_ids and sample_keys have incompatible shapes")
    if tokens.dtype.kind not in {"i", "u"}:
        raise MaskingError("input_ids must contain integers")
    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise MaskingError("update must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise MaskingError("seed must be a non-negative integer")
    if not 0.0 < mask_fraction < 1.0:
        raise MaskingError("mask_fraction must be in (0, 1)")
    positions = np.arange(tokens.shape[1], dtype=np.uint64)[None, :]
    update_key = np.uint64((update * int(STEP_KEY)) & int(UINT64_MASK))
    counters = (
        keys[:, None]
        ^ (positions * POSITION_KEY)
        ^ update_key
        ^ np.uint64(seed)
    )
    selection_random = _splitmix64(counters)
    threshold = np.uint64(int(mask_fraction * (1 << 64)))
    eligible = (tokens != PAD_ID) & (tokens != CLS_ID) & (tokens != SEP_ID)
    selected = eligible & (selection_random < threshold)
    labels = np.full(tokens.shape, -100, dtype=np.int64)
    labels[selected] = tokens[selected]
    corrupted = tokens.astype(np.int64, copy=True)

    replacement_random = _splitmix64(counters ^ STREAM_KEY)
    replacement_bucket = replacement_random % np.uint64(10)
    use_mask = selected & (replacement_bucket < np.uint64(8))
    use_random = selected & (replacement_bucket == np.uint64(8))
    corrupted[use_mask] = MASK_ID
    second_stream_key = np.uint64((2 * int(STREAM_KEY)) & int(UINT64_MASK))
    residue_random = _splitmix64(counters ^ second_stream_key)
    corrupted[use_random] = FIRST_RESIDUE_ID + (
        residue_random[use_random] % np.uint64(STANDARD_RESIDUE_COUNT)
    ).astype(np.int64)
    return MaskedBatch(
        input_ids=corrupted,
        labels=labels,
        selected_target_count=int(selected.sum()),
    )
