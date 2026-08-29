"""Paired VH/VL encoding and deterministic CDR-aware MLM primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


SPECIAL_TOKENS = ("[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]")
STANDARD_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
RESIDUES = STANDARD_RESIDUES + "X"
VOCABULARY = SPECIAL_TOKENS + tuple(RESIDUES)
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCABULARY)}
PAD_ID = TOKEN_TO_ID["[PAD]"]
MASK_ID = TOKEN_TO_ID["[MASK]"]
CLS_ID = TOKEN_TO_ID["[CLS]"]
SEP_ID = TOKEN_TO_ID["[SEP]"]
FIRST_RESIDUE_ID = TOKEN_TO_ID["A"]
STANDARD_RESIDUE_COUNT = len(STANDARD_RESIDUES)
PAIRED_HEAVY_TYPE_ID = 5
PAIRED_LIGHT_TYPE_ID = 6

REGION_SPECIAL = 0
REGION_FRAMEWORK = 1
REGION_CDR1 = 2
REGION_CDR2 = 3
REGION_CDR3 = 4

UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
POSITION_KEY = np.uint64(0xD2B74407B1CE6E93)
STEP_KEY = np.uint64(0x9E3779B97F4A7C15)
STREAM_KEY = np.uint64(0x94D049BB133111EB)


class PairedPretrainingError(ValueError):
    """Report a malformed pair, region map, or masking request."""


@dataclass(frozen=True)
class EncodedPair:
    input_ids: np.ndarray
    type_ids: np.ndarray
    region_ids: np.ndarray
    heavy_length: int
    light_length: int


@dataclass(frozen=True)
class MaskedPairBatch:
    input_ids: np.ndarray
    labels: np.ndarray
    selected_target_count: int
    selected_by_region: dict[int, int]


def _canonical_sequence(value: str, label: str) -> str:
    sequence = value.strip().upper()
    if not sequence or not set(sequence).issubset(RESIDUES):
        raise PairedPretrainingError(f"{label} is empty or noncanonical")
    return sequence


def locate_cdrs(sequence: str, cdrs: tuple[str, str, str]) -> np.ndarray:
    """Map ordered exact CDR strings to residue-level IMGT-region IDs."""
    sequence = _canonical_sequence(sequence, "sequence")
    region = np.full(len(sequence), REGION_FRAMEWORK, dtype=np.uint8)
    cursor = 0
    for region_id, raw_cdr in zip(
        (REGION_CDR1, REGION_CDR2, REGION_CDR3), cdrs, strict=True
    ):
        cdr = _canonical_sequence(raw_cdr, f"CDR{region_id - 1}")
        start = sequence.find(cdr, cursor)
        if start < 0:
            raise PairedPretrainingError("Ordered CDR is not an exact sequence substring")
        end = start + len(cdr)
        region[start:end] = region_id
        cursor = end
    return region


def encode_pair(
    heavy_sequence: str,
    light_sequence: str,
    heavy_cdrs: tuple[str, str, str],
    light_cdrs: tuple[str, str, str],
    *,
    maximum_tokens: int = 320,
) -> EncodedPair:
    """Encode ``[CLS] VH [SEP] VL [SEP]`` without learned-position changes."""
    heavy = _canonical_sequence(heavy_sequence, "heavy_sequence")
    light = _canonical_sequence(light_sequence, "light_sequence")
    heavy_region = locate_cdrs(heavy, heavy_cdrs)
    light_region = locate_cdrs(light, light_cdrs)
    tokens = np.asarray(
        [CLS_ID]
        + [TOKEN_TO_ID[value] for value in heavy]
        + [SEP_ID]
        + [TOKEN_TO_ID[value] for value in light]
        + [SEP_ID],
        dtype=np.uint8,
    )
    if len(tokens) > maximum_tokens:
        raise PairedPretrainingError("Paired sequence exceeds maximum_tokens")
    type_ids = np.asarray(
        [PAIRED_HEAVY_TYPE_ID] * (len(heavy) + 2)
        + [PAIRED_LIGHT_TYPE_ID] * (len(light) + 1),
        dtype=np.uint8,
    )
    region_ids = np.concatenate(
        (
            np.asarray([REGION_SPECIAL], dtype=np.uint8),
            heavy_region,
            np.asarray([REGION_SPECIAL], dtype=np.uint8),
            light_region,
            np.asarray([REGION_SPECIAL], dtype=np.uint8),
        )
    )
    return EncodedPair(
        input_ids=tokens,
        type_ids=type_ids,
        region_ids=region_ids,
        heavy_length=len(heavy),
        light_length=len(light),
    )


def _splitmix64(value: np.ndarray) -> np.ndarray:
    value = (value + STEP_KEY) & UINT64_MASK
    value = ((value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & UINT64_MASK
    value = ((value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & UINT64_MASK
    return value ^ (value >> np.uint64(31))


def apply_cdr3_balanced_mlm(
    input_ids: np.ndarray,
    region_ids: np.ndarray,
    sample_keys: np.ndarray,
    *,
    update: int,
    seed: int,
    overall_fraction: float = 0.15,
    cdr3_fraction: float = 0.25,
) -> MaskedPairBatch:
    """Use 25% CDR3 masking while preserving 15% expected masks per pair."""
    tokens = np.asarray(input_ids)
    regions = np.asarray(region_ids)
    keys = np.asarray(sample_keys, dtype=np.uint64)
    if (
        tokens.ndim != 2
        or regions.shape != tokens.shape
        or keys.ndim != 1
        or len(keys) != len(tokens)
    ):
        raise PairedPretrainingError("Paired masking inputs have incompatible shapes")
    if not 0.0 < overall_fraction < 1.0 or not 0.0 < cdr3_fraction < 1.0:
        raise PairedPretrainingError("Mask fractions must be in (0, 1)")
    if cdr3_fraction < overall_fraction:
        raise PairedPretrainingError("CDR3 mask fraction must not be below overall fraction")
    if update < 0 or seed < 0:
        raise PairedPretrainingError("update and seed must be non-negative")

    eligible = tokens >= FIRST_RESIDUE_ID
    cdr3 = eligible & regions.astype(np.int64, copy=False).__eq__(REGION_CDR3)
    background = eligible & ~cdr3
    eligible_count = eligible.sum(axis=1)
    cdr3_count = cdr3.sum(axis=1)
    background_count = background.sum(axis=1)
    if np.any(eligible_count == 0) or np.any(cdr3_count == 0) or np.any(background_count == 0):
        raise PairedPretrainingError("Every pair must contain CDR3 and background residues")
    background_fraction = (
        overall_fraction * eligible_count - cdr3_fraction * cdr3_count
    ) / background_count
    if np.any(background_fraction <= 0.0) or np.any(background_fraction >= 1.0):
        raise PairedPretrainingError("Balanced background mask probability is invalid")

    probabilities = np.broadcast_to(background_fraction[:, None], tokens.shape).copy()
    probabilities[cdr3] = cdr3_fraction
    probabilities[~eligible] = 0.0
    positions = np.arange(tokens.shape[1], dtype=np.uint64)[None, :]
    update_key = np.uint64((update * int(STEP_KEY)) & int(UINT64_MASK))
    counters = keys[:, None] ^ (positions * POSITION_KEY) ^ update_key ^ np.uint64(seed)
    selection_random = _splitmix64(counters)
    # Use the high 53 bits to obtain an exactly representable uniform float in
    # [0, 1).  This avoids object arrays and keeps masking fully vectorized.
    selection_uniform = (
        (selection_random >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))
    )
    selected = eligible & (selection_uniform < probabilities)

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
    return MaskedPairBatch(
        input_ids=corrupted,
        labels=labels,
        selected_target_count=int(selected.sum()),
        selected_by_region={
            region_id: int((selected & (regions == region_id)).sum())
            for region_id in (REGION_FRAMEWORK, REGION_CDR1, REGION_CDR2, REGION_CDR3)
        },
    )


def focal_mlm_loss_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Return focal MLM loss over selected tokens with FP32 probabilities."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise PairedPretrainingError("Focal MLM shapes differ")
    if not 0.0 <= gamma <= 5.0:
        raise PairedPretrainingError("gamma must be in [0, 5]")
    selected = labels.ne(-100)
    if not bool(selected.any()):
        return logits.sum() * 0.0
    selected_logits = logits.float()[selected]
    selected_labels = labels[selected]
    log_probability = F.log_softmax(selected_logits, dim=-1)
    log_pt = log_probability.gather(1, selected_labels[:, None]).squeeze(1)
    pt = log_pt.exp()
    return (-(1.0 - pt).pow(gamma) * log_pt).sum()
