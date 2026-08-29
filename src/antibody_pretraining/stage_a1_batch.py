"""Deterministic Stage-A1 MLM, retrieval, and sparse-contact batches."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import torch
from torch import Tensor

from .pair_aware_esm2 import ESM2_ALPHABET


TOKEN_INDEX = {token: index for index, token in enumerate(ESM2_ALPHABET)}


def _seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _identity_id(*parts: object) -> int:
    """Return a deterministic signed-int64-safe identity fingerprint."""

    return _seed("identity", *parts) % (2**63 - 1)


def _tokenize(sequence: str) -> list[int]:
    if not sequence or any(character.isspace() for character in sequence):
        raise ValueError("Malformed Stage-A1 protein sequence")
    unknown = TOKEN_INDEX["<unk>"]
    return [TOKEN_INDEX.get(character, unknown) for character in sequence]


def _crop_string_pair(
    left: str, right: str, *, maximum_residues: int, seed: int
) -> tuple[str, str]:
    if len(left) + len(right) <= maximum_residues:
        return left, right
    if min(len(left), len(right)) < 8 or maximum_residues < 16:
        raise ValueError("Stage-A1 STRING endpoint is too short for deterministic crop")
    left_budget = max(8, round(maximum_residues * len(left) / (len(left) + len(right))))
    left_budget = min(left_budget, maximum_residues - 8, len(left))
    right_budget = min(len(right), maximum_residues - left_budget)
    left_budget = min(len(left), maximum_residues - right_budget)

    def crop(sequence: str, length: int, endpoint: str) -> str:
        if len(sequence) == length:
            return sequence
        start = _seed(seed, endpoint, sequence) % (len(sequence) - length + 1)
        return sequence[start : start + length]

    return crop(left, left_budget, "left"), crop(right, right_budget, "right")


def _ranked_positive_pairs(
    contacts: Sequence[Sequence[int]], *, seed: int, limit: int
) -> list[tuple[int, int]]:
    unique = {(int(left), int(right)) for left, right in contacts}
    return sorted(
        unique,
        key=lambda pair: _seed(seed, "positive", pair[0], pair[1]),
    )[:limit]


def _sample_negative_pairs(
    left_length: int,
    right_length: int,
    positives: set[tuple[int, int]],
    *,
    seed: int,
    count: int,
) -> list[tuple[int, int]]:
    available = left_length * right_length - len(positives)
    target = min(count, available)
    if target < 1:
        raise ValueError("Stage-A1 contact matrix has no native noncontact")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed % (2**63 - 1))
    selected: set[tuple[int, int]] = set()
    maximum_attempts = max(100, target * 50)
    for _ in range(maximum_attempts):
        if len(selected) == target:
            break
        pair = (
            int(torch.randint(left_length, (1,), generator=generator)),
            int(torch.randint(right_length, (1,), generator=generator)),
        )
        if pair not in positives:
            selected.add(pair)
    if len(selected) != target:
        if left_length * right_length > 100_000:
            raise RuntimeError("Sparse negative sampler did not converge")
        candidates = [
            (left, right)
            for left in range(left_length)
            for right in range(right_length)
            if (left, right) not in positives
        ]
        selected = set(
            sorted(
                candidates,
                key=lambda pair: _seed(seed, "fallback", pair[0], pair[1]),
            )[:target]
        )
    return sorted(selected, key=lambda pair: _seed(seed, "publish", *pair))


@dataclass(frozen=True)
class StageA1Microbatch:
    tokens: Tensor
    chain_ids: Tensor
    labels: Tensor
    retrieval_batch_indices: Tensor
    retrieval_allowed: Tensor
    retrieval_left_exact_ids: Tensor
    retrieval_right_exact_ids: Tensor
    retrieval_left_cluster_ids: Tensor
    retrieval_right_cluster_ids: Tensor
    retrieval_positive_group_ids: Tensor
    contact_batch_indices: Tensor
    contact_left_token_indices: Tensor
    contact_right_token_indices: Tensor
    contact_labels: Tensor
    contact_distance_bins: Tensor | None
    residue_count: int


def build_stage_a1_retrieval_allowed(
    local_metadata: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    candidate_metadata: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    target_indices: Tensor,
) -> Tensor:
    """Build a fail-closed pair-level false-negative mask for retrieval."""

    if (
        any(value.ndim != 1 for value in (*local_metadata, *candidate_metadata))
        or any(value.dtype != torch.long for value in (*local_metadata, *candidate_metadata))
        or target_indices.ndim != 1
        or target_indices.dtype != torch.long
    ):
        raise ValueError("Stage-A1 retrieval identity metadata is malformed")
    local_count = len(local_metadata[0])
    candidate_count = len(candidate_metadata[0])
    if (
        any(len(value) != local_count for value in local_metadata)
        or any(len(value) != candidate_count for value in candidate_metadata)
        or len(target_indices) != local_count
        or bool((target_indices < 0).any())
        or bool((target_indices >= candidate_count).any())
    ):
        raise ValueError("Stage-A1 retrieval identity metadata is not aligned")
    left_exact, right_exact, left_cluster, right_cluster, positive_group = (
        local_metadata
    )
    (
        candidate_left_exact,
        candidate_right_exact,
        candidate_left_cluster,
        candidate_right_cluster,
        candidate_positive_group,
    ) = candidate_metadata
    exact_overlap = (
        left_exact[:, None].eq(candidate_left_exact[None, :])
        | left_exact[:, None].eq(candidate_right_exact[None, :])
        | right_exact[:, None].eq(candidate_left_exact[None, :])
        | right_exact[:, None].eq(candidate_right_exact[None, :])
    )
    cluster_overlap = (
        left_cluster[:, None].eq(candidate_left_cluster[None, :])
        | left_cluster[:, None].eq(candidate_right_cluster[None, :])
        | right_cluster[:, None].eq(candidate_left_cluster[None, :])
        | right_cluster[:, None].eq(candidate_right_cluster[None, :])
    )
    allowed = ~(
        exact_overlap
        | cluster_overlap
        | positive_group[:, None].eq(candidate_positive_group[None, :])
    )
    rows = torch.arange(local_count, device=allowed.device)
    allowed[rows, target_indices] = True
    if bool(allowed.sum(dim=1).lt(2).any()):
        raise ValueError("Stage-A1 false-negative exclusions removed every negative")
    return allowed


def build_stage_a1_microbatch(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    maximum_tokens: int = 1024,
    padding_multiple: int | None = None,
    maximum_positive_pairs_per_complex: int = 102,
    maximum_sampled_pairs_per_complex: int = 512,
    maximum_negative_to_positive_ratio: int = 4,
    contact_distance_bin_count: int | None = None,
) -> StageA1Microbatch:
    if len(records) < 2 or maximum_tokens < 20:
        raise ValueError("Stage-A1 microbatch shape is invalid")
    if padding_multiple is not None and (
        isinstance(padding_multiple, bool)
        or not isinstance(padding_multiple, int)
        or padding_multiple < 1
        or padding_multiple > maximum_tokens
    ):
        raise ValueError("Stage-A1 padding multiple is invalid")

    prepared_records: list[tuple[dict[str, Any], str, str]] = []
    maximum_observed_tokens = 0
    for record in records:
        left = str(record["sequence1"])
        right = str(record["sequence2"])
        contact_supervision = bool(record["contact_supervision"])
        if contact_supervision:
            if len(left) + len(right) + 4 > maximum_tokens:
                raise ValueError("Structural Stage-A1 patch exceeds the token budget")
        else:
            left, right = _crop_string_pair(
                left,
                right,
                maximum_residues=maximum_tokens - 4,
                seed=_seed(seed, record["example_id"], "crop"),
            )
        maximum_observed_tokens = max(
            maximum_observed_tokens, len(left) + len(right) + 4
        )
        prepared_records.append((record, left, right))

    padded_tokens = maximum_tokens
    if padding_multiple is not None:
        padded_tokens = min(
            maximum_tokens,
            math.ceil(maximum_observed_tokens / padding_multiple) * padding_multiple,
        )
        if padded_tokens < maximum_observed_tokens:
            raise ValueError("Stage-A1 dynamic padding exceeds the token budget")
    batch_size = len(records)
    tokens = torch.full(
        (batch_size, padded_tokens), TOKEN_INDEX["<pad>"], dtype=torch.long
    )
    chain_ids = torch.full((batch_size, padded_tokens), -1, dtype=torch.long)
    labels = torch.full((batch_size, padded_tokens), -100, dtype=torch.long)
    retrieval_indices: list[int] = []
    contact_batch: list[int] = []
    contact_left: list[int] = []
    contact_right: list[int] = []
    contact_labels: list[int] = []
    contact_distance_bins: list[int] = []
    retrieval_records: list[dict[str, Any]] = []
    residue_count = 0

    for batch_index, (record, left, right) in enumerate(prepared_records):
        contact_supervision = bool(record["contact_supervision"])
        first = [TOKEN_INDEX["<cls>"], *_tokenize(left), TOKEN_INDEX["<eos>"]]
        second = [TOKEN_INDEX["<cls>"], *_tokenize(right), TOKEN_INDEX["<eos>"]]
        combined = first + second
        tokens[batch_index, : len(combined)] = torch.tensor(combined)
        chain_ids[batch_index, : len(first)] = 0
        chain_ids[batch_index, len(first) : len(combined)] = 1
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_seed(seed, record["example_id"], "mask") % (2**63 - 1))
        for start, end in ((1, len(first) - 1), (len(first) + 1, len(combined) - 1)):
            count = max(1, int(math.ceil((end - start) * 0.15)))
            selected = torch.randperm(end - start, generator=generator)[:count] + start
            labels[batch_index, selected] = tokens[batch_index, selected]
            replacement = torch.rand(count, generator=generator)
            tokens[batch_index, selected[replacement < 0.8]] = TOKEN_INDEX["<mask>"]
            random_selected = selected[(replacement >= 0.8) & (replacement < 0.9)]
            if len(random_selected):
                tokens[batch_index, random_selected] = torch.randint(
                    TOKEN_INDEX["L"],
                    TOKEN_INDEX["C"] + 1,
                    (len(random_selected),),
                    generator=generator,
                )
        residue_count += len(left) + len(right)

        if bool(record["retrieval_supervision"]):
            retrieval_indices.append(batch_index)
            retrieval_records.append(record)
        if contact_supervision:
            distance_bins = None
            if contact_distance_bin_count is not None:
                shape = record.get("distance_bin_shape")
                encoded = record.get("distance_bin_u8_base64")
                if (
                    not isinstance(shape, list)
                    or shape != [len(left), len(right)]
                    or not isinstance(encoded, str)
                    or not 0 < contact_distance_bin_count < 256
                ):
                    raise ValueError("Stage-A1 distance-bin target is malformed")
                decoded = bytearray(base64.b64decode(encoded, validate=True))
                if len(decoded) != len(left) * len(right):
                    raise ValueError("Stage-A1 distance-bin matrix size differs")
                distance_bins = torch.frombuffer(decoded, dtype=torch.uint8).reshape(
                    len(left), len(right)
                )
                source_positives = [
                    (int(left_index), int(right_index))
                    for left_index, right_index in zip(
                        *torch.nonzero(
                            distance_bins < contact_distance_bin_count,
                            as_tuple=True,
                        )
                    )
                ]
            else:
                source_positives = record["contact_pairs"]
            positives = _ranked_positive_pairs(
                source_positives,
                seed=_seed(seed, record["example_id"], "contact"),
                limit=maximum_positive_pairs_per_complex,
            )
            if not positives or any(
                left_index < 0
                or right_index < 0
                or left_index >= len(left)
                or right_index >= len(right)
                for left_index, right_index in positives
            ):
                raise ValueError("Stage-A1 contact positive index is invalid")
            negative_limit = min(
                len(positives) * maximum_negative_to_positive_ratio,
                maximum_sampled_pairs_per_complex - len(positives),
            )
            negatives = _sample_negative_pairs(
                len(left),
                len(right),
                set(positives),
                seed=_seed(seed, record["example_id"], "noncontact"),
                count=negative_limit,
            )
            for (left_index, right_index), label in [
                *((pair, 1) for pair in positives),
                *((pair, 0) for pair in negatives),
            ]:
                contact_batch.append(batch_index)
                contact_left.append(1 + left_index)
                contact_right.append(len(first) + 1 + right_index)
                contact_labels.append(label)
                if distance_bins is not None:
                    contact_distance_bins.append(
                        int(distance_bins[left_index, right_index])
                    )

    if len(retrieval_indices) < 2:
        raise ValueError("Stage-A1 microbatch requires at least two retrieval pairs")
    retrieval_left_exact_ids = torch.tensor(
        [
            _identity_id(
                "exact",
                record.get("endpoint_exact1_sha256", record["sequence1"]),
            )
            for record in retrieval_records
        ],
        dtype=torch.long,
    )
    retrieval_right_exact_ids = torch.tensor(
        [
            _identity_id(
                "exact",
                record.get("endpoint_exact2_sha256", record["sequence2"]),
            )
            for record in retrieval_records
        ],
        dtype=torch.long,
    )
    retrieval_left_cluster_ids = torch.tensor(
        [
            _identity_id(
                "cluster",
                record.get(
                    "endpoint_cluster1",
                    record.get("endpoint_exact1_sha256", record["sequence1"]),
                ),
            )
            for record in retrieval_records
        ],
        dtype=torch.long,
    )
    retrieval_right_cluster_ids = torch.tensor(
        [
            _identity_id(
                "cluster",
                record.get(
                    "endpoint_cluster2",
                    record.get("endpoint_exact2_sha256", record["sequence2"]),
                ),
            )
            for record in retrieval_records
        ],
        dtype=torch.long,
    )
    retrieval_positive_group_ids = torch.tensor(
        [
            _identity_id(
                "positive_group",
                record.get("source", "unknown"),
                record["known_positive_group"],
            )
            for record in retrieval_records
        ],
        dtype=torch.long,
    )
    retrieval_metadata = (
        retrieval_left_exact_ids,
        retrieval_right_exact_ids,
        retrieval_left_cluster_ids,
        retrieval_right_cluster_ids,
        retrieval_positive_group_ids,
    )
    retrieval_allowed = build_stage_a1_retrieval_allowed(
        retrieval_metadata,
        retrieval_metadata,
        torch.arange(len(retrieval_records), dtype=torch.long),
    )
    if not contact_labels or 0 not in contact_labels or 1 not in contact_labels:
        raise ValueError("Stage-A1 microbatch requires both contact classes")
    if contact_distance_bin_count is not None and len(contact_distance_bins) != len(
        contact_labels
    ):
        raise RuntimeError("Stage-A1 distance-bin labels are not aligned")
    return StageA1Microbatch(
        tokens=tokens,
        chain_ids=chain_ids,
        labels=labels,
        retrieval_batch_indices=torch.tensor(retrieval_indices, dtype=torch.long),
        retrieval_allowed=retrieval_allowed,
        retrieval_left_exact_ids=retrieval_left_exact_ids,
        retrieval_right_exact_ids=retrieval_right_exact_ids,
        retrieval_left_cluster_ids=retrieval_left_cluster_ids,
        retrieval_right_cluster_ids=retrieval_right_cluster_ids,
        retrieval_positive_group_ids=retrieval_positive_group_ids,
        contact_batch_indices=torch.tensor(contact_batch, dtype=torch.long),
        contact_left_token_indices=torch.tensor(contact_left, dtype=torch.long),
        contact_right_token_indices=torch.tensor(contact_right, dtype=torch.long),
        contact_labels=torch.tensor(contact_labels, dtype=torch.float32),
        contact_distance_bins=(
            torch.tensor(contact_distance_bins, dtype=torch.long)
            if contact_distance_bin_count is not None
            else None
        ),
        residue_count=residue_count,
    )
