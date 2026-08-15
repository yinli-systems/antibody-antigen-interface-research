"""Test the lossless fixed-IMGT 90/90/90 candidate theorem and verifier."""

from __future__ import annotations

import hashlib

from antibody_pretraining.imgt_exact_search import (
    MINIMUM_QUALIFYING_EXACT_SEEDS,
    SEED_GROUP_COUNT,
    SEED_POSITION_GROUPS,
    assert_lossless_candidate,
    build_query_index,
    candidate_query_ordinals,
    verify_target_candidates,
)
from antibody_pretraining.imgt_position import (
    CANONICAL_LENGTH,
    GAP,
    MISSING_END,
    CanonicalAlignment,
)
from antibody_pretraining.imgt_query_export import IMGTQueryRecord
from antibody_pretraining.oas_imgt_export import IMGTTargetRecord, sequence_from_alignment


def _alignment(values: bytes) -> CanonicalAlignment:
    return CanonicalAlignment(
        encoded=values,
        residue_count=sum(value not in {GAP, MISSING_END} for value in values),
    )


def _query(ordinal: int, alignment: CanonicalAlignment) -> IMGTQueryRecord:
    sequence = sequence_from_alignment(alignment)
    return IMGTQueryRecord(
        alignment=alignment,
        chain_type="heavy",
        domain_sequence_sha256=hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        query_ordinal=ordinal,
        role="train",
    )


def _target(ordinal: int, alignment: CanonicalAlignment) -> IMGTTargetRecord:
    sequence = sequence_from_alignment(alignment)
    return IMGTTargetRecord(
        alignment=alignment,
        authority_ordinal=ordinal,
        chain_type="heavy",
        sequence_sha256=hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        split="train",
    )


def test_seed_partition_is_25_disjoint_uint64_groups_over_200_positions() -> None:
    flattened = [position for group in SEED_POSITION_GROUPS for position in group]
    assert SEED_GROUP_COUNT == 25
    assert MINIMUM_QUALIFYING_EXACT_SEEDS == 5
    assert {len(group) for group in SEED_POSITION_GROUPS} == {8}
    assert len(flattened) == CANONICAL_LENGTH
    assert sorted(flattened) == list(range(CANONICAL_LENGTH))


def test_twenty_disagreements_at_identity_boundary_are_always_recalled() -> None:
    query_alignment = _alignment(b"A" * CANONICAL_LENGTH)
    target_values = bytearray(query_alignment.encoded)
    for group in SEED_POSITION_GROUPS[:20]:
        target_values[group[0]] = ord("C")
    query = _query(3, query_alignment)
    target = _target(7, _alignment(bytes(target_values)))
    index = build_query_index([query])

    assert candidate_query_ordinals(target, index) == (3,)
    assert_lossless_candidate(query, target, index)
    matches = verify_target_candidates(target, index)
    assert len(matches) == 1
    assert matches[0].metrics.identical_positions == 180
    assert matches[0].metrics.identity == 0.9


def test_one_disagreement_per_seed_can_miss_only_below_identity_gate() -> None:
    query_alignment = _alignment(b"A" * CANONICAL_LENGTH)
    target_values = bytearray(query_alignment.encoded)
    for group in SEED_POSITION_GROUPS[:21]:
        target_values[group[0]] = ord("C")
    query = _query(0, query_alignment)
    target = _target(1, _alignment(bytes(target_values)))
    index = build_query_index([query])

    assert candidate_query_ordinals(target, index) == ()
    assert verify_target_candidates(target, index) == ()
    assert_lossless_candidate(query, target, index)


def test_gap_and_missing_end_share_the_metric_absent_state() -> None:
    query_values = bytes([MISSING_END] * 20) + b"A" * 180
    target_values = bytes([GAP] * 20) + b"A" * 180
    query = _query(11, _alignment(query_values))
    target = _target(13, _alignment(target_values))
    index = build_query_index([query])

    assert candidate_query_ordinals(target, index) == (11,)
    match = verify_target_candidates(target, index)[0]
    assert match.metrics.identity == 1.0
    assert match.metrics.query_coverage == 1.0
    assert match.metrics.target_coverage == 1.0


def test_reciprocal_coverage_is_verified_after_lossless_candidate_recall() -> None:
    query = _query(2, _alignment(b"A" * CANONICAL_LENGTH))
    target_values = bytearray(b"A" * CANONICAL_LENGTH)
    for position in range(25):
        target_values[position] = GAP
    target = _target(5, _alignment(bytes(target_values)))
    index = build_query_index([query])

    assert candidate_query_ordinals(target, index) == (2,)
    assert verify_target_candidates(target, index) == ()


def test_seed_false_positive_is_removed_by_full_position_verifier() -> None:
    query = _query(17, _alignment(b"A" * CANONICAL_LENGTH))
    target_values = bytearray(b"C" * CANONICAL_LENGTH)
    for group in SEED_POSITION_GROUPS[:MINIMUM_QUALIFYING_EXACT_SEEDS]:
        for position in group:
            target_values[position] = ord("A")
    target = _target(19, _alignment(bytes(target_values)))
    index = build_query_index([query])

    assert candidate_query_ordinals(target, index) == (17,)
    assert verify_target_candidates(target, index) == ()
