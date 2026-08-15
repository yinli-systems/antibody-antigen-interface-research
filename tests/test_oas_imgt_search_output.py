"""Test the canonical exhaustive fixed-IMGT match output schema."""

from __future__ import annotations

import pytest

from antibody_pretraining.imgt_search_output import (
    FIXED_IMGT_MATCH_RECORD,
    FixedIMGTOutputMatch,
    IMGTSearchOutputError,
    pack_fixed_imgt_match,
    unpack_fixed_imgt_match,
)


def _match(**updates: object) -> FixedIMGTOutputMatch:
    values: dict[str, object] = {
        "identical_positions": 180,
        "overlapping_positions": 180,
        "query_ordinal": 3,
        "query_positions": 180,
        "target_authority_ordinal": 17,
        "target_positions": 180,
        "target_split": "train",
        "union_positions": 180,
    }
    values.update(updates)
    return FixedIMGTOutputMatch(**values)  # type: ignore[arg-type]


def test_fixed_imgt_match_round_trips_at_inclusive_identity_boundary() -> None:
    match = _match(
        identical_positions=180,
        overlapping_positions=200,
        query_positions=200,
        target_positions=200,
        union_positions=200,
    )
    payload = pack_fixed_imgt_match(match)
    assert len(payload) == FIXED_IMGT_MATCH_RECORD.size == 23
    assert unpack_fixed_imgt_match(payload) == match


@pytest.mark.parametrize(
    "updates, message",
    [
        (
            {
                "identical_positions": 179,
                "overlapping_positions": 200,
                "query_positions": 200,
                "target_positions": 200,
                "union_positions": 200,
            },
            "90/90/90",
        ),
        (
            {
                "identical_positions": 179,
                "overlapping_positions": 179,
                "query_positions": 200,
                "target_positions": 179,
                "union_positions": 200,
            },
            "90/90/90",
        ),
        ({"target_split": "test"}, "metric accounting"),
        ({"union_positions": 181}, "metric accounting"),
        ({"query_ordinal": 2**32}, "metric accounting"),
    ],
)
def test_fixed_imgt_match_rejects_invalid_metrics_and_metadata(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(IMGTSearchOutputError, match=message):
        pack_fixed_imgt_match(_match(**updates))


def test_fixed_imgt_match_rejects_noncanonical_split_byte() -> None:
    payload = bytearray(pack_fixed_imgt_match(_match()))
    payload[12] = 7
    with pytest.raises(IMGTSearchOutputError, match="split id"):
        unpack_fixed_imgt_match(bytes(payload))
