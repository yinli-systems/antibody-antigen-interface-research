"""Canonical output schema for exhaustive fixed-IMGT near-identity matches."""

from __future__ import annotations

from dataclasses import dataclass
import struct


FIXED_IMGT_MATCH_RECORD = struct.Struct(">IQBHHHHH")
ID_TO_TARGET_SPLIT = {0: "train", 1: "validation"}
TARGET_SPLIT_TO_ID = {value: key for key, value in ID_TO_TARGET_SPLIT.items()}


class IMGTSearchOutputError(RuntimeError):
    """Report malformed or scientifically ineligible fixed-IMGT output."""


@dataclass(frozen=True)
class FixedIMGTOutputMatch:
    """One exact integer-verified 90/90/90 query-target match."""

    identical_positions: int
    overlapping_positions: int
    query_ordinal: int
    query_positions: int
    target_authority_ordinal: int
    target_positions: int
    target_split: str
    union_positions: int

    def validate(self) -> None:
        integers = (
            self.identical_positions,
            self.overlapping_positions,
            self.query_ordinal,
            self.query_positions,
            self.target_authority_ordinal,
            self.target_positions,
            self.union_positions,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise IMGTSearchOutputError("Fixed-IMGT match field is not an integer")
        if (
            not 0 <= self.query_ordinal < 2**32
            or not 0 <= self.target_authority_ordinal < 2**64
            or self.target_split not in TARGET_SPLIT_TO_ID
            or not 0 < self.query_positions <= 200
            or not 0 < self.target_positions <= 200
            or not 0 < self.union_positions <= 200
            or not 0 <= self.identical_positions <= self.overlapping_positions
            or self.overlapping_positions > min(
                self.query_positions, self.target_positions
            )
            or self.union_positions
            != self.query_positions + self.target_positions - self.overlapping_positions
        ):
            raise IMGTSearchOutputError("Fixed-IMGT match metric accounting differs")
        if (
            10 * self.identical_positions < 9 * self.union_positions
            or 10 * self.overlapping_positions < 9 * self.query_positions
            or 10 * self.overlapping_positions < 9 * self.target_positions
        ):
            raise IMGTSearchOutputError("Fixed-IMGT match does not pass 90/90/90")


def pack_fixed_imgt_match(match: FixedIMGTOutputMatch) -> bytes:
    """Serialize one match after all ordinal and metric invariants pass."""
    match.validate()
    return FIXED_IMGT_MATCH_RECORD.pack(
        match.query_ordinal,
        match.target_authority_ordinal,
        TARGET_SPLIT_TO_ID[match.target_split],
        match.identical_positions,
        match.overlapping_positions,
        match.query_positions,
        match.target_positions,
        match.union_positions,
    )


def unpack_fixed_imgt_match(payload: bytes) -> FixedIMGTOutputMatch:
    """Deserialize one record and reject every non-canonical byte string."""
    if len(payload) != FIXED_IMGT_MATCH_RECORD.size:
        raise IMGTSearchOutputError("Fixed-IMGT match byte count differs")
    (
        query_ordinal,
        target_authority_ordinal,
        split_id,
        identical_positions,
        overlapping_positions,
        query_positions,
        target_positions,
        union_positions,
    ) = FIXED_IMGT_MATCH_RECORD.unpack(payload)
    try:
        target_split = ID_TO_TARGET_SPLIT[split_id]
    except KeyError as error:
        raise IMGTSearchOutputError("Fixed-IMGT match split id is invalid") from error
    match = FixedIMGTOutputMatch(
        identical_positions=identical_positions,
        overlapping_positions=overlapping_positions,
        query_ordinal=query_ordinal,
        query_positions=query_positions,
        target_authority_ordinal=target_authority_ordinal,
        target_positions=target_positions,
        target_split=target_split,
        union_positions=union_positions,
    )
    if pack_fixed_imgt_match(match) != payload:
        raise IMGTSearchOutputError("Fixed-IMGT match is not canonical")
    return match


if FIXED_IMGT_MATCH_RECORD.size != 23:
    raise RuntimeError("Fixed-IMGT match schema differs from the frozen width")


__all__ = [
    "FIXED_IMGT_MATCH_RECORD",
    "FixedIMGTOutputMatch",
    "ID_TO_TARGET_SPLIT",
    "IMGTSearchOutputError",
    "TARGET_SPLIT_TO_ID",
    "pack_fixed_imgt_match",
    "unpack_fixed_imgt_match",
]
