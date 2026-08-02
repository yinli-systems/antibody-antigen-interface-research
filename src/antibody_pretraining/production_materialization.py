"""Shared binary schema for production OAS corpus materialization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import BinaryIO, Iterator

from .tokenizer import CHAIN_TYPE_TO_ID, ID_TO_CHAIN_TYPE


BUCKET_COUNT = 256
MAXIMUM_SEQUENCE_LENGTH = 192
MAP_RECORD = struct.Struct(">32sQBBBBHH192s")
SPLIT_TO_ID = {"validation": 0, "train": 1}
ID_TO_SPLIT = {value: key for key, value in SPLIT_TO_ID.items()}
STRATUM_TO_ID = {
    "human_heavy": 0,
    "human_light": 1,
    "nonhuman_heavy": 2,
    "nonhuman_light": 3,
}
ID_TO_STRATUM = {value: key for key, value in STRATUM_TO_ID.items()}
STATUS_FLAG_BITS = {
    "deletion": 1,
    "insertion": 2,
    "shorter": 4,
    "other_nonfatal": 8,
}


class ProductionMaterializationError(RuntimeError):
    """Report an invalid record or materialization artifact."""


@dataclass(frozen=True)
class ProductionMapRecord:
    """Represent one quality-filtered row before global deduplication."""

    authority_ordinal: int
    chain_type: str
    sequence: str
    sequence_sha256: str
    split: str
    status_flags: tuple[str, ...]
    stratum: str
    study_id: int


def status_flag_mask(flags: tuple[str, ...]) -> int:
    """Encode the bounded ANARCI status categories into one byte."""
    if len(flags) != len(set(flags)):
        raise ProductionMaterializationError("Status flags are duplicated")
    try:
        return sum(STATUS_FLAG_BITS[flag] for flag in flags)
    except KeyError as error:
        raise ProductionMaterializationError("Status flag is unknown") from error


def decode_status_flag_mask(mask: int) -> tuple[str, ...]:
    """Decode a valid bounded ANARCI status bit mask."""
    if isinstance(mask, bool) or not isinstance(mask, int) or not 0 <= mask < 256:
        raise ProductionMaterializationError("Status mask is outside uint8")
    known_mask = sum(STATUS_FLAG_BITS.values())
    if mask & ~known_mask:
        raise ProductionMaterializationError("Status mask has unknown bits")
    return tuple(name for name, bit in STATUS_FLAG_BITS.items() if mask & bit)


def pack_map_record(record: ProductionMapRecord) -> bytes:
    """Pack one validated fixed-width big-endian map record."""
    try:
        digest = bytes.fromhex(record.sequence_sha256)
        sequence = record.sequence.encode("ascii")
        split_id = SPLIT_TO_ID[record.split]
        chain_id = CHAIN_TYPE_TO_ID[record.chain_type]
        stratum_id = STRATUM_TO_ID[record.stratum]
    except (KeyError, UnicodeEncodeError, ValueError) as error:
        raise ProductionMaterializationError("Map record has invalid text fields") from error
    if len(digest) != 32 or hashlib.sha256(sequence).digest() != digest:
        raise ProductionMaterializationError("Sequence digest differs")
    if chain_id not in {1, 2, 3}:
        raise ProductionMaterializationError("Map record chain type is invalid")
    if not 80 <= len(sequence) <= MAXIMUM_SEQUENCE_LENGTH:
        raise ProductionMaterializationError("Map record sequence length differs")
    if (
        isinstance(record.authority_ordinal, bool)
        or not 0 <= record.authority_ordinal < 2**64
        or isinstance(record.study_id, bool)
        or not 0 <= record.study_id < 2**16
    ):
        raise ProductionMaterializationError("Map record integer is outside its domain")
    return MAP_RECORD.pack(
        digest,
        record.authority_ordinal,
        split_id,
        chain_id,
        stratum_id,
        status_flag_mask(record.status_flags),
        record.study_id,
        len(sequence),
        sequence.ljust(MAXIMUM_SEQUENCE_LENGTH, b"\0"),
    )


def unpack_map_record(payload: bytes) -> ProductionMapRecord:
    """Unpack and revalidate one fixed-width map record."""
    if len(payload) != MAP_RECORD.size:
        raise ProductionMaterializationError("Map record byte count differs")
    (
        digest,
        authority_ordinal,
        split_id,
        chain_id,
        stratum_id,
        flags,
        study_id,
        sequence_length,
        padded_sequence,
    ) = MAP_RECORD.unpack(payload)
    try:
        split = ID_TO_SPLIT[split_id]
        chain_type = ID_TO_CHAIN_TYPE[chain_id]
        stratum = ID_TO_STRATUM[stratum_id]
    except KeyError as error:
        raise ProductionMaterializationError("Map record enum is invalid") from error
    if (
        not 80 <= sequence_length <= MAXIMUM_SEQUENCE_LENGTH
        or any(padded_sequence[sequence_length:])
    ):
        raise ProductionMaterializationError("Map record padding differs")
    try:
        sequence = padded_sequence[:sequence_length].decode("ascii")
    except UnicodeDecodeError as error:
        raise ProductionMaterializationError("Map record sequence is not ASCII") from error
    record = ProductionMapRecord(
        authority_ordinal=authority_ordinal,
        chain_type=chain_type,
        sequence=sequence,
        sequence_sha256=digest.hex(),
        split=split,
        status_flags=decode_status_flag_mask(flags),
        stratum=stratum,
        study_id=study_id,
    )
    if pack_map_record(record) != payload:
        raise ProductionMaterializationError("Map record is not canonical")
    return record


def iter_map_records(handle: BinaryIO) -> Iterator[ProductionMapRecord]:
    """Yield complete records and reject a truncated terminal record."""
    while True:
        payload = handle.read(MAP_RECORD.size)
        if not payload:
            return
        if len(payload) != MAP_RECORD.size:
            raise ProductionMaterializationError("Map bucket ends with a partial record")
        yield unpack_map_record(payload)
