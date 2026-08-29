"""Binary schemas for an authenticated OAS fixed-IMGT search track.

Representable rows retain the complete 200-position KA-Search alignment and
the production-corpus identity metadata needed to separate train from
validation targets.  Rows that cannot be represented are never discarded:
their complete residue sequence is placed in a second fixed-width stream for
the independent gap-aware search track.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

from .imgt_position import (
    CANONICAL_LENGTH,
    GAP,
    MISSING_END,
    CanonicalAlignment,
    IMGTPositionError,
    parse_oas_anarci_numbering,
)
from .production_materialization import (
    ID_TO_SPLIT,
    MAXIMUM_SEQUENCE_LENGTH,
    ProductionMapRecord,
    SPLIT_TO_ID,
)
from .tokenizer import CHAIN_TYPE_TO_ID, ID_TO_CHAIN_TYPE


IMGT_TARGET_RECORD = struct.Struct(">32sQBBB200s")
UNUSUAL_TARGET_RECORD = struct.Struct(">32sQBBBH192s")
UNUSUAL_REASON_TO_ID = {
    "malformed_numbering": 0,
    "outside_canonical_positions": 1,
    "sequence_numbering_mismatch": 2,
}
ID_TO_UNUSUAL_REASON = {
    value: key for key, value in UNUSUAL_REASON_TO_ID.items()
}


class OASIMGTExportError(RuntimeError):
    """Report a non-canonical fixed-IMGT export record."""


@dataclass(frozen=True)
class IMGTTargetRecord:
    """One production OAS row represented on all 200 canonical positions."""

    alignment: CanonicalAlignment
    authority_ordinal: int
    chain_type: str
    sequence_sha256: str
    split: str


@dataclass(frozen=True)
class UnusualTargetRecord:
    """One full sequence handed from fixed IMGT to the gap-aware track."""

    authority_ordinal: int
    chain_type: str
    reason: str
    sequence: str
    sequence_sha256: str
    split: str


@dataclass(frozen=True)
class ClassifiedTarget:
    """Exactly one of the two exhaustive target-domain classifications."""

    representable: IMGTTargetRecord | None
    unusual: UnusualTargetRecord | None

    def validate(self) -> None:
        if (self.representable is None) == (self.unusual is None):
            raise OASIMGTExportError("Target classification is not exclusive")


def _digest_bytes(value: str) -> bytes:
    try:
        digest = bytes.fromhex(value)
    except (TypeError, ValueError) as error:
        raise OASIMGTExportError("Sequence digest is invalid") from error
    if len(digest) != 32:
        raise OASIMGTExportError("Sequence digest is not SHA-256")
    return digest


def _record_enums(split: str, chain_type: str) -> tuple[int, int]:
    try:
        split_id = SPLIT_TO_ID[split]
        chain_id = CHAIN_TYPE_TO_ID[chain_type]
    except KeyError as error:
        raise OASIMGTExportError("Target record enum is invalid") from error
    if chain_id not in {1, 2, 3}:
        raise OASIMGTExportError("Target chain type is invalid")
    return split_id, chain_id


def _validate_ordinal(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
        raise OASIMGTExportError("Target authority ordinal is outside uint64")


def sequence_from_alignment(alignment: CanonicalAlignment) -> str:
    """Recover the admitted variable-domain sequence from fixed positions."""
    alignment.validate()
    try:
        return bytes(
            value for value in alignment.encoded if value not in {GAP, MISSING_END}
        ).decode("ascii")
    except UnicodeDecodeError as error:
        raise OASIMGTExportError("Canonical alignment is not ASCII") from error


def classify_oas_target(
    numbering: object,
    production_record: ProductionMapRecord,
) -> ClassifiedTarget:
    """Classify one admitted OAS row without losing unusual insertions."""
    if not isinstance(numbering, str) or not numbering.strip():
        reason = "malformed_numbering"
    else:
        try:
            alignment = parse_oas_anarci_numbering(numbering)
        except IMGTPositionError as error:
            reason = (
                "outside_canonical_positions"
                if "outside the canonical 200" in str(error)
                else "malformed_numbering"
            )
        else:
            if sequence_from_alignment(alignment) == production_record.sequence:
                result = ClassifiedTarget(
                    representable=IMGTTargetRecord(
                        alignment=alignment,
                        authority_ordinal=production_record.authority_ordinal,
                        chain_type=production_record.chain_type,
                        sequence_sha256=production_record.sequence_sha256,
                        split=production_record.split,
                    ),
                    unusual=None,
                )
                result.validate()
                return result
            reason = "sequence_numbering_mismatch"
    result = ClassifiedTarget(
        representable=None,
        unusual=UnusualTargetRecord(
            authority_ordinal=production_record.authority_ordinal,
            chain_type=production_record.chain_type,
            reason=reason,
            sequence=production_record.sequence,
            sequence_sha256=production_record.sequence_sha256,
            split=production_record.split,
        ),
    )
    result.validate()
    return result


def pack_imgt_target(record: IMGTTargetRecord) -> bytes:
    """Pack and fully validate one representable fixed-width target."""
    _validate_ordinal(record.authority_ordinal)
    split_id, chain_id = _record_enums(record.split, record.chain_type)
    record.alignment.validate()
    sequence = sequence_from_alignment(record.alignment).encode("ascii")
    digest = _digest_bytes(record.sequence_sha256)
    if hashlib.sha256(sequence).digest() != digest:
        raise OASIMGTExportError("IMGT alignment sequence digest differs")
    return IMGT_TARGET_RECORD.pack(
        digest,
        record.authority_ordinal,
        split_id,
        chain_id,
        record.alignment.residue_count,
        record.alignment.encoded,
    )


def unpack_imgt_target(payload: bytes) -> IMGTTargetRecord:
    """Unpack one target and reject non-canonical bytes."""
    if len(payload) != IMGT_TARGET_RECORD.size:
        raise OASIMGTExportError("IMGT target byte count differs")
    digest, ordinal, split_id, chain_id, residue_count, encoded = (
        IMGT_TARGET_RECORD.unpack(payload)
    )
    try:
        split = ID_TO_SPLIT[split_id]
        chain_type = ID_TO_CHAIN_TYPE[chain_id]
    except KeyError as error:
        raise OASIMGTExportError("IMGT target enum is invalid") from error
    alignment = CanonicalAlignment(encoded=encoded, residue_count=residue_count)
    record = IMGTTargetRecord(
        alignment=alignment,
        authority_ordinal=ordinal,
        chain_type=chain_type,
        sequence_sha256=digest.hex(),
        split=split,
    )
    if pack_imgt_target(record) != payload:
        raise OASIMGTExportError("IMGT target is not canonical")
    return record


def pack_unusual_target(record: UnusualTargetRecord) -> bytes:
    """Pack one complete gap-aware handoff sequence."""
    _validate_ordinal(record.authority_ordinal)
    split_id, chain_id = _record_enums(record.split, record.chain_type)
    try:
        reason_id = UNUSUAL_REASON_TO_ID[record.reason]
        sequence = record.sequence.encode("ascii")
    except (KeyError, UnicodeEncodeError) as error:
        raise OASIMGTExportError("Unusual target text is invalid") from error
    digest = _digest_bytes(record.sequence_sha256)
    if (
        not 80 <= len(sequence) <= MAXIMUM_SEQUENCE_LENGTH
        or hashlib.sha256(sequence).digest() != digest
    ):
        raise OASIMGTExportError("Unusual target sequence or digest differs")
    return UNUSUAL_TARGET_RECORD.pack(
        digest,
        record.authority_ordinal,
        split_id,
        chain_id,
        reason_id,
        len(sequence),
        sequence.ljust(MAXIMUM_SEQUENCE_LENGTH, b"\0"),
    )


def unpack_unusual_target(payload: bytes) -> UnusualTargetRecord:
    """Unpack one handoff sequence and reject padding or digest changes."""
    if len(payload) != UNUSUAL_TARGET_RECORD.size:
        raise OASIMGTExportError("Unusual target byte count differs")
    digest, ordinal, split_id, chain_id, reason_id, length, padded = (
        UNUSUAL_TARGET_RECORD.unpack(payload)
    )
    try:
        split = ID_TO_SPLIT[split_id]
        chain_type = ID_TO_CHAIN_TYPE[chain_id]
        reason = ID_TO_UNUSUAL_REASON[reason_id]
        sequence = padded[:length].decode("ascii")
    except (KeyError, UnicodeDecodeError) as error:
        raise OASIMGTExportError("Unusual target enum or sequence is invalid") from error
    if any(padded[length:]):
        raise OASIMGTExportError("Unusual target padding differs")
    record = UnusualTargetRecord(
        authority_ordinal=ordinal,
        chain_type=chain_type,
        reason=reason,
        sequence=sequence,
        sequence_sha256=digest.hex(),
        split=split,
    )
    if pack_unusual_target(record) != payload:
        raise OASIMGTExportError("Unusual target is not canonical")
    return record


if CANONICAL_LENGTH != 200 or IMGT_TARGET_RECORD.size != 243:
    raise RuntimeError("Fixed IMGT target schema differs from the frozen width")


__all__ = [
    "ClassifiedTarget",
    "ID_TO_UNUSUAL_REASON",
    "IMGTTargetRecord",
    "IMGT_TARGET_RECORD",
    "OASIMGTExportError",
    "UNUSUAL_REASON_TO_ID",
    "UNUSUAL_TARGET_RECORD",
    "UnusualTargetRecord",
    "classify_oas_target",
    "pack_imgt_target",
    "pack_unusual_target",
    "sequence_from_alignment",
    "unpack_imgt_target",
    "unpack_unusual_target",
]
