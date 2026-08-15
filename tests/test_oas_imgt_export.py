"""Test fixed IMGT target records and explicit gap-aware handoff."""

from __future__ import annotations

import hashlib

import pytest

from antibody_pretraining.oas_imgt_export import (
    IMGT_TARGET_RECORD,
    UNUSUAL_TARGET_RECORD,
    OASIMGTExportError,
    classify_oas_target,
    pack_imgt_target,
    pack_unusual_target,
    unpack_imgt_target,
    unpack_unusual_target,
)
from antibody_pretraining.production_materialization import ProductionMapRecord


SEQUENCE = "A" * 100


def _numbering(sequence: str) -> str:
    return "{" + ", ".join(
        f"'{index} ': '{residue}'"
        for index, residue in enumerate(sequence, start=1)
    ) + "}"


def _record(sequence: str = SEQUENCE) -> ProductionMapRecord:
    return ProductionMapRecord(
        authority_ordinal=17,
        chain_type="heavy",
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        split="train",
        status_flags=(),
        stratum="human_heavy",
        study_id=3,
    )


def test_representable_target_round_trip_is_digest_bound() -> None:
    classified = classify_oas_target(_numbering(SEQUENCE), _record())
    assert classified.unusual is None
    assert classified.representable is not None
    payload = pack_imgt_target(classified.representable)
    assert len(payload) == IMGT_TARGET_RECORD.size == 243
    assert unpack_imgt_target(payload) == classified.representable
    changed = bytearray(payload)
    changed[IMGT_TARGET_RECORD.size - 200] = ord("C")
    with pytest.raises(OASIMGTExportError, match="digest differs"):
        unpack_imgt_target(bytes(changed))


@pytest.mark.parametrize(
    ("numbering", "reason"),
    [
        ("{'111Z': 'A'}", "outside_canonical_positions"),
        ("not a numbering", "malformed_numbering"),
        (_numbering("C" * 100), "sequence_numbering_mismatch"),
    ],
)
def test_unrepresentable_targets_are_complete_gap_aware_handoffs(
    numbering: str,
    reason: str,
) -> None:
    classified = classify_oas_target(numbering, _record())
    assert classified.representable is None
    assert classified.unusual is not None
    assert classified.unusual.reason == reason
    payload = pack_unusual_target(classified.unusual)
    assert len(payload) == UNUSUAL_TARGET_RECORD.size
    assert unpack_unusual_target(payload) == classified.unusual


def test_unusual_padding_change_fails_closed() -> None:
    unusual = classify_oas_target("not a numbering", _record()).unusual
    assert unusual is not None
    payload = bytearray(pack_unusual_target(unusual))
    payload[-1] = 1
    with pytest.raises(OASIMGTExportError, match="padding differs"):
        unpack_unusual_target(bytes(payload))
