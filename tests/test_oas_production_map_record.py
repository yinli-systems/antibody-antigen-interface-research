"""Test the fixed-width production OAS map-record schema."""

from __future__ import annotations

import hashlib
import io

import pytest

from antibody_pretraining.production_materialization import (
    MAP_RECORD,
    ProductionMapRecord,
    ProductionMaterializationError,
    iter_map_records,
    pack_map_record,
    unpack_map_record,
)


SEQUENCE = "ACDEFGHIKLMNPQRSTVWY" * 4


def _record(**updates: object) -> ProductionMapRecord:
    values = {
        "authority_ordinal": (7 << 32) | 11,
        "chain_type": "heavy",
        "sequence": SEQUENCE,
        "sequence_sha256": hashlib.sha256(SEQUENCE.encode("ascii")).hexdigest(),
        "split": "train",
        "status_flags": ("deletion", "shorter"),
        "stratum": "human_heavy",
        "study_id": 9,
    }
    values.update(updates)
    return ProductionMapRecord(**values)


def test_record_round_trip_is_exact_and_fixed_width() -> None:
    """Preserve every field in the protocol's 240-byte representation."""
    record = _record()
    payload = pack_map_record(record)
    assert len(payload) == MAP_RECORD.size == 240
    assert unpack_map_record(payload) == record
    assert list(iter_map_records(io.BytesIO(payload * 2))) == [record, record]


def test_record_rejects_digest_padding_and_truncation_tampering() -> None:
    """Fail closed on malformed or semantically inconsistent bytes."""
    with pytest.raises(ProductionMaterializationError, match="digest differs"):
        pack_map_record(_record(sequence_sha256="0" * 64))
    payload = bytearray(pack_map_record(_record()))
    payload[-1] = 1
    with pytest.raises(ProductionMaterializationError, match="padding differs"):
        unpack_map_record(bytes(payload))
    with pytest.raises(ProductionMaterializationError, match="partial record"):
        list(iter_map_records(io.BytesIO(pack_map_record(_record())[:-1])))


def test_record_rejects_unknown_status_and_chain_enums() -> None:
    """Keep record categories inside the frozen bounded vocabularies."""
    with pytest.raises(ProductionMaterializationError, match="Status flag"):
        pack_map_record(_record(status_flags=("unknown",)))
    with pytest.raises(ProductionMaterializationError, match="chain type is invalid"):
        pack_map_record(_record(chain_type="paired_heavy"))
