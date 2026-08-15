"""Test the authenticated AsEP fixed-IMGT query export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from antibody_pretraining.downstream_overlap import (
    DevelopmentChainReference,
    sequence_sha256,
)
from antibody_pretraining.imgt_query_export import (
    ANARCII_SOURCE_COMMIT,
    ANARCII_VERSION,
    ANARCII_WHEEL_SHA256,
    IMGT_QUERY_RECORD,
    IMGTQueryExportError,
    export_numbered_development_queries,
    numbered_development_query,
    pack_imgt_query,
    unpack_imgt_query,
)
from antibody_pretraining.near_identity_overlap import reference_key


def _reference(chain: str, instance: str, role: str, sequence: str):
    return DevelopmentChainReference(
        chain=chain,
        instance_id=instance,
        role=role,
        sequence_length=len(sequence),
        sequence_sha256=sequence_sha256(sequence),
    )


def _result(sequence: str, chain_type: str, *, insertion: str = ""):
    numbering = [((position, ""), residue) for position, residue in enumerate(sequence, 1)]
    while len(numbering) < 128:
        numbering.append(((len(numbering) + 1, ""), "-"))
    if insertion:
        numbering.insert(112, ((111, insertion), "A"))
    return {
        "chain_type": chain_type,
        "error": None,
        "numbering": numbering,
        "query_end": len(sequence) - 1,
        "query_start": 0,
        "scheme": "imgt",
        "score": 42.5,
    }


def _runtime():
    return {
        "package": "anarcii",
        "package_version": ANARCII_VERSION,
        "source_commit": ANARCII_SOURCE_COMMIT,
        "wheel_sha256": ANARCII_WHEEL_SHA256,
    }


def test_query_record_roundtrip_binds_domain_digest_and_enums() -> None:
    sequence = "A" * 128
    reference = _reference("heavy", "complex-1", "val", sequence)
    numbered = numbered_development_query(
        query_ordinal=7,
        reference=reference,
        sequence=sequence,
        result=_result(sequence, "H"),
    )
    payload = pack_imgt_query(numbered.binary)
    assert len(payload) == IMGT_QUERY_RECORD.size == 239
    restored = unpack_imgt_query(payload)
    assert restored == numbered.binary
    assert restored.domain_sequence_sha256 == hashlib.sha256(sequence.encode()).hexdigest()


def test_query_result_rejects_wrong_chain_and_unregistered_insertion() -> None:
    sequence = "A" * 128
    heavy = _reference("heavy", "complex-1", "train", sequence)
    with pytest.raises(IMGTQueryExportError, match="chain type differs"):
        numbered_development_query(
            query_ordinal=0,
            reference=heavy,
            sequence=sequence,
            result=_result(sequence, "K"),
        )
    with pytest.raises(IMGTQueryExportError, match="outside the canonical 200"):
        numbered_development_query(
            query_ordinal=0,
            reference=heavy,
            sequence=sequence,
            result=_result(sequence, "H", insertion="Z"),
        )


def test_export_is_complete_label_free_and_create_only(tmp_path: Path) -> None:
    heavy_sequence = "A" * 128
    light_sequence = "C" * 128
    heavy = _reference("heavy", "complex-1", "train", heavy_sequence)
    light = _reference("light", "complex-1", "val", light_sequence)
    rows = ((heavy, heavy_sequence), (light, light_sequence))
    numbered = {
        reference_key(heavy): _result(heavy_sequence, "H"),
        reference_key(light): _result(light_sequence, "L"),
    }
    output = tmp_path / "queries"
    receipt = export_numbered_development_queries(
        output_root=output,
        sequences=rows,
        numbered=numbered,
        runtime_provenance=_runtime(),
        expected_query_count=2,
        source_provenance={"fixture": True},
    )
    assert receipt["query_count"] == 2
    assert receipt["authorization"] == {
        "fixed_imgt_search_authorized": False,
        "large_scale_pretraining_authorized": False,
        "reason": "exact fixed-IMGT and gap-aware near-identity tracks incomplete",
    }
    manifest = json.loads((output / "imgt_query_manifest.json").read_text("ascii"))
    assert manifest["query_count"] == 2
    assert manifest["test_access"] == {
        "development_sequence_fields_accessed": True,
        "frozen_official_external_test_accessed": False,
        "model_predictions_or_scores_accessed": False,
        "task_label_values_used": False,
    }
    assert (output / "queries.imgt.bin").stat().st_size == 2 * IMGT_QUERY_RECORD.size
    with pytest.raises(FileExistsError):
        export_numbered_development_queries(
            output_root=output,
            sequences=rows,
            numbered=numbered,
            runtime_provenance=_runtime(),
            expected_query_count=2,
        )


def test_export_rejects_missing_numbering_and_count_drift(tmp_path: Path) -> None:
    sequence = "A" * 128
    reference = _reference("heavy", "complex-1", "val", sequence)
    with pytest.raises(IMGTQueryExportError, match="count or uniqueness differs"):
        export_numbered_development_queries(
            output_root=tmp_path / "count",
            sequences=((reference, sequence),),
            numbered={reference_key(reference): _result(sequence, "H")},
            runtime_provenance=_runtime(),
            expected_query_count=2,
        )
    with pytest.raises(IMGTQueryExportError, match="identities differ"):
        export_numbered_development_queries(
            output_root=tmp_path / "missing",
            sequences=((reference, sequence),),
            numbered={},
            runtime_provenance=_runtime(),
            expected_query_count=1,
        )
