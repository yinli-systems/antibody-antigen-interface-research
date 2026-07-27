"""Tests for independently reconstructed AVIDa IL-6 annotations."""

from __future__ import annotations

import hashlib

import pytest

from interaction_first.avida_antigen_independent_auditor import (
    ACCEPTED_STATUS,
    AvidaAntigenIndependentAuditError,
    _expected_annotation,
)


def _candidate(sequence: str, label: str) -> dict:
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    return {
        "annotation_status": "raw_construct_untrimmed_signal_and_tag_boundaries_pending",
        "antigen_id": f"ag_sha256:{digest}",
        "construct_sequences": [sequence],
        "sequence_digest": digest,
        "source_id": "avida_hil6_2024_11",
        "variant_id": label,
    }


def test_independent_reconstruction_uses_inclusive_uniprot_coordinates() -> None:
    reference = "M" * 212
    row = _expected_annotation(
        _candidate(reference + "HHHHHH", "IL-6_WTs"),
        reference,
    )
    assert row["annotation_status"] == ACCEPTED_STATUS
    assert row["signal_sequence"] == reference[:29]
    assert row["mature_chain_sequence"] == reference[29:212]
    assert len(row["mature_chain_sequence"]) == 183


def test_independent_reconstruction_rejects_mislabeled_mutant() -> None:
    reference = "M" * 212
    mutant = reference[:41] + "A" + reference[42:]
    with pytest.raises(AvidaAntigenIndependentAuditError, match="disagrees"):
        _expected_annotation(
            _candidate(mutant + "HHHHHH", "IL-6_M43A"),
            reference,
        )


def test_independent_reconstruction_rejects_construct_tamper() -> None:
    reference = "M" * 212
    with pytest.raises(AvidaAntigenIndependentAuditError, match="construct differs"):
        _expected_annotation(
            _candidate(reference + "HHHHHA", "IL-6_WTs"),
            reference,
        )
