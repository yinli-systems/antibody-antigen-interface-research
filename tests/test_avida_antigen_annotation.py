"""Tests for pinned AVIDa IL-6 construct-boundary interpretation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from interaction_first.avida_annotation_protocol import (
    load_avida_annotation_protocol,
)
from interaction_first.avida_antigen_annotation import (
    ACCEPTED_STATUS,
    QUARANTINED_STATUS,
    _interpret_antigen,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data_registry/annotation_protocols/avida_hil6_v1.json"


def _row(sequence: str, label: str) -> dict:
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    return {
        "antigen_id": f"ag_sha256:{digest}",
        "construct_sequences": [sequence],
        "sequence_digest": digest,
        "variant_id": label,
    }


def test_wild_type_and_declared_single_mutant_are_accepted() -> None:
    protocol, _ = load_avida_annotation_protocol(PROTOCOL)
    reference = "M" * 212
    wild_type = _interpret_antigen(
        row=_row(reference + "HHHHHH", "IL-6_WTs"),
        reference_sequence=reference,
        protocol=protocol,
    )
    mutant_sequence = reference[:41] + "A" + reference[42:]
    mutant = _interpret_antigen(
        row=_row(mutant_sequence + "HHHHHH", "IL-6_M42A"),
        reference_sequence=reference,
        protocol=protocol,
    )
    assert wild_type["annotation_status"] == ACCEPTED_STATUS
    assert wild_type["mutation_count"] == 0
    assert len(wild_type["mature_chain_sequence"]) == 183
    assert mutant["annotation_status"] == ACCEPTED_STATUS
    assert mutant["mutation_position_uniprot_one_based"] == 42
    assert mutant["reference_residue"] == "M"
    assert mutant["mutant_residue"] == "A"


def test_variant_label_disagreement_is_quarantined() -> None:
    protocol, _ = load_avida_annotation_protocol(PROTOCOL)
    reference = "M" * 212
    mutant_sequence = reference[:41] + "A" + reference[42:]
    row = _interpret_antigen(
        row=_row(mutant_sequence + "HHHHHH", "IL-6_M43A"),
        reference_sequence=reference,
        protocol=protocol,
    )
    assert row["annotation_status"] == QUARANTINED_STATUS
    assert "variant_label_sequence_disagrees" in row["quarantine_reasons"]


def test_missing_tag_is_quarantined() -> None:
    protocol, _ = load_avida_annotation_protocol(PROTOCOL)
    reference = "M" * 212
    row = _interpret_antigen(
        row=_row(reference + "HHHHHA", "IL-6_WTs"),
        reference_sequence=reference,
        protocol=protocol,
    )
    assert row["annotation_status"] == QUARANTINED_STATUS
    assert "construct_tag_differs" in row["quarantine_reasons"]
    assert "precursor_suffix_differs" in row["quarantine_reasons"]


def test_signal_region_mutation_is_quarantined() -> None:
    protocol, _ = load_avida_annotation_protocol(PROTOCOL)
    reference = "M" * 212
    mutant_sequence = reference[:9] + "A" + reference[10:]
    row = _interpret_antigen(
        row=_row(mutant_sequence + "HHHHHH", "IL-6_M10A"),
        reference_sequence=reference,
        protocol=protocol,
    )
    assert row["annotation_status"] == QUARANTINED_STATUS
    assert "mutation_outside_mature_chain" in row["quarantine_reasons"]
