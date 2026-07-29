"""Focused tests for AVIDa Tier-1 entity normalization."""

from __future__ import annotations

import hashlib

import pytest

from interaction_first.avida_hil6_normalization import (
    AvidaNormalizationError,
    _canonical_numbering,
    _normalize_antibody,
    _normalize_antigen,
)


def _candidate_antibody() -> dict[str, object]:
    construct = "SIGNALQVQLTAG"
    digest = hashlib.sha256(construct.encode()).hexdigest()
    return {
        "annotation_status": (
            "raw_construct_untrimmed_signal_and_tag_boundaries_pending"
        ),
        "antibody_format": "vhh_construct_untrimmed",
        "antibody_id": f"ab_sha256:{digest}",
        "cdr_boundaries": None,
        "clonotype_id": None,
        "construct_sequence": construct,
        "germline_j": None,
        "germline_v": None,
        "imgt_numbering": None,
        "license_route": "noncommercial_only",
        "lineage_id": None,
        "sequence_digest": digest,
        "source_id": "avida_hil6_2024_11",
        "vh_sequence": None,
        "vl_sequence": None,
    }


def _accepted_annotation() -> dict[str, object]:
    candidate = _candidate_antibody()
    domain = "QVQL"
    return {
        "annotation_status": "imgt_heavy_domain_verified",
        "antibody_id": candidate["antibody_id"],
        "construct_sequence_digest": candidate["sequence_digest"],
        "domain_sequence": domain,
        "domain_sequence_digest": hashlib.sha256(domain.encode()).hexdigest(),
        "imgt_numbering": [
            {"insertion": "", "position": 1, "residue": "Q"},
            {"insertion": "", "position": 2, "residue": "V"},
            {"insertion": "", "position": 3, "residue": "Q"},
            {"insertion": "", "position": 4, "residue": "L"},
        ],
        "quarantine_reasons": [],
        "query_end_zero_based_inclusive": 9,
        "query_start_zero_based_inclusive": 6,
        "tool_error": None,
    }


def test_accepted_vhh_is_rekeyed_by_verified_domain() -> None:
    candidate = _candidate_antibody()
    annotation = _accepted_annotation()
    normalized, rekey, quarantined = _normalize_antibody(candidate, annotation)
    expected_id = f"ab_sha256:{annotation['domain_sequence_digest']}"
    assert normalized["antibody_id"] == expected_id
    assert normalized["construct_sequence"] == candidate["construct_sequence"]
    assert normalized["vh_sequence"] == "QVQL"
    assert normalized["sequence_digest"] == annotation["domain_sequence_digest"]
    assert normalized["imgt_numbering"] == ["1:Q", "2:V", "3:Q", "4:L"]
    assert rekey["normalized_entity_id"] == expected_id
    assert rekey["admission_status"] == "admitted_tier1"
    assert quarantined is None


def test_nonconcordant_vhh_is_preserved_only_in_quarantine() -> None:
    candidate = _candidate_antibody()
    annotation = _accepted_annotation()
    annotation.update(
        annotation_status="quarantined_vhh_boundary",
        quarantine_reasons=["query_suffix_differs"],
    )
    normalized, rekey, quarantined = _normalize_antibody(candidate, annotation)
    assert normalized is None
    assert rekey["normalized_entity_id"] is None
    assert rekey["admission_status"] == "quarantined"
    assert quarantined["source_antibody_id"] == candidate["antibody_id"]
    assert quarantined["quarantine_reasons"] == ["query_suffix_differs"]


def test_antigen_is_rekeyed_by_verified_mature_chain() -> None:
    construct = "SIGNALCHAINHHHHHH"
    construct_digest = hashlib.sha256(construct.encode()).hexdigest()
    chain = "CHAIN"
    chain_digest = hashlib.sha256(chain.encode()).hexdigest()
    candidate = {
        "annotation_status": (
            "raw_construct_untrimmed_signal_and_tag_boundaries_pending"
        ),
        "antigen_id": f"ag_sha256:{construct_digest}",
        "chain_sequences": None,
        "construct_sequences": [construct],
        "sequence_digest": construct_digest,
        "uniprot_ids": [],
        "variant_id": "IL-6_WTs",
    }
    annotation = {
        "annotation_status": "mature_chain_verified_single_variant",
        "construct_sequence_digest": construct_digest,
        "mature_chain_sequence": chain,
        "mature_chain_sequence_digest": chain_digest,
        "quarantine_reasons": [],
        "uniprot_accession": "P05231",
        "variant_id": "IL-6_WTs",
    }
    normalized, rekey = _normalize_antigen(candidate, annotation)
    assert normalized["antigen_id"] == f"ag_sha256:{chain_digest}"
    assert normalized["chain_sequences"] == [chain]
    assert normalized["construct_sequences"] == [construct]
    assert normalized["uniprot_ids"] == ["P05231"]
    assert rekey["construct_sequence_digest"] == construct_digest


def test_numbering_rejects_duplicate_coordinate_or_wrong_domain() -> None:
    with pytest.raises(AvidaNormalizationError, match="duplicated"):
        _canonical_numbering(
            [
                {"position": 1, "insertion": "", "residue": "Q"},
                {"position": 1, "insertion": "", "residue": "V"},
            ],
            "QV",
        )
    with pytest.raises(AvidaNormalizationError, match="reconstruct"):
        _canonical_numbering(
            [{"position": 1, "insertion": "", "residue": "Q"}],
            "QV",
        )
