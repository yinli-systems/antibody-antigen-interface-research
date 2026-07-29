"""Tests for AVIDa VHH construct-boundary interpretation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from interaction_first.avida_annotation_protocol import (
    load_avida_annotation_protocol,
)
from interaction_first.avida_vhh_annotation import (
    ACCEPTED_STATUS,
    QUARANTINED_STATUS,
    _interpret_result,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data_registry/annotation_protocols/avida_hil6_v1.json"


def _case() -> tuple[str, str, dict]:
    prefix = "MKYLLPTAAAGLLLLAAQPAMA"
    domain = "A" * 128
    suffix = "HHHHHH"
    construct = prefix + domain + suffix
    result = {
        "chain_type": "H",
        "error": None,
        "numbering": [((position, " "), "A") for position in range(1, 129)],
        "query_end": len(prefix) + len(domain) - 1,
        "query_start": len(prefix),
        "scheme": "imgt",
        "score": 30.0,
    }
    return construct, domain, result


def _interpret(construct: str, result: dict):
    protocol, _ = load_avida_annotation_protocol(PROTOCOL)
    return _interpret_result(
        antibody_id=f"ab_sha256:{hashlib.sha256(construct.encode()).hexdigest()}",
        construct_sequence=construct,
        construct_sequence_digest=hashlib.sha256(construct.encode()).hexdigest(),
        result=result,
        protocol=protocol,
    )


def test_exact_inclusive_boundary_is_accepted() -> None:
    construct, domain, result = _case()
    row = _interpret(construct, result)
    assert row["annotation_status"] == ACCEPTED_STATUS
    assert row["domain_sequence"] == domain
    assert row["suffix_sequence"] == "HHHHHH"
    assert row["query_end_zero_based_inclusive"] == len(construct) - 7
    assert row["quarantine_reasons"] == []


def test_exclusive_end_interpretation_is_quarantined() -> None:
    construct, _, result = _case()
    result["query_end"] += 1
    row = _interpret(construct, result)
    assert row["annotation_status"] == QUARANTINED_STATUS
    assert "numbering_residues_differ" in row["quarantine_reasons"]
    assert "query_suffix_differs" in row["quarantine_reasons"]


def test_tool_failure_is_preserved_without_fallback() -> None:
    construct, _, result = _case()
    result.update(
        {
            "error": "Score less than cut off.",
            "numbering": None,
            "query_end": None,
            "query_start": None,
        }
    )
    row = _interpret(construct, result)
    assert row["annotation_status"] == QUARANTINED_STATUS
    assert row["tool_error"] == "Score less than cut off."
    assert "tool_reported_error" in row["quarantine_reasons"]
    assert row["domain_sequence"] is None


def test_duplicate_numbering_coordinate_is_quarantined() -> None:
    construct, _, result = _case()
    result["numbering"].append(((128, " "), "A"))
    row = _interpret(construct, result)
    assert row["annotation_status"] == QUARANTINED_STATUS
    assert "numbering_coordinate_duplicated" in row["quarantine_reasons"]


def test_wrong_chain_and_construct_tag_are_quarantined() -> None:
    construct, _, result = _case()
    construct = construct[:-1] + "A"
    result["chain_type"] = "L"
    row = _interpret(construct, result)
    assert row["annotation_status"] == QUARANTINED_STATUS
    assert "chain_type_differs" in row["quarantine_reasons"]
    assert "construct_suffix_differs" in row["quarantine_reasons"]
