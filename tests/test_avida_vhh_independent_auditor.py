"""Tests for independent AVIDa VHH annotation replay semantics."""

from __future__ import annotations

import hashlib

import pytest

from interaction_first.avida_vhh_independent_auditor import (
    ACCEPTED_STATUS,
    AvidaVhhIndependentAuditError,
    QUARANTINED_STATUS,
    _compare_annotation,
    _expected_annotation,
)


def _case() -> tuple[str, dict]:
    domain = "A" * 128
    construct = "MKYLLPTAAAGLLLLAAQPAMA" + domain + "HHHHHH"
    return construct, {
        "chain_type": "H",
        "error": None,
        "numbering": [((position, " "), "A") for position in range(1, 129)],
        "query_end": 22 + len(domain) - 1,
        "query_start": 22,
        "scheme": "imgt",
        "score": 30.0,
    }


def _expected(construct: str, result: dict):
    digest = hashlib.sha256(construct.encode()).hexdigest()
    return _expected_annotation(
        antibody_id=f"ab_sha256:{digest}",
        construct=construct,
        construct_digest=digest,
        result=result,
    )


def test_independent_replay_accepts_inclusive_boundary() -> None:
    construct, result = _case()
    row = _expected(construct, result)
    assert row["annotation_status"] == ACCEPTED_STATUS
    assert row["query_end_zero_based_inclusive"] == len(construct) - 7
    assert row["suffix_sequence"] == "HHHHHH"
    assert row["quarantine_reasons"] == []


def test_independent_replay_preserves_tool_error() -> None:
    construct, result = _case()
    result.update(
        {
            "chain_type": "F",
            "error": "Model predicted duplicate numbers",
            "numbering": None,
            "query_end": None,
            "query_start": None,
        }
    )
    row = _expected(construct, result)
    assert row["annotation_status"] == QUARANTINED_STATUS
    assert row["tool_error"] == "Model predicted duplicate numbers"
    assert "tool_reported_error" in row["quarantine_reasons"]


def test_independent_replay_detects_field_tamper() -> None:
    construct, result = _case()
    expected = _expected(construct, result)
    observed = dict(expected)
    observed["query_end_zero_based_inclusive"] -= 1
    with pytest.raises(AvidaVhhIndependentAuditError, match="annotation differs"):
        _compare_annotation(expected, observed)


def test_independent_replay_enforces_score_tolerance() -> None:
    construct, result = _case()
    expected = _expected(construct, result)
    observed = dict(expected)
    observed["tool_score"] += 1e-6
    with pytest.raises(AvidaVhhIndependentAuditError, match="tool score differs"):
        _compare_annotation(expected, observed)
