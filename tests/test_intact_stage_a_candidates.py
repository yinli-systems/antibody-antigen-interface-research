"""Tests for strict IntAct candidate filtering."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_intact_stage_a_candidates.py"


def _module():
    specification = importlib.util.spec_from_file_location("intact_audit", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _row(module, *, negative: str = "false") -> list[str]:
    row = ["-"] * len(module.EXPECTED_HEADER)
    row[0] = "intact:EBI-1"
    row[1] = "intact:EBI-2"
    row[2] = "uniprotkb:P12345"
    row[3] = "uniprotkb:Q8N158-2"
    row[8] = "pubmed:123"
    row[9] = "taxid:9606(human)"
    row[10] = "taxid:9606(human)"
    row[11] = 'psi-mi:"MI:0407"(direct interaction)'
    row[13] = "intact:EBI-3"
    row[14] = "intact-miscore:0.75"
    row[20] = 'psi-mi:"MI:0326"(protein)'
    row[21] = 'psi-mi:"MI:0326"(protein)'
    row[35] = negative
    return row


def test_positive_candidate_requires_binary_high_confidence_proteins() -> None:
    module = _module()
    reason, record = module._classify(_row(module))
    assert reason == "accepted"
    assert record is not None
    assert record["accession_a"] == "P12345"
    assert record["accession_b"] == "Q8N158"
    assert record["label"] == 1
    row = _row(module)
    row[15] = 'psi-mi:"MI:1060"(spoke expansion)'
    assert module._classify(row)[0] == "expanded_complex"
    row = _row(module)
    row[14] = "intact-miscore:0.59"
    assert module._classify(row)[0] == "positive_miscore"


def test_measured_negative_is_not_forced_through_positive_score_gate() -> None:
    module = _module()
    row = _row(module, negative="true")
    row[14] = "-"
    reason, record = module._classify(row)
    assert reason == "accepted"
    assert record is not None and record["label"] == 0


def test_byte_range_and_accession_helpers_are_strict() -> None:
    module = _module()
    assert module._uniprot_accession("uniprotkb:P12345-7") == "P12345"
    assert module._uniprot_accession("intact:EBI-1") is None
    assert module._miscore("foo:1|intact-miscore:0.7") == 0.7
