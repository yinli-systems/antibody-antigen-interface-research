"""Tests for the independent AVIDa candidate auditor."""

from __future__ import annotations

from pathlib import Path

import pytest

from interaction_first.avida_hil6_candidate import build_avida_hil6_candidate
from interaction_first.avida_hil6_independent_auditor import (
    AvidaIndependentAuditError,
    audit_avida_hil6_candidate,
)
from test_avida_hil6_candidate import _contract, _expectations, _fixture


SYNTHETIC_COUNTS = {
    "antibodies": 3,
    "antigens": 2,
    "binders": 2,
    "conflicting_pairs": 0,
    "exact_duplicate_groups": 0,
    "exact_duplicate_records": 0,
    "nonbinders": 2,
    "rows": 4,
    "subject_names": ["wizzy"],
    "subject_sexes": ["male"],
    "subject_species": ["alpaca"],
    "unique_pairs": 4,
}


def _audit(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    raw, registry = _fixture(tmp_path)
    candidate = tmp_path / "candidate"
    build_avida_hil6_candidate(
        raw_root=raw,
        registry_root=registry,
        contract_path=_contract(),
        output_root=candidate,
        expectations=_expectations(),
        created_at_utc="2026-07-27T16:00:00Z",
    )
    receipt = tmp_path / "independent.json"
    result = audit_avida_hil6_candidate(
        raw_root=raw,
        registered_raw_inventory=registry
        / "avida_hil6_2024_11"
        / "raw_sha256.json",
        contract_path=_contract(),
        candidate_root=candidate,
        normalizer_source_path=Path(__file__).resolve().parents[1]
        / "src/interaction_first/avida_hil6_candidate.py",
        output_receipt=receipt,
        audited_at_utc="2026-07-27T16:01:00Z",
        expected_counts=SYNTHETIC_COUNTS,
    )
    return candidate, dict(result)


def test_independent_auditor_reconstructs_every_row(tmp_path: Path) -> None:
    _, result = _audit(tmp_path)
    assert result["status"] == "pass_candidate_remains_blocked"
    assert result["semantic_comparison"] == "exact_row_order_and_content"
    assert result["counts"]["rows"] == 4


def test_independent_auditor_rejects_candidate_byte_tamper(tmp_path: Path) -> None:
    candidate, _ = _audit(tmp_path)
    with (candidate / "pairs.parquet").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(AvidaIndependentAuditError, match="size differs"):
        audit_avida_hil6_candidate(
            raw_root=tmp_path / "raw/avida_hil6_2024_11",
            registered_raw_inventory=tmp_path
            / "registry/avida_hil6_2024_11/raw_sha256.json",
            contract_path=_contract(),
            candidate_root=candidate,
            normalizer_source_path=Path(__file__).resolve().parents[1]
            / "src/interaction_first/avida_hil6_candidate.py",
            output_receipt=tmp_path / "second.json",
            expected_counts=SYNTHETIC_COUNTS,
        )
