"""Regression tests for signed MMseqs gap finalizer snapshots."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from antibody_pretraining.mmseqs_gap_aware import MMseqsGapAwareError

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/finalize_oas_mmseqs_gap_search.py"
)
SPEC = importlib.util.spec_from_file_location("oas_mmseqs_gap_finalizer", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_signed_payload_comparison_strips_only_authenticated_envelope() -> None:
    signed = {"format_version": "audit_v1", "rows": 7, "receipt_sha256": "a" * 64}
    snapshot = {"format_version": "audit_v1", "rows": 7}

    MODULE._same_signed_payload(signed, snapshot, "audit")

    with pytest.raises(MMseqsGapAwareError, match="audit differs"):
        MODULE._same_signed_payload(signed, {**snapshot, "rows": 8}, "audit")
    with pytest.raises(MMseqsGapAwareError, match="unsigned snapshot differs"):
        MODULE._same_signed_payload(signed, signed, "audit")
    with pytest.raises(MMseqsGapAwareError, match="receipt"):
        MODULE._same_signed_payload(snapshot, snapshot, "audit")
