from __future__ import annotations

from scripts.render_fair_comparison_table import render


def test_render_keeps_scopes_and_unavailable_outputs_explicit() -> None:
    text = render()
    assert "frozen SAbDab2 test" in text
    assert "official five-fold held-out" in text
    assert "FrameContact (frozen test not opened)" in text
    assert "FrameContact (development-only; not ranked here)" in text
    assert "Never convert a residue score into a contact score" in text


def test_render_contains_receipt_values() -> None:
    text = render()
    assert "0.2775" in text  # EpiFormer epitope AUPRC
    assert "0.6892" in text  # MIPE validation-best paratope AUPRC
    assert "0.6511" in text  # SPPIDER paratope AUPRC


def test_render_includes_native_contact_development_snapshot() -> None:
    text = render()
    assert "FrameContact contact development snapshot" in text
    assert "FrameContact unary-only | 21 | 0.0107 ± 0.0008" in text
    assert "FrameContact distogram | 18 | 0.0109 ± 0.0012" in text
    assert "not frozen-test results" in text
