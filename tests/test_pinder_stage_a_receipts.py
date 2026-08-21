"""Cross-check the compact PINDER summary against the exact remote receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "data_registry/materialization_receipts"
SUMMARY = RECEIPTS / "pinder_2024_02_trainval_source_audit_v1.json"
EXACT = RECEIPTS / "pinder_2024_02_source_audit_v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_summary_is_anchored_to_exact_remote_receipt() -> None:
    summary = _load(SUMMARY)
    exact = _load(EXACT)
    observed_receipt_sha256 = hashlib.sha256(EXACT.read_bytes()).hexdigest()
    assert observed_receipt_sha256 == summary["remote_audit_receipt_sha256"]
    assert summary["source_files"]["index_sha256"] == exact["files"]["index"]["sha256"]
    assert summary["source_files"]["metadata_sha256"] == exact["files"]["metadata"]["sha256"]
    assert summary["quality_filter"]["strict_model_fit_train_row_count"] == exact["quality_filter"]["counts"]["strict_model_fit"]
    assert summary["quality_filter"]["strict_model_fit_cluster_count"] == exact["quality_filter"]["counts"]["strict_model_fit_clusters"]


def test_exact_receipt_fails_closed_and_keeps_test_sealed() -> None:
    exact = _load(EXACT)
    assert exact["authorization"]["normalization"] is False
    assert exact["authorization"]["stage_a_training"] is False
    assert exact["quality_filter"]["development_decontamination_pending"] is True
    assert not any(exact["test_access"].values())
    assert exact["structure_archive"]["test_set_archive_requested"] is False
