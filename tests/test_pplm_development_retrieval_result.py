"""Contract checks for the audited PPLM full development retrieval result."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results"
    / "pretraining"
    / "abpairfm_pplm_development_retrieval_job_1459203.json"
)


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_full_pplm_result_is_complete_hash_bound_and_test_sealed() -> None:
    result = _result()
    assert result["format_version"] == "abpairfm_pplm_development_retrieval_job_v1"
    assert result["test_access"] is False
    assert result["equivalence_gate"]["status"] == "ACCEPTED_BY_VERIFIED_SMOKE_GATE"
    assert len(result["protocol_sha256"]) == 64
    for dataset in ("asep", "sabdab2"):
        record = result["datasets"][dataset]
        assert record["status"] == "FROZEN_DEVELOPMENT_RETRIEVAL_COMPLETE"
        assert record["sacct"]["state"] == "COMPLETED"
        assert record["sacct"]["exit_code"] == "0:0"
        assert record["sacct"]["stderr_bytes"] == 0
        assert record["test_access"] is False
        assert len(record["predictions"]["sha256"]) == 64
        assert len(record["summary_sha256"]) == 64


def test_full_pplm_result_has_exact_frozen_panel_cardinality() -> None:
    datasets = _result()["datasets"]
    assert datasets["asep"]["predictions"]["row_count"] == 10_880
    assert datasets["asep"]["unique_canonical_exact_pair_count"] == 170
    assert datasets["sabdab2"]["predictions"]["row_count"] == 35_136
    assert datasets["sabdab2"]["anchor_instance_count"] == 549
    assert datasets["sabdab2"]["unique_canonical_exact_pair_count"] == 413


def test_full_pplm_result_preserves_the_negative_quality_frontier_result() -> None:
    comparison = _result()["comparison_to_mint"]
    assert comparison["decision"] == (
        "MINT_RETAINS_SEQUENCE_ONLY_DEVELOPMENT_RETRIEVAL_FRONTIER"
    )
    for dataset in ("asep", "sabdab2"):
        record = comparison[dataset]
        assert record["mrr_delta_pplm_minus_mint"] < 0
        assert record["mrr_paired_bootstrap_ci95"][0] < 0
        assert record["mrr_paired_bootstrap_ci95"][1] > 0
        assert record["pplm_to_mint_gpu_hour_ratio"] > 2


def test_full_pplm_result_does_not_grant_commercial_or_contact_authority() -> None:
    result = _result()
    assert result["license_boundary"]["commercial_use_authorized"] is False
    assert "Contact" in result["claim_boundary"]
    assert "test" in result["claim_boundary"]
