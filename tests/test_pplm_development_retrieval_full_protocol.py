from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "abpairfm_pplm_development_retrieval_v1.json"


def test_full_pplm_protocol_authorizes_development_retrieval_only() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["authorization"] == {
        "development_contact": False,
        "development_pair_retrieval": True,
        "production_training": False,
        "test_access": False,
    }
    assert protocol["test_access"] is False
    assert protocol["status"] == "FROZEN_BEFORE_GPU_EVALUATION"


def test_full_pplm_protocol_pins_accepted_smoke_and_runner() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    gate = protocol["equivalence_gate"]
    assert gate["status"] == "EXACT_FEATURE_SEMANTICS_ACCEPTED"
    assert gate["smoke_job_id"] == "1459194"
    assert len(gate["required_smoke_summary_sha256"]) == 64
    assert gate["accepted_probability_max_abs_delta"] < 1e-6
    assert len(protocol["reproducibility"]["runner_sha256"]) == 64


def test_full_pplm_protocol_keeps_primary_duplicate_policy_and_bootstrap_unit() -> None:
    metrics = json.loads(PROTOCOL.read_text(encoding="utf-8"))["metrics"]
    assert metrics["duplicate_policy"]["primary"].startswith("collapse exact")
    assert metrics["bootstrap_unit"] == "canonical_pair_sha256"
    assert metrics["bootstrap_replicates"] == 10_000
