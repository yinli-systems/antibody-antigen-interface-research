from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "abpairfm_pplm_development_retrieval_smoke_v4.json"


def test_pplm_smoke_protocol_closes_claims_and_test() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["authorization"] == {
        "development_contact": False,
        "development_pair_retrieval": False,
        "production_training": False,
        "test_access": False,
    }
    assert protocol["test_access"] is False
    assert protocol["status"] == "FROZEN_BEFORE_GPU_EVALUATION"


def test_pplm_smoke_protocol_requires_two_numerical_gates() -> None:
    gate = json.loads(PROTOCOL.read_text(encoding="utf-8"))["equivalence_gate"]
    assert gate["required_before_full_development"] is True
    assert gate["required_smoke_summary_sha256"] is None
    assert gate["streaming_vs_official_full_attention_max_abs_tolerance"] == 5e-6
    assert gate["clean_room_vs_official_probability_max_abs_tolerance"] == 1e-6
    assert gate["batch_padding_vs_singleton_max_abs_tolerance"] == 2e-5


def test_pplm_smoke_protocol_pins_noncommercial_official_artifacts() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    model = protocol["model"]
    assert model["backbone_parameters"] == 651_044_309
    assert model["head_ensemble_member_count"] == 10
    assert len(model["backbone_checkpoint_sha256"]) == 64
    assert len(model["head_checkpoint_sha256"]) == 64
    assert len(model["source_archive_sha256"]) == 64
    assert "commercial-use" in protocol["claim_boundary"]
    assert [failure["job_id"] for failure in protocol["failure_evidence"]] == [
        "1459174",
        "1459176",
        "1459184",
    ]
    assert all(failure["state"] == "FAILED" for failure in protocol["failure_evidence"])
    assert set(protocol["reproducibility"]["implementation_file_sha256"]) == {
        "src/antibody_pretraining/pair_aware_attention.py",
        "src/antibody_pretraining/pair_aware_esm2.py",
    }
