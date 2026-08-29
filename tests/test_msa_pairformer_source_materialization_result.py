"""Contract checks for the MSA Pairformer source-only materialization."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results"
    / "pretraining"
    / "abpairfm_msa_pairformer_source_materialization_20260822.json"
)


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_msa_pairformer_source_is_exactly_pinned_and_hash_bound() -> None:
    result = _result()
    source = result["source"]
    materialization = result["materialization"]
    assert source["revision"] == "875363570df1ae484cf725aba382790444223005"
    assert source["package_version"] == "1.0.2"
    assert materialization["file_count"] == 69
    assert materialization["archive"]["bytes"] == 7_606_011
    assert len(materialization["archive"]["sha256"]) == 64
    assert all(len(digest) == 64 for digest in source["key_file_sha256"].values())


def test_msa_pairformer_source_result_preserves_failed_acquisition_evidence() -> None:
    failure = _result()["recovery_evidence"]["failed_remote_codeload_attempt"]
    assert failure["gzip_validation"] == "FAILED_UNEXPECTED_EOF"
    assert failure["bytes"] < _result()["materialization"]["archive"]["bytes"]
    assert len(failure["sha256"]) == 64


def test_msa_pairformer_cannot_score_until_weight_runtime_and_modality_gates_pass() -> None:
    result = _result()
    assert result["status"] == (
        "SOURCE_MATERIALIZED_WEIGHTS_RUNTIME_AND_MSA_BLOCKED"
    )
    assert result["authorization"]["development_contact_evaluation"] is False
    assert result["authorization"]["production_training"] is False
    assert result["test_access"] is False
    assert result["weight_gate"]["weights_downloaded"] is False
    assert result["weight_gate"]["weights_sha256_bound"] is False
    assert result["input_modality_gate"]["paired_msa_comparison_ready"] is False
    assert result["input_modality_gate"]["query_only_must_be_reported_separately"] is True


def test_fixed_training_environment_is_not_modified() -> None:
    runtime = _result()["fixed_runtime_audit"]
    assert runtime["fixed_python"].endswith("/contact/bin/python3.10")
    assert runtime["install_into_fixed_environment"] is False
    assert "einx" in runtime["unavailable"]
    assert "cuequivariance" in runtime["unavailable"]
