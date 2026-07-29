"""Regression checks for the committed geometry research evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "sabdab2_structure_self_attention"


def _load(name: str) -> dict:
    value = json.loads((RESULT_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


def test_a100_replay_verification_retains_all_claim_boundaries() -> None:
    replay = _load("ab03_a100_replay_verification.json")
    assert replay["status"] == "passed"
    assert replay["archive"]["regular_file_count"] == 54
    assert replay["manifests"]["data_authority"]["file_count"] == 8727
    assert replay["scientific_replay"]["gate_count"] == 48
    assert replay["scientific_replay"]["native_test_read"] is False
    assert replay["multiseed"]["seed_count"] == 3
    assert replay["multiseed"]["source_input_count"] == 3
    assert replay["multiseed"]["model_parameter_count"] == 271670
    assert replay["multiseed"]["recomputation_check_count"] == 80
    assert replay["runtime"]["torch"] == "2.5.1"
    assert replay["multiseed"]["runtime_contract"]["torch"] == "2.2.1"
    assert "does not authorize" in replay["claim_boundary"]


def test_a100_paired_summary_is_hashed_and_non_authorizing() -> None:
    summary = _load("a100_checkpoint_paired_statistics_summary.json")
    assert summary["primary_gate"]["confirmatory_superiority_authorized"] is False
    assert summary["primary_gate"]["multiseed_requirement_satisfied"] is False
    assert summary["independent_verification"] == {
        "check_count": 441,
        "status": "passed",
    }
    assert summary["resampling"]["independent_endpoint_component_count"] == 16
    assert summary["resampling"]["validation_instance_count"] == 36
    assert all(_is_sha256(value) for value in summary["artifact_hashes"].values())
    assert all(
        _is_sha256(value) for value in summary["checkpoint_authority"].values()
    )


def test_research_report_contains_sources_results_and_next_gate() -> None:
    report = (RESULT_ROOT / "FINAL_GEOMETRY_RESEARCH_REPORT.md").read_text(
        encoding="utf-8"
    )
    required_terms = (
        "AlphaFold2 and invariant point attention",
        "ProteinMPNN",
        "Geometric Vector Perceptrons",
        "SE(3)-Transformer",
        "AlphaFold3 and Pairformer",
        "Exact A100 checkpoints: per-complex paired audit",
        "cognate versus matched-partner shuffle",
        "No frozen test",
    )
    for term in required_terms:
        assert term in report
