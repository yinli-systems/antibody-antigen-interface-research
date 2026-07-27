"""Consistency tests for the initial six-step PAIRScale completion audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "pairscale"
    / "initial_six_step_completion_audit_2026-07-28.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def test_all_six_requirements_have_content_addressed_completion_evidence() -> None:
    audit = _load(AUDIT_PATH)
    requirements = audit["requirements"]
    assert [value["id"] for value in requirements] == [1, 2, 3, 4, 5, 6]
    assert all(value["status"] == "complete" for value in requirements)
    for requirement in requirements:
        assert requirement["evidence"]
        for evidence in requirement["evidence"]:
            path_value = evidence.get("path")
            digest = evidence.get("sha256") or evidence.get("result_file_sha256")
            if path_value is None or digest is None:
                continue
            path = REPOSITORY_ROOT / path_value
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_overfit_metrics_and_resume_claim_match_frozen_result() -> None:
    audit = _load(AUDIT_PATH)
    requirement = audit["requirements"][4]
    metric_evidence = next(
        value for value in requirement["evidence"] if "macro_pair_auprc" in value
    )
    result = _load(REPOSITORY_ROOT / metric_evidence["path"])
    assert result["gate"]["complete_l0_passed"] is True
    assert result["exact_resume"]["passed"] is True
    assert metric_evidence["final_checkpoint_sha256"] == result["exact_resume"][
        "uninterrupted_final_checkpoint_sha256"
    ]
    assert metric_evidence["macro_pair_auprc"] == result["metrics"][
        "macro_pair_auprc"
    ]
    assert metric_evidence["macro_precision_at_true_contact_count"] == result[
        "metrics"
    ]["macro_precision_at_true_contact_count"]


def test_registry_evidence_and_scientific_claim_boundary_are_preserved() -> None:
    audit = _load(AUDIT_PATH)
    registry_evidence = audit["requirements"][5]["evidence"][1]
    receipt = _load(REPOSITORY_ROOT / registry_evidence["path"])
    assert receipt["all_passed"] is True
    assert len(receipt["assets"]) == registry_evidence["asset_count"] == 7
    scientific_state = audit["broader_scientific_state"]
    assert scientific_state["gate_l1_passed"] is False
    assert scientific_state["model_scaling_authorized"] is False
    assert "does not complete the broader PAIRScale roadmap" in scientific_state[
        "scope_limit"
    ]
