from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "extensions/strict_partner_retrieval_v1/freeze_split.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fold(prefix: str) -> dict:
    train = {"pair_id": f"{prefix}_train", "antibody_id": f"{prefix}_a0", "antigen_id": f"{prefix}_g0"}
    val = {"pair_id": f"{prefix}_val", "antibody_id": f"{prefix}_a1", "antigen_id": f"{prefix}_g1"}
    return {
        "training_pairs": [train],
        "validation_pairs": [val],
        "quarantined_pairs": [],
        "counts": {
            "universe_pair_count": 2,
            "training_antibody_count": 1,
            "training_antigen_count": 1,
            "universe_antibody_count": 2,
            "universe_antigen_count": 2,
            "training_source_row_count": 1,
            "validation_source_row_count": 1,
            "quarantined_source_row_count": 0,
        },
    }


def _scientific(value: dict) -> dict:
    material = dict(value)
    material["scientific_payload_sha256"] = __import__("hashlib").sha256(
        json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
    ).hexdigest()
    return material


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    contract = tmp_path / "contract.json"
    assignment = tmp_path / "assignment.json"
    audit = tmp_path / "audit.json"
    contract.write_text(json.dumps({"format_version": "strict_partner_retrieval_contract_v1"}))
    outer = []
    for oi in range(3):
        fold = _fold(f"o{oi}")
        fold["inner_level"] = {"folds": [_fold(f"o{oi}i{ii}") for ii in range(3)]}
        outer.append(fold)
    avalue=_scientific({
        "artifact_role": "externally_pinned_core_feasibility_replay",
        "stage": "completed_externally_pinned_core_feasibility_replay",
        "core_feasibility_constraints_passed": True,
        "construction_invariants_passed": True,
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "authorized_for_nested_model_selection": False,
        "split_domain_gate0_requirement_satisfied": False,
        "workflow_identity": {
            "joint_assignment_mode": "externally_pinned_replay",
            "provisional_attempt_id": None,
            "pinned_replay_id": "replay_001",
            "source_provisional_attempt_id": "provisional_001",
        },
        "construction": {"outer_level": {"folds": outer}},
    })
    assignment.write_text(json.dumps(avalue,sort_keys=True))
    asha=_sha(assignment)
    uvalue=_scientific({
        "valid_audit_execution": True,
        "core_feasibility_precursor_passed": True,
        "split_domain_gate0_requirement_satisfied": False,
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "authorized_for_nested_model_selection": False,
        "approval_status": "core_feasibility_precursor_passed_non_authorizing",
        "independent_gates": {"gate_a": True,"gate_b": True},
        "boundary_count": 12,
        "exhaustive_accounting": {"threshold_crossing_edge_count": 0},
        "data_firewall": {"validation_or_test_labels_predictions_or_scores_read": 0},
        "inputs": {"assignment": {"sha256": asha}},
        "assignment_scientific_payload_sha256": avalue["scientific_payload_sha256"],
        "workflow_identity": {
            "joint_assignment_mode": "externally_pinned_replay",
            "pinned_replay_id": "replay_001",
            "source_provisional_attempt_id": "provisional_001",
        },
    })
    audit.write_text(json.dumps(uvalue,sort_keys=True))
    return contract, assignment, audit


def _run(tmp_path: Path, gate0: Path | None = None) -> dict:
    contract, assignment, audit = _inputs(tmp_path)
    output = tmp_path / "frozen.json"
    command = [sys.executable, str(SCRIPT), "--contract", str(contract), "--assignment", str(assignment), "--audit", str(audit), "--output", str(output)]
    if gate0 is not None:
        command.extend(["--gate0-authority", str(gate0)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(output.read_text())


def test_precursor_only_never_authorizes_training_or_outer(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result["status"] == "PREPARED_SPLIT_NOT_AUTHORIZED"
    assert result["model_execution_authorized"] is False
    assert result["training_and_model_selection_authorized"] is False
    assert result["outer_evaluation_authorized"] is False
    assert result["h0_execution_authorized"] is True
    assert result["fold_cache_materialization_authorized"] is True
    assert result["fold_count"] == 12


def test_valid_scoped_gate0_authorizes_training_but_not_outer(tmp_path: Path) -> None:
    contract, assignment, audit = _inputs(tmp_path)
    gate0 = tmp_path / "gate0.json"
    gate0.write_text(json.dumps({
        "format_version": "strict_partner_retrieval_gate0_authority_v1",
        "status": "AUTHORIZED_FOR_STRICT_PARTNER_RETRIEVAL_TRAINING",
        "scope": "authority_train_development_only",
        "authorized_for_training": True,
        "authorized_for_nested_model_selection": True,
        "outer_evaluation_authorized": False,
        "full_v0_confirmatory_gate0_claimed": False,
        "sealed_authority_validation_access": False,
        "test_access": False,
        "ood_access": False,
        "assignment_sha256": _sha(assignment),
        "independent_audit_sha256": _sha(audit),
        "authority_sha256": "3" * 64,
    }))
    output = tmp_path / "frozen.json"
    subprocess.run([
        sys.executable, str(SCRIPT), "--contract", str(contract), "--assignment", str(assignment),
        "--audit", str(audit), "--gate0-authority", str(gate0), "--output", str(output)
    ], check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["status"] == "FROZEN_SPLIT_AUTHORIZED"
    assert result["model_execution_authorized"] is True
    assert result["training_and_model_selection_authorized"] is True
    assert result["h0_execution_authorized"] is True
    assert result["outer_evaluation_authorized"] is False


def test_audit_assignment_binding_is_required(tmp_path: Path) -> None:
    contract, assignment, audit = _inputs(tmp_path)
    value=json.loads(audit.read_text())
    value["inputs"]["assignment"]["sha256"]="0"*64
    material=dict(value);material.pop("scientific_payload_sha256")
    value["scientific_payload_sha256"]=_scientific(material)["scientific_payload_sha256"]
    audit.write_text(json.dumps(value,sort_keys=True))
    result=subprocess.run([sys.executable,str(SCRIPT),"--contract",str(contract),"--assignment",str(assignment),"--audit",str(audit),"--output",str(tmp_path/"bad.json")],capture_output=True,text=True)
    assert result.returncode != 0
    assert "bind assignment" in result.stderr
