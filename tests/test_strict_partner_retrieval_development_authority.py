from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "extensions/strict_partner_retrieval_v1/issue_development_training_authority.py"
MEMBERSHIP = "442095581ae9bf3c7440c1a818612298b5f81e5a62ef46f431e0867b9c8b466b"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: dict) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")).hexdigest()


def scientific(value: dict) -> dict:
    result=dict(value);result["scientific_payload_sha256"]=canonical(value);return result


def library(label: str) -> dict:
    return {
        "label": label,
        "semantics": {"nonpositive_cross_pairs": "unlabeled_ranking_candidates", "measured_nonbinder_claim": False},
        "antibody_to_antigen": {"candidate_ids": ["g"], "queries": [{"query_id": "a", "positive_count": 1}]},
        "antigen_to_antibody": {"candidate_ids": ["a"], "queries": [{"query_id": "g", "positive_count": 1}]},
    }


def fixture(tmp_path: Path) -> list[str]:
    contract=tmp_path/"contract.json";assignment=tmp_path/"assignment.json";audit=tmp_path/"audit.json";prepared=tmp_path/"prepared.json";output=tmp_path/"authority.json"
    contract.write_text(json.dumps({"format_version":"strict_partner_retrieval_contract_v1","authority":{"intended_use":"authority_train_development_only","membership_sha256":MEMBERSHIP,"authority_validation_access":False,"frozen_test_access":False,"ood_access":False}}))
    av=scientific({"artifact_role":"externally_pinned_core_feasibility_replay","stage":"completed_externally_pinned_core_feasibility_replay","authorized_for_training":False,"authorized_for_model_selection":False,"authorized_for_nested_model_selection":False,"split_domain_gate0_requirement_satisfied":False,"authority":{"authority_train_membership_sha256":MEMBERSHIP,"validation_or_test_membership_fields_present":False}})
    assignment.write_text(json.dumps(av,sort_keys=True));asha=sha(assignment)
    uv=scientific({"valid_audit_execution":True,"core_feasibility_precursor_passed":True,"independent_gates":{"a":True,"b":True},"boundary_count":12,"exhaustive_accounting":{"threshold_crossing_edge_count":0},"data_firewall":{"validation_or_test_labels_predictions_or_scores_read":0},"authorized_for_training":False,"authorized_for_model_selection":False,"authorized_for_nested_model_selection":False,"split_domain_gate0_requirement_satisfied":False,"inputs":{"assignment":{"sha256":asha}},"assignment_scientific_payload_sha256":av["scientific_payload_sha256"]})
    audit.write_text(json.dumps(uv,sort_keys=True));audsha=sha(audit)
    libs=[library(f"outer_{i}") for i in range(3)]+[library(f"outer_{i}_inner_{j}") for i in range(3) for j in range(3)]
    prepared.write_text(json.dumps({"format_version":"strict_partner_retrieval_frozen_split_v1","status":"PREPARED_SPLIT_NOT_AUTHORIZED","training_and_model_selection_authorized":False,"h0_execution_authorized":True,"outer_evaluation_authorized":False,"assignment":{"sha256":asha},"independent_audit":{"sha256":audsha},"fold_count":12,"libraries":libs}))
    return [sys.executable,str(SCRIPT),"--contract",str(contract),"--assignment",str(assignment),"--audit",str(audit),"--prepared-split",str(prepared),"--output",str(output)]


def test_valid_scoped_authority_never_unlocks_outer(tmp_path: Path) -> None:
    command=fixture(tmp_path);subprocess.run(command,check=True,capture_output=True,text=True);value=json.loads((tmp_path/"authority.json").read_text())
    assert value["scope"]=="authority_train_development_only"
    assert value["authorized_for_training"] is True
    assert value["authorized_for_nested_model_selection"] is True
    assert value["outer_evaluation_authorized"] is False
    assert value["full_v0_confirmatory_gate0_claimed"] is False
    material=dict(value);claimed=material.pop("authority_sha256");assert canonical(material)==claimed


def test_nonbinder_semantics_fail_closed(tmp_path: Path) -> None:
    command=fixture(tmp_path);prepared=tmp_path/"prepared.json";value=json.loads(prepared.read_text());value["libraries"][0]["semantics"]["measured_nonbinder_claim"]=True;prepared.write_text(json.dumps(value))
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode != 0
    assert "nonbinder" in result.stderr.lower()
