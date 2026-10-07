#!/usr/bin/env python3
"""Issue scoped authority-train development training permission after audited replay.

This is deliberately not a confirmatory/full-V0 split Gate 0 and can never
unlock sealed authority-validation, test, OOD, or outer evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORMAT = "strict_partner_retrieval_gate0_authority_v1"
EXPECTED_MEMBERSHIP = "442095581ae9bf3c7440c1a818612298b5f81e5a62ef46f431e0867b9c8b466b"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_scientific(value: dict[str, Any], label: str) -> str:
    claimed = value.get("scientific_payload_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError(f"{label} scientific hash missing")
    material = dict(value)
    material.pop("scientific_payload_sha256", None)
    actual = canonical_sha(material)
    if actual != claimed:
        raise ValueError(f"{label} scientific hash mismatch")
    return actual


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--assignment", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--prepared-split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    contract = load(args.contract)
    assignment = load(args.assignment)
    audit = load(args.audit)
    prepared = load(args.prepared_split)

    if contract.get("format_version") != "strict_partner_retrieval_contract_v1":
        raise ValueError("contract format")
    authority = contract.get("authority", {})
    if authority.get("intended_use") != "authority_train_development_only":
        raise ValueError("contract is not development-only authority-train")
    if authority.get("membership_sha256") != EXPECTED_MEMBERSHIP:
        raise ValueError("contract authority membership mismatch")
    for key in ("authority_validation_access", "frozen_test_access", "ood_access"):
        if authority.get(key) is not False:
            raise ValueError(f"contract unexpectedly enables {key}")

    if assignment.get("artifact_role") != "externally_pinned_core_feasibility_replay":
        raise ValueError("assignment role")
    if assignment.get("stage") != "completed_externally_pinned_core_feasibility_replay":
        raise ValueError("assignment stage")
    assignment_scientific = verify_scientific(assignment, "assignment")
    for key in (
        "authorized_for_training",
        "authorized_for_model_selection",
        "authorized_for_nested_model_selection",
        "split_domain_gate0_requirement_satisfied",
    ):
        if assignment.get(key) is not False:
            raise ValueError(f"assignment falsely enables {key}")
    assignment_authority = assignment.get("authority", {})
    if assignment_authority.get("authority_train_membership_sha256") != EXPECTED_MEMBERSHIP:
        raise ValueError("assignment authority membership mismatch")
    if assignment_authority.get("validation_or_test_membership_fields_present") is not False:
        raise ValueError("assignment contains sealed membership fields")

    if audit.get("valid_audit_execution") is not True:
        raise ValueError("audit execution invalid")
    if audit.get("core_feasibility_precursor_passed") is not True:
        raise ValueError("audit precursor failed")
    audit_scientific = verify_scientific(audit, "audit")
    gates = audit.get("independent_gates")
    if not isinstance(gates, dict) or not gates or not all(v is True for v in gates.values()):
        raise ValueError("independent audit gates are not all true")
    if audit.get("boundary_count") != 12:
        raise ValueError("audit boundary count")
    if audit.get("exhaustive_accounting", {}).get("threshold_crossing_edge_count") != 0:
        raise ValueError("audit threshold crossings")
    if audit.get("data_firewall", {}).get(
        "validation_or_test_labels_predictions_or_scores_read"
    ) != 0:
        raise ValueError("audit read forbidden data")
    for key in (
        "authorized_for_training",
        "authorized_for_model_selection",
        "authorized_for_nested_model_selection",
        "split_domain_gate0_requirement_satisfied",
    ):
        if audit.get(key) is not False:
            raise ValueError(f"audit falsely enables {key}")

    assignment_sha = sha256(args.assignment)
    audit_sha = sha256(args.audit)
    if audit.get("inputs", {}).get("assignment", {}).get("sha256") != assignment_sha:
        raise ValueError("audit assignment byte binding")
    if audit.get("assignment_scientific_payload_sha256") != assignment_scientific:
        raise ValueError("audit assignment scientific binding")

    if prepared.get("format_version") != "strict_partner_retrieval_frozen_split_v1":
        raise ValueError("prepared split format")
    if prepared.get("status") != "PREPARED_SPLIT_NOT_AUTHORIZED":
        raise ValueError("expected non-authorized prepared split")
    if prepared.get("training_and_model_selection_authorized") is not False:
        raise ValueError("prepared split unexpectedly authorizes training")
    if prepared.get("h0_execution_authorized") is not True:
        raise ValueError("prepared split did not pass H0 audit gate")
    if prepared.get("outer_evaluation_authorized") is not False:
        raise ValueError("prepared split enables outer evaluation")
    if prepared.get("assignment", {}).get("sha256") != assignment_sha:
        raise ValueError("prepared split assignment binding")
    if prepared.get("independent_audit", {}).get("sha256") != audit_sha:
        raise ValueError("prepared split audit binding")
    if prepared.get("fold_count") != 12 or len(prepared.get("libraries", [])) != 12:
        raise ValueError("prepared split does not expose 12 libraries")
    for library in prepared["libraries"]:
        if library.get("semantics", {}).get("nonpositive_cross_pairs") != "unlabeled_ranking_candidates":
            raise ValueError("candidate semantics mismatch")
        if library.get("semantics", {}).get("measured_nonbinder_claim") is not False:
            raise ValueError("library makes nonbinder claim")
        for direction in ("antibody_to_antigen", "antigen_to_antibody"):
            queries = library.get(direction, {}).get("queries")
            candidates = library.get(direction, {}).get("candidate_ids")
            if not isinstance(queries, list) or not queries:
                raise ValueError("library has no queries")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("library has no candidates")
            for query in queries:
                if query.get("positive_count", 0) < 1:
                    raise ValueError("zero-positive retrieval query")

    payload = {
        "format_version": FORMAT,
        "status": "AUTHORIZED_FOR_STRICT_PARTNER_RETRIEVAL_TRAINING",
        "scope": "authority_train_development_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_for_training": True,
        "authorized_for_nested_model_selection": True,
        "outer_evaluation_authorized": False,
        "full_v0_confirmatory_gate0_claimed": False,
        "sealed_authority_validation_access": False,
        "test_access": False,
        "ood_access": False,
        "assignment_sha256": assignment_sha,
        "assignment_scientific_payload_sha256": assignment_scientific,
        "independent_audit_sha256": audit_sha,
        "independent_audit_scientific_payload_sha256": audit_scientific,
        "prepared_split_sha256": sha256(args.prepared_split),
        "contract_sha256": sha256(args.contract),
        "authority_train_membership_sha256": EXPECTED_MEMBERSHIP,
        "fold_count": 12,
        "candidate_semantics": "unlabeled_ranking_candidates",
        "permitted_models": ["D0", "L1", "R1", "R2"],
        "permitted_model_seeds": [17, 42, 73],
        "prohibited_actions": [
            "sealed_authority_validation_access",
            "test_access",
            "ood_access",
            "affinity_claim_from_retrieval_score",
            "biological_nonbinder_claim_for_unobserved_pair",
            "outer_evaluation_without_separate_authority",
        ],
        "limitations": [
            "Development-only authority-train experiment; not a full V0 confirmatory split Gate 0.",
            "Direct-neighborhood exclusion is C3/S2-like and is not global DataSAIL S2.",
            "Outer evaluation remains sealed until a separate model-selection authority is issued.",
        ],
    }
    payload["authority_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "file_sha256": sha256(args.output),
        "authority_sha256": payload["authority_sha256"],
        "scope": payload["scope"],
        "outer_evaluation_authorized": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
