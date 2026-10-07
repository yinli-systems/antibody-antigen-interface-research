"""Fail-closed authority gates for strict partner retrieval."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(8*1024*1024),b""):h.update(c)
    return h.hexdigest()


def load_authorized_split(assignment_path:Path,audit_path:Path)->tuple[dict[str,Any],dict[str,Any]]:
    assignment=json.loads(assignment_path.read_text());audit=json.loads(audit_path.read_text())
    if assignment.get("authorized_for_training") is not True:
        raise PermissionError("split assignment is not authorized for training")
    if audit.get("authorized_for_training") is not True:
        raise PermissionError("independent audit did not authorize training")
    if audit.get("authorized_for_nested_model_selection") is not True:
        raise PermissionError("nested model-selection authorization absent")
    if audit.get("valid_audit_execution") is not True:
        raise PermissionError("audit execution invalid")
    gates=audit.get("independent_gates",{})
    required=(
        "all_four_views_completed_for_every_boundary",
        "all_outer_and_inner_partition_invariants_passed",
        "all_twelve_boundaries_present",
        "authority_train_only_firewall_passed",
        "builder_construction_gates_were_passed",
        "exhaustive_alignment_accounting_passed",
        "independent_inventory_exactly_matches_assignment",
        "zero_direct_80_80_crossing_edges_in_every_boundary_and_view",
    )
    if any(gates.get(k) is not True for k in required):
        raise PermissionError("one or more strict split gates failed")
    auth=audit.get("authority",{})
    if auth.get("source_pool_split")!="train" or set(auth.get("sealed_splits",[]))!={"validation","test"}:
        raise PermissionError("authority/sealed split boundary differs")
    if audit.get("data_firewall",{}).get("validation_or_test_labels_read")!=0:
        raise PermissionError("sealed labels were read")
    if audit.get("claim_boundary") not in {
        "foldwise direct exact four-view 80/80 C3/S2-like development; not global S2, transitive lineage independence, or test evaluation",
        "foldwise direct-80/80 four-view homology-separated C3, S2-like development",
    }:
        raise PermissionError("claim boundary differs")
    a_hash=assignment.get("scientific_payload_sha256")
    audited=audit.get("assignment_scientific_payload_sha256")
    if not a_hash or a_hash!=audited:
        raise PermissionError("assignment/audit scientific payload mismatch")
    return assignment,audit


def require_h0_cache_authority(cache:dict[str,Any])->None:
    if cache.get("h0_execution_authorized") is not True:
        raise PermissionError("H0 execution not authorized")
    if cache.get("test_access") is not False or cache.get("ood_access") is not False:
        raise PermissionError("sealed access in H0 cache")
    if cache.get("nonpositive_semantics") not in {None,"unlabeled_ranking_candidates"}:
        raise PermissionError("candidate semantics differ")
