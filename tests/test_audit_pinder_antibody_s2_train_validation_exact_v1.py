from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SOURCE_DIR = (
    Path(__file__).parents[1]
    / "remote_tools"
    / "abf_paircontact_v1"
    / "pinder_scale_v1"
)
PARENT_SOURCE = SOURCE_DIR / "audit_pinder_antibody_s2_exact_four_view_v1.py"
PARENT_SPEC = importlib.util.spec_from_file_location(
    "audit_pinder_antibody_s2_exact_four_view_v1", PARENT_SOURCE
)
assert PARENT_SPEC is not None and PARENT_SPEC.loader is not None
PARENT = importlib.util.module_from_spec(PARENT_SPEC)
sys.modules[PARENT_SPEC.name] = PARENT
PARENT_SPEC.loader.exec_module(PARENT)

SOURCE = SOURCE_DIR / "audit_pinder_antibody_s2_train_validation_exact_v1.py"
SPEC = importlib.util.spec_from_file_location("pinder_s2_tv_audit", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(system_id: str, pdb_id: str, vh: str, vl: str, antigen: str):
    return PARENT.SourceRow(system_id, pdb_id, vh, vl, antigen)


def ledger(bucket, rows):
    systems = sorted(value.system_id for value in rows)
    pairs = sorted({value.exact_pair_id for value in rows})
    pdbs = sorted({value.pdb_id for value in rows})
    return {
        "bucket": bucket,
        "system_count": len(systems),
        "unique_exact_pair_count": len(pairs),
        "unique_pdb_count": len(pdbs),
        "system_ids": systems,
        "system_membership_sha256": PARENT.membership_sha256(systems),
        "exact_pair_ids": pairs,
        "exact_pair_membership_sha256": PARENT.membership_sha256(pairs),
        "pdb_ids": pdbs,
        "pdb_membership_sha256": PARENT.membership_sha256(pdbs),
    }


def assignment(train, validation, discarded):
    payload = {
        "format_version": MODULE.ASSIGNMENT_FORMAT_VERSION,
        "status": "PASS_PROVISIONAL_TRAIN_VALIDATION_S2_80_ASSIGNMENT",
        "approval_status": "UNAPPROVED_PENDING_EXACT_FOUR_VIEW_ZERO_CROSSING_AUDIT",
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "bucket_ledgers": {
            "train": ledger("train", train),
            "validation": ledger("validation", validation),
            "discarded": ledger("discarded", discarded),
        },
        "checks": {
            "validation_rows_read": False,
            "test_rows_read": False,
            "task_labels_read": False,
            "structures_read": False,
            "predictions_or_scores_read": False,
        },
    }
    payload["scientific_payload_sha256"] = PARENT.canonical_payload_sha256(payload)
    return payload


def test_validates_ledgers_and_builds_two_bucket_inventories():
    rows = [
        row("a", "1aaa", "AAAA", "CCCC", "DDDD"),
        row("b", "2bbb", "EEEE", "FFFF", "GGGG"),
        row("c", "3ccc", "HHHH", "", "IIII"),
    ]
    payload = assignment([rows[0]], [rows[1]], [rows[2]])
    buckets, checks = MODULE.validate_assignment(
        payload, rows, payload["scientific_payload_sha256"]
    )
    assert checks["all_source_systems_accounted_once"]
    inventories = MODULE.build_inventories(rows, buckets)
    assert [len(values) for values in inventories["vh"]] == [1, 1]
    assert [len(values) for values in inventories["vl"]] == [1, 1]


def test_rejects_pdb_split_across_train_and_validation():
    rows = [
        row("a", "1aaa", "AAAA", "CCCC", "DDDD"),
        row("b", "1aaa", "EEEE", "FFFF", "GGGG"),
    ]
    payload = assignment([rows[0]], [rows[1]], [])
    with pytest.raises(MODULE.AuditError, match="PDB is split"):
        MODULE.validate_assignment(payload, rows, payload["scientific_payload_sha256"])


def test_rejects_exact_sequence_overlap_between_train_and_validation():
    rows = [
        row("a", "1aaa", "AAAA", "CCCC", "DDDD"),
        row("b", "2bbb", "AAAA", "FFFF", "GGGG"),
    ]
    payload = assignment([rows[0]], [rows[1]], [])
    buckets, _ = MODULE.validate_assignment(
        payload, rows, payload["scientific_payload_sha256"]
    )
    with pytest.raises(MODULE.AuditError, match="exact sequence overlap.*vh"):
        MODULE.build_inventories(rows, buckets)
