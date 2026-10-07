from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SOURCE = (
    Path(__file__).parents[1]
    / "remote_tools"
    / "abf_paircontact_v1"
    / "pinder_scale_v1"
    / "audit_pinder_antibody_s2_exact_four_view_v1.py"
)
SPEC = importlib.util.spec_from_file_location("pinder_s2_exact_audit", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(system_id: str, pdb_id: str, vh: str, vl: str, antigen: str):
    return MODULE.SourceRow(system_id, pdb_id, vh, vl, antigen)


def fold_ledger(fold_index: int, rows):
    system_ids = sorted(value.system_id for value in rows)
    pair_ids = sorted({value.exact_pair_id for value in rows})
    pdb_ids = sorted({value.pdb_id for value in rows})
    return {
        "fold_index": fold_index,
        "retained_system_count": len(system_ids),
        "retained_unique_exact_pair_count": len(pair_ids),
        "retained_unique_pdb_count": len(pdb_ids),
        "retained_system_ids": system_ids,
        "retained_system_membership_sha256": MODULE.membership_sha256(system_ids),
        "retained_exact_pair_ids": pair_ids,
        "retained_exact_pair_membership_sha256": MODULE.membership_sha256(pair_ids),
        "retained_pdb_ids": pdb_ids,
        "retained_pdb_membership_sha256": MODULE.membership_sha256(pdb_ids),
    }


def discarded_ledger(rows):
    system_ids = sorted(value.system_id for value in rows)
    pair_ids = sorted({value.exact_pair_id for value in rows})
    pdb_ids = sorted({value.pdb_id for value in rows})
    return {
        "discard_policy": "discard_complete_exact_pair_and_pdb_connected_component",
        "discarded_system_count": len(system_ids),
        "discarded_unique_exact_pair_count": len(pair_ids),
        "discarded_unique_pdb_count": len(pdb_ids),
        "discarded_system_ids": system_ids,
        "discarded_system_membership_sha256": MODULE.membership_sha256(system_ids),
        "discarded_exact_pair_ids": pair_ids,
        "discarded_exact_pair_membership_sha256": MODULE.membership_sha256(pair_ids),
        "discarded_pdb_ids": pdb_ids,
        "discarded_pdb_membership_sha256": MODULE.membership_sha256(pdb_ids),
    }


def assignment(folds, discarded):
    payload = {
        "format_version": MODULE.ASSIGNMENT_FORMAT_VERSION,
        "status": "PASS_PROVISIONAL_PDB_AND_EXACT_PAIR_INDIVISIBLE_S2_80_ASSIGNMENT",
        "approval_status": "UNAPPROVED_PENDING_EXACT_FOUR_VIEW_NEEDLEMAN_WUNSCH_AUDIT",
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "checks": {"validation_rows_read": False, "test_rows_read": False},
        "input_counts": {"canonical_systems": sum(len(values) for values in folds) + len(discarded)},
        "retained_folds": [fold_ledger(index, values) for index, values in enumerate(folds)],
        "discarded_ledger": discarded_ledger(discarded),
    }
    payload["scientific_payload_sha256"] = MODULE.canonical_payload_sha256(payload)
    return payload


def source_rows():
    return [
        row("a", "1aaa", "AAAA", "CCCC", "DDDD"),
        row("b", "2bbb", "EEEE", "FFFF", "GGGG"),
        row("c", "3ccc", "HHHH", "IIII", "KKKK"),
        row("d", "4ddd", "LLLL", "", "MMMM"),
    ]


def test_validates_exhaustive_indivisible_ledgers_and_builds_four_views():
    rows = source_rows()
    payload = assignment([[rows[0]], [rows[1]], [rows[2]]], [rows[3]])
    fold_by_system, discarded, checks = MODULE.validate_assignment(
        payload, rows, payload["scientific_payload_sha256"]
    )
    assert fold_by_system == {"a": 0, "b": 1, "c": 2}
    assert discarded == {"d"}
    assert checks["exact_pairs_indivisible_across_retained_and_discarded_buckets"]
    inventories = MODULE.build_fold_inventories(rows, fold_by_system)
    assert [len(values) for values in inventories["vh"]] == [1, 1, 1]
    assert [len(values) for values in inventories["vl"]] == [1, 1, 1]
    assert [len(values) for values in inventories["concatenated_antibody"]] == [1, 1, 1]
    assert [len(values) for values in inventories["antigen"]] == [1, 1, 1]


def test_rejects_pdb_split_between_retained_and_discarded():
    rows = source_rows()
    rows[3] = row("d", "1aaa", "LLLL", "", "MMMM")
    payload = assignment([[rows[0]], [rows[1]], [rows[2]]], [rows[3]])
    with pytest.raises(MODULE.AuditError, match="PDB is split"):
        MODULE.validate_assignment(payload, rows, payload["scientific_payload_sha256"])


def test_rejects_cross_fold_exact_sequence_overlap_in_any_view():
    rows = source_rows()
    rows[1] = row("b", "2bbb", rows[0].vh, "FFFF", "GGGG")
    payload = assignment([[rows[0]], [rows[1]], [rows[2]]], [rows[3]])
    fold_by_system, _, _ = MODULE.validate_assignment(
        payload, rows, payload["scientific_payload_sha256"]
    )
    with pytest.raises(MODULE.AuditError, match="exact sequence overlap.*vh"):
        MODULE.build_fold_inventories(rows, fold_by_system)


def test_scientific_digest_is_recomputed_not_only_trusted():
    rows = source_rows()
    payload = assignment([[rows[0]], [rows[1]], [rows[2]]], [rows[3]])
    expected = payload["scientific_payload_sha256"]
    payload["authorized_for_training"] = True
    with pytest.raises(MODULE.AuditError, match="scientific payload SHA"):
        MODULE.validate_assignment(payload, rows, expected)


def test_threshold_uses_inclusive_integer_cross_products():
    metrics = MODULE.AlignmentMetrics(
        score=1.0,
        identical_residue_pair_count=8,
        aligned_residue_pair_count=10,
        alignment_column_count=10,
        aligned_region_identity=0.8,
        left_reciprocal_coverage=1.0,
        right_reciprocal_coverage=0.8,
        minimum_reciprocal_coverage=0.8,
    )
    assert MODULE.meets_threshold(metrics, 10, 12)
    assert MODULE.is_safe_length_prune(79, 100)
    assert not MODULE.is_safe_length_prune(80, 100)
