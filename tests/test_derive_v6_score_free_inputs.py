"""Tests for the create-only V6-A physical input derivation controller."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from interaction_first.score_free_authority import (
    AuthorityCounts,
    build_score_free_authority,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "derive_v6_score_free_inputs.py"
SPEC = importlib.util.spec_from_file_location(
    "derive_v6_score_free_inputs",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _frame(sequence: str) -> bytes:
    encoded = sequence.encode("ascii")
    return str(len(encoded)).encode("ascii") + b":" + encoded


def _antibody_id(vh_sequence: str, vl_sequence: str) -> str:
    return hashlib.sha256(
        b"antibody-v1\0" + _frame(vh_sequence) + _frame(vl_sequence)
    ).hexdigest()


def _manifest_row(
    instance: str,
    vh_sequence: str,
    vl_sequence: str,
    antigen_sequence: str,
) -> str:
    values = {column: "" for column in SCRIPT.SOURCE_MANIFEST_COLUMNS}
    values.update(
        {
            "INSTANCE": instance,
            "PDB_ID": instance[:4],
            "split": "train",
            "vh_seq": vh_sequence,
            "vl_seq": vl_sequence,
            "ag_seq": antigen_sequence,
            "vh_len": str(len(vh_sequence)),
            "vl_len": str(len(vl_sequence)),
            "ag_len": str(len(antigen_sequence)),
        }
    )
    return ",".join(values[column] for column in SCRIPT.SOURCE_MANIFEST_COLUMNS)


def _fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[bytes, bytes, bytes]:
    source_ids = ("row0", "row1")
    membership = SCRIPT._membership_sha256(source_ids)
    monkeypatch.setattr(
        SCRIPT,
        "EXPECTED_COUNTS",
        {
            "source_rows": 2,
            "exact_pairs": 2,
            "antibody_entities": 2,
            "antigen_entities": 2,
        },
    )
    monkeypatch.setattr(
        SCRIPT,
        "EXPECTED_AUTHORITY_MEMBERSHIP_SHA256",
        membership,
    )
    antibody_edge = tuple(
        sorted(
            (
                _antibody_id("AAAAA", ""),
                _antibody_id("CCCCC", "DDD"),
            )
        )
    )
    graphs = {view: [] for view in SCRIPT.VIEWS}
    graphs["vh"] = [antibody_edge]
    assignment = {
        "format_version": SCRIPT.ASSIGNMENT_FORMAT,
        "scientific_payload_sha256": "a" * 64,
        "authority": {
            "authority_train_membership_sha256": membership,
        },
        "canonical_inventory": {
            "authority_train_source_row_count": 2,
            "unique_exact_pair_count": 2,
            "unique_antibody_entity_count": 2,
            "unique_antigen_entity_count": 2,
        },
        "data_firewall": {
            "authorized_source_split": "train",
            "authorized_rows_materialized": 2,
            "sealed_rows_materialized": 0,
            "validation_sequence_fields_parsed": 0,
            "test_sequence_fields_parsed": 0,
        },
        "construction": {
            "source_instance_ids": ["row1", "row0"],
        },
        "exact_direct_graphs": {
            view: {
                "direct_entity_edges": [
                    {
                        "left_entity_id": left,
                        "right_entity_id": right,
                    }
                    for left, right in graphs[view]
                ],
                "lifted_direct_entity_edge_count": len(graphs[view]),
                "direct_entity_edge_membership_sha256": (
                    SCRIPT._edge_membership_sha256(graphs[view])
                ),
            }
            for view in SCRIPT.VIEWS
        },
        "outer_folds": {
            "must_not_be_reused": {
                "assignment": ["row0"],
                "contact_density": 0.99,
                "model_score": 1000,
            }
        },
    }
    assignment_bytes = _canonical(assignment)
    audit = {
        "format_version": SCRIPT.AUDIT_FORMAT,
        "valid_audit_execution": True,
        "inputs": {
            "assignment": {
                "sha256": hashlib.sha256(assignment_bytes).hexdigest(),
            }
        },
        "assignment_scientific_payload_sha256": "a" * 64,
        "independent_gates": {
            "authority_train_only_firewall_passed": True,
            "exhaustive_alignment_accounting_passed": True,
            "independent_inventory_exactly_matches_assignment": True,
        },
    }
    header = ",".join(SCRIPT.SOURCE_MANIFEST_COLUMNS)
    manifest = (
        header
        + "\n"
        + _manifest_row("row1", "CCCCC", "DDD", "KKKKK")
        + "\n"
        + "sealed-row,this,is,intentionally,not,a,valid,CSV,row\n"
        + _manifest_row("row0", "AAAAA", "", "IIIII")
        + "\n"
    ).encode("ascii")
    return assignment_bytes, _canonical(audit), manifest


def test_derivation_uses_only_authority_membership_and_exact_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignment, audit, manifest = _fixture(monkeypatch)
    artifacts = SCRIPT.derive_inputs(
        assignment_bytes=assignment,
        audit_bytes=audit,
        manifest_bytes=manifest,
        implementation_source_sha256=hashlib.sha256(
            SCRIPT_PATH.read_bytes()
        ).hexdigest(),
    )

    authority = json.loads(artifacts["train_authority_allowlist.json"])
    assert authority == {
        "authorized_split": "train",
        "format_version": SCRIPT.AUTHORITY_FORMAT,
        "instance_ids": ["row0", "row1"],
    }
    projected_manifest = artifacts["train_canonical_manifest.csv"]
    assert projected_manifest == (
        b"INSTANCE,vh_seq,vl_seq,ag_seq\n"
        b"row0,AAAAA,,IIIII\n"
        b"row1,CCCCC,DDD,KKKKK\n"
    )
    receipt = artifacts["receipt"]
    assert receipt["historical_fold_assignment_reused"] is False
    assert receipt["historical_fold_feasibility_claim_reused"] is False
    assert receipt["manifest_firewall"] == {
        "physical_source_rows_scanned": 3,
        "authorized_sequence_rows_parsed": 2,
        "non_authorized_sequence_rows_parsed": 0,
    }
    asserted_receipt_hash = receipt["receipt_sha256"]
    unhashed_receipt = dict(receipt)
    del unhashed_receipt["receipt_sha256"]
    assert hashlib.sha256(_canonical(unhashed_receipt)).hexdigest() == asserted_receipt_hash

    bundle = build_score_free_authority(
        authority_bytes=artifacts["train_authority_allowlist.json"],
        manifest_bytes=projected_manifest,
        edge_ledger_bytes={
            view: artifacts[f"edge_ledger_{view}.json"]
            for view in SCRIPT.VIEWS
        },
        expected_counts=AuthorityCounts(2, 2, 2, 2),
    )
    assert bundle.canonical_payload()["counts"] == {
        "source_rows": 2,
        "exact_pairs": 2,
        "antibody_entities": 2,
        "antigen_entities": 2,
    }
    assert bundle.authority_membership_sha256 == SCRIPT._membership_sha256(
        ("row0", "row1")
    )


@pytest.mark.parametrize(
    ("target", "expected_message"),
    (
        ("audit_binding", "binds a different assignment"),
        ("audit_gate", "raw-authority gates did not pass"),
        ("graph_digest", "edge digest differs"),
        ("authority_membership", "authority differs"),
    ),
)
def test_historical_evidence_tampering_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    expected_message: str,
) -> None:
    assignment_bytes, audit_bytes, manifest = _fixture(monkeypatch)
    assignment = json.loads(assignment_bytes)
    audit = json.loads(audit_bytes)
    if target == "audit_binding":
        audit["inputs"]["assignment"]["sha256"] = "f" * 64
    elif target == "audit_gate":
        audit["independent_gates"][
            "exhaustive_alignment_accounting_passed"
        ] = False
    elif target == "graph_digest":
        assignment["exact_direct_graphs"]["vh"][
            "direct_entity_edge_membership_sha256"
        ] = "f" * 64
        assignment_bytes = _canonical(assignment)
        audit["inputs"]["assignment"]["sha256"] = hashlib.sha256(
            assignment_bytes
        ).hexdigest()
    elif target == "authority_membership":
        assignment["authority"]["authority_train_membership_sha256"] = "f" * 64
        assignment_bytes = _canonical(assignment)
        audit["inputs"]["assignment"]["sha256"] = hashlib.sha256(
            assignment_bytes
        ).hexdigest()
    else:
        raise AssertionError(target)
    with pytest.raises(SCRIPT.InputDerivationError, match=expected_message):
        SCRIPT.derive_inputs(
            assignment_bytes=assignment_bytes,
            audit_bytes=_canonical(audit),
            manifest_bytes=manifest,
            implementation_source_sha256=hashlib.sha256(
                SCRIPT_PATH.read_bytes()
            ).hexdigest(),
        )


def test_duplicate_or_missing_authority_rows_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignment, audit, manifest = _fixture(monkeypatch)
    duplicate = manifest + _manifest_row("row0", "AAAAA", "", "IIIII").encode(
        "ascii"
    ) + b"\n"
    with pytest.raises(SCRIPT.InputDerivationError, match="duplicate identifier"):
        SCRIPT.derive_inputs(
            assignment_bytes=assignment,
            audit_bytes=audit,
            manifest_bytes=duplicate,
            implementation_source_sha256=hashlib.sha256(
                SCRIPT_PATH.read_bytes()
            ).hexdigest(),
        )
    missing = b"\n".join(
        line for line in manifest.split(b"\n") if not line.startswith(b"row1,")
    )
    with pytest.raises(SCRIPT.InputDerivationError, match="not every authorized ID"):
        SCRIPT.derive_inputs(
            assignment_bytes=assignment,
            audit_bytes=audit,
            manifest_bytes=missing,
            implementation_source_sha256=hashlib.sha256(
                SCRIPT_PATH.read_bytes()
            ).hexdigest(),
        )


def test_create_file_is_exclusive(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    SCRIPT._create_file(destination, b"first\n")
    assert destination.read_bytes() == b"first\n"
    with pytest.raises(FileExistsError):
        SCRIPT._create_file(destination, b"second\n")
    assert destination.read_bytes() == b"first\n"


def test_authenticated_reader_accepts_hardlinks_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    payload = b"externally anchored\n"
    original = tmp_path / "original.json"
    hardlink = tmp_path / "hardlink.json"
    symlink = tmp_path / "symlink.json"
    original.write_bytes(payload)
    hardlink.hardlink_to(original)
    symlink.symlink_to(original)
    expected = hashlib.sha256(payload).hexdigest()
    assert SCRIPT._read_regular_file_nofollow(
        hardlink,
        expected,
        "hard-linked input",
    ) == payload
    with pytest.raises(OSError):
        SCRIPT._read_regular_file_nofollow(
            symlink,
            expected,
            "symbolic-linked input",
        )
