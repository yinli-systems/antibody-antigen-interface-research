"""Tests for the create-only V6-C family-core audit executor."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from interaction_first import v6_c_family_core_audit as audit
from interaction_first.v6_c_family_core_registration import (
    EXPECTED_DIMENSIONS,
    EXPECTED_PARENT_REQUIRED_FIELDS,
)
from interaction_first.v6_cp_sat_formulation import DirectCPModel


def _direct_model(*, infeasible: bool) -> DirectCPModel:
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    model = cp_model.CpModel()
    value = model.new_bool_var("value")
    families = []
    model.add(value >= 0)
    families.append("domain")
    if infeasible:
        model.add(value == 0)
        families.append("zero")
        model.add(value == 1)
        families.append("one")
    return DirectCPModel(
        model=model,
        variables={("value",): value},
        authority_payload_sha256="a" * 64,
        constraint_family_counts=tuple(
            sorted(
                (family, families.count(family))
                for family in set(families)
            )
        ),
        constraint_families=tuple(families),
        formulation_sha256="b" * 64,
    )


def _make_writable(path: Path) -> None:
    if not path.exists():
        return
    for root, directories, files in os.walk(path):
        os.chmod(root, 0o700)
        for name in directories:
            os.chmod(Path(root) / name, 0o700)
        for name in files:
            os.chmod(Path(root) / name, 0o600)


def _write_receipt(
    path: Path,
    unsigned: dict[str, object],
) -> dict[str, object]:
    receipt_sha256 = audit.sha256_payload(unsigned)
    receipt = {**unsigned, "receipt_sha256": receipt_sha256}
    payload = audit.canonical_json_bytes(receipt)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "receipt_sha256": receipt_sha256,
    }


def test_projection_dimensions_count_exact_linear_nonzeros() -> None:
    direct = _direct_model(infeasible=True)
    assert audit._projection_dimensions(direct) == {
        "rows": 3,
        "columns": 1,
        "nonzeros": 3,
    }


def test_registered_receipt_requires_file_and_self_hash(
    tmp_path: Path,
) -> None:
    unsigned = {
        "terminal_status": "PROVEN_INFEASIBLE",
        "training_authorized": False,
    }
    receipt_sha256 = audit.sha256_payload(unsigned)
    receipt = {**unsigned, "receipt_sha256": receipt_sha256}
    path = tmp_path / "receipt.json"
    payload = audit.canonical_json_bytes(receipt)
    path.write_bytes(payload)
    reference = {
        "path": str(path),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "receipt_sha256": receipt_sha256,
        "required_fields": {
            "terminal_status": "PROVEN_INFEASIBLE",
            "training_authorized": False,
        },
    }
    assert audit._read_registered_receipt(
        reference,
        "fixture",
    ) == receipt
    reference["receipt_sha256"] = "c" * 64
    with pytest.raises(
        audit.V6CFamilyCoreAuditError,
        match="self hash",
    ):
        audit._read_registered_receipt(reference, "fixture")


def test_three_parent_lineage_and_nested_highs_receipt_are_bound(
    tmp_path: Path,
) -> None:
    v6_b_sha256 = "f" * 64
    primary_unsigned = {
        **EXPECTED_PARENT_REQUIRED_FIELDS["primary_cp"],
        "registration_sha256": v6_b_sha256,
    }
    primary = _write_receipt(
        tmp_path / "primary.json",
        primary_unsigned,
    )
    independent_unsigned = {
        **EXPECTED_PARENT_REQUIRED_FIELDS["independent_cp"],
        "registration_sha256": v6_b_sha256,
        "primary_result_file_sha256": primary["file_sha256"],
    }
    independent = _write_receipt(
        tmp_path / "independent.json",
        independent_unsigned,
    )
    bundle_sha256 = "e" * 64
    solver = _write_receipt(
        tmp_path / "solver.json",
        {
            "bundle_sha256": bundle_sha256,
            "complete_score_free_feasible_region_certified": False,
            "linear_projection_certified": True,
            "linear_projection_infeasible": True,
            "linear_projection_optimal": False,
            "original_dimensions": EXPECTED_DIMENSIONS,
        },
    )
    direct_unsigned = {
        **EXPECTED_PARENT_REQUIRED_FIELDS["direct_highs"],
        "bundle_export_receipt": {
            "bundle_sha256": bundle_sha256,
        },
        "cp_consensus_status": "INFEASIBLE",
        "registration_sha256": "d" * 64,
        "independent_cp_receipt": independent,
        "primary_cp_receipt": primary,
        "solver_receipt": {
            **solver,
            "linear_projection_certified": True,
            "linear_projection_infeasible": True,
            "linear_projection_optimal": False,
        },
    }
    direct = _write_receipt(
        tmp_path / "direct.json",
        direct_unsigned,
    )
    references = {}
    for label, reference in {
        "primary_cp": primary,
        "independent_cp": independent,
        "direct_highs": direct,
    }.items():
        references[label] = {
            **reference,
            "required_fields": EXPECTED_PARENT_REQUIRED_FIELDS[label],
        }
    registration = {
        "expected": {
            "bundle_sha256": bundle_sha256,
            "direct_highs_registration_sha256": "d" * 64,
            "dimensions": EXPECTED_DIMENSIONS,
        },
        "parent_results": references,
    }
    authenticated = audit._authenticate_parent_results(
        registration,
        v6_b_registration_sha256=v6_b_sha256,
    )
    assert set(authenticated) == {
        "direct_highs",
        "independent_cp",
        "primary_cp",
    }
    assert authenticated["direct_highs"]["terminal_status"] == (
        "DIRECT_HIGHS_LINEAR_INFEASIBLE_AGREEMENT"
    )
    altered_direct = _write_receipt(
        tmp_path / "direct-altered.json",
        {
            **direct_unsigned,
            "registration_sha256": "c" * 64,
        },
    )
    registration["parent_results"]["direct_highs"] = {
        **altered_direct,
        "required_fields": EXPECTED_PARENT_REQUIRED_FIELDS["direct_highs"],
    }
    with pytest.raises(
        audit.V6CFamilyCoreAuditError,
        match="linear proof differs",
    ):
        audit._authenticate_parent_results(
            registration,
            v6_b_registration_sha256=v6_b_sha256,
        )


def test_publish_is_create_only_and_non_authorizing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    output.mkdir()
    receipt = {
        "terminal_status": audit.TERMINAL_SUFFICIENT,
        "diagnostic_receipt": {"sufficient_core": ["one", "zero"]},
        "split_gate0_authorized": False,
        "training_authorized": False,
        "test_access_authorized": False,
    }
    try:
        audit._publish(output, receipt)
        published = json.loads(
            (output / "receipt.json").read_text(encoding="ascii")
        )
        claimed = published.pop("receipt_sha256")
        assert audit.sha256_payload(published) == claimed
        assert published["split_gate0_authorized"] is False
        assert published["training_authorized"] is False
        assert published["test_access_authorized"] is False
        assert hashlib.sha256(
            (output / "report.md").read_bytes()
        ).hexdigest() == published["report_sha256"]
        with pytest.raises(FileExistsError):
            audit._write_exclusive(
                output / "receipt.json",
                b"replacement",
            )
    finally:
        _make_writable(tmp_path)


@pytest.mark.parametrize(
    ("infeasible", "terminal"),
    [
        (True, audit.TERMINAL_IRREDUCIBLE),
        (False, audit.TERMINAL_NONPROOF),
    ],
)
def test_executor_publishes_exact_bounded_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    infeasible: bool,
    terminal: str,
) -> None:
    checked = {
        "limits": {
            "maximum_time_seconds_per_solve": 30,
            "minimize_core": True,
        }
    }
    registration_sha256 = audit.sha256_payload(checked)
    monkeypatch.setattr(
        audit.registration_io,
        "validate_family_core_registration",
        lambda registration: checked,
    )
    monkeypatch.setattr(
        audit,
        "_authenticate_policy",
        lambda registration: {"sha256": "a" * 64},
    )
    monkeypatch.setattr(
        audit,
        "_authenticate_v6_b_registration",
        lambda registration: ({"frozen": True}, "b" * 64),
    )
    monkeypatch.setattr(
        audit,
        "_authenticate_parent_results",
        lambda registration, v6_b_registration_sha256: {
            "agreement": True
        },
    )
    monkeypatch.setattr(
        audit,
        "_authenticate_runtime",
        lambda registration: {"ortools_version": "fixture"},
    )
    monkeypatch.setattr(
        audit,
        "_authenticate_sources",
        lambda registration, controller_path, controller_sha256: {
            "controller": {"sha256": controller_sha256}
        },
    )
    direct = _direct_model(infeasible=infeasible)
    monkeypatch.setattr(
        audit,
        "_reconstruct_direct_model",
        lambda registration, v6_b_registration: (
            direct,
            {"dimensions": audit._projection_dimensions(direct)},
        ),
    )
    output = tmp_path / f"result-{infeasible}"
    try:
        receipt = audit.execute_family_core_audit(
            registration={"ignored": True},
            registration_sha256=registration_sha256,
            controller_path=tmp_path / "controller.py",
            controller_sha256="d" * 64,
            output_directory=output,
        )
        assert receipt["terminal_status"] == terminal
        assert receipt[
            "common_linear_projection_infeasibility_diagnosed"
        ] is infeasible
        assert receipt["irreducible_family_core_proven"] is infeasible
        assert receipt["full_protocol_infeasibility_claim_authorized"] is False
        assert receipt["threshold_change_authorized"] is False
        assert receipt["split_gate0_authorized"] is False
        assert receipt["training_authorized"] is False
        assert receipt["model_selection_authorized"] is False
        assert receipt["outer_evaluation_authorized"] is False
        assert receipt["test_access_authorized"] is False
        assert (output / "receipt.json").is_file()
        assert (output / "report.md").is_file()
    finally:
        _make_writable(tmp_path)


def test_executor_preserves_failure_without_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = {
        "limits": {
            "maximum_time_seconds_per_solve": 30,
            "minimize_core": True,
        }
    }
    monkeypatch.setattr(
        audit.registration_io,
        "validate_family_core_registration",
        lambda registration: checked,
    )
    monkeypatch.setattr(
        audit,
        "_authenticate_policy",
        lambda registration: (_ for _ in ()).throw(
            audit.V6CFamilyCoreAuditError("injected parent failure")
        ),
    )
    output = tmp_path / "failure"
    try:
        with pytest.raises(
            audit.V6CFamilyCoreAuditError,
            match="injected parent failure",
        ):
            audit.execute_family_core_audit(
                registration={},
                registration_sha256=audit.sha256_payload(checked),
                controller_path=tmp_path / "controller.py",
                controller_sha256="d" * 64,
                output_directory=output,
            )
        published = json.loads(
            (output / "receipt.json").read_text(encoding="ascii")
        )
        assert (
            published["terminal_status"]
            == "V6_C_FAMILY_CORE_AUDIT_EXCEPTION"
        )
        assert published["split_gate0_authorized"] is False
        assert published["training_authorized"] is False
        assert published["test_access_authorized"] is False
    finally:
        _make_writable(tmp_path)
