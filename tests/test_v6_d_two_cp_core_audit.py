"""Focused tests for the V6-D two-CP audit executor."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from interaction_first.v6_cp_sat_formulation import DirectCPModel
from interaction_first import v6_d_two_cp_core_audit as audit


def _direct_model(*, infeasible: bool) -> DirectCPModel:
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    model = cp_model.CpModel()
    value = model.new_bool_var("value")
    model.add(value >= 0)
    families = ["domain"]
    if infeasible:
        model.add(value == 0)
        model.add(value == 1)
        families.extend(["zero", "one"])
    return DirectCPModel(
        model=model,
        variables={("value",): value},
        authority_payload_sha256="a" * 64,
        constraint_family_counts=tuple(
            sorted((family, families.count(family)) for family in set(families))
        ),
        constraint_families=tuple(families),
        formulation_sha256="b" * 64,
    )


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    os.chmod(root, 0o700)
    for path in root.rglob("*"):
        if path.is_dir():
            os.chmod(path, 0o700)
        elif not path.is_symlink():
            os.chmod(path, 0o600)


@pytest.mark.parametrize("infeasible", [False, True])
def test_executor_is_create_only_and_never_authorizes_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    infeasible: bool,
) -> None:
    registration = {
        "source": {"files": {"cp_formulation": {"sha256": "c" * 64}}},
        "limits": {
            "maximum_time_seconds_per_solve": 10,
            "minimize_core": True,
        },
    }
    monkeypatch.setattr(
        audit.registration_io,
        "validate_two_cp_core_registration",
        lambda value: value,
    )
    monkeypatch.setattr(audit, "_authenticate_policy", lambda value: {})
    monkeypatch.setattr(
        audit,
        "_authenticate_v6_b_registration",
        lambda value: ({}, "d" * 64),
    )
    monkeypatch.setattr(audit, "_authenticate_parents", lambda *a, **k: {})
    monkeypatch.setattr(audit, "_authenticate_runtime", lambda value: {})
    monkeypatch.setattr(audit, "_authenticate_sources", lambda *a, **k: {})
    monkeypatch.setattr(
        audit,
        "_reconstruct_direct_model",
        lambda *a, **k: (_direct_model(infeasible=infeasible), {}, "e" * 64, "f" * 64),
    )
    registration_sha256 = audit.sha256_payload(registration)
    output = tmp_path / f"result-{infeasible}"
    try:
        receipt = audit.execute_two_cp_core_audit(
            registration=registration,
            registration_sha256=registration_sha256,
            controller_path=Path(__file__),
            controller_sha256="a" * 64,
            output_directory=output,
        )
        assert receipt["training_authorized"] is False
        assert receipt["split_gate0_authorized"] is False
        assert receipt["test_access_authorized"] is False
        assert receipt["highs_core_audit_completed"] is False
        assert receipt["common_linear_projection_infeasibility_diagnosed"] is infeasible
        assert (output / "core_projection").exists() is infeasible
        published = json.loads((output / "receipt.json").read_text("ascii"))
        assert published["terminal_status"] == receipt["terminal_status"]
        assert os.stat(output).st_mode & 0o222 == 0
        with pytest.raises(FileExistsError):
            audit.execute_two_cp_core_audit(
                registration=registration,
                registration_sha256=registration_sha256,
                controller_path=Path(__file__),
                controller_sha256="a" * 64,
                output_directory=output,
            )
    finally:
        _make_writable(output)


def test_parent_authentication_rejects_wrong_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = {
        "registration_sha256": "a" * 64,
        "terminal_status": "PROVEN_INFEASIBLE",
        "decomposition_result": {
            "terminal_status": "PROVEN_INFEASIBLE",
            "candidate_present": False,
        },
    }
    independent = {
        "registration_sha256": "a" * 64,
        "primary_result_file_sha256": "wrong",
        "terminal_status": "INDEPENDENT_DIRECT_CP_INFEASIBLE_AGREEMENT",
        "projection_receipt": {
            "status": "INFEASIBLE",
            "accepted_projection_terminal_status": True,
        },
    }
    monkeypatch.setattr(
        audit,
        "_read_registered_receipt",
        lambda reference, label: (primary if "primary" in label else independent),
    )
    registration = {
        "expected": {"parent_ir_sha256": "parent-ir"},
        "parent_results": {
            "primary_cp": {"file_sha256": "p"},
            "independent_cp": {"file_sha256": "i"},
        },
    }
    with pytest.raises(audit.V6DTwoCPCoreAuditError, match="lineage"):
        audit._authenticate_parents(
            registration,
            v6_b_registration_sha256="a" * 64,
        )


def test_parent_authentication_binds_parent_ir_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = {
        "registration_sha256": "a" * 64,
        "terminal_status": "PROVEN_INFEASIBLE",
        "decomposition_result": {
            "terminal_status": "PROVEN_INFEASIBLE",
            "candidate_present": False,
        },
    }
    independent = {
        "registration_sha256": "a" * 64,
        "primary_result_file_sha256": "p",
        "terminal_status": "INDEPENDENT_DIRECT_CP_INFEASIBLE_AGREEMENT",
        "ir_sha256": "wrong-ir",
        "projection_receipt": {
            "status": "INFEASIBLE",
            "accepted_projection_terminal_status": True,
        },
    }
    monkeypatch.setattr(
        audit,
        "_read_registered_receipt",
        lambda reference, label: primary if "primary" in label else independent,
    )
    registration = {
        "expected": {"parent_ir_sha256": "expected-ir"},
        "parent_results": {
            "primary_cp": {"file_sha256": "p"},
            "independent_cp": {"file_sha256": "i"},
        },
    }
    with pytest.raises(audit.V6DTwoCPCoreAuditError, match="lineage"):
        audit._authenticate_parents(
            registration,
            v6_b_registration_sha256="a" * 64,
        )


def test_runtime_uses_invoked_venv_path_without_resolving_symlink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}

    class FakePath(type(Path())):
        def absolute(self) -> Path:
            observed["absolute"] = True
            return self

        def resolve(self, *args: object, **kwargs: object) -> Path:
            if str(self) == sys.executable:
                raise AssertionError("invoked Python path must not be resolved")
            return super().resolve(*args, **kwargs)

    monkeypatch.setattr(audit, "Path", FakePath)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import ortools

    ortools_path = Path(str(ortools.__file__)).resolve(strict=True)
    libc_family, libc_version = audit.platform.libc_ver()
    registration = {
        "runtime": {
            "cp_sat_runtime_receipt_path": "/runtime/receipt.json",
            "cp_sat_runtime_receipt_sha256": "a" * 64,
            "libc_family": libc_family,
            "libc_version": libc_version,
            "machine": audit.platform.machine(),
            "ortools_module_sha256": audit._sha256_file(ortools_path),
            "ortools_version": ortools.__version__,
            "python_executable": str(FakePath(sys.executable).absolute()),
            "python_version": audit.platform.python_version(),
            "system": audit.platform.system(),
        }
    }
    monkeypatch.setattr(audit.production, "read_verified", lambda *a, **k: b"x")
    receipt = audit._authenticate_runtime(registration)
    assert receipt["python_executable"] == registration["runtime"]["python_executable"]
    assert observed["absolute"] is True
