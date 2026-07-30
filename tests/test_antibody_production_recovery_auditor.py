"""Test fail-closed structure of the production loader recovery auditor."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_antibody_production_loader_recovery.py"


def _load_auditor():
    specification = importlib.util.spec_from_file_location(
        "production_recovery_auditor_test",
        AUDITOR,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )


def _fixture(tmp_path: Path):
    authorization = {
        "loader_recovery_gate": True,
        "production_pretraining": False,
        "scale_out": False,
        "scientific_claim": False,
    }
    test_access = {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }
    data = {
        "independent_audit_file_sha256": "1" * 64,
        "independent_audit_payload_sha256": "2" * 64,
        "manifest_file_sha256": "3" * 64,
        "manifest_payload_sha256": "4" * 64,
        "protocol_file_sha256": "5" * 64,
        "split": "train",
    }
    protocol = {
        "authorization": authorization,
        "format_version": "antibody_production_loader_recovery_gate_v1",
        "implementation": {"runner.py": "6" * 64},
        "inputs": {
            **data,
            "efficiency_protocol_file_sha256": "7" * 64,
            "model_family_file_sha256": "8" * 64,
            "training_protocol_file_sha256": "9" * 64,
        },
        "runs": {"stage1": {"steps": 2}},
        "test_access": test_access,
    }
    protocol_path = tmp_path / "gate.json"
    _write(protocol_path, protocol)
    protocol_sha256 = __import__("hashlib").sha256(protocol_path.read_bytes()).hexdigest()

    def trace(start: int, stop: int):
        return [
            {
                "consumed_residue_tokens": update * 10,
                "gradient_norm": update / 10,
                "learning_rate": update / 100,
                "loss": 5 - update / 10,
                "micro_steps": 1,
                "residue_tokens": 10,
                "seconds": 99.0 + update,
                "selected_targets": 2,
                "sequence_count": 4,
                "update": update,
            }
            for update in range(start, stop)
        ]

    common = {
        "allocation": {"hostname": "node"},
        "authorization": authorization,
        "data_source": data,
        "efficiency_protocol_sha256": "7" * 64,
        "format_version": "antibody_production_loader_recovery_run_v1",
        "gate_protocol_sha256": protocol_sha256,
        "implementation": protocol["implementation"],
        "model_family_sha256": "8" * 64,
        "optimization": {"deterministic_algorithms": True},
        "preset": "abf_650m",
        "runtime": {"torch": "test"},
        "test_access": test_access,
        "training_protocol_sha256": "9" * 64,
        "training_universe_count": 100,
        "world_size": 1,
    }
    receipts = {
        "continuous": {
            **common,
            "final_model_sha256": "a" * 64,
            "final_optimizer_sha256": "b" * 64,
            "membership_sha256": "c" * 64,
            "resume_used": False,
            "run_role": "continuous",
            "trace": trace(1, 5),
        },
        "stage1": {
            **common,
            "checkpoint_model_sha256": "d" * 64,
            "checkpoint_optimizer_sha256": "e" * 64,
            "checkpoint_rng_manifest_sha256": "f" * 64,
            "final_model_sha256": "d" * 64,
            "final_optimizer_sha256": "1" * 64,
            "membership_sha256": "g" * 64,
            "resume_used": False,
            "run_role": "stage1",
            "trace": trace(1, 3),
        },
        "resumed": {
            **common,
            "final_model_sha256": "a" * 64,
            "final_optimizer_sha256": "b" * 64,
            "membership_sha256": "h" * 64,
            "resume_loaded_model_sha256": "d" * 64,
            "resume_loaded_optimizer_sha256": "1" * 64,
            "resume_payload_model_sha256": "d" * 64,
            "resume_payload_optimizer_sha256": "e" * 64,
            "resume_rng_manifest_sha256": "f" * 64,
            "resume_used": True,
            "run_role": "resumed",
            "trace": trace(3, 5),
        },
    }
    roots = {}
    for role, receipt in receipts.items():
        root = tmp_path / role
        _write(root / "receipt.json", receipt)
        state = {
            "consumed_residue_tokens": 40 if role != "stage1" else 20,
            "next_update": 4 if role != "stage1" else 2,
            "sample_cursor": 16 if role != "stage1" else 8,
            "world_size": 1,
        }
        _write(root / "trainer_state.json", state)
        roots[role] = root
    return protocol_path, roots


def test_recovery_auditor_is_ascii_and_syntax_valid() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    ast.parse(source)
    assert "Uninterrupted and resumed scientific traces differ" in source
    assert "Checkpoint payload or applied state differs" in source
    assert "Final uninterrupted and resumed states differ" in source
    assert "Terminal trainer coordinates differ" in source


def test_recovery_auditor_is_create_only_and_non_authorizing() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    assert "os.O_EXCL" in source
    assert '"production_pretraining": False' in source
    assert '"scale_out": False' in source
    assert '"scientific_claim": False' in source
    assert '"sabdab2_heldout_accessed": False' in source
    assert '"downstream_labels_used": False' in source


def test_recovery_auditor_accepts_exact_fixture(tmp_path: Path) -> None:
    auditor = _load_auditor()
    protocol, roots = _fixture(tmp_path)
    result = auditor.audit_recovery(
        gate_protocol_path=protocol,
        continuous_root=roots["continuous"],
        stage1_root=roots["stage1"],
        resumed_root=roots["resumed"],
    )
    assert len(result["auditor_sha256"]) == 64
    assert result["exact_recovery"] == {
        "final_model_equal": True,
        "final_optimizer_equal": True,
        "scientific_trace_equal": True,
        "terminal_coordinates_equal": True,
    }


def test_recovery_auditor_rejects_trace_drift(tmp_path: Path) -> None:
    auditor = _load_auditor()
    protocol, roots = _fixture(tmp_path)
    receipt_path = roots["resumed"] / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    receipt["trace"][0]["loss"] += 0.001
    _write(receipt_path, receipt)
    with pytest.raises(auditor.RecoveryAuditError, match="scientific traces"):
        auditor.audit_recovery(
            gate_protocol_path=protocol,
            continuous_root=roots["continuous"],
            stage1_root=roots["stage1"],
            resumed_root=roots["resumed"],
        )


@pytest.mark.parametrize(
    "field",
    ("resume_payload_optimizer_sha256", "resume_loaded_optimizer_sha256"),
)
def test_recovery_auditor_rejects_optimizer_checkpoint_drift(
    tmp_path: Path,
    field: str,
) -> None:
    auditor = _load_auditor()
    protocol, roots = _fixture(tmp_path)
    receipt_path = roots["resumed"] / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    receipt[field] = "0" * 64
    _write(receipt_path, receipt)
    with pytest.raises(auditor.RecoveryAuditError, match="Checkpoint payload"):
        auditor.audit_recovery(
            gate_protocol_path=protocol,
            continuous_root=roots["continuous"],
            stage1_root=roots["stage1"],
            resumed_root=roots["resumed"],
        )
