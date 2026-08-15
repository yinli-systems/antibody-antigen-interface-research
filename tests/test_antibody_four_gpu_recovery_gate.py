"""Test the four-GPU real-OAS exact-recovery readiness gate."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts/freeze_antibody_four_gpu_recovery_gate.py"
AUDITOR = ROOT / "scripts/audit_antibody_four_gpu_recovery_gate.py"
SLURM = ROOT / "slurm/antibody_four_gpu_recovery_gate.sbatch"
PIPELINE = ROOT / "slurm/antibody_four_gpu_recovery_pipeline.sbatch"
SUBMISSION = (
    ROOT
    / "results/pretraining/antibody_four_gpu_recovery_submission_f1b958e.json"
)
FAILURE = (
    ROOT / "results/pretraining/antibody_four_gpu_recovery_failure_1406713.json"
)
RETRY = ROOT / "results/pretraining/antibody_four_gpu_recovery_retry_1406723.json"
LAUNCH_FAILURE = (
    ROOT / "results/pretraining/antibody_four_gpu_recovery_failure_1406723.json"
)
AUDIT_FAILURE = (
    ROOT / "results/pretraining/antibody_four_gpu_recovery_failure_1406724.json"
)


def _load(name: str, path: Path):
    for item in (ROOT / "scripts", ROOT / "src"):
        value = str(item)
        if value not in sys.path:
            sys.path.insert(0, value)
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )


def test_sources_are_ascii_syntax_valid_and_slurm_is_valid() -> None:
    for path in (FREEZER, AUDITOR):
        ast.parse(path.read_text(encoding="ascii"))
    for path in (SLURM, PIPELINE):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_freezer_scales_only_global_token_budget(monkeypatch) -> None:
    freezer = _load("four_gpu_freezer_test", FREEZER)
    base = {
        "auditor_sha256": "0" * 64,
        "authorization": {
            "loader_recovery_gate": True,
            "production_pretraining": False,
            "scale_out": False,
            "scientific_claim": False,
        },
        "claim_boundary": "eight",
        "execution": {
            "optimization": {"compile_blocks": True},
            "world_size": 8,
        },
        "runs": {
            role: {
                "batch_quantum_sequences": 512,
                "micro_batch_per_gpu": 16,
                "target_residue_tokens": 262_144,
            }
            for role in ("continuous", "stage1", "resumed")
        },
    }
    monkeypatch.setattr(freezer.parent, "freeze_gate", lambda **_: base)
    monkeypatch.setattr(freezer.parent, "_file_sha256", lambda _: "a" * 64)
    result = freezer.freeze_gate(source_root=ROOT)
    assert result["execution"]["world_size"] == 4
    assert result["hardware_identity"] == {
        "gpu_count": 4,
        "gpu_model": "NVIDIA GeForce RTX 5090",
        "node_count": 1,
    }
    assert "hardware" not in result["execution"]
    for run in result["runs"].values():
        assert run["micro_batch_per_gpu"] == 16
        assert run["target_residue_tokens"] == 131_072
        assert run["target_residue_tokens"] // 4 == 32_768
        assert run["batch_quantum_sequences"] == 256
    assert result["authorization"]["production_pretraining"] is False
    assert "official-test" in result["claim_boundary"]


def test_auditor_accepts_only_four_same_host_ranks(tmp_path: Path) -> None:
    auditor = _load("four_gpu_auditor_test", AUDITOR)
    authorization = auditor.AUTHORIZATION
    test_access = auditor.TEST_ACCESS
    receipt = {
        "authorization": authorization,
        "corpus_open": {"node_count": 1, "rank_hosts": ["node"] * 4},
        "format_version": auditor.RUN_FORMAT,
        "run_role": "continuous",
        "test_access": test_access,
        "world_size": 4,
    }
    _write(tmp_path / "receipt.json", receipt)
    _write(tmp_path / "trainer_state.json", {"world_size": 4})
    loaded = auditor._load_run(tmp_path, "continuous")
    assert loaded["receipt"]["world_size"] == 4

    receipt["world_size"] = 8
    receipt["corpus_open"]["rank_hosts"] = ["node"] * 8
    _write(tmp_path / "receipt.json", receipt)
    with pytest.raises(
        auditor.recovery.RecoveryAuditError,
        match="four-GPU identity",
    ):
        auditor._load_run(tmp_path, "continuous")


def test_auditor_binds_top_level_hardware_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    auditor = _load("four_gpu_auditor_protocol_test", AUDITOR)
    protocol = {
        "auditor_sha256": auditor.artifacts._sha256(AUDITOR),
        "authorization": auditor.AUTHORIZATION,
        "execution": {"world_size": 4},
        "format_version": auditor.GATE_FORMAT,
        "hardware_identity": {
            "gpu_count": 4,
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "node_count": 1,
        },
        "test_access": auditor.TEST_ACCESS,
    }
    gate = tmp_path / "gate.json"
    _write(gate, protocol)
    monkeypatch.setattr(
        auditor.recovery,
        "audit_recovery",
        lambda **_: {"recovery_exact": True},
    )
    monkeypatch.setattr(
        auditor.artifacts,
        "_verify_artifacts",
        lambda _: {"manifest_verified": True, "writable_paths": []},
    )
    result = auditor.audit_recovery_4gpu(
        gate_protocol_path=gate,
        continuous_root=tmp_path,
        stage1_root=tmp_path,
        resumed_root=tmp_path,
        artifact_root=tmp_path,
    )
    assert result["recovery_exact"] is True
    assert result["world_size"] == 4

    protocol["hardware_identity"]["gpu_count"] = 8
    _write(gate, protocol)
    with pytest.raises(
        auditor.recovery.RecoveryAuditError,
        match="Four-GPU gate protocol differs",
    ):
        auditor.audit_recovery_4gpu(
            gate_protocol_path=gate,
            continuous_root=tmp_path,
            stage1_root=tmp_path,
            resumed_root=tmp_path,
            artifact_root=tmp_path,
        )


def test_slurm_freezes_four_5090s_and_never_opens_downstream_test() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --gres=gpu:4" in source
    assert "--nproc-per-node=4" in source
    assert "--target-tokens 131072" in source
    assert "target_residue_tokens_per_gpu=32768" in source
    assert "--ddp-communication-dtype" not in source
    assert "official" not in source.lower()
    assert "external" in source.lower()  # SHA-256 anchors, not external test data.


def test_pipeline_freezes_runs_and_audits_on_one_four_gpu_allocation() -> None:
    source = PIPELINE.read_text(encoding="ascii")
    assert "#SBATCH --gres=gpu:4" in source
    assert "freeze_antibody_four_gpu_recovery_gate.py" in source
    assert "antibody_four_gpu_recovery_gate.sbatch" in source
    assert "audit_antibody_four_gpu_recovery_gate.py" in source
    assert source.index("freeze_antibody_four_gpu_recovery_gate.py") < source.index(
        "antibody_four_gpu_recovery_gate.sbatch"
    ) < source.index("audit_antibody_four_gpu_recovery_gate.py")
    assert "four_gpu_recovery_pipeline=PASS" in source
    assert "official" not in source.lower()


def test_submission_receipt_is_non_authorizing_and_commit_pinned() -> None:
    value = json.loads(SUBMISSION.read_text(encoding="ascii"))
    assert value["format_version"] == "antibody_four_gpu_recovery_submission_v1"
    assert value["job"] == {
        "job_id": 1406713,
        "output_root": (
            "/ssd/scxi253/model_deployment/results/pretraining/"
            "antibody_four_gpu_recovery_f1b958e_attempt_001"
        ),
        "state_at_receipt": "SUBMITTED",
    }
    assert value["source"]["commit"] == (
        "f1b958e1cd6178869c332f49c5d2e32ae8cf768b"
    )
    assert value["corpus"]["unique_train_records"] == 224_887_833
    assert value["training_shape"]["target_residue_tokens_per_gpu"] == 32_768
    assert value["authorization"]["production_pretraining"] is False
    assert not any(value["test_access"].values())


def test_failed_attempt_is_preserved_and_retry_uses_verified_v3_runtime() -> None:
    failure = json.loads(FAILURE.read_text(encoding="ascii"))
    retry = json.loads(RETRY.read_text(encoding="ascii"))
    assert failure["failure"]["state"] == "FAILED"
    assert failure["failure"]["training_updates_started"] == 0
    assert failure["resolution"]["compatibility_bypass_added"] is False
    assert retry["predecessor_failure_job_id"] == failure["failure"]["job_id"]
    assert retry["job"]["job_id"] == 1406723
    assert retry["runtime"]["schema_version"] == "frontier.runtime-manifest.v3"
    assert retry["runtime"]["verified_live_before_submission"] is True
    assert retry["runtime"]["artifact_count_at_live_verification"] == 58
    assert not any(failure["test_access"].values())
    assert not any(retry["test_access"].values())


def test_second_failure_preserves_strict_execution_validation() -> None:
    failure = json.loads(LAUNCH_FAILURE.read_text(encoding="ascii"))
    assert failure["failure"]["job_id"] == 1406723
    assert failure["failure"]["stage"] == "launch_protocol_validation"
    assert failure["failure"]["training_updates_started"] == 0
    assert failure["resolution"]["execution_validation_removed"] is False
    assert failure["resolution"][
        "hardware_identity_preserved_at_protocol_top_level"
    ] is True
    assert not any(failure["test_access"].values())


def test_third_failure_preserves_completed_recovery_evidence() -> None:
    failure = json.loads(AUDIT_FAILURE.read_text(encoding="ascii"))
    assert failure["failure"]["job_id"] == 1406724
    assert failure["failure"]["stage"] == "independent_recovery_audit"
    assert failure["failure"]["training_updates_completed"] == {
        "continuous": 4,
        "resumed": 2,
        "stage1": 2,
    }
    evidence = failure["recovery_evidence"]
    assert evidence["exact_terminal_state_observed_before_independent_audit"] is True
    assert evidence["continuous_terminal_model_sha256"] == evidence[
        "resumed_terminal_model_sha256"
    ]
    assert evidence["continuous_terminal_optimizer_sha256"] == evidence[
        "resumed_terminal_optimizer_sha256"
    ]
    assert failure["resolution"]["scientific_recovery_comparison_relaxed"] is False
    assert not any(failure["test_access"].values())
