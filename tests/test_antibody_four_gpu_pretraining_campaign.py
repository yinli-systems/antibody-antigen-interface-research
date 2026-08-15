"""Test the fail-closed four-GPU OAS pretraining campaign chain."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts/freeze_antibody_four_gpu_pretraining_campaign.py"
AUDITOR = ROOT / "scripts/audit_antibody_four_gpu_pretraining_segment.py"
SLURM = ROOT / "slurm/antibody_four_gpu_pretraining_segment.sbatch"


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


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _efficiency_audit(*, passed: bool = True) -> dict[str, object]:
    gates = {
        "artifact_integrity": passed,
        "approximate_mfu": True,
        "evaluation_noninferiority": True,
    }
    selected_shape = {
        "compiled_length_bucket_boundaries": "194",
        "name": "static_194",
        "probe_audit_file_sha256": "4" * 64,
        "probe_audit_payload_sha256": "5" * 64,
        "probe_protocol_file_sha256": "6" * 64,
        "probe_static_configuration": {
            "eligible": True,
            "gates": {"step_stability": True},
        },
    }
    return {
        "authorization": {
            "configuration_promotion": passed,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "format_version": "antibody_four_gpu_efficiency_result_v2",
        "gate_passed": passed,
        "gates": gates,
        "inputs": {"protocol_file_sha256": "a" * 64},
        "selected_shape_candidate": selected_shape,
        "pooled": {
            "modes": {
                "compiled": {
                    "approximate_6n_mfu": 0.40,
                    "residue_tokens_per_second": 88_000.0,
                }
            }
        },
        "runs": {
            label: {
                "summary": {
                    "approximate_6n_mfu": 0.40,
                    "p95_step_seconds": p95,
                }
            }
            for label, p95 in (("compiled_r1", 1.9), ("compiled_r2", 2.0))
        },
    }


def test_sources_are_ascii_syntax_valid_and_slurm_is_valid() -> None:
    for path in (FREEZER, AUDITOR):
        ast.parse(path.read_text(encoding="ascii"))
    subprocess.run(["bash", "-n", str(SLURM)], check=True)


def test_campaign_freeze_preserves_30b_token_and_evidence_boundaries(
    monkeypatch,
) -> None:
    freezer = _load("pretraining_campaign_freezer_test", FREEZER)
    recovery_gate = {
        "authorization": dict(freezer.RECOVERY_AUTHORIZATION),
        "execution": {
            "length_bucket_boundaries": "96,128,160,194",
            "optimization": dict(freezer.OPTIMIZATION),
            "world_size": 4,
        },
        "format_version": freezer.RECOVERY_GATE_FORMAT,
        "hardware_identity": dict(freezer.HARDWARE),
        "implementation": {},
        "inputs": {},
        "test_access": dict(freezer.TEST_ACCESS),
    }
    monkeypatch.setattr(freezer, "_validate_recovery", lambda **_: recovery_gate)
    monkeypatch.setattr(
        freezer,
        "_validate_efficiency",
        lambda **_: ({}, _efficiency_audit()),
    )
    monkeypatch.setattr(
        freezer,
        "_campaign_source",
        lambda _: {name: "b" * 64 for name in freezer.CAMPAIGN_SOURCE_FILES},
    )
    monkeypatch.setattr(freezer, "_sha256", lambda _: "c" * 64)
    result = freezer.freeze_campaign(
        source_root=ROOT,
        recovery_gate_path=ROOT,
        recovery_gate_file_sha256="d" * 64,
        recovery_audit_path=ROOT,
        recovery_audit_file_sha256="e" * 64,
        recovery_audit_payload_sha256="f" * 64,
        efficiency_protocol_path=ROOT,
        efficiency_protocol_file_sha256="1" * 64,
        efficiency_audit_path=ROOT,
        efficiency_audit_file_sha256="2" * 64,
        efficiency_audit_payload_sha256="3" * 64,
    )
    assert result["format_version"] == freezer.FORMAT_VERSION
    assert result["authorization"] == freezer.AUTHORIZATION
    assert result["campaign"]["campaign_token_limit"] == 30_000_000_000
    assert result["campaign"]["unique_train_residue_tokens"] == 25_259_993_406
    assert result["campaign"]["planned_repeat_exposure_residue_tokens"] == (
        4_740_006_594
    )
    assert result["campaign"]["maximum_final_overrun_residue_tokens"] == 180_224
    assert result["runs"]["stage1"]["resume"] is False
    assert result["runs"]["resumed"]["resume"] is True
    assert result["runs"]["stage1"]["campaign_token_limit"] == 30_000_000_000
    assert result["runs"]["stage1"]["steps"] == 5_400
    assert result["execution"]["length_bucket_boundaries"] == "194"
    assert result["systems_evidence"]["selected_shape_candidate"]["name"] == (
        "static_194"
    )
    assert result["prerequisites"]["shape_probe_audit_file_sha256"] == "4" * 64
    assert "official/frozen/external-test" in result["claim_boundary"]


def test_efficiency_failure_cannot_freeze_campaign(tmp_path: Path) -> None:
    freezer = _load("pretraining_campaign_efficiency_test", FREEZER)
    protocol = {
        "acceptance": {"minimum_approximate_6n_mfu": 0.35},
        "authorization": {
            "configuration_promotion": False,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "format_version": freezer.EFFICIENCY_PROTOCOL_FORMAT,
        "selected_shape_candidate": _efficiency_audit()[
            "selected_shape_candidate"
        ],
        "test_access": dict(freezer.EFFICIENCY_TEST_ACCESS),
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_sha256 = _write(protocol_path, protocol)
    audit = _efficiency_audit(passed=False)
    audit["inputs"]["protocol_file_sha256"] = protocol_sha256
    payload = dict(audit)
    audit["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    audit_path = tmp_path / "audit.json"
    audit_file_sha256 = _write(audit_path, audit)
    with pytest.raises(
        freezer.CampaignFreezeError,
        match="does not authorize promotion",
    ):
        freezer._validate_efficiency(
            protocol_path=protocol_path,
            protocol_file_sha256=protocol_sha256,
            audit_path=audit_path,
            audit_file_sha256=audit_file_sha256,
            audit_payload_sha256=audit["receipt_sha256"],
        )


def test_first_segment_audit_replays_trace_evaluation_and_artifacts(
    tmp_path: Path,
) -> None:
    auditor = _load("pretraining_segment_auditor_test", AUDITOR)
    freezer = auditor.freezer
    gate = {
        "auditor_sha256": auditor.artifacts._sha256(AUDITOR),
        "authorization": dict(freezer.AUTHORIZATION),
        "campaign": {
            "campaign_token_limit": 1_000,
            "maximum_final_overrun_residue_tokens": 200,
        },
        "campaign_source": {
            name: auditor.artifacts._sha256(ROOT / name)
            for name in freezer.CAMPAIGN_SOURCE_FILES
        },
        "execution": {
            "length_bucket_boundaries": "194",
            "optimization": dict(freezer.OPTIMIZATION),
        },
        "format_version": freezer.FORMAT_VERSION,
        "hardware_identity": dict(freezer.HARDWARE),
        "implementation": {},
        "runs": {
            "stage1": {
                "steps": 1,
            }
        },
        "test_access": dict(freezer.TEST_ACCESS),
        "validation_panel": {
            "indices_sha256": "a" * 64,
            "mask_seed": 3,
            "mask_update": 0,
            "record_count": 1,
            "seed": 7,
            "split": "validation",
        },
    }
    gate_path = tmp_path / "gate.json"
    gate_sha256 = _write(gate_path, gate)
    segment_root = tmp_path / "segment-000000"
    receipt = {
        "authorization": dict(freezer.AUTHORIZATION),
        "campaign": {
            "complete": False,
            "configured_maximum_steps": 1,
            "executed_steps": 1,
            "token_limit": 1_000,
        },
        "checkpoint_model_sha256": "1" * 64,
        "checkpoint_optimizer_sha256": "2" * 64,
        "checkpoint_rng_manifest_sha256": "3" * 64,
        "device": freezer.HARDWARE["gpu_model"],
        "final_model_sha256": "1" * 64,
        "final_optimizer_sha256": "4" * 64,
        "format_version": auditor.RUN_FORMAT,
        "gate_protocol_sha256": gate_sha256,
        "implementation": {},
        "membership_sha256": "5" * 64,
        "optimization": dict(freezer.OPTIMIZATION),
        "resume_used": False,
        "run_role": "stage1",
        "test_access": dict(freezer.TEST_ACCESS),
        "trace": [
            {
                "consumed_residue_tokens": 100,
                "gradient_norm": 1.0,
                "learning_rate": 0.001,
                "loss": 2.0,
                "micro_steps": 1,
                "residue_tokens": 100,
                "seconds": 1.0,
                "selected_targets": 10,
                "sequence_count": 1,
                "update": 1,
            }
        ],
        "world_size": 4,
    }
    receipt_path = segment_root / "run" / "receipt.json"
    receipt_sha256 = _write(receipt_path, receipt)
    state_sha256 = _write(
        segment_root / "run" / "trainer_state.json",
        {
            "checkpoint_rng_manifest_sha256": "3" * 64,
            "consumed_residue_tokens": 100,
            "gate_protocol_sha256": gate_sha256,
            "length_bucket_boundaries": [194],
            "next_update": 1,
            "sample_cursor": 1,
        },
    )
    evaluation = {
        "aggregate": {
            "masked_accuracy": 0.5,
            "masked_cross_entropy": 0.2,
            "selected_targets": 2,
        },
        "authorization": dict(auditor.EVALUATION_AUTHORIZATION),
        "checkpoint": {
            "final_model_sha256": "1" * 64,
            "receipt_file_sha256": receipt_sha256,
            "trainer_state_file_sha256": state_sha256,
        },
        "evaluated_run_authorization": dict(freezer.AUTHORIZATION),
        "format_version": auditor.EVALUATION_FORMAT,
        "gate_protocol_sha256": gate_sha256,
        "panel": dict(gate["validation_panel"]),
        "records": [
            {
                "correct": 1,
                "index": 5,
                "loss_sum": 0.4,
                "selected_targets": 2,
            }
        ],
        "test_access": dict(auditor.EVALUATION_TEST_ACCESS),
        "world_size": 4,
    }
    _write(segment_root / "evaluation.json", evaluation)
    files = sorted(path for path in segment_root.rglob("*") if path.is_file())
    manifest = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.resolve()}\n"
        for path in files
    )
    (segment_root / "artifact_sha256.txt").write_text(manifest, encoding="ascii")
    for path in segment_root.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
    result = auditor.audit_segment(
        gate_protocol_path=gate_path,
        segment_root=segment_root,
        segment_index=0,
    )
    assert result["authorization"]["next_segment_submission"] is True
    assert result["terminal"]["consumed_residue_tokens"] == 100
    assert result["validation"]["masked_cross_entropy"] == 0.2


def test_predecessor_audit_binds_exact_root_and_receipt_hashes(
    tmp_path: Path,
) -> None:
    auditor = _load("pretraining_segment_predecessor_test", AUDITOR)
    previous_root = tmp_path / "segment-000000"
    receipt = {
        "checkpoint_model_sha256": "1" * 64,
        "checkpoint_optimizer_sha256": "2" * 64,
        "checkpoint_rng_manifest_sha256": "3" * 64,
    }
    receipt_sha256 = _write(previous_root / "run" / "receipt.json", receipt)
    state = {
        "consumed_residue_tokens": 100,
        "next_update": 1,
        "sample_cursor": 7,
    }
    state_sha256 = _write(previous_root / "run" / "trainer_state.json", state)
    gate_sha256 = "a" * 64
    audit = {
        "authorization": {"next_segment_submission": True},
        "format_version": auditor.FORMAT_VERSION,
        "gate_protocol_file_sha256": gate_sha256,
        "inputs": {
            "run_receipt_file_sha256": receipt_sha256,
            "segment_root": str(previous_root.resolve()),
            "trainer_state_file_sha256": state_sha256,
        },
        "segment_index": 0,
        "terminal": {
            **receipt,
            **state,
        },
        "test_access": dict(auditor.EVALUATION_TEST_ACCESS),
    }
    payload = dict(audit)
    audit["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    audit_path = tmp_path / "segment-000000.audit.json"
    _write(audit_path, audit)
    consumed, update, _, _ = auditor._previous_terminal(
        segment_index=1,
        previous_segment_root=previous_root,
        previous_audit_path=audit_path,
        gate_sha256=gate_sha256,
    )
    assert (consumed, update) == (100, 1)

    audit["inputs"]["segment_root"] = str((tmp_path / "other").resolve())
    payload = dict(audit)
    payload.pop("receipt_sha256")
    audit["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _write(audit_path, audit)
    with pytest.raises(
        auditor.SegmentAuditError,
        match="does not authorize continuation",
    ):
        auditor._previous_terminal(
            segment_index=1,
            previous_segment_root=previous_root,
            previous_audit_path=audit_path,
            gate_sha256=gate_sha256,
        )


def test_slurm_is_four_gpu_compiled_and_never_uses_downstream_labels() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --gres=gpu:4" in source
    assert "--nproc-per-node=4" in source
    assert "--authorization-mode production_pretraining" in source
    assert "ABF_GATE_PROTOCOL_SHA256" in source
    assert "Campaign source closure differs" in source
    assert "--campaign-token-limit" in source
    assert "--compile-blocks" in source
    assert "--length-bucket-boundaries 194" in source
    assert "--length-bucket-boundaries 96,128,160,194" not in source
    assert "audit_antibody_four_gpu_pretraining_segment.py" in source
    assert "official_test" not in source
    assert "frozen_test" not in source
    assert "downstream_labels" not in source


def test_v1_or_nonstatic_efficiency_evidence_cannot_authorize_campaign(
    tmp_path: Path,
) -> None:
    freezer = _load("pretraining_campaign_static_evidence_test", FREEZER)
    selected = _efficiency_audit()["selected_shape_candidate"]
    protocol = {
        "acceptance": {"minimum_approximate_6n_mfu": 0.35},
        "authorization": {
            "configuration_promotion": False,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "format_version": freezer.EFFICIENCY_PROTOCOL_FORMAT,
        "selected_shape_candidate": selected,
        "test_access": dict(freezer.EFFICIENCY_TEST_ACCESS),
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_sha256 = _write(protocol_path, protocol)
    audit = _efficiency_audit()
    audit["inputs"]["protocol_file_sha256"] = protocol_sha256
    audit["selected_shape_candidate"] = {
        **selected,
        "compiled_length_bucket_boundaries": "96,128,160,194",
    }
    payload = dict(audit)
    audit["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    audit_path = tmp_path / "audit.json"
    audit_sha256 = _write(audit_path, audit)
    with pytest.raises(
        freezer.CampaignFreezeError,
        match="does not authorize promotion",
    ):
        freezer._validate_efficiency(
            protocol_path=protocol_path,
            protocol_file_sha256=protocol_sha256,
            audit_path=audit_path,
            audit_file_sha256=audit_sha256,
            audit_payload_sha256=audit["receipt_sha256"],
        )

    protocol["format_version"] = "antibody_four_gpu_efficiency_gate_v1"
    protocol_sha256 = _write(tmp_path / "protocol-v1.json", protocol)
    audit = _efficiency_audit()
    audit["inputs"]["protocol_file_sha256"] = protocol_sha256
    payload = dict(audit)
    audit["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    audit_sha256 = _write(tmp_path / "audit-v1.json", audit)
    with pytest.raises(
        freezer.CampaignFreezeError,
        match="does not authorize promotion",
    ):
        freezer._validate_efficiency(
            protocol_path=tmp_path / "protocol-v1.json",
            protocol_file_sha256=protocol_sha256,
            audit_path=tmp_path / "audit-v1.json",
            audit_file_sha256=audit_sha256,
            audit_payload_sha256=audit["receipt_sha256"],
        )
