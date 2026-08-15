"""Test the static-shape four-GPU OAS efficiency gate."""

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
PARENT_AUDITOR = ROOT / "scripts/audit_antibody_four_gpu_efficiency_gate.py"
FREEZER = ROOT / "scripts/freeze_antibody_four_gpu_efficiency_gate_v2.py"
AUDITOR = ROOT / "scripts/audit_antibody_four_gpu_efficiency_gate_v2.py"
PIPELINE = ROOT / "slurm/antibody_four_gpu_efficiency_pipeline.sbatch"
WRAPPER = ROOT / "slurm/antibody_four_gpu_efficiency_pipeline_v2.sbatch"
SUBMISSION = (
    ROOT / "results/pretraining/antibody_four_gpu_efficiency_v2_submission_1406904.json"
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


def _shape_receipt(protocol_sha256: str) -> tuple[dict[str, object], str]:
    gates = {
        "approximate_mfu": True,
        "memory": True,
        "replicate_stability": True,
        "replicate_state": True,
        "step_stability": True,
    }
    value: dict[str, object] = {
        "authorization": {
            "configuration_promotion": False,
            "full_efficiency_gate_candidate": True,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "configurations": {
            "static_194": {"eligible": True, "gates": gates},
        },
        "format_version": "antibody_four_gpu_shape_stability_result_v1",
        "global_gates": {
            "artifact_integrity": True,
            "gpu_identity": True,
            "initial_model": True,
            "membership": True,
            "short_horizon_numerical": True,
        },
        "inputs": {"protocol_file_sha256": protocol_sha256},
        "probe_passed": True,
        "selected_candidate_for_full_efficiency_gate": "static_194",
        "test_access": {
            "downstream_labels_used": False,
            "external_test_accessed": False,
            "frozen_test_accessed": False,
            "official_test_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    payload_sha256 = hashlib.sha256(_canonical(value)).hexdigest()
    value["receipt_sha256"] = payload_sha256
    return value, payload_sha256


def _base_gate() -> dict[str, object]:
    return {
        "authorization": {
            "loader_recovery_gate": True,
            "production_pretraining": False,
            "scale_out": False,
            "scientific_claim": False,
        },
        "corpus": {"root": "/corpus"},
        "execution": {
            "backend": "ddp",
            "length_bucket_boundaries": "96,128,160,194",
            "optimization": {
                "bucketed_padding": True,
                "compile_blocks": True,
                "compile_prewarm_buckets": True,
                "ddp_bucket_cap_mb": 25,
                "deterministic_algorithms": True,
                "disable_compiled_cudagraphs": True,
                "fused_adamw": True,
                "length_bucketing": True,
                "pinned_staging": True,
            },
            "world_size": 4,
        },
        "format_version": "antibody_production_loader_recovery_gate_v1",
        "hardware_identity": {
            "gpu_count": 4,
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "node_count": 1,
        },
        "runs": {},
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }


def test_sources_and_shell_are_syntax_valid() -> None:
    for path in (PARENT_AUDITOR, FREEZER, AUDITOR):
        ast.parse(path.read_text(encoding="ascii"))
    for path in (PIPELINE, WRAPPER):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_v2_keeps_every_original_efficiency_threshold() -> None:
    parent = _load("efficiency_parent_threshold_test", PARENT_AUDITOR)
    freezer = _load("efficiency_v2_threshold_test", FREEZER)
    assert freezer.ACCEPTANCE == parent.freezer.ACCEPTANCE
    assert freezer.ACCEPTANCE["minimum_approximate_6n_mfu"] == 0.35
    assert freezer.ACCEPTANCE["maximum_p95_over_p50_step_time"] == 1.10
    assert freezer.ACCEPTANCE["maximum_replicate_throughput_relative_range"] == 0.05
    assert not any(freezer.TEST_ACCESS.values())


def test_shape_probe_loader_rejects_failed_candidate(tmp_path: Path) -> None:
    freezer = _load("efficiency_v2_shape_reject_test", FREEZER)
    protocol_sha256 = "a" * 64
    value, _ = _shape_receipt(protocol_sha256)
    value["probe_passed"] = False
    payload = dict(value)
    payload.pop("receipt_sha256")
    value["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    path = tmp_path / "shape.json"
    path.write_bytes(_canonical(value))
    with pytest.raises(freezer.EfficiencyV2FreezeError):
        freezer._load_shape_probe(
            path,
            expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_payload_sha256=value["receipt_sha256"],
            expected_protocol_file_sha256=protocol_sha256,
        )


def test_freeze_bundle_binds_passed_probe_and_static_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freezer = _load("efficiency_v2_bundle_test", FREEZER)
    protocol_sha256 = "b" * 64
    value, payload_sha256 = _shape_receipt(protocol_sha256)
    shape_path = tmp_path / "shape.json"
    shape_path.write_bytes(_canonical(value))
    monkeypatch.setattr(
        freezer.parent,
        "_load_recovery",
        lambda *args, **kwargs: {"gate_protocol_file_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        freezer.parent.recovery_freezer,
        "freeze_gate",
        lambda *args, **kwargs: _base_gate(),
    )
    master, gates = freezer.freeze_bundle(
        source_root=ROOT,
        corpus_root=tmp_path,
        manifest_file_sha256="d" * 64,
        manifest_payload_sha256="e" * 64,
        corpus_protocol_file_sha256="f" * 64,
        independent_audit_path=tmp_path / "corpus.json",
        independent_audit_file_sha256="1" * 64,
        independent_audit_payload_sha256="2" * 64,
        recovery_audit_path=tmp_path / "recovery.json",
        recovery_audit_file_sha256="3" * 64,
        recovery_audit_payload_sha256="4" * 64,
        shape_probe_audit_path=shape_path,
        shape_probe_audit_file_sha256=hashlib.sha256(shape_path.read_bytes()).hexdigest(),
        shape_probe_audit_payload_sha256=payload_sha256,
        shape_probe_protocol_file_sha256=protocol_sha256,
    )
    assert master["format_version"] == "antibody_four_gpu_efficiency_gate_v2"
    assert master["selected_shape_candidate"]["name"] == "static_194"
    assert gates["compiled"]["execution"]["length_bucket_boundaries"] == "194"
    assert gates["safe_eager"]["execution"]["length_bucket_boundaries"] == (
        "96,128,160,194"
    )
    assert master["authorization"]["production_pretraining"] is False


def test_pipeline_routes_static_shape_only_to_compiled_mode() -> None:
    source = PIPELINE.read_text(encoding="ascii")
    wrapper = WRAPPER.read_text(encoding="ascii")
    assert "ABF_EFFICIENCY_GATE_VERSION" in source
    assert "ABF_COMPILED_LENGTH_BUCKET_BOUNDARIES" in source
    assert '--length-bucket-boundaries "${length_bucket_boundaries}"' in source
    assert "run_mode safe_eager_r1 safe_eager" in source
    assert "run_mode compiled_r1 compiled" in source
    assert "run_mode compiled_r2 compiled" in source
    assert "run_mode safe_eager_r2 safe_eager" in source
    assert "export ABF_EFFICIENCY_GATE_VERSION=v2" in wrapper
    assert "export ABF_COMPILED_LENGTH_BUCKET_BOUNDARIES=194" in wrapper
    assert "ABF_SHAPE_PROBE_AUDIT" in wrapper


def test_submission_receipt_cannot_be_mistaken_for_efficiency_or_training_pass() -> None:
    value = json.loads(SUBMISSION.read_text(encoding="ascii"))
    assert value["job"]["job_id"] == 1406904
    assert value["job"]["state_at_receipt"] == "PENDING"
    assert value["job"]["requested_gpus"] == 4
    assert value["protocol"]["compiled_length_bucket_boundaries"] == "194"
    assert value["source"]["commit"] == (
        "340e2f0d380fd7988d05766b1ac4429cd7915172"
    )
    assert value["prerequisites"]["shape_probe_passed"] is True
    assert value["authorization"] == {
        "configuration_promotion": False,
        "production_pretraining": False,
        "scientific_claim": False,
    }
    assert not any(value["test_access"].values())


def test_v2_source_closure_binds_all_direct_and_indirect_execution() -> None:
    freezer = _load("efficiency_v2_source_test", FREEZER)
    assert {
        "scripts/audit_antibody_four_gpu_efficiency_gate.py",
        "scripts/audit_antibody_four_gpu_efficiency_gate_v2.py",
        "scripts/audit_antibody_four_gpu_shape_stability_probe.py",
        "scripts/audit_antibody_production_batch_efficiency.py",
        "scripts/evaluate_antibody_production_checkpoint.py",
        "scripts/freeze_antibody_four_gpu_efficiency_gate.py",
        "scripts/freeze_antibody_four_gpu_efficiency_gate_v2.py",
        "scripts/freeze_antibody_four_gpu_recovery_gate.py",
        "scripts/freeze_antibody_four_gpu_shape_stability_probe.py",
        "scripts/run_antibody_pretraining_production_gate.py",
        "slurm/antibody_four_gpu_efficiency_pipeline.sbatch",
        "slurm/antibody_four_gpu_efficiency_pipeline_v2.sbatch",
    } == set(freezer.SOURCE_FILES)
