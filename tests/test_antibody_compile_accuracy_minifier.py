from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/antibody_compile_accuracy_minifier_v1.json"
PREDECESSOR_PATH = (
    ROOT
    / "results/pretraining/antibody_compile_backend_attribution_v1_attempt_002_audit.json"
)
WRAPPER_PATH = ROOT / "scripts/run_antibody_compile_accuracy_minifier.py"
AUDITOR_PATH = ROOT / "scripts/audit_antibody_compile_accuracy_minifier.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WRAPPER = _module(WRAPPER_PATH, "accuracy_minifier_wrapper")
AUDITOR = _module(AUDITOR_PATH, "accuracy_minifier_auditor")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_binds_every_implementation_and_rejected_predecessor() -> None:
    protocol = _load(PROTOCOL_PATH)
    predecessor = _load(PREDECESSOR_PATH)
    bindings = {
        "attribution_protocol_sha256": (
            ROOT / "protocols/antibody_compile_backend_attribution_v1.json"
        ),
        "audit_launcher_sha256": (
            ROOT / "slurm/antibody_compile_accuracy_minifier_audit.sbatch"
        ),
        "auditor_sha256": AUDITOR_PATH,
        "base_attribution_wrapper_sha256": (
            ROOT / "scripts/run_antibody_compile_backend_attribution.py"
        ),
        "base_runner_sha256": (
            ROOT / "scripts/run_antibody_pretraining_optimized.py"
        ),
        "communication_protocol_sha256": (
            ROOT / "protocols/antibody_ddp_communication_optimization_v1.json"
        ),
        "efficiency_protocol_sha256": (
            ROOT / "protocols/antibody_efficiency_gate_v1.json"
        ),
        "launcher_sha256": (
            ROOT / "slurm/antibody_compile_accuracy_minifier.sbatch"
        ),
        "model_family_sha256": ROOT / "protocols/antibody_model_family_v1.json",
        "parity_evaluation_indices_sha256": (
            ROOT
            / "results/pretraining/antibody_ddp_parity_split_v2/evaluation_indices.npy"
        ),
        "parity_protocol_sha256": (
            ROOT / "protocols/antibody_ddp_scientific_parity_gate_v2.json"
        ),
        "parity_split_receipt_file_sha256": (
            ROOT / "results/pretraining/antibody_ddp_parity_split_v2/receipt.json"
        ),
        "parity_training_indices_sha256": (
            ROOT
            / "results/pretraining/antibody_ddp_parity_split_v2/training_indices.npy"
        ),
        "training_protocol_sha256": ROOT / "protocols/antibody_training_v1.json",
        "wrapper_sha256": WRAPPER_PATH,
    }
    for label, path in bindings.items():
        assert protocol["inputs"][label] == _sha256(path)
    assert protocol["predecessor"] == {
        "audit_file_sha256": _sha256(PREDECESSOR_PATH),
        "audit_receipt_sha256": predecessor["receipt_sha256"],
        "candidate_found": False,
        "diagnostic_valid": True,
        "selected_candidate_for_strict_confirmation": None,
        "training_job_id": 1345763,
    }
    assert set(protocol["authorization"].values()) == {False}
    assert protocol["diagnostic"]["silent_no_repro_is_not_a_pass"] is True
    assert protocol["inputs"]["runtime_manifest_file_sha256"] == (
        "8e025279fc1bb795756e6d1acad59d31de981c9ea86bd1a57610dd1dbd90593a"
    )
    assert protocol["runtime"]["torch"] == "2.13.0+cu130"


def test_wrapper_accepts_only_the_frozen_dynamo_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _load(PROTOCOL_PATH)
    predecessor = _load(PREDECESSOR_PATH)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    arguments = [
        "--backend-attribution-mode",
        "dynamo_eager_gc_off",
        "--output",
        str(tmp_path / "run"),
    ]
    output, rank, world_size = WRAPPER._validate_protocol(
        protocol,
        predecessor,
        boundary="dynamo",
        protocol_sha256=_sha256(PROTOCOL_PATH),
        predecessor_sha256=_sha256(PREDECESSOR_PATH),
        base_arguments=arguments,
    )
    assert output == tmp_path / "run"
    assert (rank, world_size) == (0, 8)

    with pytest.raises(WRAPPER.AccuracyMinifierError, match="not authorized"):
        WRAPPER._validate_protocol(
            protocol,
            predecessor,
            boundary="aot",
            protocol_sha256=_sha256(PROTOCOL_PATH),
            predecessor_sha256=_sha256(PREDECESSOR_PATH),
            base_arguments=arguments,
        )


def test_wrapper_rejects_reanchoring_and_any_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _load(PROTOCOL_PATH)
    predecessor = _load(PREDECESSOR_PATH)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    arguments = [
        "--backend-attribution-mode",
        "dynamo_eager_gc_off",
        "--output",
        str(tmp_path / "run"),
    ]
    changed = copy.deepcopy(protocol)
    changed["authorization"]["aot_minifier"] = True
    with pytest.raises(WRAPPER.AccuracyMinifierError, match="authority differs"):
        WRAPPER._validate_protocol(
            changed,
            predecessor,
            boundary="dynamo",
            protocol_sha256=_sha256(PROTOCOL_PATH),
            predecessor_sha256=_sha256(PREDECESSOR_PATH),
            base_arguments=arguments,
        )

    with pytest.raises(WRAPPER.AccuracyMinifierError, match="audit digest differs"):
        WRAPPER._validate_protocol(
            protocol,
            predecessor,
            boundary="dynamo",
            protocol_sha256=_sha256(PROTOCOL_PATH),
            predecessor_sha256="0" * 64,
            base_arguments=arguments,
        )


def test_auditor_receipt_self_hash_is_create_then_verify() -> None:
    payload = {
        "authorization": {"configuration_promotion": False},
        "format_version": "fixture",
    }
    payload["receipt_sha256"] = hashlib.sha256(
        AUDITOR._canonical_json_bytes(payload)
    ).hexdigest()
    AUDITOR._validate_self_hash(payload, "Fixture")

    changed = copy.deepcopy(payload)
    changed["authorization"]["configuration_promotion"] = True
    with pytest.raises(
        AUDITOR.AccuracyMinifierAuditError, match="self-hash differs"
    ):
        AUDITOR._validate_self_hash(changed, "Fixture")


def test_scripts_are_ascii_and_disable_long_option_abbreviation() -> None:
    for path in (WRAPPER_PATH, AUDITOR_PATH):
        path.read_bytes().decode("ascii")
        source = path.read_text(encoding="ascii")
        assert "allow_abbrev=False" in source
        assert "sabdab2_heldout_accessed" in source
    for path in (
        ROOT / "slurm/antibody_compile_accuracy_minifier.sbatch",
        ROOT / "slurm/antibody_compile_accuracy_minifier_audit.sbatch",
    ):
        path.read_bytes().decode("ascii")
        assert "set -euo pipefail" in path.read_text(encoding="ascii")


def test_wrapper_forbids_minifier_semantic_environment_overrides() -> None:
    assert WRAPPER.FORBIDDEN_PRESET_ENVIRONMENT == {
        "TORCHDYNAMO_REPRO_FORWARD_ONLY",
        "TORCHDYNAMO_REPRO_IGNORE_NON_FP",
    }
