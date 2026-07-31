"""Test the confirmed-candidate eight-GPU real-OAS recovery gate."""

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
FREEZER = ROOT / "scripts/freeze_antibody_production_recovery_gate_8gpu.py"
WRAPPER = ROOT / "scripts/run_antibody_pretraining_production_candidate_gate.py"
AUDITOR = ROOT / "scripts/audit_antibody_production_recovery_gate_8gpu.py"
SLURM = ROOT / "slurm/antibody_production_recovery_gate_8gpu.sbatch"
AUDIT_SLURM = ROOT / "slurm/antibody_production_recovery_gate_8gpu_audit.sbatch"


def _load(name: str, path: Path):
    scripts = str(ROOT / "scripts")
    source = str(ROOT / "src")
    for item in (scripts, source):
        if item not in sys.path:
            sys.path.insert(0, item)
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confirmation_protocol() -> dict[str, object]:
    return {
        "confirmation_selection": {
            "selected_candidate": "compile_accuracy",
        },
        "inputs": {
            "confirmation_auditor_sha256": _sha(
                ROOT / "scripts/audit_antibody_compile_candidate_confirmation.py"
            ),
            "confirmation_wrapper_sha256": _sha(
                ROOT / "scripts/run_antibody_compile_candidate_confirmation.py"
            ),
            "runner_sha256": _sha(
                ROOT / "scripts/run_antibody_pretraining_optimized.py"
            ),
            "runner_wrapper_sha256": _sha(
                ROOT / "scripts/run_antibody_pretraining_parity_extension.py"
            ),
        },
        "optimization": {
            "execution_controls": {
                "accuracy_first_bf16_reduction": True,
                "disable_python_gc": False,
            }
        },
    }


def _confirmation_audit(protocol_sha256: str) -> dict[str, object]:
    value: dict[str, object] = {
        "authorization": {
            "configuration_promotion": True,
            "production_pretraining": False,
            "scale_16_gpu": False,
            "scale_32_gpu": False,
            "scientific_claim": False,
        },
        "confirmed_candidate": "compile_accuracy",
        "format_version": (
            "antibody_compile_candidate_confirmation_result_v1"
        ),
        "gate_passed": True,
        "inputs": {"protocol_file_sha256": protocol_sha256},
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    value["receipt_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def _confirmation_protocol_v2() -> dict[str, object]:
    return {
        "confirmation_selection": {
            "selected_candidate": "inductor_no_cudagraphs",
        },
        "inputs": {
            "confirmation_auditor_sha256": _sha(
                ROOT
                / "scripts/audit_antibody_compile_candidate_confirmation_v2.py"
            ),
            "confirmation_wrapper_sha256": _sha(
                ROOT / "scripts/run_antibody_compile_candidate_confirmation_v2.py"
            ),
            "runner_sha256": _sha(
                ROOT / "scripts/run_antibody_pretraining_optimized.py"
            ),
            "runner_wrapper_sha256": _sha(
                ROOT / "scripts/run_antibody_pretraining_parity_extension.py"
            ),
        },
        "optimization": {
            "execution_controls": {
                "compile_backend": "inductor",
                "disable_compiled_cudagraphs": True,
                "disable_python_gc": True,
            }
        },
    }


def _confirmation_audit_v2(protocol_sha256: str) -> dict[str, object]:
    value: dict[str, object] = {
        "authorization": {
            "configuration_promotion": True,
            "production_pretraining": False,
            "scale_16_gpu": False,
            "scale_32_gpu": False,
            "scientific_claim": False,
        },
        "confirmed_candidate": "inductor_no_cudagraphs",
        "format_version": "antibody_compile_candidate_confirmation_result_v2",
        "gate_passed": True,
        "inputs": {"protocol_file_sha256": protocol_sha256},
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    value["receipt_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def _confirmation_protocol_component(scope: str) -> dict[str, object]:
    candidate = f"inductor_{scope}_only"
    return {
        "component_confirmation_selection": {
            "selected_candidate": candidate,
        },
        "inputs": {
            "confirmation_auditor_sha256": _sha(
                ROOT
                / "scripts/audit_antibody_inductor_component_confirmation.py"
            ),
            "confirmation_wrapper_sha256": _sha(
                ROOT
                / "scripts/run_antibody_inductor_component_confirmation.py"
            ),
            "runner_sha256": _sha(
                ROOT / "scripts/run_antibody_pretraining_optimized.py"
            ),
            "runner_wrapper_sha256": _sha(
                ROOT / "scripts/run_antibody_pretraining_parity_extension.py"
            ),
        },
        "optimization": {
            "execution_controls": {
                "compile_backend": "inductor",
                "compile_scope": scope,
                "disable_compiled_cudagraphs": True,
                "disable_python_gc": True,
            }
        },
    }


def _confirmation_audit_component(
    protocol_sha256: str,
    scope: str,
) -> dict[str, object]:
    value: dict[str, object] = {
        "authorization": {
            "configuration_promotion": True,
            "production_pretraining": False,
            "scale_16_gpu": False,
            "scale_32_gpu": False,
            "scientific_claim": False,
        },
        "confirmed_component_candidate": f"inductor_{scope}_only",
        "format_version": (
            "antibody_inductor_component_confirmation_result_v1"
        ),
        "gate_passed": True,
        "inputs": {"protocol_file_sha256": protocol_sha256},
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    value["receipt_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def test_sources_are_ascii_and_syntax_valid() -> None:
    for path in (FREEZER, WRAPPER, AUDITOR):
        source = path.read_text(encoding="ascii")
        ast.parse(source)
        assert "SOTA claim" in source


def test_freezer_requires_passing_confirmation_and_locks_world_size(
    tmp_path: Path,
) -> None:
    freezer = _load("oas_recovery_8gpu_freezer", FREEZER)
    confirmation_protocol = _confirmation_protocol()
    protocol_path = tmp_path / "confirmation-protocol.json"
    protocol_path.write_bytes(_canonical(confirmation_protocol))
    protocol_sha256 = _sha(protocol_path)
    confirmation_audit = _confirmation_audit(protocol_sha256)
    audit_path = tmp_path / "confirmation-audit.json"
    audit_path.write_bytes(_canonical(confirmation_audit))

    class Corpus:
        manifest = {"counts": {"unique_validation_records": 1_743_380}}

        def __len__(self) -> int:
            return 224_887_833

    freezer.load_production_corpus = lambda *args, **kwargs: Corpus()
    result = freezer.freeze_gate(
        source_root=ROOT,
        corpus_root=tmp_path / "corpus",
        manifest_file_sha256="1" * 64,
        manifest_payload_sha256="2" * 64,
        corpus_protocol_file_sha256="3" * 64,
        independent_audit_path=tmp_path / "corpus-audit.json",
        independent_audit_file_sha256="4" * 64,
        independent_audit_payload_sha256="5" * 64,
        confirmation_protocol_path=protocol_path,
        confirmation_protocol_file_sha256=protocol_sha256,
        confirmation_audit_path=audit_path,
        confirmation_audit_file_sha256=_sha(audit_path),
    )
    assert result["execution"]["world_size"] == 8
    assert result["execution"]["optimization"] == {
        "bucketed_padding": False,
        "compile_blocks": True,
        "compile_prewarm_buckets": True,
        "ddp_bucket_cap_mb": 25,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": False,
        "fused_adamw": False,
        "length_bucketing": False,
        "pinned_staging": False,
    }
    assert result["runs"]["continuous"]["target_residue_tokens"] == 262_144
    assert result["runs"]["continuous"]["batch_quantum_sequences"] == 512
    assert result["candidate_selection"]["execution_controls"] == {
        "accuracy_first_bf16_reduction": True,
        "disable_python_gc": False,
    }
    assert result["authorization"]["production_pretraining"] is False


def test_freezer_rejects_failed_confirmation(tmp_path: Path) -> None:
    freezer = _load("oas_recovery_8gpu_freezer_reject", FREEZER)
    protocol = _confirmation_protocol()
    protocol_path = tmp_path / "confirmation-protocol.json"
    protocol_path.write_bytes(_canonical(protocol))
    audit = _confirmation_audit(_sha(protocol_path))
    audit["gate_passed"] = False
    audit.pop("receipt_sha256")
    audit["receipt_sha256"] = hashlib.sha256(_canonical(audit)).hexdigest()
    audit_path = tmp_path / "failed-confirmation.json"
    audit_path.write_bytes(_canonical(audit))
    try:
        freezer.freeze_gate(
            source_root=ROOT,
            corpus_root=tmp_path / "corpus",
            manifest_file_sha256="1" * 64,
            manifest_payload_sha256="2" * 64,
            corpus_protocol_file_sha256="3" * 64,
            independent_audit_path=tmp_path / "corpus-audit.json",
            independent_audit_file_sha256="4" * 64,
            independent_audit_payload_sha256="5" * 64,
            confirmation_protocol_path=protocol_path,
            confirmation_protocol_file_sha256=_sha(protocol_path),
            confirmation_audit_path=audit_path,
            confirmation_audit_file_sha256=_sha(audit_path),
        )
    except freezer.EightGpuGateFreezeError as error:
        assert "does not authorize" in str(error)
    else:
        raise AssertionError("A failed confirmation must not freeze a gate")


def test_freezer_accepts_v2_backend_controls_without_relaxing_authority(
    tmp_path: Path,
) -> None:
    freezer = _load("oas_recovery_8gpu_freezer_v2", FREEZER)
    confirmation_protocol = _confirmation_protocol_v2()
    protocol_path = tmp_path / "confirmation-protocol-v2.json"
    protocol_path.write_bytes(_canonical(confirmation_protocol))
    protocol_sha256 = _sha(protocol_path)
    audit_path = tmp_path / "confirmation-audit-v2.json"
    audit_path.write_bytes(_canonical(_confirmation_audit_v2(protocol_sha256)))

    class Corpus:
        manifest = {"counts": {"unique_validation_records": 1_743_380}}

        def __len__(self) -> int:
            return 224_887_833

    freezer.load_production_corpus = lambda *args, **kwargs: Corpus()
    result = freezer.freeze_gate(
        source_root=ROOT,
        corpus_root=tmp_path / "corpus",
        manifest_file_sha256="1" * 64,
        manifest_payload_sha256="2" * 64,
        corpus_protocol_file_sha256="3" * 64,
        independent_audit_path=tmp_path / "corpus-audit.json",
        independent_audit_file_sha256="4" * 64,
        independent_audit_payload_sha256="5" * 64,
        confirmation_protocol_path=protocol_path,
        confirmation_protocol_file_sha256=protocol_sha256,
        confirmation_audit_path=audit_path,
        confirmation_audit_file_sha256=_sha(audit_path),
    )
    assert result["candidate_selection"]["confirmation_format"] == (
        "antibody_compile_candidate_confirmation_result_v2"
    )
    assert result["candidate_selection"]["execution_controls"] == {
        "compile_backend": "inductor",
        "disable_compiled_cudagraphs": True,
        "disable_python_gc": True,
    }
    assert result["execution"]["optimization"][
        "disable_compiled_cudagraphs"
    ] is True
    assert result["authorization"]["production_pretraining"] is False


def test_freezer_carries_component_scope_into_real_oas_gate(
    tmp_path: Path,
) -> None:
    freezer = _load("oas_recovery_8gpu_freezer_component", FREEZER)
    confirmation_protocol = _confirmation_protocol_component("attention")
    protocol_path = tmp_path / "confirmation-protocol-component.json"
    protocol_path.write_bytes(_canonical(confirmation_protocol))
    protocol_sha256 = _sha(protocol_path)
    audit_path = tmp_path / "confirmation-audit-component.json"
    audit_path.write_bytes(
        _canonical(_confirmation_audit_component(protocol_sha256, "attention"))
    )

    class Corpus:
        manifest = {"counts": {"unique_validation_records": 1_743_380}}

        def __len__(self) -> int:
            return 224_887_833

    freezer.load_production_corpus = lambda *args, **kwargs: Corpus()
    result = freezer.freeze_gate(
        source_root=ROOT,
        corpus_root=tmp_path / "corpus",
        manifest_file_sha256="1" * 64,
        manifest_payload_sha256="2" * 64,
        corpus_protocol_file_sha256="3" * 64,
        independent_audit_path=tmp_path / "corpus-audit.json",
        independent_audit_file_sha256="4" * 64,
        independent_audit_payload_sha256="5" * 64,
        confirmation_protocol_path=protocol_path,
        confirmation_protocol_file_sha256=protocol_sha256,
        confirmation_audit_path=audit_path,
        confirmation_audit_file_sha256=_sha(audit_path),
    )
    assert result["candidate_selection"]["selected_candidate"] == (
        "inductor_attention_only"
    )
    assert result["candidate_selection"]["execution_controls"] == {
        "compile_backend": "inductor",
        "compile_scope": "attention",
        "disable_compiled_cudagraphs": True,
        "disable_python_gc": True,
    }
    assert result["execution"]["optimization"][
        "disable_compiled_cudagraphs"
    ] is True
    assert result["authorization"]["production_pretraining"] is False


def test_wrapper_accepts_only_the_confirmed_compile_controls(
    tmp_path: Path,
) -> None:
    wrapper = _load("oas_recovery_8gpu_wrapper", WRAPPER)
    gate = {
        "authorization": {
            "loader_recovery_gate": True,
            "production_pretraining": False,
            "scale_out": False,
            "scientific_claim": False,
        },
        "candidate_selection": {
            "base_runner_sha256": _sha(
                ROOT / "scripts/run_antibody_pretraining_production_gate.py"
            ),
            "configuration_promotion_verified": True,
            "execution_controls": {
                "accuracy_first_bf16_reduction": True,
                "disable_python_gc": False,
            },
            "selected_candidate": "compile_accuracy",
            "wrapper_sha256": _sha(WRAPPER),
        },
        "format_version": "antibody_production_loader_recovery_gate_v1",
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_bytes(_canonical(gate))
    output = tmp_path / "output"
    observed_output, controls = wrapper._validate_launch(
        gate,
        [
            "--gate-protocol",
            str(gate_path),
            "--output",
            str(output),
            "--compile-blocks",
            "--compile-prewarm-buckets",
        ],
    )
    assert observed_output == output
    assert controls["accuracy_first_bf16_reduction"] is True


def test_wrapper_requires_v2_cudagraph_control(tmp_path: Path) -> None:
    wrapper = _load("oas_recovery_8gpu_wrapper_v2", WRAPPER)
    controls = {
        "compile_backend": "inductor",
        "disable_compiled_cudagraphs": True,
        "disable_python_gc": True,
    }
    gate = {
        "authorization": {
            "loader_recovery_gate": True,
            "production_pretraining": False,
            "scale_out": False,
            "scientific_claim": False,
        },
        "candidate_selection": {
            "base_runner_sha256": _sha(
                ROOT / "scripts/run_antibody_pretraining_production_gate.py"
            ),
            "configuration_promotion_verified": True,
            "execution_controls": controls,
            "selected_candidate": "inductor_no_cudagraphs",
            "wrapper_sha256": _sha(WRAPPER),
        },
        "format_version": "antibody_production_loader_recovery_gate_v1",
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    gate_path = tmp_path / "gate-v2.json"
    gate_path.write_bytes(_canonical(gate))
    output = tmp_path / "output-v2"
    observed_output, observed_controls = wrapper._validate_launch(
        gate,
        [
            "--gate-protocol",
            str(gate_path),
            "--output",
            str(output),
            "--compile-blocks",
            "--compile-prewarm-buckets",
            "--disable-compiled-cudagraphs",
        ],
    )
    assert observed_output == output
    assert observed_controls == controls


def test_wrapper_requires_exact_component_scope(tmp_path: Path) -> None:
    wrapper = _load("oas_recovery_8gpu_wrapper_component", WRAPPER)
    controls = {
        "compile_backend": "inductor",
        "compile_scope": "ffn",
        "disable_compiled_cudagraphs": True,
        "disable_python_gc": True,
    }
    gate = {
        "authorization": {
            "loader_recovery_gate": True,
            "production_pretraining": False,
            "scale_out": False,
            "scientific_claim": False,
        },
        "candidate_selection": {
            "base_runner_sha256": _sha(
                ROOT / "scripts/run_antibody_pretraining_production_gate.py"
            ),
            "configuration_promotion_verified": True,
            "execution_controls": controls,
            "selected_candidate": "inductor_ffn_only",
            "wrapper_sha256": _sha(WRAPPER),
        },
        "format_version": "antibody_production_loader_recovery_gate_v1",
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    gate_path = tmp_path / "gate-component.json"
    gate_path.write_bytes(_canonical(gate))
    output = tmp_path / "output-component"
    observed_output, observed_controls = wrapper._validate_launch(
        gate,
        [
            "--gate-protocol",
            str(gate_path),
            "--output",
            str(output),
            "--compile-blocks",
            "--compile-prewarm-buckets",
            "--disable-compiled-cudagraphs",
        ],
    )
    assert observed_output == output
    assert observed_controls["compile_scope"] == "ffn"
    invalid = dict(controls)
    invalid["compile_scope"] = "full_block"
    assert wrapper._component_backend_controls(invalid) is False
    mismatched_gate = json.loads(json.dumps(gate))
    mismatched_gate["candidate_selection"]["execution_controls"][
        "compile_scope"
    ] = "attention"
    mismatched_path = tmp_path / "gate-component-mismatched.json"
    mismatched_path.write_bytes(_canonical(mismatched_gate))
    mismatched_arguments = [
        "--gate-protocol",
        str(mismatched_path),
        "--output",
        str(output),
        "--compile-blocks",
        "--compile-prewarm-buckets",
        "--disable-compiled-cudagraphs",
    ]
    with pytest.raises(wrapper.ProductionCandidateGateError):
        wrapper._validate_launch(mismatched_gate, mismatched_arguments)


def test_auditor_replays_v2_confirmation_selection(tmp_path: Path) -> None:
    auditor = _load("oas_recovery_8gpu_auditor_v2", AUDITOR)
    confirmation_protocol = _confirmation_protocol_v2()
    protocol_path = tmp_path / "auditor-confirmation-protocol-v2.json"
    protocol_path.write_bytes(_canonical(confirmation_protocol))
    protocol_sha256 = _sha(protocol_path)
    audit_path = tmp_path / "auditor-confirmation-audit-v2.json"
    audit_path.write_bytes(_canonical(_confirmation_audit_v2(protocol_sha256)))
    gate = {
        "candidate_selection": {
            "configuration_promotion_verified": True,
            "confirmation_audit_file_sha256": _sha(audit_path),
            "confirmation_audit_receipt_sha256": json.loads(
                audit_path.read_text(encoding="ascii")
            )["receipt_sha256"],
            "confirmation_format": (
                "antibody_compile_candidate_confirmation_result_v2"
            ),
            "confirmation_protocol_file_sha256": protocol_sha256,
            "execution_controls": {
                "compile_backend": "inductor",
                "disable_compiled_cudagraphs": True,
                "disable_python_gc": True,
            },
            "selected_candidate": "inductor_no_cudagraphs",
        }
    }
    result = auditor._validate_confirmation(
        protocol=gate,
        confirmation_protocol_path=protocol_path,
        confirmation_audit_path=audit_path,
    )
    assert result["confirmation_protocol_file_sha256"] == protocol_sha256


def test_auditor_replays_component_confirmation_selection(
    tmp_path: Path,
) -> None:
    auditor = _load("oas_recovery_8gpu_auditor_component", AUDITOR)
    confirmation_protocol = _confirmation_protocol_component("ffn")
    protocol_path = tmp_path / "auditor-confirmation-protocol-component.json"
    protocol_path.write_bytes(_canonical(confirmation_protocol))
    protocol_sha256 = _sha(protocol_path)
    audit_path = tmp_path / "auditor-confirmation-audit-component.json"
    audit_path.write_bytes(
        _canonical(_confirmation_audit_component(protocol_sha256, "ffn"))
    )
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    controls = confirmation_protocol["optimization"]["execution_controls"]
    gate = {
        "candidate_selection": {
            "configuration_promotion_verified": True,
            "confirmation_audit_file_sha256": _sha(audit_path),
            "confirmation_audit_receipt_sha256": audit["receipt_sha256"],
            "confirmation_format": (
                "antibody_inductor_component_confirmation_result_v1"
            ),
            "confirmation_protocol_file_sha256": protocol_sha256,
            "execution_controls": controls,
            "selected_candidate": "inductor_ffn_only",
        }
    }
    result = auditor._validate_confirmation(
        protocol=gate,
        confirmation_protocol_path=protocol_path,
        confirmation_audit_path=audit_path,
    )
    assert result["confirmation_protocol_file_sha256"] == protocol_sha256
    assert auditor._component_backend_controls(controls) is True
    mismatched_protocol = json.loads(json.dumps(confirmation_protocol))
    mismatched_protocol["optimization"]["execution_controls"][
        "compile_scope"
    ] = "attention"
    mismatched_protocol_path = tmp_path / "auditor-component-mismatch.json"
    mismatched_protocol_path.write_bytes(_canonical(mismatched_protocol))
    mismatched_protocol_sha256 = _sha(mismatched_protocol_path)
    mismatched_audit_path = tmp_path / "auditor-component-mismatch-audit.json"
    mismatched_audit_path.write_bytes(
        _canonical(
            _confirmation_audit_component(
                mismatched_protocol_sha256,
                "ffn",
            )
        )
    )
    mismatched_audit = json.loads(
        mismatched_audit_path.read_text(encoding="ascii")
    )
    mismatched_gate = json.loads(json.dumps(gate))
    mismatched_gate["candidate_selection"].update(
        {
            "confirmation_audit_file_sha256": _sha(mismatched_audit_path),
            "confirmation_audit_receipt_sha256": mismatched_audit[
                "receipt_sha256"
            ],
            "confirmation_protocol_file_sha256": mismatched_protocol_sha256,
            "execution_controls": mismatched_protocol["optimization"][
                "execution_controls"
            ],
        }
    )
    with pytest.raises(auditor.parent.RecoveryAuditError):
        auditor._validate_confirmation(
            protocol=mismatched_gate,
            confirmation_protocol_path=mismatched_protocol_path,
            confirmation_audit_path=mismatched_audit_path,
        )


def test_freezer_rejects_component_candidate_scope_mismatch(
    tmp_path: Path,
) -> None:
    freezer = _load("oas_recovery_8gpu_freezer_mismatch", FREEZER)
    confirmation_protocol = _confirmation_protocol_component("attention")
    confirmation_protocol["optimization"]["execution_controls"][
        "compile_scope"
    ] = "ffn"
    protocol_path = tmp_path / "confirmation-protocol-mismatch.json"
    protocol_path.write_bytes(_canonical(confirmation_protocol))
    protocol_sha256 = _sha(protocol_path)
    audit_path = tmp_path / "confirmation-audit-mismatch.json"
    audit_path.write_bytes(
        _canonical(_confirmation_audit_component(protocol_sha256, "attention"))
    )

    class Corpus:
        manifest = {"counts": {"unique_validation_records": 1_743_380}}

        def __len__(self) -> int:
            return 224_887_833

    freezer.load_production_corpus = lambda *args, **kwargs: Corpus()
    with pytest.raises(freezer.EightGpuGateFreezeError):
        freezer.freeze_gate(
            source_root=ROOT,
            corpus_root=tmp_path / "corpus",
            manifest_file_sha256="1" * 64,
            manifest_payload_sha256="2" * 64,
            corpus_protocol_file_sha256="3" * 64,
            independent_audit_path=tmp_path / "corpus-audit.json",
            independent_audit_file_sha256="4" * 64,
            independent_audit_payload_sha256="5" * 64,
            confirmation_protocol_path=protocol_path,
            confirmation_protocol_file_sha256=protocol_sha256,
            confirmation_audit_path=audit_path,
            confirmation_audit_file_sha256=_sha(audit_path),
        )


def test_launchers_use_public_partition_and_exact_three_part_recovery() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "#SBATCH --gres=gpu:8" in source
    assert "--nproc-per-node=8" in source
    assert "--target-tokens 262144" in source
    assert "--batch-quantum 512" in source
    assert "run_gate continuous 4 --skip-checkpoint" in source
    assert "run_gate stage1 2" in source
    assert (
        'run_gate resumed 2 --resume-root "${RUN_ROOT}/stage1" '
        "--skip-checkpoint"
    ) in source
    assert "--compile-blocks" in source
    assert "--compile-prewarm-buckets" in source
    for switch in (
        "--fused-adamw",
        "--pinned-staging",
        "--bucketed-padding",
        "--length-bucketing",
    ):
        assert switch not in source
    assert "COMMON_ARGS+=(--disable-compiled-cudagraphs)" in source
    assert 'controls["compile_scope"] in {"attention", "ffn"}' in source
    assert 'value["candidate_selection"].get("selected_candidate")' in source
    audit_source = AUDIT_SLURM.read_text(encoding="ascii")
    assert "audit_antibody_production_recovery_gate_8gpu.py" in audit_source
    assert '--artifact-root "${RUN_ROOT}"' in audit_source
    assert "CUDA_VISIBLE_DEVICES=" in audit_source


def test_launcher_embedded_control_gate_executes_fail_closed(
    tmp_path: Path,
) -> None:
    source = SLURM.read_text(encoding="ascii")
    embedded = source.split("<<'PY'\n", 1)[1].split("\nPY\n)", 1)[0]

    def run_gate(candidate: str, controls: dict[str, object]):
        protocol_path = tmp_path / f"{candidate}-{controls.get('compile_scope')}.json"
        protocol_path.write_bytes(
            _canonical(
                {
                    "candidate_selection": {
                        "execution_controls": controls,
                        "selected_candidate": candidate,
                    }
                }
            )
        )
        return subprocess.run(
            [sys.executable, "-", str(protocol_path)],
            input=embedded,
            check=False,
            capture_output=True,
            text=True,
        )

    attention = {
        "compile_backend": "inductor",
        "compile_scope": "attention",
        "disable_compiled_cudagraphs": True,
        "disable_python_gc": True,
    }
    passed = run_gate("inductor_attention_only", attention)
    assert passed.returncode == 0
    assert passed.stdout.strip() == "true"
    mismatched = run_gate("inductor_ffn_only", attention)
    assert mismatched.returncode != 0
    assert "Candidate execution controls differ" in mismatched.stderr
