"""Test the fail-closed compile-candidate confirmation handoff."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts/freeze_antibody_compile_candidate_confirmation.py"
WRAPPER = ROOT / "scripts/run_antibody_compile_candidate_confirmation.py"
AUDITOR = ROOT / "scripts/audit_antibody_compile_candidate_confirmation.py"
SLURM = ROOT / "slurm/antibody_compile_candidate_confirmation.sbatch"
AUDIT_SLURM = ROOT / "slurm/antibody_compile_candidate_confirmation_audit.sbatch"
DIAGNOSTIC_PROTOCOL = ROOT / "protocols/antibody_compile_numerics_diagnostic_v1.json"


def _load(name: str, path: Path):
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
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


def _diagnostic_audit(protocol_sha256: str) -> dict[str, object]:
    value: dict[str, object] = {
        "authorization": {
            "configuration_promotion": False,
            "production_pretraining": False,
            "scale_16_gpu": False,
            "scale_32_gpu": False,
            "scientific_claim": False,
        },
        "candidate_found": True,
        "comparisons": {
            "compile_accuracy_gc_off": {
                "eligible_for_strict_confirmation": True,
                "gates": {
                    "evaluation_noninferiority": True,
                    "memory": True,
                    "short_horizon_numerical": True,
                    "step_stability": True,
                    "throughput": True,
                },
            }
        },
        "diagnostic_valid": True,
        "format_version": "antibody_compile_numerics_diagnostic_result_v1",
        "inputs": {"protocol_file_sha256": protocol_sha256},
        "selected_candidate_for_strict_confirmation": (
            "compile_accuracy_gc_off"
        ),
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
        assert "os.O_EXCL" in source or path == AUDITOR


def test_freezer_selects_only_the_audited_candidate(tmp_path: Path) -> None:
    freezer = _load("compile_candidate_freezer", FREEZER)
    protocol_sha256 = hashlib.sha256(DIAGNOSTIC_PROTOCOL.read_bytes()).hexdigest()
    audit = _diagnostic_audit(protocol_sha256)
    audit_path = tmp_path / "diagnostic-audit.json"
    audit_path.write_bytes(_canonical(audit))
    result = freezer.freeze_confirmation(
        source_root=ROOT,
        diagnostic_protocol_path=DIAGNOSTIC_PROTOCOL,
        diagnostic_protocol_file_sha256=protocol_sha256,
        diagnostic_audit_path=audit_path,
        diagnostic_audit_file_sha256=hashlib.sha256(
            audit_path.read_bytes()
        ).hexdigest(),
    )
    assert result["confirmation_selection"]["selected_candidate"] == (
        "compile_accuracy_gc_off"
    )
    assert result["optimization"]["execution_controls"] == {
        "accuracy_first_bf16_reduction": True,
        "disable_python_gc": True,
    }
    assert result["batch"]["updates_per_run"] == 600
    assert result["acceptance"]["maximum_p95_over_p50_step_time"] == 1.1
    assert result["authorization"]["production_pretraining"] is False
    assert result["inputs"]["confirmation_wrapper_sha256"] == hashlib.sha256(
        WRAPPER.read_bytes()
    ).hexdigest()
    assert result["inputs"]["confirmation_auditor_sha256"] == hashlib.sha256(
        AUDITOR.read_bytes()
    ).hexdigest()


def test_freezer_rejects_a_diagnostic_without_a_candidate(tmp_path: Path) -> None:
    freezer = _load("compile_candidate_freezer_reject", FREEZER)
    protocol_sha256 = hashlib.sha256(DIAGNOSTIC_PROTOCOL.read_bytes()).hexdigest()
    audit = _diagnostic_audit(protocol_sha256)
    audit["candidate_found"] = False
    audit.pop("receipt_sha256")
    audit["receipt_sha256"] = hashlib.sha256(_canonical(audit)).hexdigest()
    audit_path = tmp_path / "failed-audit.json"
    audit_path.write_bytes(_canonical(audit))
    try:
        freezer.freeze_confirmation(
            source_root=ROOT,
            diagnostic_protocol_path=DIAGNOSTIC_PROTOCOL,
            diagnostic_protocol_file_sha256=protocol_sha256,
            diagnostic_audit_path=audit_path,
            diagnostic_audit_file_sha256=hashlib.sha256(
                audit_path.read_bytes()
            ).hexdigest(),
        )
    except freezer.ConfirmationFreezeError as error:
        assert "does not authorize" in str(error)
    else:
        raise AssertionError("A failed diagnostic must not freeze a confirmation")


def test_confirmation_auditor_excludes_unselected_optimizations() -> None:
    auditor = _load("compile_candidate_auditor", AUDITOR)
    candidate = auditor._expected_optimization("fp32_compiled")
    assert candidate["compile_blocks"] is True
    assert candidate["compile_prewarm_buckets"] is True
    assert candidate["fused_adamw"] is False
    assert candidate["pinned_staging"] is False
    assert candidate["bucketed_padding"] is False
    assert candidate["ddp_communication_dtype"] == "float32"


def test_launchers_bind_eight_gpu_crossover_and_independent_audit() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "#SBATCH --gres=gpu:8" in source
    assert "--nproc-per-node=8" in source
    assert "--steps 600" in source
    assert "run_mode safe_eager_r1 safe_eager" in source
    assert 'run_mode fp32_compiled_r2 "${SELECTED_MODE}"' in source
    assert "--compile-blocks --compile-prewarm-buckets" in source
    assert "--fused-adamw" not in source
    audit_source = AUDIT_SLURM.read_text(encoding="ascii")
    assert "audit_antibody_compile_candidate_confirmation.py" in audit_source
    assert '--diagnostic-protocol "${DIAGNOSTIC_PROTOCOL}"' in audit_source
    assert '--diagnostic-audit "${DIAGNOSTIC_AUDIT}"' in audit_source
    assert "CUDA_VISIBLE_DEVICES=" in audit_source
