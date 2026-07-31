"""Test the backend-attribution confirmation V2 handoff."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts/freeze_antibody_compile_candidate_confirmation_v2.py"
WRAPPER = ROOT / "scripts/run_antibody_compile_candidate_confirmation_v2.py"
AUDITOR = ROOT / "scripts/audit_antibody_compile_candidate_confirmation_v2.py"
SLURM = ROOT / "slurm/antibody_compile_candidate_confirmation_v2.sbatch"
AUDIT_SLURM = (
    ROOT / "slurm/antibody_compile_candidate_confirmation_v2_audit.sbatch"
)
ATTRIBUTION_PROTOCOL = (
    ROOT / "protocols/antibody_compile_backend_attribution_v1.json"
)


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


def _attribution_audit(protocol_sha256: str) -> dict[str, object]:
    protocol = json.loads(ATTRIBUTION_PROTOCOL.read_text(encoding="ascii"))
    value: dict[str, object] = {
        "authorization": protocol["authorization"],
        "candidate_found": True,
        "comparisons": {
            "inductor_no_cudagraphs": {
                "eligible_for_strict_confirmation": True,
                "gates": {
                    "evaluation_noninferiority": True,
                    "memory": True,
                    "short_horizon_numerical": True,
                    "step_stability": True,
                    "throughput": True,
                },
                "promotable_backend": True,
            }
        },
        "diagnostic_valid": True,
        "format_version": "antibody_compile_backend_attribution_result_v1",
        "inputs": {"protocol_file_sha256": protocol_sha256},
        "selected_candidate_for_strict_confirmation": (
            "inductor_no_cudagraphs"
        ),
        "test_access": protocol["test_access"],
    }
    value["receipt_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def _freeze(tmp_path: Path) -> tuple[object, dict[str, object], Path]:
    freezer = _load("compile_candidate_v2_freezer", FREEZER)
    protocol_sha256 = hashlib.sha256(ATTRIBUTION_PROTOCOL.read_bytes()).hexdigest()
    audit = _attribution_audit(protocol_sha256)
    audit_path = tmp_path / "attribution-audit.json"
    audit_path.write_bytes(_canonical(audit))
    frozen = freezer.freeze_confirmation_v2(
        source_root=ROOT,
        attribution_protocol_path=ATTRIBUTION_PROTOCOL,
        attribution_protocol_file_sha256=protocol_sha256,
        attribution_audit_path=audit_path,
        attribution_audit_file_sha256=hashlib.sha256(
            audit_path.read_bytes()
        ).hexdigest(),
    )
    return freezer, frozen, audit_path


def test_sources_are_ascii_and_syntax_valid() -> None:
    for path in (FREEZER, WRAPPER, AUDITOR):
        source = path.read_text(encoding="ascii")
        ast.parse(source)
        assert "os.O_EXCL" in source or path == AUDITOR


def test_freezer_preserves_backend_and_cudagraph_controls(
    tmp_path: Path,
) -> None:
    _, frozen, _ = _freeze(tmp_path)
    assert frozen["confirmation_selection"]["selected_candidate"] == (
        "inductor_no_cudagraphs"
    )
    assert frozen["confirmation_selection"]["selected_mode_flags"] == {
        "compile_backend": "inductor",
        "compile_blocks": True,
        "disable_compiled_cudagraphs": True,
        "disable_python_gc": True,
    }
    assert frozen["optimization"]["execution_controls"] == {
        "compile_backend": "inductor",
        "disable_compiled_cudagraphs": True,
        "disable_python_gc": True,
    }
    assert frozen["batch"]["updates_per_run"] == 600
    assert frozen["acceptance"]["maximum_p95_over_p50_step_time"] == 1.1
    assert frozen["acceptance"]["warmup_updates_excluded"] == 50
    assert frozen["authorization"]["production_pretraining"] is False
    for key, path in (
        ("confirmation_wrapper_sha256", WRAPPER),
        ("confirmation_auditor_sha256", AUDITOR),
        ("confirmation_launcher_sha256", SLURM),
        ("confirmation_audit_launcher_sha256", AUDIT_SLURM),
    ):
        assert frozen["inputs"][key] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_freezer_rejects_missing_candidate_and_failed_gate(
    tmp_path: Path,
) -> None:
    freezer = _load("compile_candidate_v2_freezer_reject", FREEZER)
    protocol_sha256 = hashlib.sha256(ATTRIBUTION_PROTOCOL.read_bytes()).hexdigest()
    for case in ("candidate", "gate"):
        audit = _attribution_audit(protocol_sha256)
        if case == "candidate":
            audit["candidate_found"] = False
        else:
            audit["comparisons"]["inductor_no_cudagraphs"]["gates"][
                "evaluation_noninferiority"
            ] = False
        audit.pop("receipt_sha256")
        audit["receipt_sha256"] = hashlib.sha256(_canonical(audit)).hexdigest()
        audit_path = tmp_path / f"failed-{case}.json"
        audit_path.write_bytes(_canonical(audit))
        try:
            freezer.freeze_confirmation_v2(
                source_root=ROOT,
                attribution_protocol_path=ATTRIBUTION_PROTOCOL,
                attribution_protocol_file_sha256=protocol_sha256,
                attribution_audit_path=audit_path,
                attribution_audit_file_sha256=hashlib.sha256(
                    audit_path.read_bytes()
                ).hexdigest(),
            )
        except freezer.ConfirmationV2FreezeError as error:
            assert "does not authorize" in str(error)
        else:
            raise AssertionError(f"Invalid attribution passed: {case}")


def test_wrapper_rejects_cudagraph_control_drift(tmp_path: Path) -> None:
    _, frozen, _ = _freeze(tmp_path)
    wrapper = _load("compile_candidate_v2_wrapper", WRAPPER)
    arguments = [
        "--output",
        str(tmp_path / "run"),
        "--compile-blocks",
        "--compile-prewarm-buckets",
    ]
    try:
        wrapper._validate_launch(
            frozen,
            mode="inductor_no_cudagraphs",
            arguments=arguments,
        )
    except wrapper.CompileCandidateConfirmationV2Error as error:
        assert "Compile controls differ" in str(error)
    else:
        raise AssertionError("Missing CUDAGraph control must fail closed")


def test_auditor_expects_only_the_selected_optimization() -> None:
    auditor = _load("compile_candidate_v2_auditor", AUDITOR)
    candidate = auditor._expected_optimization("fp32_compiled")
    assert candidate["compile_blocks"] is True
    assert candidate["compile_prewarm_buckets"] is True
    assert candidate["disable_compiled_cudagraphs"] is True
    assert candidate["fused_adamw"] is False
    assert candidate["pinned_staging"] is False
    assert candidate["bucketed_padding"] is False
    eager = auditor._expected_optimization("safe_eager")
    assert eager["disable_compiled_cudagraphs"] is False


def test_launchers_bind_eight_gpu_crossover_and_independent_audit() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "#SBATCH --gres=gpu:8" in source
    assert "--nproc-per-node=8" in source
    assert "--steps 600" in source
    assert "run_mode safe_eager_r1 safe_eager" in source
    assert "run_mode fp32_compiled_r2 inductor_no_cudagraphs" in source
    assert "--disable-compiled-cudagraphs" in source
    assert "--fused-adamw" not in source
    audit_source = AUDIT_SLURM.read_text(encoding="ascii")
    assert "audit_antibody_compile_candidate_confirmation_v2.py" in audit_source
    assert '--attribution-protocol "${ATTRIBUTION_PROTOCOL}"' in audit_source
    assert '--attribution-audit "${ATTRIBUTION_AUDIT}"' in audit_source
    assert "CUDA_VISIBLE_DEVICES=" in audit_source
