"""Test the fail-closed compiler-backend attribution chain."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/run_antibody_compile_backend_attribution.py"
AUDITOR = ROOT / "scripts/audit_antibody_compile_backend_attribution.py"
PROTOCOL = ROOT / "protocols/antibody_compile_backend_attribution_v1.json"
SLURM = ROOT / "slurm/antibody_compile_backend_attribution.sbatch"
AUDIT_SLURM = ROOT / "slurm/antibody_compile_backend_attribution_audit.sbatch"


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


def _base_arguments(output: Path, *, compiled: bool, no_cudagraphs: bool):
    arguments = [
        "--output",
        str(output),
        "--steps",
        "500",
        "--backend",
        "ddp",
    ]
    if compiled:
        arguments.extend(("--compile-blocks", "--compile-prewarm-buckets"))
    if no_cudagraphs:
        arguments.append("--disable-compiled-cudagraphs")
    return arguments


def test_sources_are_ascii_and_syntax_valid() -> None:
    for path in (WRAPPER, AUDITOR):
        source = path.read_text(encoding="ascii")
        ast.parse(source)
        assert "os.O_EXCL" in source
    for path in (SLURM, AUDIT_SLURM):
        path.read_text(encoding="ascii")


def test_protocol_binds_every_implementation_and_input() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="ascii"))
    bindings = {
        "audit_launcher_sha256": AUDIT_SLURM,
        "auditor_sha256": AUDITOR,
        "base_runner_sha256": ROOT
        / "scripts/run_antibody_pretraining_optimized.py",
        "communication_protocol_sha256": ROOT
        / "protocols/antibody_ddp_communication_optimization_v1.json",
        "efficiency_protocol_sha256": ROOT
        / "protocols/antibody_efficiency_gate_v1.json",
        "failed_parent_audit_file_sha256": ROOT
        / "results/pretraining/antibody_compile_numerics_diagnostic_v1_1343455/independent-audit-v2.json",
        "launcher_sha256": SLURM,
        "model_family_sha256": ROOT
        / "protocols/antibody_model_family_v1.json",
        "parity_evaluation_indices_sha256": ROOT
        / "results/pretraining/antibody_ddp_parity_split_v2/evaluation_indices.npy",
        "parity_protocol_sha256": ROOT
        / "protocols/antibody_ddp_scientific_parity_gate_v2.json",
        "parity_split_receipt_file_sha256": ROOT
        / "results/pretraining/antibody_ddp_parity_split_v2/receipt.json",
        "parity_training_indices_sha256": ROOT
        / "results/pretraining/antibody_ddp_parity_split_v2/training_indices.npy",
        "training_protocol_sha256": ROOT / "protocols/antibody_training_v1.json",
        "wrapper_sha256": WRAPPER,
    }
    for key, path in bindings.items():
        assert protocol["inputs"][key] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    failed = json.loads(
        bindings["failed_parent_audit_file_sha256"].read_text(encoding="ascii")
    )
    assert failed["candidate_found"] is False
    assert protocol["failed_parent_diagnostic"]["audit_receipt_sha256"] == (
        failed["receipt_sha256"]
    )


def test_wrapper_accepts_only_registered_backend_controls(tmp_path: Path) -> None:
    wrapper = _load("compile_backend_wrapper", WRAPPER)
    protocol = json.loads(PROTOCOL.read_text(encoding="ascii"))
    cases = {
        "safe_eager": (False, False, None),
        "dynamo_eager_gc_off": (True, False, "eager"),
        "aot_eager_gc_off": (True, False, "aot_eager"),
        "inductor_no_cudagraphs_gc_off": (True, True, "inductor"),
    }
    for mode, (compiled, no_cudagraphs, backend) in cases.items():
        output = tmp_path / mode
        observed_output, flags = wrapper._validate_launch(
            protocol,
            mode=mode,
            arguments=_base_arguments(
                output,
                compiled=compiled,
                no_cudagraphs=no_cudagraphs,
            ),
            scripts_root=ROOT / "scripts",
        )
        assert observed_output == output
        assert flags["compile_backend"] == backend
    try:
        wrapper._validate_launch(
            protocol,
            mode="inductor_no_cudagraphs_gc_off",
            arguments=_base_arguments(
                tmp_path / "bad",
                compiled=True,
                no_cudagraphs=False,
            ),
            scripts_root=ROOT / "scripts",
        )
    except wrapper.BackendAttributionError as error:
        assert "CUDAGraph" in str(error)
    else:
        raise AssertionError("Inductor attribution must disable CUDAGraphs")


def test_auditor_preserves_backend_specific_receipt_controls() -> None:
    auditor = _load("compile_backend_auditor", AUDITOR)
    protocol = json.loads(PROTOCOL.read_text(encoding="ascii"))
    eager = auditor._expected_optimization(protocol["modes"]["safe_eager"])
    inductor = auditor._expected_optimization(
        protocol["modes"]["inductor_no_cudagraphs_gc_off"]
    )
    assert eager["compile_blocks"] is False
    assert eager["disable_compiled_cudagraphs"] is False
    assert inductor["compile_blocks"] is True
    assert inductor["disable_compiled_cudagraphs"] is True
    assert auditor.PROMOTABLE_RUNS == ("inductor_no_cudagraphs",)


def test_launchers_freeze_five_runs_and_cpu_only_audit() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --gres=gpu:8" in source
    assert "--nproc-per-node=8" in source
    assert source.count("run_mode ") == 5
    assert "run_mode dynamo_eager dynamo_eager_gc_off true false" in source
    assert "run_mode aot_eager aot_eager_gc_off true false" in source
    assert (
        "run_mode inductor_no_cudagraphs "
        "inductor_no_cudagraphs_gc_off true true"
    ) in source
    assert source.count("run_mode safe_eager_") == 2
    audit_source = AUDIT_SLURM.read_text(encoding="ascii")
    assert "CUDA_VISIBLE_DEVICES=" in audit_source
    assert "audit_antibody_compile_backend_attribution.py" in audit_source
