"""Validate the fail-closed compile numerics and Python-GC diagnostic."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/antibody_compile_numerics_diagnostic_v1.json"
WRAPPER = ROOT / "scripts/run_antibody_compile_numerics_diagnostic.py"
AUDITOR = ROOT / "scripts/audit_antibody_compile_numerics_diagnostic.py"
RUNNER = ROOT / "slurm/antibody_compile_numerics_diagnostic.sbatch"
AUDIT_RUNNER = ROOT / "slurm/antibody_compile_numerics_diagnostic_audit.sbatch"


def _load_auditor():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    specification = importlib.util.spec_from_file_location(
        "compile_numerics_auditor", AUDITOR
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_protocol_binds_failed_parent_and_forbids_promotion() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="ascii"))
    assert value["format_version"] == (
        "antibody_compile_numerics_diagnostic_protocol_v1"
    )
    assert value["failed_parent_gate"] == {
        "audit_file_sha256": hashlib.sha256(
            (
                ROOT
                / "results/pretraining/antibody_compile_only_confirmation_v1_1343024"
                / "antibody_compile_only_confirmation_v1_1343024.json"
            ).read_bytes()
        ).hexdigest(),
        "audit_job_id": 1_343_025,
        "audit_receipt_sha256": (
            "9887b28baedac3ea465f4add59a5e340e4c01d4402877c0c4b0d6be65a65d317"
        ),
        "failed_gates": ["evaluation_noninferiority", "step_stability"],
        "training_job_id": 1_343_024,
    }
    assert set(value["authorization"].values()) == {False}
    assert value["test_access"]["sabdab2_heldout_accessed"] is False
    assert value["selection"]["required_successor"].startswith(
        "new source-bound two-repetition"
    )


def test_protocol_factorizes_accuracy_and_gc_controls() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="ascii"))
    assert value["measurement"]["run_order"] == [
        "safe_eager_r1",
        "compile_default",
        "compile_gc_off",
        "compile_accuracy",
        "compile_accuracy_gc_off",
        "safe_eager_r2",
    ]
    assert value["batch"] == {
        "batch_quantum_sequences": 512,
        "global_residue_token_target_per_update": 262144,
        "micro_batch_per_gpu": 16,
        "updates_per_run": 600,
    }
    modes = value["modes"]
    assert modes["safe_eager"] == {
        "accuracy_first_bf16_reduction": False,
        "compile_blocks": False,
        "disable_python_gc": False,
    }
    assert modes["compile_default"]["compile_blocks"] is True
    assert modes["compile_gc_off"]["disable_python_gc"] is True
    assert modes["compile_accuracy"]["accuracy_first_bf16_reduction"] is True
    assert modes["compile_accuracy_gc_off"] == {
        "accuracy_first_bf16_reduction": True,
        "compile_blocks": True,
        "disable_python_gc": True,
    }


def test_protocol_binds_every_execution_input() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="ascii"))
    bindings = {
        "audit_launcher_sha256": "slurm/antibody_compile_numerics_diagnostic_audit.sbatch",
        "auditor_sha256": "scripts/audit_antibody_compile_numerics_diagnostic.py",
        "base_runner_sha256": "scripts/run_antibody_pretraining_optimized.py",
        "communication_protocol_sha256": "protocols/antibody_ddp_communication_optimization_v1.json",
        "diagnostic_wrapper_sha256": "scripts/run_antibody_compile_numerics_diagnostic.py",
        "efficiency_protocol_sha256": "protocols/antibody_efficiency_gate_v1.json",
        "extension_protocol_sha256": "protocols/antibody_ddp_scientific_parity_extension_600_seed17_v1.json",
        "failed_parent_audit_file_sha256": "results/pretraining/antibody_compile_only_confirmation_v1_1343024/antibody_compile_only_confirmation_v1_1343024.json",
        "launcher_sha256": "slurm/antibody_compile_numerics_diagnostic.sbatch",
        "mfu_protocol_sha256": "protocols/antibody_foundation_scale_v2.json",
        "model_family_sha256": "protocols/antibody_model_family_v1.json",
        "parity_evaluation_indices_sha256": "results/pretraining/antibody_ddp_parity_split_v2/evaluation_indices.npy",
        "parity_protocol_sha256": "protocols/antibody_ddp_scientific_parity_gate_v2.json",
        "parity_split_receipt_file_sha256": "results/pretraining/antibody_ddp_parity_split_v2/receipt.json",
        "parity_training_indices_sha256": "results/pretraining/antibody_ddp_parity_split_v2/training_indices.npy",
        "parity_wrapper_sha256": "scripts/run_antibody_pretraining_parity_extension.py",
        "training_protocol_sha256": "protocols/antibody_training_v1.json",
    }
    for key, relative in bindings.items():
        assert value["inputs"][key] == hashlib.sha256(
            (ROOT / relative).read_bytes()
        ).hexdigest()


def test_wrapper_is_ascii_syntax_valid_and_restores_process_state() -> None:
    source = WRAPPER.read_text(encoding="ascii")
    ast.parse(source)
    assert "gc.collect()" in source
    assert "gc.disable()" in source
    assert "gc.enable()" in source
    assert "allow_bf16_reduced_precision_reduction" in source
    assert 'raise CompileNumericsDiagnosticError(' in source
    assert "stats_delta" in source
    assert "os.O_EXCL" in source
    assert '"configuration_promotion": False' in source


def test_launcher_uses_one_allocation_and_exact_six_run_design() -> None:
    source = RUNNER.read_text(encoding="ascii")
    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --gres=gpu:8" in source
    assert "--steps 600" in source
    assert "--target-tokens 262144" in source
    assert "--ddp-communication-dtype float32" in source
    for command in (
        "run_mode safe_eager_r1 safe_eager false",
        "run_mode compile_default compile_default true",
        "run_mode compile_gc_off compile_gc_off true",
        "run_mode compile_accuracy compile_accuracy true",
        "run_mode compile_accuracy_gc_off compile_accuracy_gc_off true",
        "run_mode safe_eager_r2 safe_eager false",
    ):
        assert command in source
    assert "--compile-blocks --compile-prewarm-buckets" in source
    assert "--skip-checkpoint" in source
    assert "chmod -R a-w" in source


def test_auditor_and_audit_launcher_fail_closed() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    ast.parse(source)
    assert "eligible_for_strict_confirmation" in source
    assert "selected_candidate_for_strict_confirmation" in source
    assert "maximum_masked_cross_entropy_relative_increase" in source
    assert "maximum_p95_over_p50_step_time" in source
    assert '"configuration_promotion": False' not in source
    launcher = AUDIT_RUNNER.read_text(encoding="ascii")
    assert 'test ! -e "${OUTPUT}"' in launcher
    assert "export CUDA_VISIBLE_DEVICES=" in launcher
    assert 'exec "${PYTHON_BIN}" "${AUDITOR}"' in launcher
    assert "|| true" not in launcher


def test_auditor_derives_exact_eager_and_compiled_contracts() -> None:
    module = _load_auditor()
    eager = module._expected_optimization(False)
    compiled = module._expected_optimization(True)
    assert eager["compile_blocks"] is False
    assert eager["compile_prewarm_buckets"] is False
    assert compiled["compile_blocks"] is True
    assert compiled["compile_prewarm_buckets"] is True
    for field in ("bucketed_padding", "fused_adamw", "pinned_staging"):
        assert eager[field] is False
        assert compiled[field] is False


def test_slow_step_overlap_is_index_aligned() -> None:
    module = _load_auditor()
    left = {"trace": [{"seconds": float(index + 1)} for index in range(20)]}
    right = {"trace": [{"seconds": float(index + 1) * 2.0} for index in range(20)]}
    value = module._slow_step_overlap(left, right, warmup=0)
    assert value["pearson_step_time"] == 1.0
    assert value["slowest_10_percent_jaccard"] == 1.0
