"""Validate the fail-closed efficiency factor-attribution experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/antibody_efficiency_factor_attribution_v1.json"
EXTENSION = (
    ROOT / "protocols/antibody_ddp_efficiency_factor_extension_200_seed17_v1.json"
)
RUNNER = ROOT / "slurm/antibody_efficiency_factor_attribution.sbatch"
AUDIT_RUNNER = ROOT / "slurm/antibody_efficiency_factor_attribution_audit.sbatch"
AUDITOR = ROOT / "scripts/audit_antibody_efficiency_factor_attribution.py"


def _load_auditor():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    specification = importlib.util.spec_from_file_location("factor_attribution_audit", AUDITOR)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_protocol_binds_failed_parent_and_forbids_scale_authority() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="ascii"))
    assert value["format_version"] == "antibody_efficiency_factor_attribution_protocol_v1"
    assert value["failed_parent_gate"] == {
        "audit_file_sha256": hashlib.sha256(
            (ROOT / "results/pretraining/antibody_production_batch_efficiency_v3_1342869.json").read_bytes()
        ).hexdigest(),
        "audit_job_id": 1_342_962,
        "audit_receipt_sha256": "812863e3493d204bf65205f96d5811b8e97d217d39722c5dd9f749948f1e30e0",
        "failed_gates": ["evaluation_noninferiority", "step_stability"],
        "parent_protocol_sha256": hashlib.sha256(
            (ROOT / "protocols/antibody_production_batch_efficiency_v3.json").read_bytes()
        ).hexdigest(),
        "terminal_state": "FAILED",
    }
    assert set(value["authorization"].values()) == {False}
    assert value["test_access"]["sabdab2_heldout_accessed"] is False
    assert value["selection"]["required_successor"].startswith("new source-bound 600-update")


def test_protocol_factorizes_compile_optimizer_and_data_paths() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="ascii"))
    assert value["measurement"]["run_order"] == [
        "safe_eager_r1",
        "compile_only",
        "fused_adamw_only",
        "data_pipeline_only",
        "compile_data_pipeline",
        "full_bundle",
        "safe_eager_r2",
    ]
    modes = value["modes"]
    assert not any(modes["safe_eager"].values())
    assert modes["compile_only"] == {
        "bucketed_padding": False,
        "compile_blocks": True,
        "compile_prewarm_buckets": True,
        "fused_adamw": False,
        "pinned_staging": False,
    }
    assert modes["fused_adamw_only"]["fused_adamw"] is True
    assert sum(modes["fused_adamw_only"].values()) == 1
    assert modes["data_pipeline_only"]["bucketed_padding"] is True
    assert modes["data_pipeline_only"]["pinned_staging"] is True
    assert modes["data_pipeline_only"]["compile_blocks"] is False
    assert modes["compile_data_pipeline"]["fused_adamw"] is False
    assert all(modes["full_bundle"].values())


def test_protocol_binds_all_execution_inputs() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="ascii"))
    bindings = {
        "communication_protocol_sha256": "protocols/antibody_ddp_communication_optimization_v1.json",
        "efficiency_protocol_sha256": "protocols/antibody_efficiency_gate_v1.json",
        "extension_protocol_sha256": "protocols/antibody_ddp_efficiency_factor_extension_200_seed17_v1.json",
        "mfu_protocol_sha256": "protocols/antibody_foundation_scale_v2.json",
        "model_family_sha256": "protocols/antibody_model_family_v1.json",
        "parent_auditor_sha256": "scripts/audit_antibody_production_batch_efficiency.py",
        "parity_evaluation_indices_sha256": "results/pretraining/antibody_ddp_parity_split_v2/evaluation_indices.npy",
        "parity_protocol_sha256": "protocols/antibody_ddp_scientific_parity_gate_v2.json",
        "parity_split_receipt_file_sha256": "results/pretraining/antibody_ddp_parity_split_v2/receipt.json",
        "parity_training_indices_sha256": "results/pretraining/antibody_ddp_parity_split_v2/training_indices.npy",
        "runner_sha256": "scripts/run_antibody_pretraining_optimized.py",
        "runner_wrapper_sha256": "scripts/run_antibody_pretraining_parity_extension.py",
        "training_protocol_sha256": "protocols/antibody_training_v1.json",
    }
    for key, relative in bindings.items():
        assert value["inputs"][key] == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_extension_changes_only_the_systems_run_length() -> None:
    value = json.loads(EXTENSION.read_text(encoding="ascii"))
    assert value["execution"] == {
        "gpu_count": 8,
        "parent_updates_per_run": 500,
        "run_seed": 17,
        "updates_per_run": 200,
    }
    assert value["parent_parity_protocol_sha256"] == hashlib.sha256(
        (ROOT / "protocols/antibody_ddp_scientific_parity_gate_v2.json").read_bytes()
    ).hexdigest()
    assert set(value["authorization"].values()) == {False}


def test_run_launcher_uses_one_allocation_and_exact_factor_flags() -> None:
    source = RUNNER.read_text(encoding="ascii")
    assert "#SBATCH --gres=gpu:8" in source
    assert "#SBATCH --nodes=1" in source
    assert "--steps 200" in source
    assert "--target-tokens 262144" in source
    assert "--ddp-communication-dtype float32" in source
    assert "--ddp-communication-dtype bfloat16" not in source
    for command in (
        "run_mode safe_eager_r1 safe_eager",
        "run_mode compile_only compile_only",
        "run_mode fused_adamw_only fused_adamw_only",
        "run_mode data_pipeline_only data_pipeline_only",
        "run_mode compile_data_pipeline compile_data_pipeline",
        "run_mode full_bundle full_bundle",
        "run_mode safe_eager_r2 safe_eager",
    ):
        assert command in source
    assert "--compile-blocks --compile-prewarm-buckets)" in source
    assert "--fused-adamw)" in source
    assert "--pinned-staging --bucketed-padding)" in source
    assert "--skip-checkpoint" in source


def test_audit_launcher_is_source_bound_create_only_and_fail_closed() -> None:
    source = AUDIT_RUNNER.read_text(encoding="ascii")
    assert "ABF_SOURCE_ROOT" in source
    assert "ABF_RUN_ROOT" in source
    assert "ABF_AUDIT_OUTPUT" in source
    assert "ABF_AUDITOR_SHA256" in source
    assert 'test ! -e "${OUTPUT}"' in source
    assert "export CUDA_VISIBLE_DEVICES=" in source
    assert 'exec "${PYTHON_BIN}" "${AUDITOR}"' in source
    assert "|| true" not in source


def test_auditor_derives_each_optimization_contract() -> None:
    module = _load_auditor()
    protocol = json.loads(PROTOCOL.read_text(encoding="ascii"))
    for mode, flags in protocol["modes"].items():
        observed = module._expected_optimization(flags)
        assert observed["compile_blocks"] is flags["compile_blocks"]
        assert observed["fused_adamw"] is flags["fused_adamw"]
        assert observed["pinned_staging"] is flags["pinned_staging"]
        assert observed["ddp_communication_dtype"] == "float32"
        assert observed["ddp_batched_grad_copy"] is False


def test_short_horizon_and_pooled_reference_are_exact() -> None:
    module = _load_auditor()
    reference = {
        "trace": [
            {"loss": 2.0, "gradient_norm": 1.0},
            {"loss": 1.5, "gradient_norm": 2.0},
        ]
    }
    candidate = {
        "trace": [
            {"loss": 2.001, "gradient_norm": 1.005},
            {"loss": 1.499, "gradient_norm": 2.01},
        ]
    }
    result = module._short_horizon(
        reference,
        candidate,
        limit=2,
        loss_limit=0.005,
        gradient_limit=0.01,
    )
    assert result["passed"] is True
    summaries = {
        "safe_eager_r1": {"nonpadding_residue_tokens": 100, "measured_seconds": 2.0},
        "safe_eager_r2": {"nonpadding_residue_tokens": 200, "measured_seconds": 4.0},
    }
    pooled = module._pooled_reference(summaries)
    assert pooled["residue_tokens_per_second"] == 50.0
    expected = 6 * module.PARAMETER_COUNT * 50 / (8 * module.PEAK_DENSE_BF16_TFLOPS * 1.0e12)
    assert math.isclose(pooled["approximate_6n_mfu"], expected)
