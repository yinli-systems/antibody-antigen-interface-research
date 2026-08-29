"""Test the non-authorizing four-GPU OAS shape-stability probe."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts/freeze_antibody_four_gpu_shape_stability_probe.py"
AUDITOR = ROOT / "scripts/audit_antibody_four_gpu_shape_stability_probe.py"
PIPELINE = ROOT / "slurm/antibody_four_gpu_shape_stability_probe.sbatch"
SUBMISSION = (
    ROOT / "results/pretraining/antibody_four_gpu_shape_stability_submission_1406872.json"
)
COMPLETION = (
    ROOT / "results/pretraining/antibody_four_gpu_shape_stability_completion_1406872.json"
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


def _base_gate() -> dict[str, object]:
    optimization = {
        "bucketed_padding": True,
        "compile_blocks": True,
        "compile_prewarm_buckets": True,
        "ddp_bucket_cap_mb": 25,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": True,
        "fused_adamw": True,
        "length_bucketing": True,
        "pinned_staging": True,
    }
    return {
        "execution": {
            "backend": "ddp",
            "length_bucket_boundaries": "96,128,160,194",
            "optimization": optimization,
            "world_size": 4,
        },
        "runs": {},
    }


def test_sources_are_ascii_syntax_valid_and_pipeline_is_valid() -> None:
    for path in (FREEZER, AUDITOR):
        source = path.read_text(encoding="ascii")
        ast.parse(source)
    subprocess.run(["bash", "-n", str(PIPELINE)], check=True)


def test_candidate_gate_changes_only_frozen_shape_contract() -> None:
    freezer = _load("shape_probe_freezer_gate_test", FREEZER)
    bucketed = freezer._candidate_gate(_base_gate(), "bucketed")
    static = freezer._candidate_gate(_base_gate(), "static_194")
    assert bucketed["execution"]["length_bucket_boundaries"] == "96,128,160,194"
    assert static["execution"]["length_bucket_boundaries"] == "194"
    for gate in (bucketed, static):
        assert gate["runs"]["continuous"] == {
            "batch_quantum_sequences": 256,
            "checkpoint": False,
            "micro_batch_per_gpu": 16,
            "preset": "abf_650m",
            "resume": False,
            "run_seed": 17,
            "steps": 300,
            "target_residue_tokens": 131072,
        }
        assert gate["execution"]["optimization"]["compile_blocks"] is True
        assert gate["execution"]["optimization"]["length_bucketing"] is True


def test_acceptance_keeps_failed_efficiency_thresholds_unchanged() -> None:
    freezer = _load("shape_probe_freezer_acceptance_test", FREEZER)
    assert freezer.ACCEPTANCE["minimum_approximate_6n_mfu"] == 0.35
    assert freezer.ACCEPTANCE["maximum_p95_over_p50_step_time"] == 1.10
    assert freezer.ACCEPTANCE["maximum_memory_fraction"] == 0.80
    assert freezer.ACCEPTANCE["maximum_replicate_throughput_relative_range"] == 0.05
    assert not any(freezer.TEST_ACCESS.values())
    assert freezer.AUTHORIZATION == {
        "configuration_promotion": False,
        "full_efficiency_gate_candidate": False,
        "production_pretraining": False,
        "scientific_claim": False,
    }


def test_configuration_summary_fails_raw_step_instability() -> None:
    freezer = _load("shape_probe_freezer_summary_test", FREEZER)
    auditor = _load("shape_probe_auditor_summary_test", AUDITOR)
    summaries = {
        "a": {
            "maximum_memory_fraction": 0.6,
            "measured_seconds": 100.0,
            "nonpadding_residue_tokens": 10_000_000,
            "p95_over_p50": 1.09,
            "residue_tokens_per_second": 100_000.0,
        },
        "b": {
            "maximum_memory_fraction": 0.6,
            "measured_seconds": 100.0,
            "nonpadding_residue_tokens": 10_000_000,
            "p95_over_p50": 1.11,
            "residue_tokens_per_second": 101_000.0,
        },
    }
    receipts = {
        name: {"final_model_sha256": "a", "final_optimizer_sha256": "b"}
        for name in summaries
    }
    value = auditor._configuration_summary(
        ["a", "b"], summaries, receipts, freezer.ACCEPTANCE
    )
    assert value["gates"]["approximate_mfu"] is True
    assert value["gates"]["step_stability"] is False
    assert value["eligible"] is False


def test_pipeline_is_four_gpu_real_oas_abba_without_checkpoint_or_evaluation() -> None:
    source = PIPELINE.read_text(encoding="ascii")
    assert "#SBATCH --gres=gpu:4" in source
    assert "run_candidate bucketed_r1 bucketed 96,128,160,194" in source
    assert "run_candidate static_194_r1 static_194 194" in source
    assert "run_candidate static_194_r2 static_194 194" in source
    assert "run_candidate bucketed_r2 bucketed 96,128,160,194" in source
    assert source.index("run_candidate bucketed_r1") < source.index(
        "run_candidate static_194_r1"
    )
    assert source.index("run_candidate static_194_r1") < source.index(
        "run_candidate static_194_r2"
    )
    assert source.index("run_candidate static_194_r2") < source.index(
        "run_candidate bucketed_r2"
    )
    assert "--steps 300" in source
    assert "--skip-checkpoint" in source
    assert "evaluate_antibody_production_checkpoint.py" not in source
    assert "--production-corpus-root" in source
    assert "ABF_RECOVERY_AUDIT" in source


def test_source_closure_binds_every_imported_and_executed_probe_file() -> None:
    freezer = _load("shape_probe_freezer_sources_test", FREEZER)
    assert set(freezer.SOURCE_FILES) == {
        "scripts/audit_antibody_four_gpu_efficiency_gate.py",
        "scripts/audit_antibody_four_gpu_shape_stability_probe.py",
        "scripts/audit_antibody_production_batch_efficiency.py",
        "scripts/freeze_antibody_four_gpu_efficiency_gate.py",
        "scripts/freeze_antibody_four_gpu_recovery_gate.py",
        "scripts/freeze_antibody_four_gpu_shape_stability_probe.py",
        "scripts/run_antibody_pretraining_production_gate.py",
        "slurm/antibody_four_gpu_shape_stability_probe.sbatch",
    }


def test_result_authorization_can_only_advance_to_full_efficiency_gate() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    assert '"configuration_promotion": False' in source
    assert '"production_pretraining": False' in source
    assert '"scientific_claim": False' in source
    assert '"full_efficiency_gate_candidate": passed' in source
    assert '"selected_candidate_for_full_efficiency_gate"' in source


def test_submission_receipt_binds_exact_source_protocol_and_nonproduction_scope() -> None:
    value = json.loads(SUBMISSION.read_text(encoding="ascii"))
    assert value["job"]["job_id"] == 1406872
    assert value["job"]["requested_gpus"] == 4
    assert value["job"]["state_at_receipt"] == "PENDING"
    assert value["source"]["commit"] == (
        "095c15ac9adb59e0b055005d307d47accf175cec"
    )
    assert value["preflight"]["protocol_file_sha256"] == (
        "915973752eac8f286bab171b51a33950e33b1e2c8d01ad483986c9f24b4436ca"
    )
    assert value["protocol"]["candidates"] == {
        "bucketed": "96,128,160,194",
        "static_194": "194",
    }
    assert value["authorization"]["production_pretraining"] is False
    assert not any(value["test_access"].values())


def test_completion_receipt_only_promotes_static_shape_to_full_gate() -> None:
    value = json.loads(COMPLETION.read_text(encoding="ascii"))
    assert value["job"] == {
        "elapsed": "00:41:02",
        "exit_code": "0:0",
        "job_id": 1406872,
        "output_root": (
            "/ssd/scxi253/model_deployment/results/pretraining/"
            "antibody_four_gpu_shape_stability_095c15a_attempt_001"
        ),
        "state": "COMPLETED",
    }
    assert value["outcome"]["selected_candidate"] == "static_194"
    assert value["outcome"]["static_194"]["eligible"] is True
    assert value["outcome"]["bucketed"]["eligible"] is False
    assert value["outcome"]["static_194"]["pooled_approximate_6n_mfu"] >= 0.35
    assert max(value["outcome"]["static_194"]["run_p95_over_p50"]) <= 1.10
    assert value["authorization"] == {
        "configuration_promotion": False,
        "full_efficiency_gate_candidate": True,
        "production_pretraining": False,
        "scientific_claim": False,
    }
    assert not any(value["test_access"].values())
