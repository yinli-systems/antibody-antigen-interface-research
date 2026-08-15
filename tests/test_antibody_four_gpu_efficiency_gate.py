"""Test the real-OAS four-GPU efficiency and equivalence gate."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts/freeze_antibody_four_gpu_efficiency_gate.py"
AUDITOR = ROOT / "scripts/audit_antibody_four_gpu_efficiency_gate.py"
EVALUATOR = ROOT / "scripts/evaluate_antibody_production_checkpoint.py"
PIPELINE = ROOT / "slurm/antibody_four_gpu_efficiency_pipeline.sbatch"
SUBMISSION = (
    ROOT / "results/pretraining/antibody_four_gpu_efficiency_submission_1406763.json"
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
        json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _recovery_receipt(*, exact: bool = True) -> dict[str, object]:
    payload = {
        "artifact_audit": {
            "manifest_verified": True,
            "writable_paths": [],
        },
        "authorization": {"production_pretraining": False},
        "exact_recovery": {
            "final_model_equal": exact,
            "final_optimizer_equal": True,
            "scientific_trace_equal": True,
            "terminal_coordinates_equal": True,
        },
        "format_version": "antibody_production_loader_recovery_audit_4gpu_v1",
        "gate_protocol_file_sha256": "a" * 64,
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "world_size": 4,
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def test_sources_are_ascii_syntax_valid_and_pipeline_is_valid() -> None:
    for path in (FREEZER, AUDITOR):
        source = path.read_text(encoding="ascii")
        ast.parse(source)
    subprocess.run(["bash", "-n", str(PIPELINE)], check=True)


def test_recovery_prerequisite_is_fail_closed(tmp_path: Path) -> None:
    freezer = _load("four_gpu_efficiency_freezer_test", FREEZER)
    receipt = _recovery_receipt()
    path = tmp_path / "recovery.json"
    path.write_bytes(_canonical(receipt))
    loaded = freezer._load_recovery(
        path,
        expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_payload_sha256=receipt["receipt_sha256"],
    )
    assert loaded["world_size"] == 4

    receipt = _recovery_receipt(exact=False)
    path.write_bytes(_canonical(receipt))
    with pytest.raises(
        freezer.EfficiencyFreezeError,
        match="does not authorize efficiency work",
    ):
        freezer._load_recovery(
            path,
            expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_payload_sha256=receipt["receipt_sha256"],
        )


def test_mode_gates_change_only_frozen_optimization_and_use_checkpoints() -> None:
    freezer = _load("four_gpu_efficiency_mode_test", FREEZER)
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
    base = {"execution": {"optimization": optimization}, "runs": {}}
    eager = freezer._mode_gate(base, compiled=False)
    compiled = freezer._mode_gate(base, compiled=True)
    assert eager["execution"]["optimization"]["compile_blocks"] is False
    assert compiled["execution"]["optimization"]["compile_blocks"] is True
    assert compiled["execution"]["optimization"]["disable_compiled_cudagraphs"] is True
    for gate in (eager, compiled):
        run = gate["runs"]["continuous"]
        assert run["steps"] == 600
        assert run["checkpoint"] is True
        assert run["target_residue_tokens"] == 131_072


def test_validation_panel_anchor_replays_evaluator_membership() -> None:
    freezer = _load("four_gpu_efficiency_panel_test", FREEZER)
    evaluator = _load("four_gpu_efficiency_evaluator_test", EVALUATOR)
    panel = evaluator._panel_indices(
        record_count=1_743_380,
        sample_count=freezer.PANEL_COUNT,
        seed=freezer.PANEL_SEED,
    )
    assert hashlib.sha256(panel.tobytes()).hexdigest() == (
        freezer.PANEL_INDICES_SHA256
    )


def test_bootstrap_and_mfu_use_four_gpu_denominator() -> None:
    auditor = _load("four_gpu_efficiency_auditor_test", AUDITOR)
    trace = []
    consumed = 0
    for update in range(1, 601):
        consumed += 131_072
        trace.append(
            {
                "consumed_residue_tokens": consumed,
                "gradient_norm": 1.0,
                "learning_rate": 1.0e-4,
                "loss": 1.0,
                "micro_steps": 1,
                "residue_tokens": 131_072,
                "seconds": 1.0,
                "selected_targets": 1,
                "sequence_count": 1,
                "update": update,
            }
        )
    summary = auditor._throughput({"trace": trace}, 100)
    expected = (
        6
        * auditor.PARAMETER_COUNT
        * 131_072
        / (4 * auditor.PEAK_DENSE_BF16_TFLOPS * 1.0e12)
    )
    assert summary["approximate_6n_mfu"] == pytest.approx(expected)
    indices = np.arange(32, dtype=np.int64)
    values = np.ones(32, dtype=np.float64)
    result = auditor._bootstrap_noninferiority(
        (indices, values, values),
        (indices, values, values),
        iterations=100,
        confidence=0.95,
        seed=17,
    )
    assert result["cross_entropy_relative_increase_upper"] == 0.0
    assert result["masked_accuracy_decrease_upper"] == 0.0


def test_pipeline_is_real_oas_abba_and_never_skips_checkpoints() -> None:
    source = PIPELINE.read_text(encoding="ascii")
    assert "#SBATCH --gres=gpu:4" in source
    assert "run_mode safe_eager_r1 safe_eager" in source
    assert "run_mode compiled_r1 compiled" in source
    assert "run_mode compiled_r2 compiled" in source
    assert "run_mode safe_eager_r2 safe_eager" in source
    assert source.index("run_mode safe_eager_r1") < source.index("run_mode compiled_r1")
    assert source.index("run_mode compiled_r1") < source.index("run_mode compiled_r2")
    assert source.index("run_mode compiled_r2") < source.index("run_mode safe_eager_r2")
    assert "--steps 600" in source
    assert "--panel-sample-count 8192" in source
    assert "evaluate_antibody_production_checkpoint.py" in source
    assert "--skip-checkpoint" not in source
    assert "ABF_RECOVERY_AUDIT" in source
    assert "--production-corpus-root" in source


def test_gate_sources_remain_non_authorizing() -> None:
    freezer = _load("four_gpu_efficiency_boundary_test", FREEZER)
    assert not any(freezer.TEST_ACCESS.values())
    source = AUDITOR.read_text(encoding="ascii")
    assert '"production_pretraining": False' in source
    assert '"configuration_promotion": passed' in source
    assert 'receipt.get("checkpoint_optimizer_sha256") is None' in source
    assert (
        'receipt.get("checkpoint_model_sha256")\n'
        '            != receipt.get("final_model_sha256")'
    ) in source


def test_submission_receipt_binds_exact_source_and_nonproduction_scope() -> None:
    value = json.loads(SUBMISSION.read_text(encoding="ascii"))
    assert value["job"]["job_id"] == 1406763
    assert value["job"]["state_at_receipt"] == "RUNNING"
    assert value["source"]["commit"] == (
        "ff00fb0d160145dee5622d4f30949efdf1041f70"
    )
    assert value["source"]["deployment_file_count"] == 499
    assert value["recovery_prerequisite"]["passed"] is True
    assert value["protocol"]["crossover_run_order"] == [
        "safe_eager_r1",
        "compiled_r1",
        "compiled_r2",
        "safe_eager_r2",
    ]
    assert value["authorization"]["production_pretraining"] is False
    assert not any(value["test_access"].values())
