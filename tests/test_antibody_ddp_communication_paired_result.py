import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / (
    "results/pretraining/"
    "antibody_ddp_communication_paired_gate_attempt_013_v1.json"
)
RUNS = (
    "safe_eager_r1",
    "fp32_compiled_r1",
    "bf16_selected_r1",
    "bf16_selected_r2",
    "fp32_compiled_r2",
    "safe_eager_r2",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(run: str) -> dict[str, object]:
    path = ROOT / (
        "results/pretraining/"
        f"antibody_ddp_communication_paired_gate_attempt_013_{run}_receipt.json"
    )
    return json.loads(path.read_text(encoding="ascii"))


def test_result_binds_every_raw_run_artifact() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    assert set(result["runs"]) == set(RUNS)
    for run in RUNS:
        evidence = result["runs"][run]
        prefix = ROOT / (
            "results/pretraining/"
            f"antibody_ddp_communication_paired_gate_attempt_013_{run}"
        )
        assert _sha256(Path(f"{prefix}_receipt.json")) == evidence["receipt_sha256"]
        assert _sha256(Path(f"{prefix}_trainer_state.json")) == (
            evidence["trainer_state_sha256"]
        )
        assert _sha256(Path(f"{prefix}_gpu.csv")) == evidence["telemetry_sha256"]
        assert evidence["job_id"] == "1337792"
        assert evidence["slurm_state"] == "COMPLETED"


def test_every_run_preserves_exact_eight_rank_state_parity() -> None:
    for run in RUNS:
        receipt = _receipt(run)
        assert len(receipt["trace"]) == 500
        assert receipt["optimization"]["ddp_static_graph"] is False
        assert receipt["optimization"]["audit_gradient_rank_parity"] is True
        assert len(set(receipt["initial_model_rank_sha256"])) == 1
        assert len(set(receipt["final_model_rank_sha256"])) == 1
        assert len(set(receipt["final_optimizer_rank_sha256"])) == 1
        assert receipt["ddp_logging_data"]["num_buckets_reduced"] == 65
        assert "can_set_static_graph" not in receipt["ddp_logging_data"]


def test_counterbalanced_efficiency_evidence_is_strong_and_repeatable() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    assert result["membership_equal"] is True
    assert all(result["replicate_exact"].values())
    assert max(result["replicate_throughput_relative_ranges"].values()) < 0.02
    assert result["pooled_throughput_relative_to_safe_eager"][
        "bf16_selected"
    ] > 1.50
    assert min(
        result["round_selected_throughput_relative_to_safe_eager"].values()
    ) > 1.47
    assert max(
        summary["p95_over_p50"] for summary in result["summaries"].values()
    ) < 1.05
    assert max(
        summary["maximum_memory_fraction"]
        for summary in result["summaries"].values()
    ) < 0.67


def test_numerical_failure_blocks_recovery_and_scale() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    assert result["gate_passed"] is False
    assert result["gates"] == {
        "allocation": True,
        "membership": True,
        "memory": True,
        "numerical_correctness": False,
        "rank_parity": True,
        "replicate_exact": True,
        "replicate_throughput_range": True,
        "stability": True,
        "throughput_each_round": True,
        "throughput_pooled": True,
    }
    assert not any(
        comparison["passed"]
        for comparison in result["comparisons_to_safe_eager"].values()
    )
    assert result["authorization"] == {
        "exact_recovery": False,
        "pilot_16_gpu": False,
        "pilot_32_gpu": False,
        "production_pretraining": False,
        "scientific_claim": False,
    }


def test_result_binds_frozen_auditor_and_allocation() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    assert result["source"]["auditor_sha256"] == _sha256(
        ROOT / "scripts/audit_antibody_ddp_communication_paired_gate.py"
    )
    assert result["allocation"]["receipt_sha256"] == _sha256(
        ROOT
        / "results/pretraining/"
        "antibody_ddp_communication_paired_gate_attempt_013_allocation.json"
    )
    assert result["allocation"]["job_id"] == "1337792"
    assert result["allocation"]["hostname"] == "wqd10nah10g2"
