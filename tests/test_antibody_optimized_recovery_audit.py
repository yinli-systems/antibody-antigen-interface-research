import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_antibody_optimized_recovery.py"
ATTEMPT_ONE = (
    ROOT / "results/pretraining/antibody_optimized_recovery_attempt_001_v1.json"
)
ATTEMPT_TWO = (
    ROOT / "results/pretraining/antibody_optimized_recovery_attempt_002_v1.json"
)
ATTEMPT_THREE = (
    ROOT / "results/pretraining/antibody_optimized_recovery_attempt_003_v1.json"
)


def test_optimized_recovery_auditor_is_ascii_and_syntax_valid() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    ast.parse(source)
    assert "continuous_state == resumed_state" in source
    assert '"prefix_trace": _trace_equal' in source
    assert '"resumed_trace": _trace_equal' in source
    assert '"length_bucket_boundaries": EXPECTED_BUCKETS' in source
    assert '"--require-enabled-compiled-cudagraphs"' in source
    assert '"--require-deterministic-algorithms"' in source
    assert '"--require-same-hostname"' in source
    assert '"run_hostname"' in source
    assert '"--rng-manifest"' in source
    assert '"rng_manifest"' in source
    assert '"loaded_model"' in source
    assert '"loaded_optimizer"' in source
    assert 'f"rng/rank-{expected_rank:05d}.json"' in source
    assert "Recovery RNG rank file differs" in source
    assert "recorded_state_sha256 != observed_state_sha256" in source


def test_optimized_recovery_auditor_fails_closed_before_scale() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    assert "passed = all(exact.values())" in source
    assert '"pilot_16_gpu": passed' in source
    assert '"pilot_32_gpu": False' in source
    assert '"production_pretraining": False' in source
    assert '"scientific_claim": False' in source
    assert '"recovery_passed": passed' in source
    assert "os.O_EXCL" in source


def test_first_optimized_recovery_attempt_preserves_nondeterminism() -> None:
    import json

    result = json.loads(ATTEMPT_ONE.read_text(encoding="ascii"))
    assert result["recovery_passed"] is False
    assert result["exact_equal"] == {
        "final_model": False,
        "final_optimizer": False,
        "prefix_trace": False,
        "resumed_trace": False,
        "trainer_state": True,
    }
    assert result["trace_differences"][
        "prefix_maximum_loss_absolute_difference"
    ] > 0.0001
    assert result["authorization"]["pilot_16_gpu"] is False
    assert result["authorization"]["pilot_32_gpu"] is False


def test_second_attempt_is_deterministic_before_but_not_after_resume() -> None:
    import json

    result = json.loads(ATTEMPT_TWO.read_text(encoding="ascii"))
    assert result["recovery_passed"] is False
    assert result["required_deterministic_algorithms"] is True
    assert result["required_enabled_compiled_cudagraphs"] is True
    assert result["required_same_hostname"] is True
    assert result["run_hostname"] == "wqd10nah10g2"
    assert result["exact_equal"] == {
        "final_model": False,
        "final_optimizer": False,
        "prefix_trace": True,
        "resumed_trace": False,
        "trainer_state": True,
    }
    assert result["trace_differences"][
        "prefix_maximum_loss_absolute_difference"
    ] == 0.0
    assert result["trace_differences"][
        "resumed_maximum_loss_absolute_difference"
    ] > 0.001
    assert result["authorization"]["pilot_16_gpu"] is False


def test_third_attempt_authenticates_rng_but_preserves_graph_rng_failure() -> None:
    import json

    result = json.loads(ATTEMPT_THREE.read_text(encoding="ascii"))
    assert result["recovery_passed"] is False
    assert result["job_ids"] == ["1336809", "1336810", "1336811"]
    assert result["run_hostname"] == "wqd10nah10g2"
    assert result["exact_equal"] == {
        "final_model": False,
        "final_optimizer": False,
        "prefix_trace": True,
        "resumed_trace": False,
        "rng_manifest": True,
        "trainer_state": True,
    }
    assert result["input_sha256"]["rng_manifest"] == (
        "7d948089f9ff537bfc8f1e1be7a2844fcdce86555883e60c2c5779e34ae40923"
    )
    assert result["trace_differences"][
        "prefix_maximum_loss_absolute_difference"
    ] == 0.0
    assert result["trace_differences"][
        "resumed_maximum_loss_absolute_difference"
    ] > 0.001
    assert result["authorization"]["pilot_16_gpu"] is False
    assert result["authorization"]["pilot_32_gpu"] is False
