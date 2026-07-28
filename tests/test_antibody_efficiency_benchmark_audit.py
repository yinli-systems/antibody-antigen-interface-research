import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_antibody_efficiency_benchmark.py"
ATTEMPT_ONE = (
    ROOT
    / "results/pretraining/antibody_efficiency_benchmark_attempt_001_v1.json"
)
ATTEMPT_TWO = (
    ROOT
    / "results/pretraining/antibody_efficiency_benchmark_attempt_002_v1.json"
)
ATTEMPT_THREE = (
    ROOT
    / "results/pretraining/antibody_efficiency_benchmark_attempt_003_v1.json"
)
ATTEMPT_FOUR = (
    ROOT
    / "results/pretraining/antibody_efficiency_benchmark_attempt_004_v1.json"
)
CUDAGRAPH_DISABLED = (
    ROOT
    / "results/pretraining/antibody_efficiency_benchmark_cudagraph_disabled_v1.json"
)
BUCKET144_BENCHMARK = (
    ROOT / "results/pretraining/antibody_efficiency_benchmark_bucket144_v1.json"
)


def test_efficiency_benchmark_auditor_is_ascii_and_syntax_valid() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    ast.parse(source)
    assert "len(trace) != 500" in source
    assert '"memory"' in source
    assert '"membership"' in source
    assert '"stability"' in source
    assert '"throughput"' in source
    assert '"--require-disabled-compiled-cudagraphs"' in source
    assert '"--expected-length-bucket-boundaries"' in source
    assert '"disable_compiled_cudagraphs": True' in source


def test_efficiency_benchmark_auditor_preserves_failed_results() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    assert '"gate_passed": passed' in source
    assert '"production_pretraining": False' in source
    assert '"recovery_required_before_scale": True' in source
    assert '"scale_16_gpu": False' in source
    assert "os.O_EXCL" in source
    assert "raise RuntimeError(f\"Efficiency benchmark failed" not in source


def test_variable_width_attempt_preserves_stability_failure() -> None:
    import json

    result = json.loads(ATTEMPT_ONE.read_text(encoding="ascii"))
    assert result["label"] == "variable_width_compile_attempt"
    assert result["gate_passed"] is False
    assert result["gates"] == {
        "membership": True,
        "memory": True,
        "stability": False,
        "throughput": True,
    }
    assert result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] > 1.29
    assert result["summaries"]["compiled"]["p95_over_p50"] > 1.24
    assert result["authorization"]["scale_16_gpu"] is False


def test_bucketed_lazy_compile_attempt_preserves_late_compile_failure() -> None:
    import json

    result = json.loads(ATTEMPT_TWO.read_text(encoding="ascii"))
    assert result["label"] == "bucketed_lazy_compile_attempt"
    assert result["gate_passed"] is False
    assert result["gates"] == {
        "membership": True,
        "memory": True,
        "stability": False,
        "throughput": False,
    }
    assert result["summaries"]["compiled"]["maximum_step_seconds"] > 7.0
    assert result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] < 0.81
    assert result["authorization"]["scale_16_gpu"] is False


def test_prewarm_attempt_preserves_near_threshold_failure() -> None:
    import json

    result = json.loads(ATTEMPT_THREE.read_text(encoding="ascii"))
    assert result["label"] == "prewarmed_bucket_attempt"
    assert result["gate_passed"] is False
    assert result["gates"] == {
        "membership": True,
        "memory": True,
        "stability": False,
        "throughput": False,
    }
    assert 1.24 < result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] < 1.25
    assert result["summaries"]["compiled"]["p95_over_p50"] > 1.15
    assert result["authorization"]["scale_16_gpu"] is False


def test_low_frequency_telemetry_attempt_preserves_exact_failure() -> None:
    import json

    result = json.loads(ATTEMPT_FOUR.read_text(encoding="ascii"))
    assert result["label"] == "low_frequency_telemetry_attempt"
    assert result["gate_passed"] is False
    assert result["gates"] == {
        "membership": True,
        "memory": True,
        "stability": False,
        "throughput": True,
    }
    assert result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] > 1.25
    assert 1.10 < result["summaries"]["compiled"][
        "p95_over_p50"
    ] < 1.101
    assert result["authorization"]["scale_16_gpu"] is False


def test_cudagraph_disabled_candidate_preserves_stability_failure() -> None:
    import json

    result = json.loads(CUDAGRAPH_DISABLED.read_text(encoding="ascii"))
    assert result["label"] == "compiled_cudagraph_disabled_candidate"
    assert result["required_disabled_compiled_cudagraphs"] is True
    assert result["gate_passed"] is False
    assert result["gates"] == {
        "membership": True,
        "memory": True,
        "stability": False,
        "throughput": True,
    }
    assert result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] > 1.28
    assert result["summaries"]["compiled"]["p95_over_p50"] > 1.12
    assert result["authorization"]["scale_16_gpu"] is False


def test_bucket144_candidate_passes_all_frozen_eight_gpu_gates() -> None:
    import json

    result = json.loads(BUCKET144_BENCHMARK.read_text(encoding="ascii"))
    assert result["label"] == "narrow_bucket_compiled_candidate"
    assert result["expected_length_bucket_boundaries"] == [96, 128, 144, 194]
    assert result["required_disabled_compiled_cudagraphs"] is True
    assert result["gate_passed"] is True
    assert all(result["gates"].values())
    assert result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] > 1.30
    assert result["summaries"]["compiled"]["p95_over_p50"] < 1.051
    assert result["summaries"]["compiled"]["maximum_memory_fraction"] < 0.60
    assert result["authorization"] == {
        "production_pretraining": False,
        "recovery_required_before_scale": True,
        "scale_16_gpu": False,
        "scale_32_gpu": False,
    }
