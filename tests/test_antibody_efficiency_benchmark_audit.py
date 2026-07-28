import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_antibody_efficiency_benchmark.py"
ATTEMPT_ONE = (
    ROOT
    / "results/pretraining/antibody_efficiency_benchmark_attempt_001_v1.json"
)


def test_efficiency_benchmark_auditor_is_ascii_and_syntax_valid() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    ast.parse(source)
    assert "len(trace) != 500" in source
    assert '"memory"' in source
    assert '"membership"' in source
    assert '"stability"' in source
    assert '"throughput"' in source


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
