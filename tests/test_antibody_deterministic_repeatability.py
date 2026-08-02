import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_antibody_deterministic_repeatability.py"
RESULT = ROOT / "results/pretraining/antibody_deterministic_repeatability_v1.json"


def test_deterministic_repeatability_auditor_is_ascii_and_valid() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    ast.parse(source)
    assert '"deterministic_algorithms": True' in source
    assert '"final_model"' in source
    assert '"final_optimizer"' in source
    assert '"trainer_state_coordinates"' in source
    assert '"trainer_state_execution_mode"' in source
    assert "all(repeatability.values())" in source


def test_deterministic_repeatability_auditor_does_not_authorize_scale() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    assert '"benchmark_500_updates": passed' in source
    assert '"pilot_16_gpu": False' in source
    assert '"pilot_32_gpu": False' in source
    assert '"production_pretraining": False' in source
    assert "if not passed:" in source
    assert "os.O_EXCL" in source


def test_deterministic_repeatability_result_is_bitwise_exact() -> None:
    import json

    result = json.loads(RESULT.read_text(encoding="ascii"))
    assert result["gate_passed"] is True
    assert result["membership_equal"] is True
    assert all(result["repeatability"].values())
    assert result["comparison_to_eager"]["passed"] is True
    assert result["summaries"]["compiled_cudagraph"][
        "throughput_relative_to_eager"
    ] > 1.41
    assert result["authorization"]["benchmark_500_updates"] is True
    assert result["authorization"]["pilot_16_gpu"] is False
