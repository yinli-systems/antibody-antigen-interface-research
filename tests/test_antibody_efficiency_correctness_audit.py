import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_antibody_efficiency_correctness.py"
RESULT = ROOT / "results/pretraining/antibody_efficiency_correctness_v1.json"


def test_efficiency_correctness_auditor_is_ascii_and_syntax_valid() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    ast.parse(source)
    assert "membership_equal" in source
    assert "maximum_final_loss_absolute_difference" in source
    assert "maximum_gradient_norm_relative_difference" in source
    assert "maximum_parameter_moment_relative_difference" in source


def test_efficiency_correctness_auditor_fails_closed() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    assert '"benchmark_500_updates": passed' in source
    assert '"production_pretraining": False' in source
    assert '"scale_16_gpu": False' in source
    assert '"scale_32_gpu": False' in source
    assert "if not passed:" in source
    assert "os.O_EXCL" in source


def test_committed_efficiency_correctness_result_passes_without_scale_claim() -> None:
    import json

    result = json.loads(RESULT.read_text(encoding="ascii"))
    assert result["membership_equal"] is True
    assert result["stability_passed"] is True
    assert all(
        comparison["passed"]
        for comparison in result["comparisons_to_eager"].values()
    )
    assert result["summaries"]["compiled"][
        "residue_tokens_per_second"
    ] > 200_000
    assert result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] > 1.7
    assert result["authorization"] == {
        "benchmark_500_updates": True,
        "production_pretraining": False,
        "scale_16_gpu": False,
        "scale_32_gpu": False,
    }
