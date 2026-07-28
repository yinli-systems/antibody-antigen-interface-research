import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_antibody_cudagraph_ablation_correctness.py"
RESULT = (
    ROOT
    / "results/pretraining/antibody_cudagraph_ablation_correctness_v1.json"
)


def test_cudagraph_ablation_auditor_is_ascii_and_syntax_valid() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    ast.parse(source)
    assert 'CANDIDATES = ("eager", "compiled_no_cudagraph")' in source
    assert '"disable_compiled_cudagraphs": True' in source
    assert '"--disable-compiled-cudagraphs" in arguments' in source
    assert '"--expected-length-bucket-boundaries"' in source


def test_cudagraph_ablation_auditor_fails_closed() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    assert "passed = membership_equal and comparison_passed" in source
    assert '"diagnostic_stability_passed"' in source
    assert '"production_pretraining": False' in source
    assert '"scale_16_gpu": False' in source
    assert '"scale_32_gpu": False' in source
    assert "if not passed:" in source
    assert "os.O_EXCL" in source


def test_cudagraph_ablation_correctness_authorizes_only_long_benchmark() -> None:
    import json

    result = json.loads(RESULT.read_text(encoding="ascii"))
    assert result["membership_equal"] is True
    assert result["comparison_to_eager"]["passed"] is True
    assert result["comparison_to_eager"][
        "final_loss_absolute_difference"
    ] < 0.00012
    assert result["comparison_to_eager"][
        "final_gradient_norm_relative_difference"
    ] < 0.00072
    assert result["authorization"] == {
        "benchmark_500_updates": True,
        "production_pretraining": False,
        "scale_16_gpu": False,
        "scale_32_gpu": False,
    }
