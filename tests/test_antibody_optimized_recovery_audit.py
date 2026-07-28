import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_antibody_optimized_recovery.py"


def test_optimized_recovery_auditor_is_ascii_and_syntax_valid() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    ast.parse(source)
    assert "continuous_state == resumed_state" in source
    assert '"prefix_trace": _trace_equal' in source
    assert '"resumed_trace": _trace_equal' in source
    assert '"length_bucket_boundaries": EXPECTED_BUCKETS' in source


def test_optimized_recovery_auditor_fails_closed_before_scale() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    assert "if not all(exact.values()):" in source
    assert '"pilot_16_gpu": True' in source
    assert '"pilot_32_gpu": False' in source
    assert '"production_pretraining": False' in source
    assert '"scientific_claim": False' in source
    assert "os.O_EXCL" in source
