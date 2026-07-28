import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_antibody_deterministic_repeatability.py"


def test_deterministic_repeatability_auditor_is_ascii_and_valid() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    ast.parse(source)
    assert '"deterministic_algorithms": True' in source
    assert '"final_model"' in source
    assert '"final_optimizer"' in source
    assert '"trainer_state"' in source
    assert "all(repeatability.values())" in source


def test_deterministic_repeatability_auditor_does_not_authorize_scale() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    assert '"benchmark_500_updates": passed' in source
    assert '"pilot_16_gpu": False' in source
    assert '"pilot_32_gpu": False' in source
    assert '"production_pretraining": False' in source
    assert "if not passed:" in source
    assert "os.O_EXCL" in source
