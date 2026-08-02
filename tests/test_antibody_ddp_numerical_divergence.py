import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_antibody_ddp_numerical_divergence.py"
RESULT = ROOT / (
    "results/pretraining/antibody_ddp_numerical_divergence_attribution_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_attribution_auditor_is_ascii_and_fail_closed() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    ast.parse(source)
    assert "maximum_loss(20)" in source
    assert "maximum_loss(500)" in source
    assert '"exact_recovery": False' in source
    assert '"pilot_16_gpu": False' in source
    assert '"pilot_32_gpu": False' in source
    assert '"production_pretraining": False' in source
    assert "os.O_EXCL" in source


def test_attribution_localizes_early_correctness_and_long_drift() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    assert result["conclusion"] == (
        "early_numerical_correctness_passes_but_long_horizon_"
        "trajectory_equivalence_fails"
    )
    assert result["early_20_update_thresholds_passed"] is True
    assert result["long_horizon_thresholds_failed"] is True
    assert all(result["replicate_exact"].values())
    for comparison in result["comparisons"].values():
        assert comparison["early_20_updates"]["passed_trace_thresholds"] is True
    assert result["comparisons"]["compiler_r1"]["full_500_updates"][
        "maximum_loss_absolute_difference"
    ] > 0.20
    assert result["comparisons"]["communication_increment_r1"][
        "full_500_updates"
    ]["maximum_loss_absolute_difference"] > 0.17


def test_attribution_binds_failed_gate_and_auditor() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    failed = ROOT / (
        "results/pretraining/"
        "antibody_ddp_communication_paired_gate_attempt_013_v1.json"
    )
    assert result["failed_gate_result_sha256"] == _sha256(failed)
    assert result["source"]["auditor_sha256"] == _sha256(AUDITOR)
    assert result["authorization"] == {
        "exact_recovery": False,
        "pilot_16_gpu": False,
        "pilot_32_gpu": False,
        "production_pretraining": False,
        "scientific_claim": False,
    }
