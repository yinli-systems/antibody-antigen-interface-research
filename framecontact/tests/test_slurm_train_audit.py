from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "slurm_evaluate_framecontact_train_audit_v1.sbatch"
)


def test_train_audit_slurm_is_complete_five_seed_matrix_and_test_free() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-14%15" in text
    assert "seeds=(17 42 73 101 137)" in text
    assert "framecontact_gauge_marginal" in text
    assert "framecontact_gauge_hybrid" in text
    assert "framecontact_gauge_hybrid_contrastive" in text
    assert "--training-partner-panel" in text
    assert "--test-partner-panel" not in text
    assert "--external" not in text


def test_train_audit_slurm_requires_completed_development_receipt() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "FRAMECONTACT_DEVELOPMENT_RESULT.json" in text
    assert "FRAMECONTACT_SELECTED.pt" in text
    assert "RUN_RECEIPT.json" in text
    assert "sha256sum -c SOURCE_SHA256SUMS.txt" in text
    assert "Refusing" not in text
    assert "Partial train-audit output exists" in text
