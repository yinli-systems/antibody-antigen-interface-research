"""Validate automatic freeze-and-run recovery pipeline orchestration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "slurm/antibody_production_recovery_pipeline.sbatch"


def test_pipeline_uses_public_one_gpu_allocation() -> None:
    source = PIPELINE.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "vip_gpu_5090_scxi253" not in source


def test_pipeline_derives_and_validates_final_evidence_hashes() -> None:
    source = PIPELINE.read_text(encoding="ascii")
    assert "MANIFEST_FILE_SHA256=$(sha256sum" in source
    assert "AUDIT_FILE_SHA256=$(sha256sum" in source
    assert '["manifest_sha256"]' in source
    assert '["receipt_sha256"]' in source
    assert 'test "${#digest}" = 64' in source
    assert "freeze_antibody_production_recovery_gate.py" in source


def test_pipeline_delegates_same_allocation_recovery_after_freeze() -> None:
    source = PIPELINE.read_text(encoding="ascii")
    assert "ABF_GATE_PROTOCOL=$GATE_PROTOCOL" in source
    assert "antibody_production_loader_recovery_gate.sbatch" in source
    assert source.index("freeze_antibody_production_recovery_gate.py") < source.index(
        "antibody_production_loader_recovery_gate.sbatch"
    )
