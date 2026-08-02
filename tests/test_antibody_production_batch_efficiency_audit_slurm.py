"""Validate the source-bound production-batch audit launcher."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm/antibody_production_batch_efficiency_audit.sbatch"


def test_audit_launcher_is_create_only_and_runtime_bound() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem" not in source
    assert "ABF_SOURCE_ROOT" in source
    assert "ABF_RUN_ROOT" in source
    assert "ABF_AUDIT_OUTPUT" in source
    assert "ABF_PROTOCOL_SHA256" in source
    assert "ABF_AUDITOR_SHA256" in source
    assert "ABF_RUNTIME_MANIFEST" in source
    assert "ABF_RUNTIME_MANIFEST_SHA256" in source
    assert "ABF_PROTOCOL_BASENAME" in source
    assert "ABF_AUDITOR_BASENAME" in source
    assert 'test ! -e "${OUTPUT}"' in source
    assert "generate_runtime_manifest.py\" verify" in source
    assert "--expected-file-sha256" in source
    assert "export CUDA_VISIBLE_DEVICES=" in source
    assert "audit_antibody_production_batch_efficiency.py" in source


def test_audit_launcher_preserves_fail_closed_exit_status() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert 'exec "${PYTHON_BIN}" "${AUDITOR}"' in source
    assert "|| true" not in source
    assert "--run-root" in source
    assert "--protocol" in source
    assert "--output" in source
