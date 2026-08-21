"""Validate the narrow stale validation-panel anchor recovery contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_freezer():
    path = ROOT / "remote_tools/freeze_antibody_panel_anchor_recovery.py"
    spec = importlib.util.spec_from_file_location("panel_anchor_freezer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_panel_anchor_matches_current_sampler() -> None:
    freezer = load_freezer()
    observed = freezer.live_anchor(
        ROOT,
        record_count=1_215_730,
        sample_count=8_192,
        seed=2_026_081_501,
    )
    assert observed == freezer.CORRECTED_ANCHOR
    assert observed != freezer.STALE_ANCHOR


def test_recovery_job_is_gpu_constrained_and_self_watching() -> None:
    script = (
        ROOT / "remote_tools/antibody_panel_anchor_recovery.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu_4090" in script
    assert "#SBATCH --gres=gpu:4" in script
    assert "hp_4090" not in script
    assert "/ssd/scxi253/model_deployment/envs/contact/bin/python3.10" in script
    assert '--dependency="afternotok:${NEXT_JOB_ID}"' in script
    assert '--panel-indices-sha256 "${PANEL[2]}"' in script
    assert "audit_antibody_panel_anchor_recovery.py" in script


def test_recovery_auditor_restores_base_validator() -> None:
    source = (
        ROOT / "remote_tools/audit_antibody_panel_anchor_recovery.py"
    ).read_text(encoding="utf-8")
    assert "base._validate_evaluation = corrected_validator" in source
    assert "finally:" in source
    assert "base._validate_evaluation = original_validator" in source
    assert 'result["inputs"]["segment_root"] = str(segment_root.resolve())' in source
