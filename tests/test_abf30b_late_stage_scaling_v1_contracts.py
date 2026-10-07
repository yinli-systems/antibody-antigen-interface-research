from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "remote_tools/abf30b_late_stage_scaling_v1"


def load(name: str):
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_selection_and_analysis_order_are_fixed() -> None:
    freeze = load("freeze_late_stage_scaling_protocol_v1")
    analyze = load("analyze_late_stage_scaling_v1")
    assert freeze.EXPECTED_SELECTED_SEGMENTS == (58, 63, 69)
    assert analyze.NEW_CHECKPOINTS == ("segment_000058", "segment_000063")
    assert analyze.FINAL_CHECKPOINT == "segment_000069"


def test_slurm_contract_uses_only_gpu_4090_and_no_distance_loss() -> None:
    scripts = sorted(TOOLS.glob("*.sbatch"))
    assert scripts
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert "#SBATCH --partition=gpu_4090" in text
        assert "hp_4090" not in text
        assert "5090" not in text
    training = (TOOLS / "slurm_abf_checkpoint_development_v1.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --array=0-11%12" in training
    assert "--distance-loss-weight 0.0" in training
    assert "--contrastive-loss-weight 0.0" in training


def test_submission_has_no_external_measurement_stage() -> None:
    submit = (TOOLS / "submit_late_stage_scaling_v1.py").read_text(encoding="utf-8")
    assert '"external_measurement_submitted": False' in submit
    assert '"old_sabdab2_test_loaded": False' in submit
    assert '"temporal_holdout_loaded": False' in submit
    assert "SABLE" not in submit


def test_analyzer_verifies_run_hashes_and_zero_distance_loss() -> None:
    analyze = (TOOLS / "analyze_late_stage_scaling_v1.py").read_text(encoding="utf-8")
    for needle in (
        'receipt.get("result_sha256") != sha256_file(result_path)',
        'receipt.get("selected_checkpoint_sha256") != sha256_file(checkpoint_path)',
        'authority.get("distance_label_manifest_sha256") is not None',
        'finite(row.get("train", {}).get("distance_loss", 0.0)) != 0.0',
        '"full_scaling_curve": False',
        '"independent_test": False',
    ):
        assert needle in analyze
