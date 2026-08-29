"""Static custody checks for the SAbDab2 validation-only postprocess chain."""

from __future__ import annotations

from pathlib import Path


def test_postprocess_chain_freezes_all_ensembles_before_selection() -> None:
    root = Path(__file__).resolve().parents[1]
    ensemble = (root / "scripts/slurm_freeze_sabdab2_ensemble_array.sbatch").read_text()
    selection = (root / "scripts/slurm_select_sabdab2_development.sbatch").read_text()
    experiments = (
        "sequence_dense",
        "spatial_topology",
        "distance",
        "distance_direction",
        "full_pose",
        "frozen_frame_ipa",
        "framecontact",
        "framecontact_interleaved",
    )
    assert "#SBATCH --array=0-7%4" in ensemble
    assert "freeze_framecontact_development_ensemble.py" in ensemble
    assert "freeze_framecontact_development_selection.py" in selection
    for experiment in experiments:
        assert experiment in ensemble
        assert experiment in selection


def test_postprocess_chain_has_no_test_authority_or_test_scorer() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = "\n".join(
        (root / path).read_text()
        for path in (
            "scripts/slurm_freeze_sabdab2_ensemble_array.sbatch",
            "scripts/slurm_select_sabdab2_development.sbatch",
        )
    )
    prohibited = (
        "TEST_PARTNER_PANEL",
        "SABDAB2_TEST_SCORE_CONSUMED",
        "evaluate_framecontact_sabdab2_ensemble_test.py",
        "freeze_sabdab2_frozen_test_opening.py",
        "split_role=test",
        "--test-partner-panel",
    )
    assert not any(value in payload for value in prohibited)
    assert "test_loaded" not in payload
