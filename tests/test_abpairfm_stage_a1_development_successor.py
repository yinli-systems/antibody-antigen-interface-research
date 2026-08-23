from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_development_successor_is_fail_closed_and_test_free() -> None:
    text = (
        ROOT
        / "scripts"
        / "continue_abpairfm_stage_a1_development_evaluation.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH -p gpu_4090" in text
    assert "hp_4090" not in text
    assert "#SBATCH --mem" not in text
    assert "/ssd/scxi253/model_deployment/envs/contact/bin/python3.10" in text
    assert "PASS_STAGE_A1_LENGTH_BAND_PROMOTION_CHECKPOINT_VERIFIED" in text
    assert 'get("test_access") is not False' in text
    assert "verify_dcp_generation" in text
    assert "stage_a1_updates\") != 200" in text
    assert "submit_abpairfm_stage_a1_frozen_development_retrieval.sbatch" in text
    assert "submit_abpairfm_stage_a1_frozen_development_contact.sbatch" in text


def test_development_successor_has_no_test_output_or_argument() -> None:
    text = (
        ROOT
        / "scripts"
        / "continue_abpairfm_stage_a1_development_evaluation.sbatch"
    ).read_text(encoding="utf-8")
    command_lines = [line for line in text.splitlines() if "sbatch --parsable" in line]
    assert all("test" not in line.lower() for line in command_lines)
    assert "development_retrieval_v1" in text
    assert "development_contact_v1" in text
