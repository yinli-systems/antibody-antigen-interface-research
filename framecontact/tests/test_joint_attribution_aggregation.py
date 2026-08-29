from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_distance_auxiliaries_are_explicitly_fail_closed() -> None:
    receipt = json.loads((ROOT / "protocols/FRAMECONTACT_ASEP_DISTANCE_AUXILIARY_BLOCKED_V1_RECEIPT.json").read_text())
    assert receipt["status"] == "blocked_before_model_fitting"
    assert receipt["distance_bin_auxiliary_executed"] is False
    assert receipt["distogram_auxiliary_executed"] is False
    assert receipt["ten_angstrom_neighborhood_auxiliary_executed"] is False
    assert receipt["official_test_scores_read_for_this_protocol"] is False


def test_attribution_aggregators_are_development_only() -> None:
    source = (ROOT / "scripts/aggregate_asep_joint_attribution_dev_v1.py").read_text()
    case_source = (ROOT / "scripts/aggregate_asep_joint_attribution_case_study_v1.py").read_text()
    assert "official_test" in source and "external_test" in source
    assert "three conditions and five seeds" in source
    assert "no checkpoint or architecture selection" in case_source


def test_followup_aggregation_wrappers_use_absolute_logs() -> None:
    for name in (
        "slurm_aggregate_asep_joint_coupling_ratio_v1.sbatch",
        "slurm_aggregate_asep_joint_attribution_case_study_v1.sbatch",
    ):
        source = (ROOT / name).read_text()
        assert "#SBATCH --output=/ssd/" in source
        assert "#SBATCH --error=/ssd/" in source
