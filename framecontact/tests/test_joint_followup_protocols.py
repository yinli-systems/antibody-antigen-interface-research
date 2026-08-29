from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(protocol_name: str, receipt_name: str) -> tuple[dict, dict]:
    protocol_path = ROOT / "protocols" / protocol_name
    receipt_path = ROOT / "protocols" / receipt_name
    protocol = json.loads(protocol_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    assert receipt["protocol_file_sha256"] == _sha256(protocol_path)
    assert protocol["data"]["test_role_forbidden"] is True
    assert protocol["model_selection"]["test_role_forbidden"] is True
    assert receipt["official_test_used_for_selection"] is False
    assert receipt["official_test_scores_read_for_this_protocol"] is False
    return protocol, receipt


def test_ratio_joint_protocol_is_development_only() -> None:
    protocol, _ = _load(
        "framecontact_asep_joint_coupling_ratio_v1.json",
        "FRAMECONTACT_ASEP_JOINT_COUPLING_RATIO_V1_RECEIPT.json",
    )
    assert protocol["data"]["split_method"] == "epitope_ratio"
    assert len(protocol["development_matrix"]) == 6
    assert tuple(protocol["model_selection"]["seeds"]) == (17, 42, 73, 101, 137)


def test_aggregation_attribution_protocol_registers_three_reductions() -> None:
    protocol, _ = _load(
        "framecontact_asep_joint_attribution_dev_v1.json",
        "FRAMECONTACT_ASEP_JOINT_ATTRIBUTION_DEV_V1_RECEIPT.json",
    )
    assert protocol["data"]["split_method"] == "epitope_group"
    assert protocol["development_matrix"] == [
        "framecontact_joint_a1_noisy_or",
        "framecontact_joint_attribution_mean_probability",
        "framecontact_joint_attribution_length_normalized_noisy_or",
    ]


def test_followup_wrappers_use_absolute_logs_and_no_test_role() -> None:
    for name in (
        "slurm_train_framecontact_asep_joint_coupling_ratio_v1.sbatch",
        "slurm_train_framecontact_asep_joint_attribution_dev_v1.sbatch",
        "slurm_evaluate_joint_attribution_case_study_v1.sbatch",
    ):
        text = (ROOT / name).read_text()
        assert "#SBATCH --output=/ssd/" in text
        assert "#SBATCH --error=/ssd/" in text
        assert "official" not in text.lower() or "official_test" not in text.lower()
