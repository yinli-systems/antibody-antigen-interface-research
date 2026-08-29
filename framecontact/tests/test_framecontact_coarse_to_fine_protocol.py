"""Fail-closed contracts for the post-ICG coarse-to-fine experiment."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/framecontact_coarse_to_fine_exploratory_v1.json"
RECEIPT = ROOT / "protocols/FRAMECONTACT_COARSE_TO_FINE_EXPLORATORY_V1_RECEIPT.json"
TRAINER = ROOT / "scripts/train_framecontact.py"
SLURM = ROOT / "slurm/train_framecontact_coarse_to_fine_exploratory_v1.sbatch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _experiments() -> dict[str, dict[str, object]]:
    tree = ast.parse(TRAINER.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXPERIMENTS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, dict)
            return value
    raise AssertionError("EXPERIMENTS was not found")


def test_protocol_is_hash_bound_post_hoc_and_test_closed() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["protocol_file_sha256"] == _sha256(PROTOCOL)
    assert receipt["protocol_canonical_sha256"] == _canonical_sha256(protocol)
    assert protocol["adaptation_disclosure"]["prior_icg_validation_accessed"] is True
    assert protocol["data"]["test_role_forbidden"] is True
    assert protocol["model_selection"]["test_role_forbidden"] is True
    assert protocol["evidence_boundary_at_freeze"][
        "coarse_to_fine_training_started"
    ] is False
    assert protocol["evidence_boundary_at_freeze"][
        "sabdab2_antigen_fold_test_predictions_accessed"
    ] is False
    assert protocol["evidence_boundary_at_freeze"][
        "external_test_predictions_accessed"
    ] is False
    assert protocol["model_selection"]["seeds"] == [17, 42, 73, 101, 137]


def test_four_conditions_change_one_factor_at_a_time() -> None:
    experiments = _experiments()
    p0 = experiments["framecontact_icg_m4_interaction_gate"]
    p1 = experiments["framecontact_plan_b_soft_crop"]
    p2 = experiments["framecontact_plan_b_soft_crop_joint"]
    p3 = experiments["framecontact_plan_b_soft_crop_far_patch"]
    assert p0 == {
        key: value for key, value in p1.items() if key != "coarse_contact_crop_mode"
    }
    assert p1["coarse_contact_crop_mode"] == "soft_unary"
    ignored = {"pair_output_mode"}
    assert {key: value for key, value in p1.items() if key not in ignored} == {
        key: value for key, value in p2.items() if key not in ignored
    }
    assert p1["pair_output_mode"] == "binary_contact"
    assert p2["pair_output_mode"] == "joint_contact_distogram"
    assert p2 == p3

    protocol = json.loads(PROTOCOL.read_text())
    weights = protocol["training_objective"]["loss_weights_by_condition"]
    for name in (
        "framecontact_plan_b_soft_crop_joint",
        "framecontact_plan_b_soft_crop_far_patch",
    ):
        assert weights[name]["distogram"] == 0.25
        assert weights[name]["contact_distance_consistency"] == 0.1
    assert weights["framecontact_plan_b_soft_crop_joint"][
        "far_patch_exclusion"
    ] == 0.0
    assert weights["framecontact_plan_b_soft_crop_far_patch"][
        "far_patch_exclusion"
    ] == 0.05


def test_soft_crop_and_far_patch_contracts_are_fail_closed() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    crop = protocol["coarse_to_fine_contract"]
    assert crop["first_pass_loss"].startswith("none")
    assert crop["all_authenticated_pairs_retained"] is True
    assert crop["probability_floor"] > 0
    far = protocol["far_patch_exclusion_contract"]
    assert far["cross_monomer_alignment_required"] is False
    assert far["label_routing_forbidden"] is True
    assert ">=" in far["constraint"]
    assert protocol["post_selection_train_audit"]["required"] is True


def test_slurm_covers_exact_matrix_and_has_no_test_path() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    text = SLURM.read_text()
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
    for experiment in protocol["development_matrix"]:
        assert experiment in text
    assert "#SBATCH --array=0-19%5" in text
    assert "--far-patch-exclusion-loss-weight" in text
    assert "--coarse-contact-crop-floor 0.05" in text
    assert "coarse_to_fine_exploratory_v1.json" in text
    assert "frozen_test" not in text.lower()
    assert "test.csv" not in text
