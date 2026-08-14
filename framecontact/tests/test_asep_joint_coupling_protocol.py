"""Fail-closed contracts for the teacher-requested joint-coupling audit."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/framecontact_asep_joint_coupling_dev_v1.json"
RECEIPT = ROOT / "protocols/FRAMECONTACT_ASEP_JOINT_COUPLING_DEV_V1_RECEIPT.json"
TRAINER = ROOT / "scripts/train_framecontact.py"
SLURM = ROOT / "slurm_train_framecontact_asep_joint_coupling_dev_v1.sbatch"
SINGLE_SLURM = (
    ROOT / "slurm_train_framecontact_asep_joint_coupling_dev_v1_single.sbatch"
)


def _experiments() -> dict[str, dict[str, object]]:
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXPERIMENTS"
            for target in node.targets
        ):
            experiments = ast.literal_eval(node.value)
            assert isinstance(experiments, dict)
            return experiments
    raise AssertionError("EXPERIMENTS was not found")


def test_joint_protocol_is_hash_bound_and_development_only() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    canonical = (
        json.dumps(
            protocol,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    assert receipt["protocol_file_sha256"] == hashlib.sha256(
        PROTOCOL.read_bytes()
    ).hexdigest()
    assert receipt["protocol_canonical_sha256"] == hashlib.sha256(
        canonical
    ).hexdigest()
    assert protocol["data"]["test_role_forbidden"] is True
    assert protocol["model_selection"]["test_role_forbidden"] is True
    assert protocol["confirmatory_order"]["official_test"] == (
        "Not authorized by this protocol."
    )
    assert receipt["joint_coupling_fitting_started_at_freeze"] is False
    assert protocol["model_selection"]["seeds"] == [17, 42, 73, 101, 137]


def test_core_ablation_changes_one_coupling_factor_at_a_time() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    experiments = _experiments()
    a0, a1, a2, a3, a4, a5 = protocol["development_matrix"]
    weights = protocol["training_objective"]["loss_weights_by_condition"]

    assert experiments[a0] == experiments[a1]
    assert weights[a0]["paratope"] == weights[a0]["epitope"] == 0.0
    assert weights[a1]["paratope"] == weights[a1]["epitope"] == 0.25
    assert experiments[a0]["contact_marginalization"] == "noisy_or"

    assert experiments[a2] == experiments[a3]
    assert weights[a2]["pair_node_consistency"] == 0.0
    assert weights[a3]["pair_node_consistency"] == 0.25
    assert experiments[a2]["residue_output_mode"] == "unary_only"

    a3_without_crop = dict(experiments[a3])
    a4_without_crop = dict(experiments[a4])
    assert a3_without_crop.get("coarse_contact_crop_mode", "none") == "none"
    assert a4_without_crop.pop("coarse_contact_crop_mode") == "soft_unary"
    assert a3_without_crop == a4_without_crop

    a4_without_pair_mode = dict(experiments[a4])
    a5_without_pair_mode = dict(experiments[a5])
    assert a4_without_pair_mode.pop("pair_output_mode") == "binary_contact"
    assert a5_without_pair_mode.pop("pair_output_mode") == "joint_contact_distogram"
    assert a4_without_pair_mode == a5_without_pair_mode
    assert weights[a5]["distogram"] > 0.0
    assert weights[a5]["contact_distance_consistency"] > 0.0


def test_joint_slurm_covers_exact_matrix_with_pinned_authorities() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt_sha = hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
    text = SLURM.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
    subprocess.run(["bash", "-n", str(SINGLE_SLURM)], check=True)
    assert "#SBATCH --array=0-29%30" in text
    assert "#SBATCH --partition=gpu_5090" in text
    for experiment in protocol["development_matrix"]:
        assert experiment in text
    assert receipt_sha in text
    assert "--pair-node-consistency-loss-weight" in text
    assert "--coarse-contact-crop-floor 0.05" in text
    assert "sha256sum" not in text
    assert "frozen_test" not in text.lower()
    assert "test.csv" not in text.lower()


def test_trainer_exposes_pair_node_consistency_as_protocol_authority() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert '"--pair-node-consistency-loss-weight"' in text
    assert '("pair_node_consistency_loss_weight", "pair_node_consistency")' in text
    assert "pair_node_consistency=arguments.pair_node_consistency_loss_weight" in text
