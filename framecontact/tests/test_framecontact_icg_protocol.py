"""Fail-closed contracts for the FrameContact-ICG development experiment."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/framecontact_icg_exploratory_v1.json"
RECEIPT = ROOT / "protocols/FRAMECONTACT_ICG_EXPLORATORY_V1_RECEIPT.json"
TRAINER = ROOT / "scripts/train_framecontact.py"
SLURM = ROOT / "slurm/train_framecontact_icg_exploratory_v1.sbatch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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


def test_protocol_and_receipt_are_hash_bound_and_development_only() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["protocol_file_sha256"] == _sha256(PROTOCOL)
    assert receipt["protocol_canonical_sha256"] == _canonical_sha256(protocol)
    assert protocol["data"]["test_role_forbidden"] is True
    assert protocol["model_selection"]["test_role_forbidden"] is True
    assert protocol["evidence_boundary_at_freeze"][
        "sabdab2_antigen_fold_test_predictions_accessed"
    ] is False
    assert protocol["evidence_boundary_at_freeze"][
        "external_test_predictions_accessed"
    ] is False
    assert protocol["model_selection"]["seeds"] == [17, 42, 73, 101, 137]
    assert protocol["data"]["cross_chain_input_geometry"].startswith("none")


def test_gate_and_residue_ablation_changes_are_explicit() -> None:
    experiments = _experiments()
    m2 = experiments["framecontact_icg_m2_ordinary"]
    m3 = experiments["framecontact_icg_m3_full_gate"]
    m4 = experiments["framecontact_icg_m4_interaction_gate"]
    ignored = {"contact_gate_source"}
    assert {key: value for key, value in m2.items() if key not in ignored} == {
        key: value for key, value in m3.items() if key not in ignored
    }
    assert {key: value for key, value in m3.items() if key not in ignored} == {
        key: value for key, value in m4.items() if key not in ignored
    }
    assert m2["contact_gate_source"] == "none"
    assert m3["contact_gate_source"] == "full_contact"
    assert m4["contact_gate_source"] == "interaction_only"
    assert m2["contact_marginalization"] == "noisy_or"

    m5 = experiments["framecontact_icg_m5_contact_marginal"]
    assert m5["contact_gate_source"] == m4["contact_gate_source"]
    assert m5["residue_output_mode"] == "contact_marginal_only"
    assert m5["contact_marginalization"] == "mean_probability"


def test_joint_consistency_is_an_independent_factor() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    experiments = _experiments()
    no_consistency = "framecontact_icg_m6a_joint_no_consistency"
    consistency = "framecontact_icg_m6_joint_consistent"
    assert experiments[no_consistency] == experiments[consistency]
    weights = protocol["training_objective"]["loss_weights_by_condition"]
    assert weights[no_consistency]["distogram"] == weights[consistency]["distogram"]
    assert weights[no_consistency]["contact_distance_consistency"] == 0.0
    assert weights[consistency]["contact_distance_consistency"] > 0.0
    assert experiments[consistency]["pair_output_mode"] == "joint_contact_distogram"
    assert experiments["framecontact_icg_m7_distogram_only"][
        "pair_output_mode"
    ] == "distogram"


def test_slurm_is_development_only_and_covers_the_frozen_matrix() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    text = SLURM.read_text()
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
    for experiment in protocol["development_matrix"]:
        assert experiment in text
    assert "#SBATCH --array=0-39%5" in text
    assert "framecontact_icg_exploratory_v1.json" in text
    assert "FRAMECONTACT_ICG_EXPLORATORY_V1_RECEIPT.json" in text
    assert "ICG_V1_SOURCE_MANIFEST_SHA256" in text
    assert "--checkpoint-wrong-partners" in text
    assert "--contact-gate-warmup-fraction 0.05" in text
    assert "--contact-gate-ramp-fraction 0.10" in text
    assert "frozen_test" not in text.lower()
    assert "test.csv" not in text
