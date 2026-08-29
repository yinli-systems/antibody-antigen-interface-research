"""Fail-closed contracts for the executable joint-coupling v2 audit."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/framecontact_asep_joint_coupling_dev_v2.json"
RECEIPT = ROOT / "protocols/FRAMECONTACT_ASEP_JOINT_COUPLING_DEV_V2_RECEIPT.json"
TRAINER = ROOT / "scripts/train_framecontact.py"
SLURM = ROOT / "slurm_train_framecontact_asep_joint_coupling_dev_v2.sbatch"
SINGLE_SLURM = (
    ROOT / "slurm_train_framecontact_asep_joint_coupling_dev_v2_single.sbatch"
)
SMOKE_SLURM = ROOT / "slurm_smoke_framecontact_asep_joint_coupling_dev_v2.sbatch"
AGGREGATE_SLURM = (
    ROOT / "slurm_aggregate_asep_joint_coupling_dev_v2.sbatch"
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


def test_v2_protocol_is_hash_bound_and_development_only() -> None:
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
    assert receipt["v2_matrix_fitting_started_at_freeze"] is False
    assert receipt["single_batch_preflight"]["a3_optimizer_steps"] == 1
    assert receipt["single_batch_preflight"]["a3_output_root_created"] is False
    assert receipt["single_batch_preflight"]["v1_a5_state"] == (
        "FAILED_BEFORE_TRAINING"
    )
    assert protocol["data"]["test_role_forbidden"] is True
    assert protocol["model_selection"]["test_role_forbidden"] is True
    assert protocol["confirmatory_order"]["official_test"] == (
        "Not authorized by this protocol."
    )
    assert receipt["official_test_used_for_selection"] is False
    assert receipt["official_test_scores_read_for_this_protocol"] is False


def test_v2_replaces_only_the_ineligible_spatial_mechanism() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    experiments = _experiments()
    a0, a1, a2, a3, a4, a5 = protocol["development_matrix"]
    weights = protocol["training_objective"]["loss_weights_by_condition"]

    assert experiments[a0] == experiments[a1]
    assert weights[a0]["paratope"] == weights[a0]["epitope"] == 0.0
    assert weights[a1]["paratope"] == weights[a1]["epitope"] == 0.25
    assert experiments[a2] == experiments[a3]
    assert weights[a2]["pair_node_consistency"] == 0.0
    assert weights[a3]["pair_node_consistency"] == 0.25

    a3_without_crop = dict(experiments[a3])
    a4_without_crop = dict(experiments[a4])
    assert a4_without_crop.pop("coarse_contact_crop_mode") == "soft_unary"
    assert a3_without_crop == a4_without_crop

    # A4 versus A5 is a strict loss-only ablation: topology, contact labels,
    # soft routing, compute path and binary pair output all remain identical.
    assert experiments[a4] == experiments[a5]
    assert experiments[a5]["pair_output_mode"] == "binary_contact"
    a4_weights = dict(weights[a4])
    a5_weights = dict(weights[a5])
    assert a4_weights.pop("node_spatial_smoothness") == 0.0
    assert a5_weights.pop("node_spatial_smoothness") == 0.10
    assert a4_weights == a5_weights
    assert weights[a5]["distogram"] == 0.0
    assert weights[a5]["contact_distance_consistency"] == 0.0
    assert "no cross-chain distance target" in protocol["training_objective"][
        "node_spatial_smoothness"
    ]


def test_v2_slurm_covers_exact_matrix_and_spatial_authority() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt_sha = hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
    text = SLURM.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
    subprocess.run(["bash", "-n", str(SINGLE_SLURM)], check=True)
    subprocess.run(["bash", "-n", str(SMOKE_SLURM)], check=True)
    assert "#SBATCH --array=0-29%30" in text
    assert "#SBATCH --partition=gpu_5090" in text
    for experiment in protocol["development_matrix"]:
        assert experiment in text
    assert receipt_sha in text
    assert '--node-spatial-smoothness-loss-weight "${node_spatial_weight}"' in text
    assert "--node-spatial-neighbor-radius 12.0" in text
    assert "--node-spatial-neighbor-k 8" in text
    assert "frozen_test" not in text.lower()
    assert "test.csv" not in text.lower()


def test_v2_aggregate_slurm_is_pinned_and_fail_closed() -> None:
    text = AGGREGATE_SLURM.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(AGGREGATE_SLURM)], check=True)
    assert "FRAMECONTACT_AGGREGATION_ROOT:?" in text
    assert "--bootstrap-resamples 50000" in text
    assert "Refusing to overwrite" in text
    assert "aggregate_asep_joint_coupling_dev_v2.py" in text
    assert "frozen_test" not in text.lower()
    assert "test.csv" not in text.lower()


def test_v2_gpu_smoke_gates_a3_and_graph_spatial_a5() -> None:
    text = SMOKE_SLURM.read_text(encoding="utf-8")
    assert "framecontact_joint_a3_coupled_heads" in text
    assert "framecontact_joint_a5_graph_spatial" in text
    assert "--node-spatial-smoothness-loss-weight" in text
    assert "--single-batch-smoke-only" in text
    assert "Single-batch smoke unexpectedly created an output root" in text
    assert "frozen_test" not in text.lower()
    assert "test.csv" not in text.lower()


def test_trainer_exposes_spatial_loss_and_graph_authority() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert '"--node-spatial-smoothness-loss-weight"' in text
    assert '"node_spatial_smoothness_loss_weight", "node_spatial_smoothness"' in text
    assert '"node_spatial_neighbor_radius": "node_spatial_neighbor_radius_angstrom"' in text
    assert '"node_spatial_neighbor_k": "node_spatial_neighbor_k"' in text
    assert "node_spatial_smoothness=" in text
