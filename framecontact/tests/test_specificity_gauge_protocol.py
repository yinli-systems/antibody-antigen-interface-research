"""The gauge protocol, the sbatch matrix, and the trainer registry must agree."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import re
from pathlib import Path

import pytest

from interaction_first.decision_matrix import ARM_FACTORS, require_single_variable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "protocols" / "framecontact_specificity_gauge_exploratory_v1.json"
)
SBATCH_PATH = (
    REPOSITORY_ROOT
    / "slurm"
    / "train_framecontact_specificity_gauge_exploratory_v1.sbatch"
)
GAUGE_ARMS = (
    "framecontact_gauge_marginal",
    "framecontact_gauge_hybrid",
    "framecontact_gauge_hybrid_contrastive",
)
RECEIPT_PATH = (
    REPOSITORY_ROOT
    / "protocols"
    / "FRAMECONTACT_SPECIFICITY_GAUGE_EXPLORATORY_V1_RECEIPT.json"
)


@pytest.fixture(scope="module")
def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sbatch() -> str:
    return SBATCH_PATH.read_text(encoding="utf-8")


def _sbatch_flag(sbatch: str, flag: str) -> str:
    match = re.search(rf"{re.escape(flag)}\s+([^\s\\]+)", sbatch)
    assert match is not None, f"{flag} is absent from the sbatch"
    return match.group(1)


def test_protocol_matrix_matches_the_sbatch_array(protocol: dict, sbatch: str) -> None:
    assert protocol["development_matrix"] == list(GAUGE_ARMS)
    listed = re.search(r"experiments=\(\n(.*?)\n\)", sbatch, re.DOTALL)
    assert listed is not None
    assert [line.strip() for line in listed.group(1).splitlines()] == list(GAUGE_ARMS)
    # Three arms times five seeds, throttled to five concurrent tasks.
    assert "--array=0-14%5" in sbatch
    assert protocol["model_selection"]["seeds"] == [17, 42, 73, 101, 137]


def test_frozen_receipt_authenticates_the_protocol_and_parents(protocol: dict) -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert receipt["protocol_file_sha256"] == hashlib.sha256(
        PROTOCOL_PATH.read_bytes()
    ).hexdigest()
    for parent_name, receipt_name in (
        ("residue_head_protocol", "residue_head"),
        ("antigen_fold_protocol", "antigen_fold"),
    ):
        parent = protocol["parents"][parent_name]
        parent_path = REPOSITORY_ROOT / parent["path"]
        expected = hashlib.sha256(parent_path.read_bytes()).hexdigest()
        assert parent["sha256"] == expected
        assert receipt["parent_sha256"][receipt_name] == expected
    assert receipt["single_variable_discipline_validated"] is True
    assert receipt["development_only_boundary_validated"] is True
    assert receipt["test_prediction_boundary_validated"] is True


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="PyTorch is unavailable"
)
def test_every_gauge_arm_is_registered_in_the_trainer() -> None:
    specification = importlib.util.spec_from_file_location(
        "train_framecontact_for_test", REPOSITORY_ROOT / "scripts" / "train_framecontact.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    for arm in GAUGE_ARMS:
        assert arm in module.EXPERIMENTS, arm
        assert module.EXPERIMENTS[arm]["pair_gauge_mode"] == "doubly_centered"
    assert (
        module.EXPERIMENTS["framecontact_gauge_marginal"]["residue_output_mode"]
        == "gauge_marginal_only"
    )
    assert (
        module.EXPERIMENTS["framecontact_gauge_hybrid"]["residue_output_mode"]
        == "gauge_hybrid"
    )


def test_every_gauge_arm_sits_in_the_factor_space() -> None:
    for arm in GAUGE_ARMS:
        assert arm in ARM_FACTORS


def test_each_gauge_arm_changes_exactly_one_factor(protocol: dict) -> None:
    """The protocol claims single-variable discipline; hold it to that."""

    discipline = protocol["single_variable_discipline"]
    assert (
        require_single_variable(
            "framecontact_contact_marginal", "framecontact_gauge_marginal"
        )
        == "residue_aggregation"
    )
    assert (
        require_single_variable(
            "framecontact_contact_hybrid", "framecontact_gauge_hybrid"
        )
        == "residue_aggregation"
    )
    assert (
        require_single_variable(
            "framecontact_gauge_hybrid", "framecontact_gauge_hybrid_contrastive"
        )
        == "residue_contrastive"
    )
    assert "forbidden_comparison" in discipline


def test_only_the_contrastive_arm_carries_the_contrastive_weight(
    protocol: dict, sbatch: str
) -> None:
    weights = protocol["training_objective"]["loss_weights_by_condition"]
    assert set(weights) == set(GAUGE_ARMS)
    assert weights["framecontact_gauge_marginal"]["residue_partner_contrastive"] == 0.0
    assert weights["framecontact_gauge_hybrid"]["residue_partner_contrastive"] == 0.0
    assert (
        weights["framecontact_gauge_hybrid_contrastive"][
            "residue_partner_contrastive"
        ]
        == 0.25
    )
    assert "residue_contrastive_weight=0.25" in sbatch
    assert "residue_contrastive_weight=0.0" in sbatch


def test_sbatch_execution_flags_match_the_protocol(protocol: dict, sbatch: str) -> None:
    implementation = protocol["implementation"]
    for flag, key, cast in (
        ("--d-model", "d_model", int),
        ("--num-heads", "num_heads", int),
        ("--num-self-layers", "num_self_layers", int),
        ("--num-rbf", "num_rbf", int),
        ("--neighbor-k", "neighbor_k", int),
        ("--neighbor-radius", "neighbor_radius_angstrom", float),
        ("--sequence-window", "sequence_window", int),
        ("--dropout", "dropout", float),
        ("--learning-rate", "learning_rate", float),
        ("--weight-decay", "weight_decay", float),
        ("--batch-size", "batch_size", int),
        ("--evaluation-batch-size", "evaluation_batch_size", int),
        ("--calibration-max-pairs", "calibration_max_pairs", int),
    ):
        assert cast(_sbatch_flag(sbatch, flag)) == cast(implementation[key]), flag
    assert _sbatch_flag(sbatch, "--precision") == implementation["precision"]
    assert ("--offload-saved-tensors" in sbatch) == implementation[
        "offload_saved_tensors"
    ]


def test_shared_loss_weights_match_the_residue_head_protocol(protocol: dict) -> None:
    """The gauge arms must inherit the objective they are compared against."""

    residue_head = json.loads(
        (
            REPOSITORY_ROOT
            / "protocols"
            / "framecontact_residue_head_exploratory_v1.json"
        ).read_text(encoding="utf-8")
    )
    baseline = residue_head["training_objective"]["loss_weights"]
    for arm, weights in protocol["training_objective"][
        "loss_weights_by_condition"
    ].items():
        for name, value in baseline.items():
            assert weights[name] == value, f"{arm} changed the {name} weight"
    assert (
        protocol["training_objective"]["specificity_score_source"]
        == residue_head["training_objective"]["specificity_score_source"]
    )


def test_protocol_records_the_test_boundary(protocol: dict) -> None:
    boundary = protocol["evidence_boundary_at_freeze"]
    assert boundary["sabdab2_antigen_fold_test_predictions_accessed"] is False
    assert boundary["asep_official_test_predictions_accessed"] is False
    assert boundary["this_gauge_ablation_training_started"] is False
    assert protocol["data"]["test_role_forbidden"] is True


def test_protocol_records_the_selection_metric_instability(protocol: dict) -> None:
    """The known epoch-noise problem must travel with the protocol."""

    limitation = protocol["model_selection"]["known_limitation"]
    assert "epoch" in limitation
    assert protocol["reporting_rule"]["stability_rule"]
