"""Validate the unique frozen partner-geometry R2 construction path."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from interaction_first.pair_geometry_schema import PairGeometrySchema
from interaction_first.r2_model_contract import (
    R2ModelContractError,
    build_frozen_r2_model,
    load_r2_model_contract,
)


ROOT = Path(__file__).parents[1]
CONTRACT_PATH = (
    ROOT
    / "protocols"
    / "sabdab2_partner_geometry_r2_model_contract_v1.json"
)


def _schema() -> PairGeometrySchema:
    return PairGeometrySchema(
        distance_bin_edges_angstrom=(1.5, 3.0, 4.5, 6.0, 8.0, 12.0, 20.0),
        contact_cutoff_angstrom=4.5,
        orientation_axis_names=("ab_azimuth", "ab_polar", "ag_azimuth"),
        orientation_axis_is_cyclic=(True, False, True),
        orientation_bin_edges_radians=(
            (-2.0, -1.0, 0.0, 1.0, 2.0),
            (0.4, 0.8, 1.2, 1.6, 2.0),
            (-2.0, -1.0, 0.0, 1.0, 2.0),
        ),
        label_format_version="pair_geometry_labels_v2",
        orientation_schema_version="test_orientation_schema_v1",
        primary_contact_definition="test_positive_or_complete_contact",
        distance_definition="test_complete_pair_distance",
        orientation_definition=(
            "right_handed_N_CA_C_backbone_frames_with_CA_to_CA_directions"
        ),
        orientation_bin_edge_convention=(
            "internal_right_closed_thresholds_num_bins_minus_one"
        ),
    )


def _load() -> tuple[dict, str]:
    digest = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    return load_r2_model_contract(
        CONTRACT_PATH,
        expected_file_sha256=digest,
        repository_root=ROOT,
    )


def test_repository_contract_authenticates_every_source() -> None:
    contract, digest = _load()
    assert digest == hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    assert contract["source_bindings"]["r2_model_contract"]["sha256"] == (
        hashlib.sha256(
            (ROOT / "src" / "interaction_first" / "r2_model_contract.py").read_bytes()
        ).hexdigest()
    )
    assert contract["model"]["candidate_id"] == "R2"
    assert contract["model"]["intra_chain_geometry"]["feature_mode"] == (
        "distance_direction"
    )
    assert contract["model"]["pair_representation"]["num_update_layers"] == 2
    assert contract["authorization"]["scientific_training"] is False
    assert contract["authorization"]["external_test_access"] is False


def test_factory_builds_only_explicit_partner_specific_r2() -> None:
    contract, _ = _load()
    schema = _schema()
    model = build_frozen_r2_model(
        contract,
        geometry_schema=schema,
        expected_geometry_schema_sha256=schema.sha256,
    )
    assert model.antibody_input_dim == 512
    assert model.antigen_input_dim == 1280
    assert model.num_pair_update_layers == 2
    assert len(model.fusion_layers) == 2
    assert len(model.pair_update_layers) == 2
    assert model.contact_logit_mode == "unary_plus_interaction_residual"
    assert model.intra_chain_geometry is not None
    antibody_encoder = model.intra_chain_geometry.antibody_encoder
    antigen_encoder = model.intra_chain_geometry.antigen_encoder
    assert antibody_encoder is not antigen_encoder
    assert antibody_encoder.geometry_feature_mode == "distance_direction"
    assert antigen_encoder.geometry_feature_mode == "distance_direction"
    antibody_parameters = {id(value) for value in antibody_encoder.parameters()}
    antigen_parameters = {id(value) for value in antigen_encoder.parameters()}
    assert antibody_parameters.isdisjoint(antigen_parameters)


def test_frozen_model_runs_native_pair_forward_and_backward() -> None:
    torch = pytest.importorskip("torch")
    contract, _ = _load()
    schema = _schema()
    model = build_frozen_r2_model(
        contract,
        geometry_schema=schema,
        expected_geometry_schema_sha256=schema.sha256,
    )
    model.train()
    antibody = torch.randn(1, 3, 512)
    antigen = torch.randn(1, 4, 1280)
    antibody_mask = torch.tensor([[True, True, True]])
    antigen_mask = torch.tensor([[True, True, True, True]])
    candidate = antibody_mask.unsqueeze(2) & antigen_mask.unsqueeze(1)

    def backbone(length: int) -> object:
        ca = torch.stack(
            (
                torch.arange(length, dtype=torch.float32) * 3.8,
                torch.zeros(length),
                torch.zeros(length),
            ),
            dim=-1,
        )
        return torch.stack(
            (
                ca + torch.tensor([0.0, 1.0, 0.0]),
                ca,
                ca + torch.tensor([1.0, 0.0, 0.0]),
            ),
            dim=1,
        ).unsqueeze(0)

    output = model(
        antibody,
        antigen,
        antibody_mask,
        antigen_mask,
        candidate,
        antibody_backbone=backbone(3),
        antigen_backbone=backbone(4),
        antibody_chain_ids=torch.zeros((1, 3), dtype=torch.long),
        antigen_chain_ids=torch.zeros((1, 4), dtype=torch.long),
        enabled_heads={"contact", "distance", "orientation", "interaction"},
    )
    assert output["contact_logits"].shape == (1, 3, 4)
    assert output["pair_representations"].shape == (1, 3, 4, 128)
    assert output["pair_update_order"] == (
        "antigen_axis_within_antibody_row",
        "antibody_axis_within_antigen_column",
        "feed_forward",
    )
    assert output["partner_score_source"] == (
        "double_centered_independent_specificity_interaction_residual"
    )
    assert output["contact_logit_source"] == (
        "antibody_unary_plus_antigen_unary_plus_interaction_residual"
    )
    loss = (
        output["contact_logits"][candidate].square().mean()
        + output["partner_score"].square().mean()
    )
    loss.backward()
    assert model.specificity_head is not None
    assert model.specificity_head.weight.grad is not None
    assert bool(torch.isfinite(model.specificity_head.weight.grad).all())


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("model", "d_model"), 512, "d_model"),
        (
            ("model", "intra_chain_geometry", "feature_mode"),
            "full_pose",
            "intra-chain geometry",
        ),
        (
            ("model", "intra_chain_geometry", "num_layers"),
            True,
            "intra-chain geometry",
        ),
        (
            ("model", "pair_representation", "num_update_layers"),
            1,
            "pair representation",
        ),
        (
            ("model", "prediction", "contact_logit_mode"),
            "distance_mass",
            "prediction contract",
        ),
        (
            ("authorization", "external_test_access"),
            True,
            "authorization",
        ),
    ),
)
def test_contract_mutations_fail_closed(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    contract, _ = _load()
    altered = copy.deepcopy(contract)
    target = altered
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    schema = _schema()
    with pytest.raises(R2ModelContractError, match=message):
        build_frozen_r2_model(
            altered,
            geometry_schema=schema,
            expected_geometry_schema_sha256=schema.sha256,
        )


def test_contract_and_geometry_schema_digests_are_externally_anchored(
    tmp_path,
) -> None:
    contract, digest = _load()
    with pytest.raises(R2ModelContractError, match="file digest differs"):
        load_r2_model_contract(
            CONTRACT_PATH,
            expected_file_sha256="f" * 64,
            repository_root=ROOT,
        )
    schema = _schema()
    with pytest.raises(R2ModelContractError, match="schema digest differs"):
        build_frozen_r2_model(
            contract,
            geometry_schema=schema,
            expected_geometry_schema_sha256="e" * 64,
        )

    injected = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    injected["model"]["bound_complex_inter_chain_distance"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(injected), encoding="ascii")
    with pytest.raises(R2ModelContractError, match="fields differ"):
        load_r2_model_contract(
            path,
            expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            repository_root=ROOT,
        )


def test_contract_symbolic_link_is_rejected(tmp_path) -> None:
    linked_contract = tmp_path / "linked_contract.json"
    linked_contract.symlink_to(CONTRACT_PATH)
    with pytest.raises(R2ModelContractError, match="regular file"):
        load_r2_model_contract(
            linked_contract,
            expected_file_sha256=hashlib.sha256(
                CONTRACT_PATH.read_bytes()
            ).hexdigest(),
            repository_root=ROOT,
        )
