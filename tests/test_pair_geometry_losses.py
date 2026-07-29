"""Adversarial tests for the prospective pair-geometry objective."""

from __future__ import annotations

import importlib.util
from dataclasses import replace

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def _schema():
    from interaction_first.pair_geometry_schema import PairGeometrySchema

    return PairGeometrySchema(
        distance_bin_edges_angstrom=(2.0, 4.5, 8.0),
        contact_cutoff_angstrom=4.5,
        orientation_axis_names=("ab_azimuth", "ab_polar"),
        orientation_axis_is_cyclic=(True, False),
        orientation_bin_edges_radians=(
            (-1.0, 1.0),
            (1.0, 2.0),
        ),
        label_format_version="pair_geometry_labels_v2",
        orientation_schema_version="test_orientation_schema_v1",
        primary_contact_definition="test_positive_or_complete_contact",
        distance_definition="test_complete_pair_distance",
        orientation_definition="test_right_handed_frames",
        orientation_bin_edge_convention=(
            "internal_right_closed_thresholds_num_bins_minus_one"
        ),
    )


def _outputs(*, executed=("contact", "distance", "orientation", "interaction")):
    import torch

    schema = _schema()
    candidate = torch.tensor(
        [
            [[True, True, False], [True, True, False]],
            [[True, False, True], [False, True, True]],
        ]
    )
    torch.manual_seed(27)
    outputs = {
        "pair_orientation": "antibody_by_antigen",
        "executed_heads": executed,
        "geometry_schema": schema,
        "geometry_schema_metadata": schema.canonical_metadata(),
        "geometry_schema_sha256": schema.sha256,
        "candidate_pair_mask": candidate,
        "epitope_mask": candidate.any(dim=1),
        "paratope_mask": candidate.any(dim=2),
    }
    if "distance" in executed:
        distance = torch.randn(2, 2, 3, 4, requires_grad=True)
        outputs["distance_bin_logits"] = distance
    if "contact" in executed:
        distance = outputs["distance_bin_logits"]
        contact = (
            torch.logsumexp(distance[..., :2], dim=-1)
            - torch.logsumexp(distance[..., 2:], dim=-1)
        )
        outputs["contact_logits"] = contact
        outputs["epitope_probability"] = torch.tensor(
            [[0.2, 0.8, 0.0], [0.7, 0.0, 0.3]],
            requires_grad=True,
        )
        outputs["paratope_probability"] = torch.tensor(
            [[0.6, 0.3], [0.4, 0.5]],
            requires_grad=True,
        )
    if "orientation" in executed:
        outputs["orientation_bin_logits"] = torch.randn(
            2,
            2,
            3,
            2,
            3,
            requires_grad=True,
        )
    if "interaction" in executed:
        outputs["partner_score"] = torch.tensor([1.4, 0.3], requires_grad=True)
    return outputs


def _targets(outputs):
    import torch

    distance = torch.full((2, 2, 3), -1, dtype=torch.long)
    distance[0, 0, 0] = 0
    distance[0, 0, 1] = 2
    distance[0, 1, 0] = 3
    distance[1, 0, 2] = 1
    distance[1, 1, 2] = 2
    distance_mask = distance >= 0

    contact = torch.full((2, 2, 3), -1.0)
    contact[distance_mask] = (distance[distance_mask] < 2).float()
    contact_mask = distance_mask.clone()
    contact[0, 1, 1] = 1.0
    contact_mask[0, 1, 1] = True

    orientation_shape = (2, 2, 3, 2)
    orientation = torch.full(orientation_shape, -1, dtype=torch.long)
    orientation[0, 0, 0] = torch.tensor([2, 1])
    orientation[1, 0, 2, 1] = 0
    orientation_mask = orientation >= 0
    return {
        "contact_targets": contact,
        "contact_mask": contact_mask,
        "distance_bin_targets": distance,
        "distance_mask": distance_mask,
        "orientation_bin_targets": orientation,
        "orientation_mask": orientation_mask,
        "target_schema": outputs["geometry_schema"],
    }


def _macro_expected(per_entry, mask):
    import torch

    per_complex = []
    for index in range(int(mask.shape[0])):
        if bool(mask[index].any()):
            per_complex.append(per_entry[index][mask[index]].mean())
    return torch.stack(per_complex).mean() if per_complex else per_entry.sum() * 0.0


def test_contact_bce_and_brier_are_equal_per_complex_reductions() -> None:
    import torch
    import torch.nn.functional as F

    from interaction_first.pair_geometry_losses import (
        masked_contact_bce_with_logits,
        masked_contact_brier_loss,
    )

    logits = torch.tensor(
        [[[0.2, -0.7, 9.0]], [[-1.0, 0.4, 2.0]]],
        requires_grad=True,
    )
    targets = torch.tensor([[[1.0, 0.0, -1.0]], [[0.0, -1.0, -1.0]]])
    mask = targets >= 0
    dense = targets.clamp_min(0.0)
    entry_bce = F.binary_cross_entropy_with_logits(
        logits,
        dense,
        pos_weight=torch.tensor(2.5),
        reduction="none",
    )
    expected_bce = _macro_expected(entry_bce, mask)
    bce, bce_count = masked_contact_bce_with_logits(
        logits,
        targets,
        mask,
        positive_weight=2.5,
        return_valid_complex_count=True,
    )
    torch.testing.assert_close(bce, expected_bce)
    assert bce_count == 2

    entry_brier = (torch.sigmoid(logits) - dense).square()
    expected_brier = _macro_expected(entry_brier, mask)
    brier, brier_count = masked_contact_brier_loss(
        logits,
        targets,
        mask,
        return_valid_complex_count=True,
    )
    torch.testing.assert_close(brier, expected_brier)
    assert brier_count == 2
    assert not torch.allclose(bce, brier)


def test_distance_orientation_and_objective_report_macro_counts() -> None:
    import torch
    import torch.nn.functional as F

    from interaction_first.pair_geometry_losses import pair_geometry_objective

    outputs = _outputs()
    targets = _targets(outputs)
    total, components = pair_geometry_objective(outputs, **targets)
    distance_entry = F.cross_entropy(
        outputs["distance_bin_logits"].movedim(-1, 1),
        targets["distance_bin_targets"].clamp_min(0),
        reduction="none",
    )
    orientation_entry = F.cross_entropy(
        outputs["orientation_bin_logits"].movedim(-1, 1),
        targets["orientation_bin_targets"].clamp_min(0),
        reduction="none",
    )
    torch.testing.assert_close(
        components["distance_ce"],
        _macro_expected(distance_entry, targets["distance_mask"]),
    )
    torch.testing.assert_close(
        components["orientation_ce"],
        _macro_expected(orientation_entry, targets["orientation_mask"]),
    )
    assert components["valid_complex_counts"] == {
        "contact_bce": 2,
        "distance_ce": 2,
        "orientation_ce": 2,
        "contact_brier": 0,
    }
    assert total.ndim == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_orientation", "pair_orientation"),
        ("empty_candidate_item", "Every batch item"),
        ("epitope_mismatch", "epitope_mask must equal support"),
        ("paratope_mismatch", "paratope_mask must equal support"),
        ("mutable_schema_metadata", "geometry_schema_metadata"),
        ("distance_bin_count", "class count"),
        ("orientation_axis_count", "dimensions violate"),
        ("contact_mapping", "distance-bin mapping"),
        ("schema_semantics", "target_schema must exactly equal"),
        ("contact_distance_contradiction", "contradict"),
    ],
)
def test_objective_rejects_adversarial_contract_violations(
    mutation: str,
    message: str,
) -> None:
    import torch

    from interaction_first.pair_geometry_losses import pair_geometry_objective
    from interaction_first.pair_geometry_schema import PairGeometrySchema

    outputs = _outputs()
    targets = _targets(outputs)
    if mutation == "wrong_orientation":
        outputs["pair_orientation"] = "antigen_by_antibody"
    elif mutation == "empty_candidate_item":
        outputs["candidate_pair_mask"][1].fill_(False)
    elif mutation == "epitope_mismatch":
        outputs["epitope_mask"][0, 0] = ~outputs["epitope_mask"][0, 0]
    elif mutation == "paratope_mismatch":
        outputs["paratope_mask"][0, 0] = ~outputs["paratope_mask"][0, 0]
    elif mutation == "mutable_schema_metadata":
        outputs["geometry_schema_metadata"] = dict(
            outputs["geometry_schema_metadata"]
        )
    elif mutation == "distance_bin_count":
        outputs["distance_bin_logits"] = torch.randn(2, 2, 3, 5)
    elif mutation == "orientation_axis_count":
        outputs["orientation_bin_logits"] = torch.randn(2, 2, 3, 3, 3)
    elif mutation == "contact_mapping":
        outputs["contact_logits"] = outputs["contact_logits"] + 0.1
    elif mutation == "schema_semantics":
        source = targets["target_schema"]
        targets["target_schema"] = PairGeometrySchema(
            distance_bin_edges_angstrom=(2.1, 4.5, 8.0),
            contact_cutoff_angstrom=4.5,
            orientation_axis_names=source.orientation_axis_names,
            orientation_axis_is_cyclic=source.orientation_axis_is_cyclic,
            orientation_bin_edges_radians=source.orientation_bin_edges_radians,
            label_format_version=source.label_format_version,
            orientation_schema_version=source.orientation_schema_version,
            primary_contact_definition=source.primary_contact_definition,
            distance_definition=source.distance_definition,
            orientation_definition=source.orientation_definition,
            orientation_bin_edge_convention=(
                source.orientation_bin_edge_convention
            ),
        )
    elif mutation == "contact_distance_contradiction":
        targets["contact_targets"][0, 0, 0] = 0.0
    else:
        raise AssertionError(f"Unhandled mutation: {mutation}")
    with pytest.raises((TypeError, ValueError), match=message):
        pair_geometry_objective(outputs, **targets)


def test_zero_weight_heads_skip_outputs_and_irrelevant_targets() -> None:
    import torch

    from interaction_first.pair_geometry_losses import (
        PairGeometryLossWeights,
        pair_geometry_objective,
    )

    outputs = _outputs(executed=("distance",))
    targets = _targets({**outputs, "geometry_schema": _schema()})
    weights = PairGeometryLossWeights(
        contact_bce=0.0,
        distance_ce=1.0,
        orientation_ce=0.0,
    )
    total, components = pair_geometry_objective(
        outputs,
        distance_bin_targets=targets["distance_bin_targets"],
        distance_mask=targets["distance_mask"],
        target_schema=outputs["geometry_schema"],
        contact_targets="intentionally malformed but disabled",
        orientation_bin_targets="intentionally malformed but disabled",
        weights=weights,
    )
    total.backward()
    assert outputs["distance_bin_logits"].grad is not None
    assert float(components["contact_bce"].detach()) == 0.0
    assert float(components["orientation_ce"].detach()) == 0.0


def test_node_support_is_candidate_derived_and_unsupported_gradient_is_zero() -> None:
    import torch

    from interaction_first.pair_geometry_losses import (
        PairGeometryLossWeights,
        pair_geometry_objective,
    )

    outputs = _outputs()
    weights = PairGeometryLossWeights(
        contact_bce=0.0,
        distance_ce=0.0,
        orientation_ce=0.0,
        epitope_consistency=1.0,
    )
    node = torch.full((2, 3), 0.5, requires_grad=True)
    mask = outputs["candidate_pair_mask"].any(dim=1).clone()
    mask[0, 1:] = False
    total, _ = pair_geometry_objective(
        outputs,
        weights=weights,
        epitope_node_probabilities=node,
        epitope_consistency_mask=mask,
    )
    total.backward()
    assert node.grad is not None
    assert bool((node.grad[mask] != 0.0).any())
    assert torch.equal(node.grad[~mask], torch.zeros_like(node.grad[~mask]))

    unsupported = ~outputs["candidate_pair_mask"].any(dim=1)
    bad_mask = mask | unsupported
    if bool(unsupported.any()):
        with pytest.raises(ValueError, match="unsupported"):
            pair_geometry_objective(
                outputs,
                weights=weights,
                epitope_node_probabilities=node.detach(),
                epitope_consistency_mask=bad_mask,
            )


def test_model_head_gating_and_objective_integrate() -> None:
    import torch

    from interaction_first.pair_geometry import ProspectivePairGeometryModel
    from interaction_first.pair_geometry_losses import (
        PairGeometryLossWeights,
        pair_geometry_objective,
    )

    schema = _schema()
    model = ProspectivePairGeometryModel(
        antibody_input_dim=5,
        antigen_input_dim=7,
        d_model=12,
        pair_dim=8,
        num_heads=3,
        num_fusion_layers=1,
        ffn_dim=24,
        geometry_schema=schema,
        pair_chunk_size=2,
        dropout=0.0,
    )
    antibody = torch.randn(2, 3, 5)
    antigen = torch.randn(2, 4, 7)
    antibody_mask = torch.tensor([[True, True, True], [True, True, False]])
    antigen_mask = torch.tensor(
        [[True, True, True, True], [True, True, True, False]]
    )
    candidates = antibody_mask.unsqueeze(2) & antigen_mask.unsqueeze(1)
    outputs = model(
        antibody,
        antigen,
        antibody_mask,
        antigen_mask,
        candidates,
        enabled_heads={"distance"},
        return_pair_representations=False,
    )
    distance_targets = torch.zeros(candidates.shape, dtype=torch.long)
    weights = PairGeometryLossWeights(
        contact_bce=0.0,
        distance_ce=1.0,
        orientation_ce=0.0,
    )
    total, _ = pair_geometry_objective(
        outputs,
        distance_bin_targets=distance_targets,
        distance_mask=candidates,
        target_schema=schema,
        weights=weights,
    )
    total.backward()
    assert model.distance_head.weight.grad is not None
    assert model.orientation_head.weight.grad is None
    assert model.interaction_head.weight.grad is None


def test_real_geometry_labels_round_trip_through_model_and_objective() -> None:
    import numpy as np
    import torch

    from interaction_first.geometry_labels import (
        LABEL_BUILDER_SOURCE_SHA256,
        ResidueAtomInput,
        build_pair_geometry_labels,
    )
    from interaction_first.pair_geometry import ProspectivePairGeometryModel
    from interaction_first.pair_geometry_losses import pair_geometry_objective
    from interaction_first.pair_geometry_schema import (
        adapt_geometry_labels,
        schema_from_geometry_labels,
    )

    def atom_table(role, origins):
        coordinates = np.full((len(origins), 4, 3), np.nan, dtype=np.float64)
        names = np.full((len(origins), 4), "", dtype="<U4")
        elements = np.full((len(origins), 4), "", dtype="<U2")
        resolved = np.ones((len(origins), 4), dtype=np.bool_)
        for index, origin in enumerate(origins):
            ca = np.asarray(origin, dtype=np.float64)
            coordinates[index] = np.stack(
                (
                    ca + np.asarray((0.0, 1.0, 0.0)),
                    ca,
                    ca + np.asarray((1.0, 0.0, 0.0)),
                    ca + np.asarray((0.0, 0.0, 1.0)),
                )
            )
            names[index] = ("N", "CA", "C", "CB")
            elements[index] = ("N", "C", "C", "C")
        chain = "H" if role == "antibody" else "A"
        return ResidueAtomInput(
            coordinates=coordinates,
            atom_names=names,
            atom_elements=elements,
            atom_resolved_mask=resolved,
            heavy_atom_complete_mask=np.ones(len(origins), dtype=np.bool_),
            chain_ids=np.asarray([chain] * len(origins), dtype="<U2"),
            residue_ids=np.asarray(
                [str(index + 1) for index in range(len(origins))],
                dtype="<U8",
            ),
        )

    labels = build_pair_geometry_labels(
        atom_table("antibody", [(0.0, 0.0, 0.0), (0.0, 8.0, 0.0)]),
        atom_table("antigen", [(3.0, 0.5, 0.5), (9.0, 8.5, 0.5)]),
        provenance={
            "structure_id": "synthetic_round_trip",
            "source_sha256": "a" * 64,
            "assembly_id": "1",
            "model_id": "1",
            "parser_name": "unit_test_parser",
            "parser_version": "1.0",
            "coordinate_unit": "angstrom",
            "altloc_policy": "highest_occupancy_then_lexicographic",
            "occupancy_policy": "require_positive_finite_occupancy",
            "assembly_generation_policy": "declared_biological_assembly_1",
            "label_builder_source_sha256": LABEL_BUILDER_SOURCE_SHA256,
        },
    )
    schema = schema_from_geometry_labels(labels)
    adapted = adapt_geometry_labels(labels, schema)
    assert adapted.geometry_schema.sha256 == schema.sha256
    assert adapted.orientation_bin_targets.shape[-1] == 5

    torch.manual_seed(101)
    model = ProspectivePairGeometryModel(
        antibody_input_dim=5,
        antigen_input_dim=7,
        d_model=12,
        pair_dim=8,
        num_heads=3,
        num_fusion_layers=1,
        ffn_dim=24,
        geometry_schema=schema,
        pair_chunk_size=1,
        dropout=0.0,
    )
    candidate = torch.from_numpy(labels.pair_resolved_mask.copy()).unsqueeze(0)
    outputs = model(
        torch.randn(1, 2, 5),
        torch.randn(1, 2, 7),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        candidate,
        return_pair_representations=False,
    )
    total, components = pair_geometry_objective(
        outputs,
        contact_targets=torch.from_numpy(
            adapted.contact_targets.copy()
        ).unsqueeze(0),
        contact_mask=torch.from_numpy(adapted.contact_mask.copy()).unsqueeze(0),
        distance_bin_targets=torch.from_numpy(
            adapted.distance_bin_targets.copy()
        ).unsqueeze(0),
        distance_mask=torch.from_numpy(adapted.distance_mask.copy()).unsqueeze(0),
        orientation_bin_targets=torch.from_numpy(
            adapted.orientation_bin_targets.copy()
        ).unsqueeze(0),
        orientation_mask=torch.from_numpy(
            adapted.orientation_mask.copy()
        ).unsqueeze(0),
        target_schema=adapted.geometry_schema,
    )
    total.backward()
    assert bool(torch.isfinite(total))
    assert components["valid_complex_counts"] == {
        "contact_bce": 1,
        "distance_ce": 1,
        "orientation_ce": 1,
        "contact_brier": 0,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contact_bce", -0.1),
        ("distance_ce", float("nan")),
        ("orientation_ce", float("inf")),
        ("contact_brier", True),
        ("matched_decoy_ranking", "0.5"),
    ],
)
def test_loss_weights_fail_closed(field: str, value: object) -> None:
    from interaction_first.pair_geometry_losses import PairGeometryLossWeights

    weights = replace(PairGeometryLossWeights(), **{field: value})
    with pytest.raises(ValueError, match="finite nonnegative"):
        weights.validate()
