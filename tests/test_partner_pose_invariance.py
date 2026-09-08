"""Tests for the executable partner-pose contract.

The suite proves three things in order, because the first two are what make the
third trustworthy: the probe fails a model that reads bound inter-chain
geometry, it refuses to pass a model that ignores the backbone, and only then
does it certify the real R2 model.
"""

from __future__ import annotations

import importlib.util

import pytest

from interaction_first.partner_pose_invariance import (
    DEFAULT_TRANSLATION,
    InvarianceVerdict,
    PartnerPoseInvarianceError,
    activate_zero_initialised_parameters,
    assert_partner_pose_invariant,
    probe_partner_pose_invariance,
    random_rotation,
    rigid_transform,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def _schema():
    from interaction_first.pair_geometry_schema import PairGeometrySchema

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


def _r2_model(*, intra_chain_geometry_layers: int = 1):
    import torch

    from interaction_first.pair_geometry import ProspectivePairGeometryModel

    torch.manual_seed(20260908)
    model = ProspectivePairGeometryModel(
        antibody_input_dim=7,
        antigen_input_dim=11,
        d_model=16,
        pair_dim=12,
        num_heads=4,
        num_fusion_layers=2,
        ffn_dim=32,
        geometry_schema=_schema(),
        num_distance_bins=8,
        contact_distance_bin_count=3,
        num_orientation_axes=3,
        num_orientation_bins=6,
        num_pair_update_layers=1,
        pair_chunk_size=4,
        dropout=0.0,
        centering_iterations=32,
        intra_chain_geometry_layers=intra_chain_geometry_layers,
    )
    # Double precision so the verdict reflects structure, not float32 round-off:
    # a 120 A translation alone costs about 1e-6 in single precision.
    return model.eval().to(torch.float64)


def _inputs():
    import torch

    torch.manual_seed(4)
    batch, antibody_length, antigen_length = 2, 5, 6
    antibody_mask = torch.ones(batch, antibody_length, dtype=torch.bool)
    antigen_mask = torch.ones(batch, antigen_length, dtype=torch.bool)
    return {
        "antibody_features": torch.randn(batch, antibody_length, 7, dtype=torch.float64),
        "antigen_features": torch.randn(batch, antigen_length, 11, dtype=torch.float64),
        "antibody_mask": antibody_mask,
        "antigen_mask": antigen_mask,
        "candidate_pair_mask": antibody_mask.unsqueeze(2) & antigen_mask.unsqueeze(1),
        "antibody_backbone": torch.randn(batch, antibody_length, 3, 3, dtype=torch.float64),
        "antigen_backbone": torch.randn(batch, antigen_length, 3, 3, dtype=torch.float64),
    }


def _probe(model, inputs, moved="antigen_backbone"):
    import torch

    def run(replacement):
        with torch.no_grad():
            return model(**{**inputs, moved: replacement})

    return probe_partner_pose_invariance(run, inputs[moved])


# --- the probe must have teeth -------------------------------------------


def test_a_model_reading_inter_chain_distance_is_caught() -> None:
    """A model built the way the contract forbids must fail the probe."""
    import torch

    inputs = _inputs()

    def leaky(**keyword):
        antibody = keyword["antibody_backbone"][:, :, 1, :]
        antigen = keyword["antigen_backbone"][:, :, 1, :]
        distance = torch.cdist(antibody, antigen)
        return {"contact_logits": -distance}

    report = probe_partner_pose_invariance(
        lambda replacement: leaky(**{**inputs, "antigen_backbone": replacement}),
        inputs["antigen_backbone"],
    )

    assert report.verdict is InvarianceVerdict.PARTNER_POSE_DEPENDENT
    assert report.max_deviation > 1.0
    with pytest.raises(PartnerPoseInvarianceError, match="bound inter-chain geometry"):
        assert_partner_pose_invariant(report)


def test_a_model_ignoring_the_backbone_is_inconclusive_not_a_pass() -> None:
    """Zero-initialised geometry is invariant to everything; that is not a pass."""
    import torch

    inputs = _inputs()
    report = probe_partner_pose_invariance(
        lambda replacement: {"contact_logits": inputs["antibody_features"].sum() * torch.ones(1)},
        inputs["antigen_backbone"],
    )

    assert report.verdict is InvarianceVerdict.INCONCLUSIVE
    assert report.max_deviation == 0.0
    with pytest.raises(PartnerPoseInvarianceError, match="ignored the partner backbone"):
        assert_partner_pose_invariant(report)


def test_an_untrained_r2_model_is_reported_inconclusive() -> None:
    """The real model, before its geometry bias is activated, proves nothing."""
    report = _probe(_r2_model(), _inputs())

    assert report.verdict is InvarianceVerdict.INCONCLUSIVE
    assert report.sensitivity == 0.0


# --- the real model ------------------------------------------------------


def test_the_r2_model_is_invariant_to_the_antigen_pose() -> None:
    model = _r2_model()
    activated = activate_zero_initialised_parameters(model)
    inputs = _inputs()

    report = _probe(model, inputs, moved="antigen_backbone")

    assert activated > 0
    assert report.sensitivity > 1e-3, "the geometry pathway must be live"
    assert report.verdict is InvarianceVerdict.INVARIANT
    assert report.max_deviation < 1e-12
    assert_partner_pose_invariant(report)


def test_the_r2_model_is_invariant_to_the_antibody_pose() -> None:
    """The property is symmetric: neither chain's pose may reach the outputs."""
    model = _r2_model()
    activate_zero_initialised_parameters(model)
    inputs = _inputs()

    report = _probe(model, inputs, moved="antibody_backbone")

    assert report.sensitivity > 1e-3
    assert report.verdict is InvarianceVerdict.INVARIANT


def test_every_rigid_perturbation_is_exercised() -> None:
    model = _r2_model()
    activate_zero_initialised_parameters(model)

    report = _probe(model, _inputs())

    assert set(report.deviations) == {"translation", "rotation", "rotation_and_translation"}


# --- helpers -------------------------------------------------------------


def test_random_rotation_is_orthogonal_with_unit_determinant() -> None:
    import torch

    rotation = random_rotation(11)

    assert torch.allclose(rotation @ rotation.T, torch.eye(3, dtype=rotation.dtype), atol=1e-12)
    assert float(torch.det(rotation)) == pytest.approx(1.0, abs=1e-12)


def test_random_rotation_is_deterministic() -> None:
    import torch

    assert torch.equal(random_rotation(5), random_rotation(5))
    assert not torch.equal(random_rotation(5), random_rotation(6))


def test_rigid_transform_preserves_internal_distances() -> None:
    import torch

    torch.manual_seed(2)
    backbone = torch.randn(1, 4, 3, 3, dtype=torch.float64)

    moved = rigid_transform(
        backbone, rotation=random_rotation(3, dtype=backbone.dtype),
        translation=DEFAULT_TRANSLATION,
    )

    before = torch.cdist(backbone[0, :, 1, :], backbone[0, :, 1, :])
    after = torch.cdist(moved[0, :, 1, :], moved[0, :, 1, :])
    assert torch.allclose(before, after, atol=1e-12)


def test_rigid_transform_rejects_malformed_input() -> None:
    import torch

    with pytest.raises(PartnerPoseInvarianceError):
        rigid_transform(torch.randn(4, 3))
    with pytest.raises(PartnerPoseInvarianceError):
        rigid_transform(torch.randn(1, 4, 3, 3, dtype=torch.float64), translation=(1.0, 2.0))


def test_probe_rejects_a_run_that_changes_the_output_set() -> None:
    import torch

    backbone = torch.randn(1, 3, 3, 3, dtype=torch.float64)
    calls = {"n": 0}

    def unstable(replacement):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"a": replacement.sum()}
        return {"b": replacement.sum()}

    with pytest.raises(PartnerPoseInvarianceError, match="set of model outputs"):
        probe_partner_pose_invariance(unstable, backbone)


def test_probe_rejects_an_empty_output_mapping() -> None:
    import torch

    with pytest.raises(PartnerPoseInvarianceError, match="non-empty mapping"):
        probe_partner_pose_invariance(lambda _: {}, torch.randn(1, 3, 3, 3, dtype=torch.float64))


def test_report_serialises_for_a_receipt() -> None:
    model = _r2_model()
    activate_zero_initialised_parameters(model)

    payload = _probe(model, _inputs()).as_json()

    assert payload["verdict"] == "invariant"
    assert set(payload) >= {"verdict", "sensitivity", "deviations", "max_deviation", "tolerance"}


def test_activation_is_deterministic_and_reports_a_count() -> None:
    import torch

    first, second = _r2_model(), _r2_model()

    count = activate_zero_initialised_parameters(first)
    assert activate_zero_initialised_parameters(second) == count
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)
