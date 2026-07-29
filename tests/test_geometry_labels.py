"""Focused tests for the prospective, data-agnostic geometry-label builder."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from interaction_first.geometry_labels import (
    DEFAULT_DISTANCE_BIN_EDGES_ANGSTROM,
    FORMAT_VERSION,
    LABEL_BUILDER_SOURCE_SHA256,
    ORIENTATION_AXIS_NAMES,
    ORIENTATION_BIN_COUNT,
    ORIENTATION_BIN_EDGE_CONVENTION,
    ORIENTATION_SCHEMA_VERSION,
    ResidueAtomInput,
    build_pair_geometry_labels,
)


def _provenance() -> dict[str, str]:
    return {
        "structure_id": "synthetic_geometry_fixture",
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
    }


def _atom_table(
    *,
    role: str,
    origins: list[tuple[float, float, float]],
) -> ResidueAtomInput:
    residue_count = len(origins)
    coordinates = np.full((residue_count, 4, 3), np.nan, dtype=np.float64)
    atom_names = np.full((residue_count, 4), "", dtype="<U4")
    atom_elements = np.full((residue_count, 4), "", dtype="<U2")
    resolved = np.zeros((residue_count, 4), dtype=np.bool_)
    for residue_index, origin in enumerate(origins):
        ca = np.asarray(origin, dtype=np.float64)
        coordinates[residue_index] = np.stack(
            (
                ca + np.asarray((0.0, 1.0, 0.0)),
                ca,
                ca + np.asarray((1.0, 0.0, 0.0)),
                ca + np.asarray((4.49, 0.0, 0.0)),
            )
        )
        atom_names[residue_index] = ("N", "CA", "C", "H")
        atom_elements[residue_index] = ("N", "C", "C", "H")
        resolved[residue_index] = True
    chain = "H" if role == "antibody" else "A"
    return ResidueAtomInput(
        coordinates=coordinates,
        atom_names=atom_names,
        atom_elements=atom_elements,
        atom_resolved_mask=resolved,
        heavy_atom_complete_mask=np.ones(residue_count, dtype=np.bool_),
        chain_ids=np.asarray([chain] * residue_count, dtype="<U2"),
        residue_ids=np.asarray(
            [str(index + 1) for index in range(residue_count)],
            dtype="<U8",
        ),
    )


def _inputs() -> tuple[ResidueAtomInput, ResidueAtomInput]:
    antibody = _atom_table(
        role="antibody",
        origins=[(0.0, 0.0, 0.0), (0.0, 10.0, 0.0)],
    )
    antigen = _atom_table(
        role="antigen",
        origins=[(5.5, 0.0, 0.0), (10.0, 10.0, 0.0)],
    )
    return antibody, antigen


def test_exact_heavy_atom_contacts_bins_and_sensitivity_labels() -> None:
    antibody, antigen = _inputs()
    labels = build_pair_geometry_labels(
        antibody,
        antigen,
        provenance=_provenance(),
        distance_bin_edges=(4.5, 5.0, 6.0, 8.0),
    )

    assert labels.minimum_heavy_atom_distance.shape == (2, 2)
    assert labels.minimum_heavy_atom_distance[0, 0] == pytest.approx(4.5)
    assert labels.minimum_heavy_atom_distance[1, 1] == pytest.approx(9.0)
    assert bool(labels.primary_contact_labels[0, 0])
    assert not bool(labels.primary_contact_labels[1, 1])
    assert labels.distance_bin_indices[0, 0] == 0
    assert labels.distance_bin_indices[1, 1] == 4
    assert labels.sensitivity_cutoffs.tolist() == [5.0, 6.0, 8.0]
    assert labels.sensitivity_contact_labels[:, 0, 0].tolist() == [
        True,
        True,
        True,
    ]
    assert labels.sensitivity_contact_labels[:, 1, 1].tolist() == [
        False,
        False,
        False,
    ]
    assert bool(labels.pair_resolved_mask.all())
    assert bool(labels.sensitivity_contact_masks.all())


def test_hydrogen_coordinates_do_not_change_minimum_heavy_atom_distance() -> None:
    antibody, antigen = _inputs()
    baseline = build_pair_geometry_labels(
        antibody,
        antigen,
        provenance=_provenance(),
    )
    changed_coordinates = antibody.coordinates.copy()
    changed_coordinates[0, 3] = antigen.coordinates[0, 1]
    changed = build_pair_geometry_labels(
        replace(antibody, coordinates=changed_coordinates),
        antigen,
        provenance=_provenance(),
    )
    np.testing.assert_array_equal(
        baseline.minimum_heavy_atom_distance,
        changed.minimum_heavy_atom_distance,
    )
    assert baseline.artifact_sha256 != changed.artifact_sha256


def test_orientation_uses_right_handed_backbone_frames() -> None:
    antibody, antigen = _inputs()
    labels = build_pair_geometry_labels(
        antibody,
        antigen,
        provenance=_provenance(),
    )

    np.testing.assert_allclose(
        labels.antibody_to_antigen_direction[0, 0],
        np.asarray((1.0, 0.0, 0.0)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        labels.antigen_to_antibody_direction[0, 0],
        np.asarray((-1.0, 0.0, 0.0)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        labels.relative_frame_rotation[0, 0],
        np.eye(3),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        labels.orientation_angles[0, 0, :4],
        np.asarray((0.0, np.pi / 2.0, -np.pi, np.pi / 2.0)),
        atol=1e-12,
    )
    assert np.isnan(labels.orientation_angles[0, 0, 4])
    assert labels.orientation_component_valid_mask[0, 0].tolist() == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert labels.orientation_axis_names.tolist() == list(ORIENTATION_AXIS_NAMES)
    assert labels.orientation_axis_is_cyclic.tolist() == [
        True,
        False,
        True,
        False,
        True,
    ]
    assert labels.orientation_bin_edges.shape == (5, ORIENTATION_BIN_COUNT - 1)
    assert str(labels.orientation_bin_edge_convention) == (
        ORIENTATION_BIN_EDGE_CONVENTION
    )
    assert labels.orientation_bin_indices[0, 0, 4] == -1
    assert bool(labels.orientation_valid_mask.all())


def test_unresolved_residues_are_masked_and_never_labeled_negative() -> None:
    antibody, antigen = _inputs()
    coordinates = antigen.coordinates.copy()
    names = antigen.atom_names.copy()
    elements = antigen.atom_elements.copy()
    resolved = antigen.atom_resolved_mask.copy()
    coordinates[1] = np.nan
    names[1] = ""
    elements[1] = ""
    resolved[1] = False
    complete = antigen.heavy_atom_complete_mask.copy()
    complete[1] = False
    antigen = replace(
        antigen,
        coordinates=coordinates,
        atom_names=names,
        atom_elements=elements,
        atom_resolved_mask=resolved,
        heavy_atom_complete_mask=complete,
    )
    labels = build_pair_geometry_labels(
        antibody,
        antigen,
        provenance=_provenance(),
    )

    assert not bool(labels.pair_resolved_mask[:, 1].any())
    assert bool(np.isnan(labels.minimum_heavy_atom_distance[:, 1]).all())
    assert bool((labels.distance_bin_indices[:, 1] == -1).all())
    assert not bool(labels.primary_contact_mask[:, 1].any())
    assert not bool(labels.sensitivity_contact_masks[:, :, 1].any())
    assert not bool(labels.orientation_valid_mask[:, 1].any())
    assert bool(np.isnan(labels.orientation_angles[:, 1]).all())


def test_partial_missing_backbone_has_explicit_orientation_mask() -> None:
    antibody, antigen = _inputs()
    coordinates = antibody.coordinates.copy()
    names = antibody.atom_names.copy()
    elements = antibody.atom_elements.copy()
    resolved = antibody.atom_resolved_mask.copy()
    coordinates[1, 0] = np.nan
    names[1, 0] = ""
    elements[1, 0] = ""
    resolved[1, 0] = False
    complete = antibody.heavy_atom_complete_mask.copy()
    complete[1] = False
    antibody = replace(
        antibody,
        coordinates=coordinates,
        atom_names=names,
        atom_elements=elements,
        atom_resolved_mask=resolved,
        heavy_atom_complete_mask=complete,
    )
    labels = build_pair_geometry_labels(
        antibody,
        antigen,
        provenance=_provenance(),
    )

    assert labels.antibody_frame_valid_mask.tolist() == [True, False]
    assert not bool(labels.orientation_valid_mask[1].any())
    assert bool(labels.pair_resolved_mask[1].all())
    assert not bool(labels.distance_bin_mask[1].any())


def test_artifact_arrays_and_provenance_are_immutable_and_hash_bound() -> None:
    antibody, antigen = _inputs()
    labels = build_pair_geometry_labels(
        antibody,
        antigen,
        provenance=_provenance(),
    )
    repeated = build_pair_geometry_labels(
        antibody,
        antigen,
        provenance=_provenance(),
    )

    assert labels.artifact_sha256 == repeated.artifact_sha256
    assert labels.provenance["label_format_version"] == FORMAT_VERSION
    assert (
        labels.provenance["orientation_schema_version"]
        == ORIENTATION_SCHEMA_VERSION
    )
    assert len(labels.input_hashes) == 16
    assert "orientation_bin_indices_sha256" in labels.output_hashes
    with pytest.raises(ValueError):
        labels.primary_contact_labels[0, 0] = False
    with pytest.raises(ValueError):
        labels.primary_contact_labels.setflags(write=True)
    with pytest.raises(ValueError):
        labels.orientation_axis_names.setflags(write=True)
    with pytest.raises(TypeError):
        labels.provenance["structure_id"] = "changed"
    with pytest.raises(TypeError):
        labels.input_hashes["new"] = "value"

    changed_provenance = _provenance()
    changed_provenance["structure_id"] = "another_fixture"
    changed = build_pair_geometry_labels(
        antibody,
        antigen,
        provenance=changed_provenance,
    )
    assert labels.artifact_sha256 != changed.artifact_sha256


@pytest.mark.parametrize(
    "required_field",
    [
        "coordinate_unit",
        "altloc_policy",
        "occupancy_policy",
        "assembly_generation_policy",
        "label_builder_source_sha256",
    ],
)
def test_coordinate_and_structure_policies_are_required_provenance(
    required_field: str,
) -> None:
    antibody, antigen = _inputs()
    provenance = _provenance()
    provenance.pop(required_field)

    with pytest.raises(ValueError, match="missing required"):
        build_pair_geometry_labels(
            antibody,
            antigen,
            provenance=provenance,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_provenance", "missing required"),
        ("bad_source_hash", "64 lowercase hex"),
        ("bad_builder_hash", "64 lowercase hex"),
        ("wrong_builder_hash", "does not match"),
        ("bad_coordinate_unit", "must equal 'angstrom'"),
        ("conflicting_reserved", "conflicting reserved"),
        ("integer_coordinates", "floating-point"),
        ("object_atom_names", "fixed-width Unicode"),
        ("float_resolved_mask", "boolean"),
        ("float_complete_mask", "one-dimensional boolean"),
        ("complete_shape_mismatch", "aligned with residues"),
        ("shape_mismatch", "align with coordinates"),
        ("duplicate_residue", "duplicate chain/residue"),
        ("overlapping_chain", "role mapping ambiguous"),
        ("duplicate_atom", "duplicate atom names"),
        ("resolved_nan", "must be finite"),
        ("unresolved_finite", "must be NaN"),
        ("unresolved_metadata", "must have empty names"),
        ("invalid_element", "invalid atom element"),
        ("bad_backbone_element", "inconsistent element"),
        ("all_unresolved", "no resolved non-hydrogen"),
        ("no_frame", "no valid resolved N-CA-C"),
        ("degenerate_frame", "implausible CA-to-C"),
        ("implausible_n_ca_bond", "implausible N-to-CA"),
        ("near_collinear_frame", "ill-conditioned near-collinear"),
        ("extreme_coordinate", "supported physical range"),
        ("bad_bins", "strictly increasing"),
        ("bins_without_primary_cutoff", "exact 4.5"),
    ],
)
def test_ambiguous_or_invalid_inputs_fail_closed(
    mutation: str,
    message: str,
) -> None:
    antibody, antigen = _inputs()
    provenance = _provenance()
    bins = (4.5, 5.0, 6.0, 8.0)

    if mutation == "missing_provenance":
        provenance.pop("assembly_id")
    elif mutation == "bad_source_hash":
        provenance["source_sha256"] = "ABC"
    elif mutation == "bad_builder_hash":
        provenance["label_builder_source_sha256"] = "ABC"
    elif mutation == "wrong_builder_hash":
        provenance["label_builder_source_sha256"] = "c" * 64
    elif mutation == "bad_coordinate_unit":
        provenance["coordinate_unit"] = "nanometer"
    elif mutation == "conflicting_reserved":
        provenance["pair_orientation"] = "antigen_by_antibody"
    elif mutation == "integer_coordinates":
        antibody = replace(antibody, coordinates=antibody.coordinates.astype(np.int64))
    elif mutation == "object_atom_names":
        antibody = replace(
            antibody,
            atom_names=antibody.atom_names.astype(object),
        )
    elif mutation == "float_resolved_mask":
        antibody = replace(
            antibody,
            atom_resolved_mask=antibody.atom_resolved_mask.astype(np.float32),
        )
    elif mutation == "float_complete_mask":
        antibody = replace(
            antibody,
            heavy_atom_complete_mask=(
                antibody.heavy_atom_complete_mask.astype(np.float32)
            ),
        )
    elif mutation == "complete_shape_mismatch":
        antibody = replace(
            antibody,
            heavy_atom_complete_mask=np.ones(3, dtype=np.bool_),
        )
    elif mutation == "shape_mismatch":
        antibody = replace(antibody, atom_elements=antibody.atom_elements[:, :-1])
    elif mutation == "duplicate_residue":
        residue_ids = antibody.residue_ids.copy()
        residue_ids[1] = residue_ids[0]
        antibody = replace(antibody, residue_ids=residue_ids)
    elif mutation == "overlapping_chain":
        antigen = replace(
            antigen,
            chain_ids=np.asarray(["H", "H"], dtype="<U2"),
        )
    elif mutation == "duplicate_atom":
        names = antibody.atom_names.copy()
        names[0, 1] = "N"
        antibody = replace(antibody, atom_names=names)
    elif mutation == "resolved_nan":
        coordinates = antibody.coordinates.copy()
        coordinates[0, 0] = np.nan
        antibody = replace(antibody, coordinates=coordinates)
    elif mutation == "unresolved_finite":
        resolved = antibody.atom_resolved_mask.copy()
        resolved[0, 3] = False
        antibody = replace(antibody, atom_resolved_mask=resolved)
    elif mutation == "unresolved_metadata":
        coordinates = antibody.coordinates.copy()
        elements = antibody.atom_elements.copy()
        resolved = antibody.atom_resolved_mask.copy()
        coordinates[0, 3] = np.nan
        elements[0, 3] = ""
        resolved[0, 3] = False
        antibody = replace(
            antibody,
            coordinates=coordinates,
            atom_elements=elements,
            atom_resolved_mask=resolved,
        )
    elif mutation == "invalid_element":
        elements = antibody.atom_elements.copy()
        elements[0, 3] = "Xx"
        antibody = replace(antibody, atom_elements=elements)
    elif mutation == "bad_backbone_element":
        elements = antibody.atom_elements.copy()
        elements[0, 1] = "Ca"
        antibody = replace(antibody, atom_elements=elements)
    elif mutation == "all_unresolved":
        coordinates = antibody.coordinates.copy()
        names = antibody.atom_names.copy()
        elements = antibody.atom_elements.copy()
        resolved = antibody.atom_resolved_mask.copy()
        coordinates[:] = np.nan
        names[:] = ""
        elements[:] = ""
        resolved[:] = False
        antibody = replace(
            antibody,
            coordinates=coordinates,
            atom_names=names,
            atom_elements=elements,
            atom_resolved_mask=resolved,
        )
    elif mutation == "no_frame":
        names = antibody.atom_names.copy()
        names[:, 0] = "O"
        antibody = replace(antibody, atom_names=names)
    elif mutation == "degenerate_frame":
        coordinates = antibody.coordinates.copy()
        coordinates[0, 2] = coordinates[0, 1]
        antibody = replace(antibody, coordinates=coordinates)
    elif mutation == "implausible_n_ca_bond":
        coordinates = antibody.coordinates.copy()
        coordinates[0, 0] = coordinates[0, 1] + np.asarray((0.0, 3.0, 0.0))
        antibody = replace(antibody, coordinates=coordinates)
    elif mutation == "near_collinear_frame":
        coordinates = antibody.coordinates.copy()
        coordinates[0, 0] = coordinates[0, 1] + np.asarray((1.0, 1e-6, 0.0))
        antibody = replace(antibody, coordinates=coordinates)
    elif mutation == "extreme_coordinate":
        coordinates = antibody.coordinates.copy()
        coordinates[0, 0, 0] = 1e7
        antibody = replace(antibody, coordinates=coordinates)
    elif mutation == "bad_bins":
        bins = (4.5, 4.5, 8.0)
    elif mutation == "bins_without_primary_cutoff":
        bins = (5.0, 6.0, 8.0)
    else:
        raise AssertionError(f"Unhandled mutation {mutation}")

    with pytest.raises((TypeError, ValueError), match=message):
        build_pair_geometry_labels(
            antibody,
            antigen,
            provenance=provenance,
            distance_bin_edges=bins,
        )


def _single_pair_at_observed_distance(
    distance: float,
) -> tuple[ResidueAtomInput, ResidueAtomInput]:
    """Construct one complete pair whose exact heavy-atom minimum is known."""
    antibody = _atom_table(role="antibody", origins=[(0.0, 0.0, 0.0)])
    antigen = _atom_table(
        role="antigen",
        origins=[(distance + 1.0, 0.0, 0.0)],
    )
    return antibody, antigen


@pytest.mark.parametrize(
    ("distance", "expected_bin", "expected_sensitivity"),
    [
        (4.5, 0, [True, True, True]),
        (5.0, 1, [True, True, True]),
        (6.0, 2, [False, True, True]),
        (8.0, 3, [False, False, True]),
    ],
)
def test_contact_cutoffs_are_inclusive(
    distance: float,
    expected_bin: int,
    expected_sensitivity: list[bool],
) -> None:
    antibody, antigen = _single_pair_at_observed_distance(distance)
    labels = build_pair_geometry_labels(
        antibody,
        antigen,
        provenance=_provenance(),
        distance_bin_edges=(4.5, 5.0, 6.0, 8.0),
    )

    assert labels.minimum_heavy_atom_distance[0, 0] == pytest.approx(distance)
    assert bool(labels.primary_contact_labels[0, 0]) == (distance <= 4.5)
    assert labels.sensitivity_contact_labels[:, 0, 0].tolist() == (
        expected_sensitivity
    )
    assert int(labels.distance_bin_indices[0, 0]) == expected_bin
    assert int(labels.distance_bin_count) == 5
    assert int(labels.primary_contact_distance_bin_count) == 1


@pytest.mark.parametrize(
    ("edge_index", "distance"),
    list(enumerate(DEFAULT_DISTANCE_BIN_EDGES_ANGSTROM)),
)
def test_every_default_distance_bin_edge_is_right_closed(
    edge_index: int,
    distance: float,
) -> None:
    antibody, antigen = _single_pair_at_observed_distance(distance)
    labels = build_pair_geometry_labels(
        antibody,
        antigen,
        provenance=_provenance(),
    )

    assert labels.minimum_heavy_atom_distance[0, 0] == pytest.approx(distance)
    assert int(labels.distance_bin_indices[0, 0]) == edge_index


@pytest.mark.parametrize("hydrogen_isotope", ["D", "T"])
def test_deuterium_and_tritium_are_excluded_from_contacts(
    hydrogen_isotope: str,
) -> None:
    antibody, antigen = _single_pair_at_observed_distance(9.0)
    coordinates = antibody.coordinates.copy()
    elements = antibody.atom_elements.copy()
    coordinates[0, 3] = antigen.coordinates[0, 1]
    elements[0, 3] = hydrogen_isotope
    antibody = replace(
        antibody,
        coordinates=coordinates,
        atom_elements=elements,
    )

    labels = build_pair_geometry_labels(
        antibody,
        antigen,
        provenance=_provenance(),
    )

    assert labels.minimum_heavy_atom_distance[0, 0] == pytest.approx(9.0)
    assert not bool(labels.primary_contact_labels[0, 0])


def test_partial_observation_can_prove_positive_but_never_create_negative() -> None:
    antibody, antigen = _single_pair_at_observed_distance(9.0)
    incomplete = antibody.heavy_atom_complete_mask.copy()
    incomplete[0] = False
    antibody = replace(antibody, heavy_atom_complete_mask=incomplete)

    negative_candidate = build_pair_geometry_labels(
        antibody,
        antigen,
        provenance=_provenance(),
    )

    assert bool(negative_candidate.pair_resolved_mask[0, 0])
    assert not bool(negative_candidate.primary_contact_labels[0, 0])
    assert not bool(negative_candidate.primary_contact_mask[0, 0])
    assert not bool(negative_candidate.pair_negative_distance_valid_mask[0, 0])
    assert not bool(negative_candidate.minimum_heavy_atom_distance_mask[0, 0])
    assert not bool(negative_candidate.distance_bin_mask[0, 0])
    assert int(negative_candidate.distance_bin_indices[0, 0]) == -1
    assert not bool(negative_candidate.sensitivity_contact_masks[:, 0, 0].any())

    antibody_positive, antigen_positive = _single_pair_at_observed_distance(4.5)
    incomplete_positive = antibody_positive.heavy_atom_complete_mask.copy()
    incomplete_positive[0] = False
    antibody_positive = replace(
        antibody_positive,
        heavy_atom_complete_mask=incomplete_positive,
    )
    observed_positive = build_pair_geometry_labels(
        antibody_positive,
        antigen_positive,
        provenance=_provenance(),
    )

    assert bool(observed_positive.primary_contact_labels[0, 0])
    assert bool(observed_positive.primary_contact_mask[0, 0])
    assert bool(observed_positive.sensitivity_contact_masks[:, 0, 0].all())
    assert not bool(observed_positive.minimum_heavy_atom_distance_mask[0, 0])
    assert not bool(observed_positive.distance_bin_mask[0, 0])
    assert int(observed_positive.distance_bin_indices[0, 0]) == -1


def test_partial_observation_has_cutoff_specific_positive_validity() -> None:
    antibody, antigen = _single_pair_at_observed_distance(6.0)
    incomplete = antibody.heavy_atom_complete_mask.copy()
    incomplete[0] = False
    antibody = replace(antibody, heavy_atom_complete_mask=incomplete)

    labels = build_pair_geometry_labels(
        antibody,
        antigen,
        provenance=_provenance(),
    )

    assert not bool(labels.primary_contact_mask[0, 0])
    assert labels.sensitivity_contact_labels[:, 0, 0].tolist() == [
        False,
        True,
        True,
    ]
    assert labels.sensitivity_contact_masks[:, 0, 0].tolist() == [
        False,
        True,
        True,
    ]


def test_orientation_poles_mask_only_undefined_components() -> None:
    antibody = _atom_table(role="antibody", origins=[(0.0, 0.0, 0.0)])
    antigen = _atom_table(role="antigen", origins=[(0.0, 0.0, 5.0)])

    labels = build_pair_geometry_labels(
        antibody,
        antigen,
        provenance=_provenance(),
    )

    assert bool(labels.orientation_valid_mask[0, 0])
    assert labels.orientation_component_valid_mask[0, 0].tolist() == [
        False,
        True,
        False,
        True,
        True,
    ]
    assert bool(np.isnan(labels.orientation_angles[0, 0, [0, 2]]).all())
    np.testing.assert_allclose(
        labels.orientation_angles[0, 0, [1, 3, 4]],
        np.asarray((0.0, np.pi, 0.0)),
        atol=1e-12,
    )
    assert labels.orientation_bin_indices[0, 0, [0, 2]].tolist() == [-1, -1]
    assert labels.orientation_bin_masks[0, 0].tolist() == (
        labels.orientation_component_valid_mask[0, 0].tolist()
    )


def test_cyclic_features_are_continuous_across_azimuth_branch_cut() -> None:
    antibody = _atom_table(role="antibody", origins=[(0.0, 0.0, 0.0)])
    antigen_above = _atom_table(
        role="antigen",
        origins=[(-5.0, 1e-6, 0.0)],
    )
    antigen_below = _atom_table(
        role="antigen",
        origins=[(-5.0, -1e-6, 0.0)],
    )
    above = build_pair_geometry_labels(
        antibody,
        antigen_above,
        provenance=_provenance(),
    )
    below = build_pair_geometry_labels(
        antibody,
        antigen_below,
        provenance=_provenance(),
    )

    assert above.orientation_angles[0, 0, 0] > 0.0
    assert below.orientation_angles[0, 0, 0] < 0.0
    np.testing.assert_allclose(
        above.orientation_circular_features[0, 0, 0],
        below.orientation_circular_features[0, 0, 0],
        atol=5e-7,
    )
    assert {
        int(above.orientation_bin_indices[0, 0, 0]),
        int(below.orientation_bin_indices[0, 0, 0]),
    } == {0, ORIENTATION_BIN_COUNT - 1}
    assert bool(above.orientation_axis_is_cyclic[0])
