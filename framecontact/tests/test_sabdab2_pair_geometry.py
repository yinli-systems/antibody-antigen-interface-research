"""Tests for authenticated pair distances and sparse triangle routing inputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("numpy", "scipy", "torch")),
    reason="pair-geometry adapter dependencies are unavailable",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cdr_segments_require_one_ordered_placement_and_expand_within_chain() -> None:
    from interaction_first.sabdab2_pair_geometry import (
        _expand_mask,
        _ordered_segment_mask,
    )
    from interaction_first.sabdab2_structure import StructureCacheError

    exact = _ordered_segment_mask("AACDEFGH", ("CD", "GH"))
    assert exact == [False, False, True, True, False, False, True, True]
    assert _expand_mask(exact, 1) == [False, True, True, True, True, True, True, True]
    with pytest.raises(StructureCacheError, match="unique ordered placement"):
        _ordered_segment_mask("ACDACD", ("ACD",))


def test_numbered_cdr_mask_survives_audited_axis_deletions() -> None:
    from interaction_first.sabdab2_pair_geometry import _numbered_cdr_mask
    from interaction_first.sabdab2_structure import ResidueSpec, StructureCacheError

    numbers = (27, 28, 56, 57, 105, 112, 112, 117)
    sequence = "ABCDEFGH"
    source = tuple(
        ResidueSpec(amino_acid=amino_acid, number=number, insertion_code="")
        for amino_acid, number in zip(sequence, numbers, strict=True)
    )
    projected = source[:6] + source[7:]
    mask = _numbered_cdr_mask(source, projected, ("AB", "CD", "EFGH"))
    assert mask == [True] * len(projected)
    with pytest.raises(StructureCacheError, match="explicit release string"):
        _numbered_cdr_mask(source, projected, ("AB", "XX", "EFGH"))


def test_heavy_atom_distance_is_capped_but_preserves_contact_equivalence() -> None:
    import numpy as np

    from interaction_first.sabdab2_pair_geometry import (
        _capped_minimum_heavy_atom_distances,
    )

    class Atom:
        element = "C"

        def __init__(self, coordinate):
            self.coordinate = np.asarray(coordinate, dtype=np.float32)

        def get_name(self):
            return "CA"

        def get_coord(self):
            return self.coordinate

    class Residue:
        def __init__(self, *coordinates):
            self.atoms = tuple(Atom(value) for value in coordinates)

        def get_atoms(self):
            return iter(self.atoms)

    distances, mask = _capped_minimum_heavy_atom_distances(
        (Residue((0, 0, 0), (100, 100, 100)), None),
        (Residue((4, 0, 0)), Residue((25, 0, 0))),
        distance_cap=20.0,
    )
    assert mask.tolist() == [[True, True], [False, False]]
    assert distances[0, 0] == pytest.approx(4.0)
    assert distances[0, 1] > 20.0
    assert (distances < 4.5).tolist() == [[True, False], [False, False]]


def test_heavy_atom_coordinates_include_all_alternate_locations() -> None:
    import numpy as np

    from interaction_first.sabdab2_pair_geometry import _heavy_atom_coordinates

    class Atom:
        element = "C"

        def __init__(self, coordinate):
            self.coordinate = np.asarray(coordinate, dtype=np.float32)

        def get_name(self):
            return "CB"

        def get_coord(self):
            return self.coordinate

        def is_disordered(self):
            return 0

    class DisorderedAtom:
        def __init__(self):
            self.children = (Atom((1, 2, 3)), Atom((4, 5, 6)))

        def is_disordered(self):
            return 2

        def disordered_get_list(self):
            return self.children

    class Residue:
        def is_disordered(self):
            return 1

        def get_atoms(self):
            return iter((DisorderedAtom(),))

    coordinates = _heavy_atom_coordinates(Residue())
    np.testing.assert_array_equal(
        coordinates,
        np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    )


def test_contact_boundary_reconciliation_is_narrow_audited_and_exact() -> None:
    import numpy as np

    from interaction_first.sabdab2_pair_geometry import _reconcile_contact_boundary
    from interaction_first.sabdab2_structure import StructureCacheError

    raw = np.asarray([[4.5014, 4.4990, 7.0]], dtype=np.float32)
    mask = np.ones_like(raw, dtype=np.bool_)
    labels = np.asarray([[True, False, False]], dtype=np.bool_)
    reconciled, audit = _reconcile_contact_boundary(
        raw,
        mask,
        labels,
        contact_cutoff=4.5,
        coordinate_quantization_tolerance=0.002,
    )
    assert (reconciled < 4.5).tolist() == labels.tolist()
    assert audit["boundary_reconciliation_count"] == 2
    assert audit["positive_contact_reconciliation_count"] == 1
    assert audit["negative_contact_reconciliation_count"] == 1
    assert audit["maximum_boundary_adjustment_angstrom"] == pytest.approx(
        0.0014, abs=1e-6
    )
    np.testing.assert_array_equal(reconciled[:, 2], raw[:, 2])

    with pytest.raises(StructureCacheError, match="outside the coordinate"):
        _reconcile_contact_boundary(
            np.asarray([[4.51]], dtype=np.float32),
            np.ones((1, 1), dtype=np.bool_),
            np.ones((1, 1), dtype=np.bool_),
            contact_cutoff=4.5,
            coordinate_quantization_tolerance=0.002,
        )


def test_explicit_instance_selection_validation_precedes_authority_reads(
    tmp_path: Path,
) -> None:
    from interaction_first.sabdab2_pair_geometry import materialize_pair_geometry_cache

    common = dict(
        structure_cache_root=tmp_path / "structure",
        structure_manifest_sha256="0" * 64,
        contact_cache_root=tmp_path / "contact",
        contact_manifest_sha256="0" * 64,
        cif_root=tmp_path / "cif",
        metadata_path=tmp_path / "metadata.csv",
        metadata_sha256="0" * 64,
        canonical_manifest_path=tmp_path / "canonical.csv",
        canonical_manifest_sha256="0" * 64,
        output_root=tmp_path / "output",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        materialize_pair_geometry_cache(
            **common, max_instances=1, instance_ids=("case",)
        )
    with pytest.raises(ValueError, match="positive integer"):
        materialize_pair_geometry_cache(**common, max_instances=0)


def test_development_selection_is_digest_bound_and_excludes_test(
    tmp_path: Path,
) -> None:
    from interaction_first.sabdab2_pair_geometry import (
        _authenticated_selection,
        freeze_pair_geometry_development_selection,
    )
    from interaction_first.sabdab2_structure import canonical_json_bytes

    structure_root = tmp_path / "structure"
    contact_root = tmp_path / "contact"
    structure_root.mkdir()
    contact_root.mkdir()
    records = [
        {"instance_id": "train-case"},
        {"instance_id": "validation-case"},
        {"instance_id": "test-case"},
    ]
    structure_manifest = structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    contact_manifest = contact_root / "CONTACT_CACHE_MANIFEST.json"
    structure_manifest.write_bytes(canonical_json_bytes({"records": records}))
    contact_manifest.write_bytes(canonical_json_bytes({"records": records}))
    split = tmp_path / "split.csv"
    split.write_text(
        "instance_id,role\n"
        "train-case,train\n"
        "validation-case,validation\n"
        "test-case,test\n",
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    result = freeze_pair_geometry_development_selection(
        split_path=split,
        split_sha256=_sha256(split),
        structure_cache_root=structure_root,
        structure_manifest_sha256=_sha256(structure_manifest),
        contact_cache_root=contact_root,
        contact_manifest_sha256=_sha256(contact_manifest),
        output_path=selection,
    )
    assert result["counts"] == {
        "selected": 2,
        "train": 1,
        "validation": 1,
        "excluded_test": 1,
    }
    identifiers, authority = _authenticated_selection(
        selection, result["selection_sha256"]
    )
    assert identifiers == ["train-case", "validation-case"]
    assert authority["selection_excluded_roles"] == ["test"]


def test_dataset_authenticates_artifact_and_collates_pair_geometry(tmp_path: Path) -> None:
    import numpy as np
    import torch

    from interaction_first.sabdab2_pair_geometry import (
        PAIR_GEOMETRY_ARTIFACT_FORMAT,
        PAIR_GEOMETRY_CACHE_FORMAT,
        Sabdab2PairGeometryBatchCollator,
        Sabdab2PairGeometryDataset,
    )
    from interaction_first.sabdab2_structure import canonical_json_bytes

    class Base:
        records = ({"instance_id": "case"},)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            antibody_backbone = torch.zeros(2, 3, 3)
            antigen_backbone = torch.zeros(3, 3, 3)
            return {
                "instance_id": "case",
                "antibody_features": torch.zeros(2, 4),
                "antigen_features": torch.zeros(3, 4),
                "antibody_backbone": antibody_backbone,
                "antigen_backbone": antigen_backbone,
                "antibody_chain_ids": torch.zeros(2, dtype=torch.long),
                "antigen_chain_ids": torch.zeros(3, dtype=torch.long),
                "antibody_frame_valid_mask": torch.ones(2, dtype=torch.bool),
                "antigen_frame_valid_mask": torch.ones(3, dtype=torch.bool),
                "antibody_missing_frame_mask": torch.zeros(2, dtype=torch.bool),
                "antigen_missing_frame_mask": torch.zeros(3, dtype=torch.bool),
                "antibody_labels": torch.tensor([1.0, 0.0]),
                "antigen_labels": torch.tensor([1.0, 0.0, 0.0]),
                "pair_labels": torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                "pair_candidate_mask": torch.ones(2, 3, dtype=torch.bool),
            }

    artifact = tmp_path / "artifacts" / "case.npz"
    artifact.parent.mkdir()
    np.savez_compressed(
        artifact,
        format_version=np.asarray(PAIR_GEOMETRY_ARTIFACT_FORMAT),
        instance_id=np.asarray("case"),
        pair_distances=np.asarray([[4.0, 9.0, 21.0], [8.0, 11.0, 21.0]], dtype=np.float32),
        pair_distance_mask=np.ones((2, 3), dtype=np.bool_),
        antibody_cdr_mask=np.asarray([True, False]),
        antigen_surface_mask=np.asarray([True, True, False]),
    )
    manifest = {
        "format_version": PAIR_GEOMETRY_CACHE_FORMAT,
        "checks": {"contact_labels_exactly_reconstructed_for_every_observed_pair": True},
        "records": [
            {
                "instance_id": "case",
                "artifact_path": "artifacts/case.npz",
                "artifact_sha256": _sha256(artifact),
            }
        ],
    }
    manifest_path = tmp_path / "PAIR_GEOMETRY_CACHE_MANIFEST.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    dataset = Sabdab2PairGeometryDataset(
        Base(), tmp_path, manifest_sha256=_sha256(manifest_path)
    )
    item = dataset[0]
    batch = Sabdab2PairGeometryBatchCollator()([item])
    assert batch["pair_distances"].shape == (1, 2, 3)
    assert torch.equal(batch["antibody_cdr_mask"], torch.tensor([[True, False]]))
    assert torch.equal(
        batch["antigen_surface_mask"], torch.tensor([[True, True, False]])
    )
    assert batch["pair_distance_mask"].all()
