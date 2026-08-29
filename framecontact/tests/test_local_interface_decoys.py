from __future__ import annotations

import numpy as np
import pytest

from interaction_first.local_interface_decoys import (
    LOCAL_INTERFACE_DECOY_FORMAT,
    LocalInterfaceDecoyError,
    build_local_motif_samples,
    extract_surface_patch,
)


def _toy_complex() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    antibody = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 3.8, 0.0], [0.0, 7.6, 0.0]],
        dtype=np.float64,
    )
    antigen = np.asarray(
        [
            [3.8, 0.0, 0.0],
            [3.8, 3.8, 0.0],
            [3.8, 7.6, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 3.8, 0.0],
            [100.0, 7.6, 0.0],
        ],
        dtype=np.float64,
    )
    contacts = np.zeros((3, 6), dtype=bool)
    contacts[1, 1] = True
    return antibody, antigen, contacts


def test_local_motif_builder_separates_native_noncontact_and_matched_decoy() -> None:
    antibody, antigen, contacts = _toy_complex()
    samples = build_local_motif_samples(
        antibody,
        antigen,
        contacts,
        np.ones(3, dtype=bool),
        np.ones(6, dtype=bool),
        np.asarray([False, True, False, False, False, False]),
        antigen_sasa=np.ones(6),
        antigen_residues=list("ACDEFG"),
        patch_radius=8.0,
        graph_radius=4.5,
    )
    assert samples
    sample = samples[0]
    assert sample.positive_mask[1, 1]
    assert sample.native_local_noncontact_mask[1, 0]
    assert sample.matched_local_decoy_mask[1, 3]
    assert not np.any(sample.matched_local_decoy_mask & contacts)
    assert sample.decoy_center >= 3
    assert sample.descriptor_distance >= 0.0
    assert sample.positive_descriptor.composition
    assert LOCAL_INTERFACE_DECOY_FORMAT.endswith("v1")


def test_patch_requires_surface_centre_and_records_distance_bin() -> None:
    _, antigen, _ = _toy_complex()
    descriptor = extract_surface_patch(
        antigen,
        np.ones(6, dtype=bool),
        1,
        radius=8.0,
        graph_radius=4.5,
        sasa=np.arange(1.0, 7.0),
        residues=list("ACDEFG"),
        centre_distance=7.0,
    )
    assert descriptor.residue_count == 3
    assert descriptor.surface_area == pytest.approx(6.0)
    assert descriptor.distance_bin == 2

    with pytest.raises(LocalInterfaceDecoyError, match="surface"):
        extract_surface_patch(
            antigen,
            np.asarray([True, False, True, True, True, True]),
            1,
            radius=8.0,
            graph_radius=4.5,
        )


def test_builder_is_empty_when_no_non_epitope_surface_decoy_exists() -> None:
    antibody, antigen, contacts = _toy_complex()
    samples = build_local_motif_samples(
        antibody,
        antigen,
        contacts,
        np.ones(3, dtype=bool),
        np.ones(6, dtype=bool),
        np.asarray([False, True, True, True, True, True]),
        patch_radius=8.0,
        graph_radius=4.5,
    )
    assert samples == ()


def test_builder_excludes_distance_boundary_decoys() -> None:
    antibody, antigen, contacts = _toy_complex()
    # Move the otherwise valid decoy centre into the registered 4.5--6 A
    # ambiguity band.  It must not become a hard negative.
    antigen = antigen.copy()
    antigen[3] = [5.0, 0.0, 0.0]
    samples = build_local_motif_samples(
        antibody,
        antigen,
        contacts,
        np.ones(3, dtype=bool),
        np.ones(6, dtype=bool),
        np.asarray([False, True, False, False, False, False]),
        patch_radius=8.0,
        graph_radius=4.5,
    )
    assert samples
    assert all(sample.decoy_center != 3 for sample in samples)


def test_local_motif_dataset_attaches_masks_without_touching_labels() -> None:
    torch = pytest.importorskip("torch")
    from interaction_first.local_interface_decoys import LocalMotifMaskDataset

    antibody, antigen, contacts = _toy_complex()
    backbone_ab = torch.from_numpy(np.repeat(antibody[:, None, :], 3, axis=1)).float()
    backbone_ag = torch.from_numpy(np.repeat(antigen[:, None, :], 3, axis=1)).float()

    class _Base:
        records = [{"instance_id": "toy"}]

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {
                "instance_id": "toy",
                "pair_labels": torch.from_numpy(contacts.astype(np.float32)),
                "antibody_cdr_mask": torch.ones(3, dtype=torch.bool),
                "antigen_surface_mask": torch.ones(6, dtype=torch.bool),
                "antigen_labels": torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float32),
                "antibody_backbone": backbone_ab,
                "antigen_backbone": backbone_ag,
            }

    item = LocalMotifMaskDataset(_Base())[0]
    assert torch.equal(item["pair_labels"], torch.from_numpy(contacts.astype(np.float32)))
    assert item["local_motif_positive_mask"].dtype is torch.bool
    assert item["local_motif_matched_decoy_mask"].any()
    assert item["local_motif_sample_count"] > 0


def test_local_motif_loss_ranks_positive_over_decoy() -> None:
    torch = pytest.importorskip("torch")
    from interaction_first.framecontact_objective import (
        compute_local_motif_hard_negative_loss,
    )

    logits = torch.zeros(1, 2, 3)
    logits[0, 0, 0] = 3.0
    logits[0, 1, 1] = 3.0
    logits[0, 0, 2] = -3.0
    positive = torch.zeros_like(logits, dtype=torch.bool)
    native_noncontact = torch.zeros_like(logits, dtype=torch.bool)
    decoy = torch.zeros_like(logits, dtype=torch.bool)
    positive[0, 0, 0] = True
    positive[0, 1, 1] = True
    native_noncontact[0, 0, 1] = True
    decoy[0, 0, 2] = True
    loss, diagnostics = compute_local_motif_hard_negative_loss(
        logits, positive, native_noncontact, decoy, margin=0.25
    )
    assert float(loss) < 0.2
    assert diagnostics["eligible_complexes"] == 1
    assert float(diagnostics["positive_score"]) > float(
        diagnostics["matched_decoy_score"]
    )
