from __future__ import annotations

from scripts.build_abpairfm_development_transfer_manifest import _axis_slices


def test_axis_slices_are_relative_to_training_clusters() -> None:
    rows = [
        {
            "instance_id": "train",
            "role": "train",
            "antigen_cluster_id": "ag-seen",
            "antibody_cluster_id": "ab-seen",
        },
        {
            "instance_id": "antigen-only",
            "role": "validation",
            "antigen_cluster_id": "ag-new",
            "antibody_cluster_id": "ab-seen",
        },
        {
            "instance_id": "antibody-only",
            "role": "validation",
            "antigen_cluster_id": "ag-seen",
            "antibody_cluster_id": "ab-new",
        },
        {
            "instance_id": "double",
            "role": "validation",
            "antigen_cluster_id": "ag-new",
            "antibody_cluster_id": "ab-new",
        },
    ]
    slices = _axis_slices(rows)
    assert slices["unseen_antigen__seen_antibody"] == ["antigen-only"]
    assert slices["seen_antigen__unseen_antibody"] == ["antibody-only"]
    assert slices["unseen_antigen__unseen_antibody"] == ["double"]
    assert slices["seen_antigen__seen_antibody"] == []
