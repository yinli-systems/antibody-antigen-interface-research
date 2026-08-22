from __future__ import annotations

import torch

from antibody_pretraining.stage_a1_batch import build_stage_a1_microbatch


def _record(index: int, *, structural: bool = True, direct: bool = True) -> dict:
    residues = "ACDEFGHIKLMNPQRSTVWY"
    rotated = residues[index:] + residues[:index]
    return {
        "contact_pairs": [[0, 0], [2, 3]] if structural else [],
        "contact_supervision": structural,
        "example_id": f"example-{index}",
        "known_positive_group": f"group-{index}",
        "retrieval_supervision": structural or direct,
        "sequence1": rotated[:9],
        "sequence2": rotated[9:18],
    }


def test_stage_a1_microbatch_is_deterministic_and_contact_aligned() -> None:
    records = [_record(index) for index in range(4)] + [
        _record(index, structural=False, direct=index < 6) for index in range(4, 7)
    ]
    first = build_stage_a1_microbatch(records, seed=1701, maximum_tokens=64)
    second = build_stage_a1_microbatch(records, seed=1701, maximum_tokens=64)
    assert torch.equal(first.tokens, second.tokens)
    assert torch.equal(first.contact_labels, second.contact_labels)
    assert first.tokens.shape == (7, 64)
    assert len(first.retrieval_batch_indices) == 6
    assert int(first.contact_labels.sum()) == 8
    assert bool((first.contact_labels == 0).any())
    assert int(first.contact_left_token_indices.min()) >= 1
    assert int(first.contact_right_token_indices.min()) >= 12
    assert len(first.retrieval_left_cluster_ids) == 6
    assert len(first.retrieval_positive_group_ids) == 6
    assert first.residue_count == 7 * 18


def test_stage_a1_masks_shared_endpoint_false_negatives() -> None:
    records = [_record(index) for index in range(4)]
    records[1]["sequence1"] = records[0]["sequence1"]
    batch = build_stage_a1_microbatch(records, seed=17, maximum_tokens=64)
    assert not bool(batch.retrieval_allowed[0, 1])
    assert not bool(batch.retrieval_allowed[1, 0])
    assert bool(torch.diagonal(batch.retrieval_allowed).all())


def test_stage_a1_masks_shared_source_cluster_false_negatives() -> None:
    records = [_record(index) for index in range(4)]
    records[0]["endpoint_cluster1"] = "cluster-shared"
    records[1]["endpoint_cluster2"] = "cluster-shared"
    batch = build_stage_a1_microbatch(records, seed=17, maximum_tokens=64)
    assert not bool(batch.retrieval_allowed[0, 1])
    assert not bool(batch.retrieval_allowed[1, 0])
    assert bool(torch.diagonal(batch.retrieval_allowed).all())
