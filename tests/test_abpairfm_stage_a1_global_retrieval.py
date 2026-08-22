from __future__ import annotations

import torch

from antibody_pretraining.stage_a1_batch import build_stage_a1_microbatch
from scripts import run_abpairfm_stage_a1_multitask_pilot as runner


def _record(index: int) -> dict:
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    rotated = sequence[index:] + sequence[:index]
    return {
        "contact_pairs": [[0, 0], [1, 1]],
        "contact_supervision": True,
        "example_id": f"record-{index}",
        "known_positive_group": f"group-{index}",
        "retrieval_supervision": True,
        "sequence1": rotated[:9],
        "sequence2": rotated[9:18],
        "source": "test",
    }


def test_global_retrieval_mask_uses_remote_cluster_ids(monkeypatch) -> None:
    batch = build_stage_a1_microbatch(
        [_record(0), _record(1)], seed=17, maximum_tokens=64
    )
    remote_metadata = (
        batch.retrieval_left_exact_ids + 100,
        batch.retrieval_right_exact_ids + 200,
        torch.tensor(
            [
                int(batch.retrieval_left_cluster_ids[0]),
                int(batch.retrieval_left_cluster_ids[1]) + 300,
            ]
        ),
        batch.retrieval_right_cluster_ids + 400,
        batch.retrieval_positive_group_ids + 500,
    )
    responses = iter(
        [
            torch.tensor([2, 2]),
            *[
                torch.cat([local, remote])
                for local, remote in zip(
                    (
                        batch.retrieval_left_exact_ids,
                        batch.retrieval_right_exact_ids,
                        batch.retrieval_left_cluster_ids,
                        batch.retrieval_right_cluster_ids,
                        batch.retrieval_positive_group_ids,
                    ),
                    remote_metadata,
                )
            ],
        ]
    )
    monkeypatch.setattr(
        runner,
        "_all_gather_equal_shape",
        lambda _tensor, _world_size: next(responses),
    )
    allowed, targets = runner._global_retrieval_mask(batch, rank=0, world_size=2)
    assert allowed.shape == (2, 4)
    assert torch.equal(targets, torch.tensor([0, 1]))
    assert not bool(allowed[0, 2])
    assert bool(allowed[0, 3])
    assert bool(allowed[0, 0])
    assert bool(allowed[1, 1])
