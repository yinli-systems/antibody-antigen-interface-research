"""Tests for the fail-closed frontier batching boundary."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from interaction_first.frontier_batching import (  # noqa: E402
    FrontierBatchCollator,
    FrontierExample,
)


def _example(
    instance_id: str,
    antibody_id: str,
    antigen_id: str,
    antigen_length: int,
    antibody_length: int,
    supervised: bool = True,
) -> FrontierExample:
    antigen = torch.arange(antigen_length * 5, dtype=torch.float32).reshape(
        antigen_length, 5
    )
    antibody = torch.arange(antibody_length * 7, dtype=torch.float32).reshape(
        antibody_length, 7
    )
    antibody_chain_ids = torch.zeros(antibody_length, dtype=torch.long)
    antibody_chain_ids[(antibody_length + 1) // 2 :] = 1
    if not supervised:
        return FrontierExample(
            instance_id=instance_id,
            antibody_id=antibody_id,
            antigen_id=antigen_id,
            antigen_embeddings=antigen,
            antibody_embeddings=antibody,
            antibody_chain_ids=antibody_chain_ids,
        )
    target = torch.zeros(antigen_length, antibody_length, dtype=torch.int64)
    target[0, 0] = 1
    mask = torch.zeros(antigen_length, antibody_length, dtype=torch.bool)
    mask[0, 0] = True
    if antigen_length > 1:
        mask[1, 0] = True
    return FrontierExample(
        instance_id=instance_id,
        antibody_id=antibody_id,
        antigen_id=antigen_id,
        antigen_embeddings=antigen,
        antibody_embeddings=antibody,
        antibody_chain_ids=antibody_chain_ids,
        contact_target=target,
        contact_mask=mask,
    )


def test_collator_pads_shapes_masks_and_targets() -> None:
    first = _example("one", "ab-1", "ag-1", antigen_length=3, antibody_length=2)
    second = _example(
        "two",
        "ab-2",
        "ag-2",
        antigen_length=2,
        antibody_length=4,
        supervised=False,
    )

    batch = FrontierBatchCollator()([first, second])

    assert batch.batch_size == 2
    assert batch.antigen_embeddings.shape == (2, 3, 5)
    assert batch.antibody_embeddings.shape == (2, 4, 7)
    assert batch.contact_target.shape == (2, 3, 4)
    assert batch.contact_target.dtype == torch.float32
    assert batch.antigen_lengths.tolist() == [3, 2]
    assert batch.antibody_lengths.tolist() == [2, 4]
    assert batch.antigen_mask.tolist() == [[True, True, True], [True, True, False]]
    assert batch.antibody_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    assert batch.antibody_chain_ids.tolist() == [
        [0, 1, -1, -1],
        [0, 0, 1, 1],
    ]
    assert torch.count_nonzero(batch.antigen_embeddings[1, 2]) == 0
    assert torch.count_nonzero(batch.antibody_embeddings[0, 2:]) == 0
    assert batch.contact_target[0, 0, 0].item() == 1.0
    assert torch.count_nonzero(batch.contact_target[1]) == 0
    assert torch.count_nonzero(batch.contact_mask[1]) == 0


def test_pair_mask_is_exact_outer_product_and_contact_mask_is_a_subset() -> None:
    first = _example("one", "ab-1", "ag-1", antigen_length=3, antibody_length=2)
    second = _example("two", "ab-2", "ag-2", antigen_length=1, antibody_length=4)
    batch = FrontierBatchCollator()([first, second])

    expected_pair_mask = batch.antigen_mask.unsqueeze(2) & batch.antibody_mask.unsqueeze(1)
    assert torch.equal(batch.pair_mask, expected_pair_mask)
    assert batch.pair_mask[0].sum().item() == 6
    assert batch.pair_mask[1].sum().item() == 4
    assert not torch.any(batch.contact_mask & ~batch.pair_mask)
    assert batch.contact_mask[0, 0, 0]
    assert batch.contact_mask[0, 1, 0]
    assert not batch.contact_mask[0, 0, 1]


def test_unresolved_entries_remain_zero_and_do_not_enter_contact_mask() -> None:
    target = torch.tensor([[1, 0], [0, 0]], dtype=torch.uint8)
    mask = torch.tensor([[True, False], [True, False]], dtype=torch.bool)
    example = FrontierExample(
        instance_id="resolved",
        antibody_id="ab",
        antigen_id="ag",
        antibody_embeddings=torch.ones(2, 3),
        antigen_embeddings=torch.ones(2, 4),
        antibody_chain_ids=torch.tensor([0, 1]),
        contact_target=target,
        contact_mask=mask,
    )

    batch = FrontierBatchCollator()([example])

    assert batch.contact_mask.tolist() == [[[True, False], [True, False]]]
    assert batch.contact_target.tolist() == [[[1.0, 0.0], [0.0, 0.0]]]


@pytest.mark.parametrize(
    "bad_target",
    [
        torch.tensor([[0.5, 0.0]]),
        torch.tensor([[-1, 0]]),
        torch.tensor([[2, 0]]),
        torch.tensor([[float("nan"), 0.0]]),
    ],
)
def test_fractional_sentinel_and_nonfinite_contact_labels_are_rejected(
    bad_target: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="exact binary|finite"):
        FrontierExample(
            instance_id="bad-label",
            antibody_id="ab",
            antigen_id="ag",
            antibody_embeddings=torch.ones(2, 3),
            antigen_embeddings=torch.ones(1, 4),
            antibody_chain_ids=torch.tensor([0, 1]),
            contact_target=bad_target,
            contact_mask=torch.ones(1, 2, dtype=torch.bool),
        )


def test_hidden_positive_under_unresolved_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="hidden labels"):
        FrontierExample(
            instance_id="hidden",
            antibody_id="ab",
            antigen_id="ag",
            antibody_embeddings=torch.ones(2, 3),
            antigen_embeddings=torch.ones(1, 4),
            antibody_chain_ids=torch.tensor([0, 1]),
            contact_target=torch.tensor([[1, 0]]),
            contact_mask=torch.tensor([[False, True]]),
        )


def test_zero_length_and_nonfinite_embeddings_are_rejected() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        FrontierExample(
            instance_id="empty",
            antibody_id="ab",
            antigen_id="ag",
            antibody_embeddings=torch.empty(0, 3),
            antigen_embeddings=torch.ones(1, 4),
            antibody_chain_ids=torch.empty(0, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="finite"):
        FrontierExample(
            instance_id="nonfinite",
            antibody_id="ab",
            antigen_id="ag",
            antibody_embeddings=torch.tensor([[1.0, float("inf")]]),
            antigen_embeddings=torch.ones(1, 4),
            antibody_chain_ids=torch.tensor([0]),
        )


def test_authorized_positive_graph_preserves_multi_positive_edges() -> None:
    examples = [
        _example("one", "ab-shared", "ag-1", 2, 2, supervised=False),
        _example("two", "ab-2", "ag-2", 2, 2, supervised=False),
    ]
    graph = [
        ("ab-shared", "ag-1"),
        ("ab-shared", "ag-2"),
        ("ab-2", "ag-2"),
        ("ab-outside", "ag-outside"),
    ]

    batch = FrontierBatchCollator(graph)(examples)

    assert batch.positive_pair_mask.tolist() == [[True, True], [False, True]]
    assert batch.positive_pairs == tuple(sorted(graph))
    assert batch.antibody_ids == ("ab-shared", "ab-2")
    assert batch.antigen_ids == ("ag-1", "ag-2")


def test_incomplete_authorized_positive_graph_fails_closed() -> None:
    example = _example("one", "ab-1", "ag-1", 2, 2, supervised=False)
    with pytest.raises(ValueError, match="omits matched batch edges"):
        FrontierBatchCollator([("other-ab", "other-ag")])([example])


def test_inconsistent_dimensions_duplicate_instances_and_malformed_ids_fail() -> None:
    first = _example("same", "ab-1", "ag-1", 2, 2, supervised=False)
    second = FrontierExample(
        instance_id="same",
        antibody_id="ab-2",
        antigen_id="ag-2",
        antibody_embeddings=torch.ones(2, 8),
        antigen_embeddings=torch.ones(2, 6),
        antibody_chain_ids=torch.tensor([0, 1]),
    )
    with pytest.raises(ValueError, match="unique|consistent"):
        FrontierBatchCollator()([first, second])
    with pytest.raises(ValueError, match="whitespace"):
        _example(" padded ", "ab", "ag", 2, 2, supervised=False)


@pytest.mark.parametrize(
    ("chain_ids", "message"),
    [
        (torch.tensor([0, -1]), "heavy-chain ID 0 or light-chain ID 1"),
        (torch.tensor([0, 2]), "heavy-chain ID 0 or light-chain ID 1"),
        (torch.tensor([1, 1]), "contain a heavy chain"),
        (torch.tensor([0, 1], dtype=torch.int32), "torch.int64"),
        (torch.tensor([[0, 1]]), "align exactly"),
        (torch.tensor([0]), "align exactly"),
    ],
)
def test_chain_identity_rejects_separators_invalid_chains_and_misalignment(
    chain_ids: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        FrontierExample(
            instance_id="bad-chain",
            antibody_id="ab",
            antigen_id="ag",
            antibody_embeddings=torch.ones(2, 3),
            antigen_embeddings=torch.ones(1, 4),
            antibody_chain_ids=chain_ids,
        )
