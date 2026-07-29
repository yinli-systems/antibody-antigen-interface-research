"""Tests for train-only PAIRScale development holdout execution."""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import pytest

from interaction_first.development_holdout import (
    ComponentHoldoutAssignment,
)
from interaction_first.development_subset import AuthorizedInstance
from interaction_first.pairscale_holdout_training import (
    load_examples,
    run_development_holdout_training,
    split_dataset_indices,
)


@dataclass(frozen=True)
class _Record:
    instance_id: str
    antibody_id: str
    antigen_id: str


class _LazyDataset:
    def __init__(self) -> None:
        self.records = (
            _Record("i0", "a0", "g0"),
            _Record("i1", "a1", "g1"),
            _Record("i2", "a2", "g2"),
            _Record("i3", "a3", "g3"),
        )
        self.loaded: list[int] = []

    def __len__(self) -> int:
        return len(self.records)

    def identity(self, index: int) -> _Record:
        return self.records[index]

    def __getitem__(self, index: int) -> str:
        self.loaded.append(index)
        return self.records[index].instance_id


def _assignment() -> ComponentHoldoutAssignment:
    return ComponentHoldoutAssignment(
        assignment_file_sha256="1" * 64,
        assignment_content_sha256="2" * 64,
        source_subset_sha256="3" * 64,
        policy_sha256="4" * 64,
        train_instances=(
            AuthorizedInstance("i0", "a0", "g0"),
            AuthorizedInstance("i2", "a2", "g2"),
        ),
        holdout_instances=(
            AuthorizedInstance("i1", "a1", "g1"),
            AuthorizedInstance("i3", "a3", "g3"),
        ),
    )


def test_index_split_does_not_deserialize_holdout_or_train_tensors() -> None:
    dataset = _LazyDataset()
    train, holdout = split_dataset_indices(dataset, _assignment())
    assert train == (0, 2)
    assert holdout == (1, 3)
    assert dataset.loaded == []
    assert load_examples(dataset, train) == ("i0", "i2")
    assert dataset.loaded == [0, 2]


def test_index_split_rejects_inventory_and_endpoint_disagreement() -> None:
    dataset = _LazyDataset()
    incomplete = ComponentHoldoutAssignment(
        **{
            **_assignment().__dict__,
            "holdout_instances": (
                AuthorizedInstance("i1", "a1", "g1"),
            ),
        }
    )
    with pytest.raises(ValueError, match="inventories differ"):
        split_dataset_indices(dataset, incomplete)
    changed = ComponentHoldoutAssignment(
        **{
            **_assignment().__dict__,
            "holdout_instances": (
                AuthorizedInstance("i1", "wrong", "g1"),
                AuthorizedInstance("i3", "a3", "g3"),
            ),
        }
    )
    with pytest.raises(ValueError, match="endpoint identity differs"):
        split_dataset_indices(dataset, changed)


def test_runner_loads_holdout_only_inside_train_gate_branch() -> None:
    source = inspect.getsource(run_development_holdout_training)
    gate_position = source.index("if train_gate_passed:")
    holdout_load_position = source.index(
        "holdout_examples = load_examples(dataset, holdout_indices)"
    )
    access_receipt_position = source.index("HOLDOUT_ACCESS_RECEIPT.json")
    evaluation_position = source.index("HOLDOUT_EVALUATION.json")
    assert gate_position < holdout_load_position
    assert holdout_load_position < access_receipt_position < evaluation_position
    assert source.count(
        "holdout_examples = load_examples(dataset, holdout_indices)"
    ) == 1
