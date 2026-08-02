"""The trainer's memory-saving views must satisfy every enabled loss.

`_cognate_training_view` and `_wrong_training_view` prune the model output
before it reaches the objective.  A key dropped there raises `KeyError` only
once the loss that reads it is switched on, so a pruning mistake survives every
run of the default configuration and then kills the one arm that needed it.
That is not hypothetical: the frozen v4 source shipped without
``epitope_unary_logits`` in the cognate view, and the triangle arm therefore
failed 53 seconds after it was finally scheduled.

These tests drive the real views with a real model forward for each loss
configuration in the decision matrix, so a missing key fails here instead of on
the cluster.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _trainer_views():
    sys.path.insert(0, str(_SCRIPTS))
    import train_framecontact

    return train_framecontact


def _batch(batch_size: int = 2):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_framecontact import _batch as build

    return build(batch_size)


def _model(**overrides):
    from interaction_first.framecontact import FrameContactModel

    arguments = {
        "antibody_input_dim": 12,
        "antigen_input_dim": 10,
        "d_model": 16,
        "num_heads": 4,
        "num_self_layers": 1,
        "neighbor_k": 4,
        "neighbor_radius": None,
        "dropout": 0.0,
    }
    arguments.update(overrides)
    return FrameContactModel(**arguments).eval()


def _triangle_batch(batch_size: int = 2):
    import torch

    batch = _batch(batch_size)
    antibody_length = batch["antibody_mask"].shape[1]
    antigen_length = batch["antigen_mask"].shape[1]
    batch["antibody_cdr_mask"] = batch["antibody_mask"].clone()
    batch["antigen_surface_mask"] = batch["antigen_mask"].clone()
    batch["antibody_labels"] = torch.zeros(batch_size, antibody_length)
    batch["antibody_labels"][:, 0] = 1.0
    batch["antigen_labels"] = torch.zeros(batch_size, antigen_length)
    batch["antigen_labels"][:, 0] = 1.0
    batch["pair_distances"] = torch.full(
        (batch_size, antibody_length, antigen_length), 9.0
    )
    batch["pair_distance_mask"] = (
        batch["antibody_mask"][:, :, None] & batch["antigen_mask"][:, None, :]
    )
    batch["pair_labels"] = torch.zeros(batch_size, antibody_length, antigen_length)
    batch["pair_labels"][:, 0, 0] = 1.0
    return batch


def test_cognate_view_carries_every_key_the_triangle_loss_reads() -> None:
    """The exact failure that stopped the triangle arm on the cluster."""

    trainer = _trainer_views()
    from interaction_first.framecontact_objective import (
        FrameContactLossWeights,
        FrameContactObjective,
    )

    model = _model(pair_output_mode="distogram")
    batch = _triangle_batch()
    view = trainer._cognate_training_view(model(batch))

    objective = FrameContactObjective(
        weights=FrameContactLossWeights(
            specificity=0.0, distogram=0.5, triangle=0.1
        ),
        antigen_patch_top_seeds=1,
        antigen_patch_radius=12.0,
        triangle_neighbor_radius=12.0,
        triangle_neighbor_k=4,
    )
    losses = objective(view, batch)

    assert "triangle_loss" in losses
    total = float(losses["loss"].detach())
    assert total == total


def test_wrong_view_carries_every_key_the_residue_contrastive_loss_reads() -> None:
    """The decoy residue logits must survive the wrong-partner view."""

    trainer = _trainer_views()
    from interaction_first.framecontact_objective import (
        FrameContactLossWeights,
        FrameContactObjective,
    )

    model = _model()
    batch = _triangle_batch()
    cognate = trainer._cognate_training_view(model(batch))
    wrong = {
        "fixed_antibody_wrong_antigen": [
            trainer._wrong_training_view(model(_triangle_batch()))
        ],
        "fixed_antigen_wrong_antibody": [
            trainer._wrong_training_view(model(_triangle_batch()))
        ],
    }

    objective = FrameContactObjective(
        weights=FrameContactLossWeights(
            specificity=0.0, residue_partner_contrastive=0.5
        ),
    )
    losses = objective(cognate, batch, wrong_partner_outputs=wrong)

    assert float(losses["residue_partner_contrastive_loss"].detach()) >= 0.0


def test_checkpointed_wrong_view_matches_the_plain_wrong_view() -> None:
    """Activation checkpointing must not silently change the key set."""

    import torch

    trainer = _trainer_views()

    model = _model()
    batch = _triangle_batch(1)
    with torch.enable_grad():
        plain = trainer._wrong_training_view(model(batch))
        checkpointed = trainer._checkpointed_wrong_training_view(model, batch)

    assert set(plain) == set(checkpointed)
    for key in plain:
        assert torch.allclose(
            plain[key].float(), checkpointed[key].float(), atol=1e-5
        ), key


def test_view_key_sets_are_declared_not_scattered() -> None:
    """The three views share one declaration so they cannot drift apart."""

    trainer = _trainer_views()

    assert "epitope_unary_logits" in trainer._COGNATE_VIEW_REQUIRED
    assert "paratope_unary_logits" in trainer._COGNATE_VIEW_REQUIRED
    assert "paratope_logits" in trainer._WRONG_VIEW_OPTIONAL
    assert "epitope_logits" in trainer._WRONG_VIEW_OPTIONAL
    # Every optional key needs a rule telling the checkpointed path whether the
    # model will emit it, or the two wrong-partner paths silently diverge.
    assert set(trainer._WRONG_VIEW_OPTIONAL) == set(
        trainer._WRONG_VIEW_OPTIONAL_SOURCE
    )
