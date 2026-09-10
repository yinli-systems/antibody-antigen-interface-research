"""Balancers, and the control that provably does nothing on this model.

The measured cosine between the node and pair gradients on the shared trunk is
+0.70, with none of 240 complexes negative. PCGrad's projection fires only on a
negative dot product, so on this model it should be the identity -- which makes
it a negative control for the whole conflict-removal family rather than a
competitor. These tests establish that mechanically, before any training run
establishes it empirically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.multitask_balancers import (  # noqa: E402
    BALANCERS,
    build_balancer,
)


class _Model:
    def __init__(self, shared, heads) -> None:
        self._all = list(shared) + list(heads)

    def parameters(self):
        return self._all


def _setup():
    shared = torch.nn.Parameter(torch.ones(4))
    head = torch.nn.Parameter(torch.ones(2))
    return shared, head, _Model([shared], [head])


def test_the_default_adds_the_losses() -> None:
    shared, head, model = _setup()

    build_balancer("none").step(
        {"node": (shared * 2).sum() + head.sum(), "pair": (shared * 3).sum()}, [shared], model
    )

    assert torch.equal(shared.grad, torch.full((4,), 5.0))
    assert torch.equal(head.grad, torch.ones(2))


def test_pcgrad_is_the_identity_when_the_gradients_agree() -> None:
    """The control property. On this model the dot product is never negative."""
    shared, head, model = _setup()

    result = build_balancer("pcgrad").step(
        {"node": (shared * 2).sum() + head.sum(), "pair": (shared * 3).sum()}, [shared], model
    )

    assert torch.equal(shared.grad, torch.full((4,), 5.0))
    assert result["pcgrad_conflict_rate"] == 0.0


def test_pcgrad_projects_when_the_gradients_oppose() -> None:
    """And it is a faithful implementation, not a no-op that always agrees."""
    shared, head, model = _setup()

    result = build_balancer("pcgrad").step(
        {"node": (shared * 2).sum() + head.sum(), "pair": (shared * -3).sum()}, [shared], model
    )

    assert torch.allclose(shared.grad, torch.zeros(4), atol=1e-6)
    assert result["pcgrad_conflict_rate"] > 0.0


def test_pcgrad_leaves_head_parameters_alone() -> None:
    """A head only one loss reaches cannot conflict with anything."""
    shared, head, model = _setup()

    build_balancer("pcgrad").step(
        {"node": (shared * 2).sum() + (head * 7).sum(), "pair": (shared * -3).sum()},
        [shared], model,
    )

    assert torch.equal(head.grad, torch.full((2,), 7.0))


def test_gradnorm_moves_the_task_weights() -> None:
    shared, head, model = _setup()

    result = build_balancer("gradnorm").step(
        {"node": (shared * 2).sum() + head.sum(), "pair": (shared * 3).sum()}, [shared], model
    )

    assert result["weight_node"] != 1.0 or result["weight_pair"] != 1.0
    assert abs(result["weight_node"] + result["weight_pair"] - 2.0) < 1e-6


def test_gradnorm_quietens_the_louder_task() -> None:
    """The scale hypothesis in one step: the larger gradient is down-weighted."""
    shared, head, model = _setup()
    balancer = build_balancer("gradnorm")

    for _ in range(6):
        shared.grad = None
        head.grad = None
        result = balancer.step(
            {"node": (shared * 1).sum() + head.sum(), "pair": (shared * 10).sum()},
            [shared], model,
        )

    assert result["weight_pair"] < result["weight_node"]


def test_weights_stay_positive() -> None:
    """A task driven to a negative weight would be trained away from."""
    shared, head, model = _setup()
    balancer = build_balancer("gradnorm")

    for _ in range(20):
        shared.grad = None
        head.grad = None
        result = balancer.step(
            {"node": (shared * 0.01).sum() + head.sum(), "pair": (shared * 50).sum()},
            [shared], model,
        )

    assert all(v > 0 for k, v in result.items() if k.startswith("weight_"))


def test_an_unknown_balancer_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown balancer"):
        build_balancer("cagrad")


def test_the_registry_names_what_is_implemented() -> None:
    """FAMO and CAGrad are named in the protocol and not implemented here.

    The protocol says an approximation whose behaviour is not the published
    method's must not be shipped, so their absence is visible rather than
    silently filled in.
    """
    assert sorted(BALANCERS) == ["gradnorm", "none", "pcgrad"]
