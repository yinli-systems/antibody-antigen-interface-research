"""Multi-task balancers, and the negative control that should do nothing here.

The measured facts about this model are that the node and pair gradients agree
on the shared trunk -- cosine +0.70, none of 240 complexes negative -- and that
the pair gradient is about half again as large. If the failure is scale rather
than conflict, then rebalancing magnitude should help and removing conflict
should not, and the two families of method separate cleanly.

``pcgrad`` is included as the **negative control**. Its projection fires only
when two task gradients have a negative dot product, which on this model is
almost never, so it should behave like the unmodified objective. A result where
it helps as much as ``gradnorm`` would say the cosine measurement is missing
something -- most likely per-layer conflict cancelling in the aggregate.
"""

from __future__ import annotations

from typing import Any, Sequence


def _flat(gradients: Sequence[Any], parameters: Sequence[Any]):
    import torch

    return torch.cat([
        torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
        for g, p in zip(gradients, parameters)
    ])


def _assign(parameters: Sequence[Any], flat) -> None:
    """Write a flat gradient back onto ``.grad`` of each parameter."""
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        piece = flat[offset:offset + size].view_as(parameter)
        parameter.grad = piece.clone() if parameter.grad is None else parameter.grad + piece
        offset += size


class NoBalancer:
    """The default objective: the two losses are simply added."""

    name = "none"

    def step(self, losses, shared, model) -> dict[str, float]:  # noqa: ARG002
        total = sum(losses.values())
        total.backward()
        return {f"weight_{k}": 1.0 for k in losses}


def _pairwise_geometry(names: Sequence[str], flats: Sequence[Any]) -> dict[str, float]:
    """Cosine and norms between task gradients on the shared trunk."""
    import torch

    geometry: dict[str, float] = {}
    for index, name in enumerate(names):
        geometry[f"grad_norm_{name}"] = float(flats[index].norm())
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            denominator = flats[first].norm() * flats[second].norm()
            geometry[f"cosine_{names[first]}_{names[second]}"] = (
                float(torch.dot(flats[first], flats[second]) / denominator)
                if float(denominator) > 0.0 else 0.0
            )
    return geometry


class MeasuringBalancer:
    """Applies the unmodified summed gradient and reports the geometry.

    Numerically this is ``none``. It exists because ``none`` takes the ordinary
    backward path and so cannot report anything, and a diagnostic arm whose
    trajectory is not identical to the control would be measuring its own
    perturbation. Any divergence between this arm and ``none`` is a bug in the
    separated-loss path, which makes it a test of the other balancers too.
    """

    name = "measure"

    def step(self, losses, shared, model) -> dict[str, float]:  # noqa: ARG002
        import torch

        names = list(losses)
        flats = [
            _flat(torch.autograd.grad(losses[name], shared, retain_graph=True,
                                      allow_unused=True), shared)
            for name in names
        ]
        geometry = _pairwise_geometry(names, flats)
        sum(losses.values()).backward()
        return geometry


class PCGrad:
    """Project conflicting task gradients apart (Yu et al., 2020).

    The projection is applied only when the dot product is negative, which is
    what makes this a control here rather than a treatment.
    """

    name = "pcgrad"

    def __init__(self) -> None:
        self.conflicts = 0
        self.steps = 0

    def step(self, losses, shared, model) -> dict[str, float]:  # noqa: ARG002
        import torch

        names = list(losses)
        flats = []
        for name in names:
            grads = torch.autograd.grad(losses[name], shared, retain_graph=True, allow_unused=True)
            flats.append(_flat(grads, shared))

        projected = [g.clone() for g in flats]
        self.steps += 1
        # Counted for this step, not cumulatively: the caller averages what is
        # returned over the epoch, and the mean of a running average is not a
        # quantity anyone can interpret. Over one step the two agree, which is
        # why the unit tests read the same either way.
        conflicts_here = 0
        for i in range(len(flats)):
            for j in range(len(flats)):
                if i == j:
                    continue
                dot = torch.dot(projected[i], flats[j])
                if float(dot) < 0.0:
                    conflicts_here += 1
                    projected[i] = projected[i] - dot / (flats[j].norm() ** 2 + 1e-12) * flats[j]
        self.conflicts += conflicts_here

        # The cosine is already paid for: both task gradients are in hand. It is
        # reported because the conflict measurement that made this a control was
        # taken on a decomposition that excluded the partner-ranking term, and
        # the objective trained here includes it. Reporting it in situ is the
        # only way the control's premise gets checked on the split it applies to.
        geometry = _pairwise_geometry(names, flats)
        _assign(shared, sum(projected))
        # Head parameters take their own task's gradient untouched: a head only
        # one loss reaches cannot conflict with anything. Identity comparison,
        # because a tensor's __eq__ is elementwise and set membership on tensors
        # does not mean what it looks like it means.
        shared_ids = {id(p) for p in shared}
        heads = [p for p in model.parameters() if id(p) not in shared_ids and p.requires_grad]
        if heads:
            sum(losses.values()).backward(inputs=heads)
        return {"pcgrad_conflict_rate": float(conflicts_here), **geometry}


class GradNorm:
    """Rebalance task weights toward equal gradient norm (Chen et al., 2018).

    Weights are updated from the ratio of each task's gradient norm to the mean,
    corrected by its relative inverse training rate, so a task that is both
    loud and already converging is quietened.
    """

    name = "gradnorm"

    def __init__(self, alpha: float = 1.5, learning_rate: float = 0.025) -> None:
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.initial: dict[str, float] = {}
        self.weights: dict[str, float] = {}

    def step(self, losses, shared, model) -> dict[str, float]:
        import torch

        names = list(losses)
        if not self.weights:
            self.weights = {name: 1.0 for name in names}
        for name in names:
            value = float(losses[name].detach())
            if name not in self.initial and value > 0:
                self.initial[name] = value

        norms = {}
        for name in names:
            grads = torch.autograd.grad(
                self.weights[name] * losses[name], shared, retain_graph=True, allow_unused=True
            )
            norms[name] = float(_flat(grads, shared).norm())

        ready = all(self.initial.get(name, 0.0) > 0 for name in names)
        if ready:
            ratios = {
                name: float(losses[name].detach()) / self.initial[name] for name in names
            }
            mean_ratio = sum(ratios.values()) / len(ratios)
            mean_norm = sum(norms.values()) / len(norms)
            for name in names:
                relative = (ratios[name] / mean_ratio) if mean_ratio > 0 else 1.0
                target = mean_norm * (relative ** self.alpha)
                # Move the weight so this task's gradient norm approaches target.
                if norms[name] > 0:
                    self.weights[name] *= float(
                        1.0 + self.learning_rate * (target - norms[name]) / norms[name]
                    )
                self.weights[name] = max(self.weights[name], 1e-3)
            scale = len(names) / sum(self.weights.values())
            for name in names:
                self.weights[name] *= scale

        total = sum(self.weights[name] * losses[name] for name in names)
        total.backward()
        return {f"weight_{name}": self.weights[name] for name in names}


BALANCERS = {
    "none": NoBalancer,
    "measure": MeasuringBalancer,
    "pcgrad": PCGrad,
    "gradnorm": GradNorm,
}


def build_balancer(name: str):
    if name not in BALANCERS:
        raise ValueError(f"unknown balancer {name!r}; available: {sorted(BALANCERS)}")
    return BALANCERS[name]()
