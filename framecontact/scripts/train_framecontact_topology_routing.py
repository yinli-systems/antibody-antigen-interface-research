#!/usr/bin/env python3
"""Train one isolated adaptive topology-routing condition on development roles."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import train_framecontact as confirmatory_trainer  # noqa: E402
from interaction_first.topology_routing_experiments import (  # noqa: E402
    TOPOLOGY_ROUTING_EXPERIMENTS,
)
from interaction_first.topology_routed_attention import (  # noqa: E402
    TopologyRoutedFrameContactModel,
)


CONFIRMATORY_EXPERIMENT_NAMES = tuple(confirmatory_trainer.EXPERIMENTS)


def install_isolated_registry() -> None:
    """Replace, never extend, the process-local confirmatory registry."""
    overlap = set(CONFIRMATORY_EXPERIMENT_NAMES) & set(
        TOPOLOGY_ROUTING_EXPERIMENTS
    )
    if overlap:
        raise RuntimeError(
            f"Topology-routing names collide with confirmatory names: {overlap}"
        )
    confirmatory_trainer.EXPERIMENTS.clear()
    confirmatory_trainer.EXPERIMENTS.update(
        {
            name: dict(specification)
            for name, specification in TOPOLOGY_ROUTING_EXPERIMENTS.items()
        }
    )


def main() -> None:
    install_isolated_registry()
    confirmatory_trainer.FrameContactModel = TopologyRoutedFrameContactModel
    confirmatory_trainer.main()


if __name__ == "__main__":
    main()
