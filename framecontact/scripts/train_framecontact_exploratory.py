#!/usr/bin/env python3
"""Train one isolated local-pose exploratory condition on development roles."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import train_framecontact as confirmatory_trainer  # noqa: E402
from interaction_first.exploratory_experiments import (  # noqa: E402
    EXPLORATORY_EXPERIMENTS,
)


CONFIRMATORY_EXPERIMENT_NAMES = tuple(confirmatory_trainer.EXPERIMENTS)


def install_isolated_registry() -> None:
    """Replace, never extend, the process-local trainer registry."""
    overlap = set(CONFIRMATORY_EXPERIMENT_NAMES) & set(EXPLORATORY_EXPERIMENTS)
    if overlap:
        raise RuntimeError(f"Exploratory names collide with confirmatory names: {overlap}")
    confirmatory_trainer.EXPERIMENTS.clear()
    confirmatory_trainer.EXPERIMENTS.update(
        {name: dict(specification) for name, specification in EXPLORATORY_EXPERIMENTS.items()}
    )


def main() -> None:
    install_isolated_registry()
    confirmatory_trainer.main()


if __name__ == "__main__":
    main()
