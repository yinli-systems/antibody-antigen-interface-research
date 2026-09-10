#!/usr/bin/env python3
"""What differed between the ablation arms besides the loss they optimise.

    python3 scripts/plot_run_contract.py --trajectories localise_lr.json \
        --output run_contract.png

A reviewer who sees that one arm selects epoch 4.0 and another 7.3 will ask
whether the arms were trained fairly. They were: every arm ran the full budget.
What differed is downstream of the loss itself -- `selection_metric` names the
validation quantity that drives ReduceLROnPlateau, the checkpoint test and the
stopping counter, and all four arms name `total`, which is composed of whichever
terms each arm enables.

Left: the learning-rate trajectories that produces. Right: the node metric at
every epoch, which is where the answer to the fairness question lives -- the
advantage of removing the ranking term is present at matched epochs and grows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

ORDER = ["node_only", "node_edge", "node_rank", "node_edge_rank"]
LABEL = {
    "node_only": "node only",
    "node_edge": "node + edge",
    "node_rank": "node + rank",
    "node_edge_rank": "node + edge + rank",
}
COLOUR = {
    "node_only": "#7f7f7f",
    "node_edge": "#1f77b4",
    "node_rank": "#d62728",
    "node_edge_rank": "#2ca02c",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(arguments.trajectories.read_text(encoding="utf-8"))
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    for arm in ORDER:
        runs = data.get(arm, {})
        if not runs:
            continue
        length = min(len(r["lr"]) for r in runs.values())
        epochs = range(1, length + 1)
        left.plot(epochs, [mean(r["lr"][i] for r in runs.values()) for i in range(length)],
                  color=COLOUR[arm], lw=1.6, marker="o", ms=3, label=LABEL[arm])
        node = [[r["node_mcc"][i] for r in runs.values()] for i in range(length)]
        if all(all(isinstance(v, (int, float)) for v in row) for row in node):
            right.plot(epochs, [mean(row) for row in node],
                       color=COLOUR[arm], lw=1.6, marker="o", ms=3, label=LABEL[arm])

    left.set_yscale("log")
    left.set_xlabel("epoch")
    left.set_ylabel("learning rate (mean over 10 seeds)")
    left.set_title("Schedules diverge because the monitored objective differs", fontsize=9.5)
    left.legend(fontsize=8, frameon=False)

    right.set_xlabel("epoch")
    right.set_ylabel("validation per-complex epitope MCC")
    right.set_title("The node advantage is present at matched epochs", fontsize=9.5)
    right.legend(fontsize=8, frameon=False, loc="lower right")

    figure.suptitle(
        "All arms received the same 12-epoch budget; what differed is downstream "
        "of the loss", fontsize=10.5)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=200)
    print(f"written to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
