#!/usr/bin/env python3
"""The node/contact-edge gradient cosine across the first epoch.

    python3 scripts/plot_gradient_trajectory.py --trace asep=/path/asep_seed*.json \
        --trace sabdab2=/path/sabdab2_seed*.json --output figure.png

Three numbers cannot show a transient; a trajectory can. Each panel plots the
per-step cosine, smoothed, against training progress, with the crossing step
marked -- the step after which the smoothed cosine never returns below zero.

Progress is normalised to the fraction of the epoch, because the two cohorts
have different numbers of steps in one epoch (1383 and 3229) and the raw step
axis would make the shorter one look slower to settle when it is not. The raw
crossing steps are annotated so nothing is hidden by the normalisation.

The lower panel is the gradient norm ratio on the same axis. A cosine says which
way the two objectives point; the ratio says how far each one steps, and the
loss-scale account rests on the second, not the first.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The smoothing window and the crossing definition come from the analysis, not
# from a copy of it. A figure drawn with a window of 25 against a table computed
# with 10 puts different crossing steps in the caption and the text, which is
# how a hand-kept duplicate of a source of truth goes wrong.
from aggregate_step_trace import COSINE_KEY as COSINE  # noqa: E402
from aggregate_step_trace import SMOOTHING, smooth  # noqa: E402

NODE_NORM = "grad_norm_node"
EDGE_NORM = "grad_norm_edge"


def crossing_of(smoothed) -> int | None:
    """The step after which the smoothed cosine never returns below zero."""
    for index in range(len(smoothed)):
        if smoothed[index] >= 0.0 and all(v >= 0.0 for v in smoothed[index:]):
            return index
    return None


def read(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload["steps"]
    return {
        "cosine": [float(entry[COSINE]) for entry in steps],
        "ratio": [float(entry[EDGE_NORM]) / float(entry[NODE_NORM])
                  if float(entry[NODE_NORM]) > 0 else float("nan") for entry in steps],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", action="append", required=True, metavar="NAME=GLOB")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cohorts = {}
    for specification in arguments.trace:
        name, _, pattern = specification.partition("=")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"no traces matching {pattern}", file=sys.stderr)
            continue
        cohorts[name] = [read(Path(p)) for p in paths]
    if not cohorts:
        return 2

    figure, axes = plt.subplots(
        2, len(cohorts), figsize=(5.2 * len(cohorts), 6.0),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    if len(cohorts) == 1:
        axes = axes.reshape(2, 1)

    colours = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    for column, (name, runs) in enumerate(cohorts.items()):
        top, bottom = axes[0][column], axes[1][column]
        steps = len(runs[0]["cosine"])
        crossings = []
        for index, run in enumerate(runs):
            smoothed = smooth(run["cosine"])
            progress = [i / len(smoothed) for i in range(len(smoothed))]
            top.plot(progress, smoothed, lw=1.1, color=colours[index % len(colours)],
                     label=f"seed {index + 1}")
            crossing = crossing_of(smoothed)
            if crossing is not None:
                crossings.append(crossing)
                top.axvline(crossing / len(smoothed), color=colours[index % len(colours)],
                            lw=0.8, ls=":", alpha=0.7)
            bottom.plot(progress, smooth(run["ratio"]), lw=1.1,
                        color=colours[index % len(colours)])

        top.axhline(0.0, color="black", lw=0.9)
        top.set_title(f"{name}   ({steps} steps in one epoch)", fontsize=10)
        top.set_ylabel(
            rf"cos($g_{{\rm node}}$, $g_{{\rm edge}}$)  ({SMOOTHING}-step mean)",
            fontsize=9,
        )
        top.legend(fontsize=8, loc="lower right", frameon=False)
        if crossings:
            top.annotate(
                "crossings at steps " + ", ".join(str(c) for c in sorted(crossings)),
                xy=(0.03, 0.94), xycoords="axes fraction", fontsize=8, va="top",
            )
        # The transient occupies a few percent of the epoch; a linear axis over
        # the whole epoch renders it as a spike at the origin, so the region it
        # lives in is shown as an inset rather than described in the caption.
        inset = top.inset_axes([0.30, 0.12, 0.36, 0.42])
        window = max(int(0.12 * steps), 60)
        for index, run in enumerate(runs):
            smoothed = smooth(run["cosine"])[:window]
            inset.plot(range(len(smoothed)), smoothed, lw=1.0,
                       color=colours[index % len(colours)])
        inset.axhline(0.0, color="black", lw=0.8)
        inset.set_title(f"first {window} steps", fontsize=7)
        inset.tick_params(labelsize=6)

        bottom.axhline(1.0, color="black", lw=0.9, ls="--")
        bottom.set_ylabel(r"$\|g_{\rm edge}\| / \|g_{\rm node}\|$", fontsize=9)
        bottom.set_xlabel("fraction of the first epoch", fontsize=9)

    figure.suptitle(
        "Node and contact-edge gradients oppose each other early, then agree",
        fontsize=11.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=200)
    print(f"written to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
