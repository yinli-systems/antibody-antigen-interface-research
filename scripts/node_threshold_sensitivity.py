#!/usr/bin/env python3
"""How much a node metric depends on where its threshold sits.

    python3 scripts/node_threshold_sensitivity.py --run-root out/esm2 \
        --role epitope --json sensitivity.json

Published epitope numbers are usually single points -- one MCC, one F1 -- and a
point on a threshold sweep is a statement about the threshold as much as about
the model. Comparing two such points taken at different operating points
measures the thresholds.

This reports the same metric at three operating points that can be named without
knowing anyone else's convention:

fixed
    The 0.5 the evaluation already uses.

best
    The maximum over the sweep. An upper bound no honest comparison can exceed,
    and the distance from ``fixed`` is how much the convention is worth.

prevalence matched
    The threshold that predicts as many positives as the split actually holds.
    It needs no tuning against the metric and is the one operating point two
    methods can be placed at without agreeing on anything else.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

STEPS = 199


def sweep(labels, scores, steps: int = STEPS) -> list[dict]:
    """Matthews correlation at every threshold on a uniform grid."""
    import numpy

    labels = numpy.asarray(labels, dtype=bool)
    scores = numpy.asarray(scores, dtype=float)
    positives = int(labels.sum())
    total = int(labels.size)
    rows = []
    for index in range(1, steps + 1):
        threshold = index / (steps + 1)
        predicted = scores >= threshold
        true_positive = int((predicted & labels).sum())
        false_positive = int((predicted & ~labels).sum())
        false_negative = positives - true_positive
        true_negative = total - true_positive - false_positive - false_negative
        denominator = (
            (true_positive + false_positive)
            * (true_positive + false_negative)
            * (true_negative + false_positive)
            * (true_negative + false_negative)
        )
        matthews = (
            ((true_positive * true_negative) - (false_positive * false_negative))
            / (denominator ** 0.5)
        ) if denominator > 0 else 0.0
        rows.append(
            {
                "threshold": threshold,
                "mcc": float(matthews),
                "predicted_positive": true_positive + false_positive,
                "precision": (true_positive / (true_positive + false_positive))
                if (true_positive + false_positive) else 0.0,
                "recall": (true_positive / positives) if positives else 0.0,
            }
        )
    return rows


def operating_points(rows: list[dict], labels) -> dict:
    """Pick out the three points the docstring names."""
    import numpy

    positives = int(numpy.asarray(labels, dtype=bool).sum())
    fixed = min(rows, key=lambda row: abs(row["threshold"] - 0.5))
    best = max(rows, key=lambda row: row["mcc"])
    matched = min(rows, key=lambda row: abs(row["predicted_positive"] - positives))
    return {"fixed": fixed, "best": best, "prevalence_matched": matched}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evaluation", default="validation")
    parser.add_argument("--role", default="epitope", choices=("epitope", "paratope"))
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    import pandas

    per_seed = {}
    for directory in sorted(arguments.run_root.glob("seed_*")):
        predictions = directory / arguments.evaluation / "node_predictions.parquet"
        if not predictions.is_file():
            continue
        frame = pandas.read_parquet(predictions)
        frame = frame.loc[frame["role"] == arguments.role]
        if frame.empty:
            continue
        rows = sweep(frame["label"].to_numpy(), frame["score"].to_numpy())
        per_seed[int(directory.name.split("_")[1])] = operating_points(
            rows, frame["label"].to_numpy()
        )

    if not per_seed:
        print(f"no {arguments.role} predictions under {arguments.run_root}", file=sys.stderr)
        return 2

    summary = {}
    for name in ("fixed", "best", "prevalence_matched"):
        values = [entry[name]["mcc"] for entry in per_seed.values()]
        summary[name] = {
            "mcc_mean": statistics.mean(values),
            "mcc_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "threshold_mean": statistics.mean(entry[name]["threshold"] for entry in per_seed.values()),
            "precision_mean": statistics.mean(entry[name]["precision"] for entry in per_seed.values()),
            "recall_mean": statistics.mean(entry[name]["recall"] for entry in per_seed.values()),
        }

    spread = summary["best"]["mcc_mean"] - summary["fixed"]["mcc_mean"]
    report = {
        "format_version": "node_threshold_sensitivity_v1",
        "role": arguments.role,
        "evaluated_on": arguments.evaluation,
        "seed_count": len(per_seed),
        "operating_points": summary,
        "best_minus_fixed": spread,
        "best_over_fixed": (summary["best"]["mcc_mean"] / summary["fixed"]["mcc_mean"])
        if summary["fixed"]["mcc_mean"] else None,
        "per_seed": {str(seed): entry for seed, entry in sorted(per_seed.items())},
        "claim_boundary": (
            "A threshold sweep on stored predictions. It says how far a single-point "
            "metric can move with its operating point; it does not identify anyone "
            "else's operating point."
        ),
    }

    print(f"{arguments.role} nodes, {len(per_seed)} seeds, {arguments.evaluation}")
    print(f"{'operating point':<20s} {'threshold':>10s} {'mcc':>10s} {'precision':>10s} {'recall':>8s}")
    for name in ("fixed", "best", "prevalence_matched"):
        entry = summary[name]
        print(f"{name:<20s} {entry['threshold_mean']:>10.3f} {entry['mcc_mean']:>10.6f}"
              f" {entry['precision_mean']:>10.4f} {entry['recall_mean']:>8.4f}")
    print(f"\nbest exceeds fixed by {spread:+.6f}"
          + (f"  ({report['best_over_fixed']:.2f}x)" if report["best_over_fixed"] else ""))

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
