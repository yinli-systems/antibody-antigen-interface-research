#!/usr/bin/env python3
"""Stage two of the checkpoint-selection confirmation, and the `d_model` choice.

Registered in ``protocols/CHECKPOINT_SELECTION_CONFIRMATION_V1.md``. Two things
are computed here and nothing else is:

1. ``M(d)`` -- the mean per-complex epitope MCC on **group validation** over the
   paired seeds, at ``d_model = d``, with the threshold chosen per seed on that
   same validation split to maximise that quantity.
2. ``d*`` -- by the rule the protocol states so that it cannot be exercised: the
   larger model only when the paired bootstrap interval of ``M(256) - M(128)``
   excludes zero, and 128 otherwise.

Per-complex, not pooled. The two are different numbers on this data and the
protocol names the per-complex one, so the pooled figure is not computed at all
rather than being computed and set aside.

    python3 scripts/aggregate_selection_metric_confirmation.py \
        --arm 128=/path/seltotal/d128 --arm 256=/path/seltotal/d256 \
        --json confirmation.json

The group **test** split is not read here and cannot be: only directories named
on the command line are opened, and the protocol permits validation only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from aggregate_baseline_margin import bootstrap_mean_interval  # noqa: E402

GRID_STEPS = 199
ROLE = "epitope"


def matthews(true_positive: int, false_positive: int,
             true_negative: int, false_negative: int) -> float:
    """MCC, defined as zero when a margin is empty rather than undefined."""
    numerator = true_positive * true_negative - false_positive * false_negative
    denominator = math.sqrt(
        float(true_positive + false_positive) * float(true_positive + false_negative)
        * float(true_negative + false_positive) * float(true_negative + false_negative)
    )
    return 0.0 if denominator == 0.0 else numerator / denominator


def macro_mcc_curve(frame) -> tuple[list[float], list[float]]:
    """Mean per-complex MCC at every threshold on a uniform grid.

    Complexes are averaged, not pooled: each contributes one MCC regardless of
    how many residues it has, which is the quantity the protocol names and the
    one the external tables report.
    """
    import numpy

    thresholds = [(index + 1) / (GRID_STEPS + 1) for index in range(GRID_STEPS)]
    per_complex = []
    for _, group in frame.groupby("instance_id", sort=True):
        labels = group["label"].to_numpy().astype(bool)
        scores = group["score"].to_numpy().astype(float)
        positives = int(labels.sum())
        negatives = int((~labels).sum())
        row = []
        for threshold in thresholds:
            predicted = scores >= threshold
            true_positive = int(numpy.logical_and(predicted, labels).sum())
            false_positive = int(predicted.sum()) - true_positive
            row.append(matthews(true_positive, false_positive,
                                negatives - false_positive, positives - true_positive))
        per_complex.append(row)
    if not per_complex:
        raise ValueError("no complexes in the prediction frame")
    means = numpy.asarray(per_complex, dtype=float).mean(axis=0)
    return thresholds, [float(value) for value in means]


def read_arm(root: Path) -> dict[int, dict]:
    """Per seed: the best achievable mean per-complex MCC and where it sits."""
    import pandas

    results: dict[int, dict] = {}
    for directory in sorted(root.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
        parquet = directory / "validation" / "node_predictions.parquet"
        if not parquet.is_file():
            continue
        frame = pandas.read_parquet(parquet)
        frame = frame[frame["role"] == ROLE]
        thresholds, curve = macro_mcc_curve(frame)
        best = max(range(len(curve)), key=lambda index: curve[index])
        results[int(directory.name.split("_")[1])] = {
            "mcc": curve[best],
            "threshold": thresholds[best],
            "mcc_at_half": curve[min(range(len(curve)),
                                     key=lambda i: abs(thresholds[i] - 0.5))],
            "complex_count": int(frame["instance_id"].nunique()),
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", action="append", required=True, metavar="D=PATH",
                        help="d_model and its run root, e.g. 128=/path/d128")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    arms: dict[int, dict] = {}
    for specification in arguments.arm:
        width, _, path = specification.partition("=")
        if not path:
            print(f"--arm wants D=PATH, got {specification!r}", file=sys.stderr)
            return 2
        arms[int(width)] = read_arm(Path(path))

    if set(arms) != {128, 256}:
        print(f"the rule is stated over d_model 128 and 256; got {sorted(arms)}",
              file=sys.stderr)
        return 2

    paired = sorted(set(arms[128]) & set(arms[256]))
    if len(paired) < 2:
        print(f"only {len(paired)} seed(s) present in both arms", file=sys.stderr)
        return 2

    differences = [arms[256][seed]["mcc"] - arms[128][seed]["mcc"] for seed in paired]
    lower, upper = bootstrap_mean_interval(differences)
    means = {width: sum(arms[width][seed]["mcc"] for seed in paired) / len(paired)
             for width in (128, 256)}

    decided = lower > 0.0 or upper < 0.0
    chosen = max((128, 256), key=lambda width: means[width]) if decided else 128
    positive = sum(1 for value in differences if value > 0.0)

    print(f"seeds paired : {len(paired)}  {paired}")
    print(f"M(128)       : {means[128]:.6f}")
    print(f"M(256)       : {means[256]:.6f}")
    print(f"M(256)-M(128): {sum(differences) / len(differences):+.6f}   "
          f"positive on {positive}/{len(paired)} seeds")
    print(f"bootstrap CI : [{lower:+.6f}, {upper:+.6f}]")
    print(f"decided      : {decided}   (the interval "
          f"{'excludes' if decided else 'contains'} zero)")
    print(f"d* = {chosen}" + ("" if decided else "   by the tie convention, not by comparison"))
    print()
    print(f"{'seed':>5}  {'M128':>9}  {'thr128':>7}  {'M256':>9}  {'thr256':>7}  {'diff':>9}")
    for seed in paired:
        one, two = arms[128][seed], arms[256][seed]
        print(f"{seed:>5}  {one['mcc']:>9.5f}  {one['threshold']:>7.3f}  "
              f"{two['mcc']:>9.5f}  {two['threshold']:>7.3f}  "
              f"{two['mcc'] - one['mcc']:>+9.5f}")

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps({
            "format_version": "checkpoint_selection_confirmation_stage_two_v1",
            "quantity": "mean per-complex epitope MCC on group validation, "
                        "threshold chosen per seed on that same split",
            "seeds": paired,
            "mean_by_d_model": means,
            "paired_difference_mean": sum(differences) / len(differences),
            "paired_difference_positive_seeds": positive,
            "bootstrap_ci95": [lower, upper],
            "decided": decided,
            "d_star": chosen,
            "rule": "d* = argmax M(d) when the paired interval excludes zero, else 128",
            "per_seed": {str(width): {str(seed): arms[width][seed] for seed in paired}
                         for width in (128, 256)},
            "claim_boundary": (
                "Group validation only. The group test split is permanently frozen "
                "and is not read here. This fixes d_model for the single registered "
                "ratio-test confirmation and authorises nothing else."
            ),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
