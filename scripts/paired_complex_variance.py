#!/usr/bin/env python3
"""Separate seed repeatability from complex-level uncertainty.

    python3 scripts/paired_complex_variance.py \
        --arm aligned=/path/esm_if1_only --arm baseline=/path/esm2 \
        --json variance.json

A margin averaged over seeds says the arm is reliably better *given this
validation set*. It says nothing about whether the advantage is spread across
complexes or carried by a few, and those are different claims with different
implications for a new antigen.

Two variance components, computed from the same paired predictions:

seed repeatability
    Per seed, the mean over complexes of the paired difference. Its spread is
    how much the answer depends on training randomness.

complex spread
    Per complex, the mean over seeds of the paired difference. Its spread is how
    much the answer depends on which complexes were drawn, and the fraction of
    complexes favouring the arm is reported with an exact sign bound.

The two are not independent samples of one thing. A `seed x complex` table has
`S x C` cells but nowhere near that many degrees of freedom, and treating the
cells as independent -- which a naive bootstrap over rows does -- understates
the interval. The complex-level bootstrap here resamples **complexes**, keeping
each complex's whole row of seeds together, which is the unit that would differ
in a new dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from aggregate_baseline_margin import clopper_pearson_lower_bound  # noqa: E402

ROLE = "epitope"


def per_complex_auprc(directory: Path) -> dict[str, float] | None:
    """Per-complex pair AUPRC from one run's predictions."""
    import pandas

    from interaction_first.metrics import average_precision

    parquet = directory / "validation" / "pair_predictions.parquet"
    if not parquet.is_file():
        return None
    frame = pandas.read_parquet(parquet)
    out = {}
    for identifier, group in frame.groupby("instance_id", sort=True):
        value = average_precision(
            group["label"].to_numpy().astype(bool), group["score"].to_numpy().astype(float)
        )
        if isinstance(value, float):
            out[str(identifier)] = value
    return out


def bootstrap_over(values, iterations: int = 10000, seed: int = 20260911):
    import numpy

    array = numpy.asarray(values, dtype=float)
    generator = numpy.random.default_rng(seed)
    draws = generator.integers(0, array.size, size=(iterations, array.size))
    means = array[draws].mean(axis=1)
    return float(numpy.quantile(means, 0.025)), float(numpy.quantile(means, 0.975))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=ROOT",
                        help="exactly two: the arm and its baseline, in that order")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    if len(arguments.arm) != 2:
        print("exactly two --arm entries are required", file=sys.stderr)
        return 2

    import numpy

    arms = {}
    for specification in arguments.arm:
        name, _, root = specification.partition("=")
        per_seed = {}
        for directory in sorted(Path(root).glob("seed_*"),
                                key=lambda p: int(p.name.split("_")[1])):
            values = per_complex_auprc(directory)
            if values:
                per_seed[int(directory.name.split("_")[1])] = values
        arms[name] = per_seed

    (arm_name, arm), (base_name, base) = arms.items()
    seeds = sorted(set(arm) & set(base))
    if len(seeds) < 2:
        print(f"only {len(seeds)} seed(s) in both arms", file=sys.stderr)
        return 2
    complexes = sorted(set.intersection(*[set(arm[s]) for s in seeds],
                                        *[set(base[s]) for s in seeds]))
    if not complexes:
        print("the arms share no complexes", file=sys.stderr)
        return 2

    # difference[seed][complex]; every cell present, so both marginals are over
    # the same table and neither is an average of a different sample.
    table = numpy.array([[arm[s][c] - base[s][c] for c in complexes] for s in seeds])

    by_seed = table.mean(axis=1)
    by_complex = table.mean(axis=0)
    favourable = int((by_complex > 0).sum())

    report = {
        "format_version": "paired_complex_variance_v1",
        "arm": arm_name, "baseline": base_name,
        "seeds": seeds, "complex_count": len(complexes),
        "overall_mean": float(table.mean()),
        "seed_level": {
            "means": [float(v) for v in by_seed],
            "sd": float(by_seed.std(ddof=1)),
            "ci95_over_seeds": list(bootstrap_over(by_seed)),
        },
        "complex_level": {
            "sd": float(by_complex.std(ddof=1)),
            "ci95_over_complexes": list(bootstrap_over(by_complex)),
            "favourable_complexes": favourable,
            "favourable_fraction": favourable / len(complexes),
            "sign_lower_bound": clopper_pearson_lower_bound(favourable, len(complexes)),
            "quantiles": {
                str(q): float(numpy.quantile(by_complex, q))
                for q in (0.05, 0.25, 0.5, 0.75, 0.95)
            },
        },
        "claim_boundary": (
            "Validation predictions only. The seed interval describes repeatability "
            "under training randomness; the complex interval describes sensitivity to "
            "which complexes were drawn. Neither removes the fact that this split "
            "informed model selection."
        ),
    }

    print(f"{arm_name} - {base_name}   {len(seeds)} seeds x {len(complexes)} complexes")
    print(f"  overall mean paired difference  {report['overall_mean']:+.6f}")
    print()
    print(f"  seed level     sd {report['seed_level']['sd']:.6f}   "
          f"CI [{report['seed_level']['ci95_over_seeds'][0]:+.6f}, "
          f"{report['seed_level']['ci95_over_seeds'][1]:+.6f}]")
    print(f"  complex level  sd {report['complex_level']['sd']:.6f}   "
          f"CI [{report['complex_level']['ci95_over_complexes'][0]:+.6f}, "
          f"{report['complex_level']['ci95_over_complexes'][1]:+.6f}]")
    print()
    print(f"  complexes favouring the arm  {favourable}/{len(complexes)} "
          f"({report['complex_level']['favourable_fraction']:.1%})   "
          f"sign bound {report['complex_level']['sign_lower_bound']:.4f}")
    quantiles = report["complex_level"]["quantiles"]
    print("  per-complex difference quantiles  "
          + "  ".join(f"p{int(float(q) * 100)}:{v:+.4f}" for q, v in quantiles.items()))
    print()
    print("  The complex-level interval is the wider one and is the one that "
          "describes a new antigen.")

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
