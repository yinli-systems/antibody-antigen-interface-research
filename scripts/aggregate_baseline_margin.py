#!/usr/bin/env python3
"""Turn several seed replicates into a statement about the baseline margin.

The single-seed ablation put the pair head above the predicted-marginal outer
product. One observation cannot establish that. This aggregates per-seed
evaluations into an interval and a decision.

    python3 scripts/aggregate_baseline_margin.py --run-root /path/to/seeds

Two criteria are reported, both fixed in advance:

significant
    The exact Clopper-Pearson one-sided 95% lower bound on the fraction of seeds
    with a positive margin exceeds 0.5. For a unanimous result this bound is
    ``alpha ** (1 / n)``, which first clears 0.5 at five seeds, so fewer than
    five replicates cannot satisfy it however large the margin is.

meaningful
    The lower end of the percentile bootstrap interval for the mean margin
    exceeds the smallest effect worth caring about.

The margin, its interval and the per-seed values are reported whether or not
either criterion holds. A null result is recorded as completely as a positive
one.
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

from interaction_first.pair_evaluation import (  # noqa: E402
    PairEvaluationError,
    evaluate_pair_predictions,
)

sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
from report_pair_evaluation import load_run  # noqa: E402

DEFAULT_ALPHA = 0.05
DEFAULT_SESOI = 0.0


def clopper_pearson_lower_bound(successes: int, trials: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Exact one-sided lower confidence bound on a binomial rate."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise PairEvaluationError("invalid success/trial counts")
    if successes == 0:
        return 0.0
    from scipy import stats

    return float(stats.beta.ppf(alpha, successes, trials - successes + 1))


def bootstrap_mean_interval(values, iterations: int = 10000, seed: int = 20260909,
                            confidence: float = 0.95):
    """Percentile bootstrap interval for the mean."""
    import numpy

    array = numpy.asarray(values, dtype=float)
    if array.size == 0:
        raise PairEvaluationError("no values to bootstrap")
    generator = numpy.random.default_rng(seed)
    draws = generator.integers(0, array.size, size=(iterations, array.size))
    means = array[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(numpy.quantile(means, tail)), float(numpy.quantile(means, 1.0 - tail))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, required=True,
                        help="directory holding seed_*/validation subdirectories")
    parser.add_argument("--baseline", default="outer_product")
    parser.add_argument("--metric", default="auprc")
    parser.add_argument("--scope", default="macro", choices=("macro", "micro"))
    parser.add_argument("--sesoi", type=float, default=DEFAULT_SESOI,
                        help="smallest margin worth caring about")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    seed_dirs = sorted(
        (path for path in arguments.run_root.glob("seed_*") if (path / "validation").is_dir()),
        key=lambda path: int(path.name.split("_")[1]),
    )
    if not seed_dirs:
        print(f"no seed_*/validation directories under {arguments.run_root}", file=sys.stderr)
        return 2

    per_seed = []
    for directory in seed_dirs:
        seed = int(directory.name.split("_")[1])
        try:
            report = evaluate_pair_predictions(load_run(directory / "validation"))
        except PairEvaluationError as error:
            print(f"seed {seed}: {error}", file=sys.stderr)
            continue
        model = report[arguments.scope].get(arguments.metric)
        baseline = (
            report["baselines"].get(arguments.baseline, {}).get(arguments.scope, {})
            .get(arguments.metric)
        )
        if not isinstance(model, float) or not isinstance(baseline, float):
            print(f"seed {seed}: {arguments.metric} unavailable", file=sys.stderr)
            continue
        per_seed.append({"seed": seed, "model": model, "baseline": baseline,
                         "margin": model - baseline})

    if len(per_seed) < 2:
        print("at least two usable seeds are required", file=sys.stderr)
        return 2

    margins = [entry["margin"] for entry in per_seed]
    count = len(margins)
    positive = sum(1 for value in margins if value > 0.0)
    mean = sum(margins) / count
    variance = sum((value - mean) ** 2 for value in margins) / (count - 1)
    sd = math.sqrt(variance)
    lower, upper = bootstrap_mean_interval(margins)
    bound = clopper_pearson_lower_bound(positive, count)

    significant = bound > 0.5
    meaningful = lower > arguments.sesoi
    report = {
        "format_version": "pair_baseline_margin_replication_v1",
        "scope": arguments.scope,
        "metric": arguments.metric,
        "baseline": arguments.baseline,
        "seed_count": count,
        "positive_seed_count": positive,
        "mean_margin": mean,
        "sd_margin": sd,
        "effect_size": (mean / sd) if sd > 0 else None,
        "bootstrap_ci95": [lower, upper],
        "sign_lower_bound": bound,
        "sesoi": arguments.sesoi,
        "significant": significant,
        "meaningful": meaningful,
        "verdict": "established" if (significant and meaningful) else "not established",
        "per_seed": per_seed,
        "claim_boundary": (
            "Development-only replication on the validation split. It establishes a "
            "margin over a baseline, not a benchmark result, and authorizes no test "
            "access or model selection."
        ),
    }

    print(f"metric      : {arguments.scope} {arguments.metric} vs {arguments.baseline}")
    print(f"seeds       : {count}  positive {positive}/{count}")
    print(f"mean margin : {mean:+.6f}  sd {sd:.6f}"
          + (f"  d {report['effect_size']:.3f}" if report["effect_size"] else ""))
    print(f"bootstrap CI: [{lower:+.6f}, {upper:+.6f}]")
    print(f"sign bound  : {bound:.4f} (needs > 0.5)   significant: {significant}")
    print(f"vs SESOI {arguments.sesoi:+.6f}                     meaningful : {meaningful}")
    print(f"VERDICT     : {report['verdict']}")
    print()
    for entry in per_seed:
        print(f"  seed {entry['seed']:>3d}  model {entry['model']:.6f}  "
              f"baseline {entry['baseline']:.6f}  margin {entry['margin']:+.6f}")

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
