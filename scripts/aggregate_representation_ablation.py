#!/usr/bin/env python3
"""Compare two representation arms that share their seeds.

    python3 scripts/aggregate_representation_ablation.py --run-root /path/to/arms

The arms differ only in which frozen encoders feed the model, and each seed
appears in both. That pairing is the point: seed-to-seed variation in this
setup is large -- the ten-seed replication measured a macro-AUPRC sd of 0.00168
against a mean of 0.02246, a coefficient of variation of 7.5% -- and most of it
is shared between arms trained from the same seed. Differencing within a seed
removes that shared part, so a paired test sees an effect an unpaired one at the
same n would miss.

Reported, whether or not it favours the treatment:

paired difference
    Mean of ``treatment - control`` per seed, with a percentile bootstrap
    interval over seeds and the count of seeds favouring the treatment.

sign bound
    The exact Clopper-Pearson one-sided 95% lower bound on the fraction of seeds
    favouring the treatment. For a unanimous result this is ``alpha ** (1 / n)``,
    which first clears 0.5 at five seeds.

A null result is recorded as completely as a positive one. A treatment that is
merely not worse is a null result, not a success.
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
from aggregate_baseline_margin import (  # noqa: E402
    bootstrap_mean_interval,
    clopper_pearson_lower_bound,
)
from report_pair_evaluation import load_run  # noqa: E402

CONTROL = "esm2"
TREATMENT = "esm2_antiberty_esmif1"


def arm_scores(root: Path, metric: str, scope: str) -> dict[int, float]:
    """Return the metric per seed for one arm."""
    scores: dict[int, float] = {}
    for directory in sorted(root.glob("seed_*")):
        evaluation = directory / "validation"
        if not evaluation.is_dir():
            continue
        seed = int(directory.name.split("_")[1])
        try:
            report = evaluate_pair_predictions(load_run(evaluation))
        except PairEvaluationError as error:
            print(f"{root.name} seed {seed}: {error}", file=sys.stderr)
            continue
        value = report[scope].get(metric)
        if isinstance(value, float):
            scores[seed] = value
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, required=True,
                        help="directory holding one subdirectory per arm")
    parser.add_argument("--control", default=CONTROL)
    parser.add_argument("--treatment", default=TREATMENT)
    parser.add_argument("--metric", default="auprc")
    parser.add_argument("--scope", default="macro", choices=("macro", "micro"))
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    control = arm_scores(arguments.run_root / arguments.control, arguments.metric, arguments.scope)
    treatment = arm_scores(
        arguments.run_root / arguments.treatment, arguments.metric, arguments.scope
    )
    shared = sorted(set(control) & set(treatment))
    unpaired = sorted(set(control) ^ set(treatment))
    if len(shared) < 2:
        print(f"at least two shared seeds are required; found {shared}", file=sys.stderr)
        return 2

    per_seed = [
        {
            "seed": seed,
            "control": control[seed],
            "treatment": treatment[seed],
            "difference": treatment[seed] - control[seed],
        }
        for seed in shared
    ]
    differences = [entry["difference"] for entry in per_seed]
    count = len(differences)
    favourable = sum(1 for value in differences if value > 0.0)
    mean = sum(differences) / count
    sd = math.sqrt(sum((value - mean) ** 2 for value in differences) / (count - 1))
    lower, upper = bootstrap_mean_interval(differences)
    bound = clopper_pearson_lower_bound(favourable, count)
    control_mean = sum(control[seed] for seed in shared) / count

    decided = bool(lower > 0.0 or upper < 0.0)
    report = {
        "format_version": "representation_ablation_v1",
        "scope": arguments.scope,
        "metric": arguments.metric,
        "control": arguments.control,
        "treatment": arguments.treatment,
        "seed_count": count,
        "unpaired_seeds": unpaired,
        "favourable_seed_count": favourable,
        "control_mean": control_mean,
        "treatment_mean": sum(treatment[seed] for seed in shared) / count,
        "mean_difference": mean,
        # Two relative estimands, both correct and not interchangeable. The
        # first divides the mean paired difference by the control's mean; the
        # second averages each seed's own ratio. On a small baseline they can
        # differ noticeably, so each is named rather than one being called "the"
        # relative difference.
        "relative_difference": (mean / control_mean) if control_mean else None,
        "relative_estimand": "ratio_of_means: mean paired difference / control arm mean",
        "relative_difference_mean_of_paired_ratios": (
            sum((row["treatment"] - row["control"]) / row["control"]
                for row in per_seed if row["control"]) / len(per_seed)
            if all(row["control"] for row in per_seed) else None
        ),
        # The rule this comparison was registered under. Three different rules
        # are in use across this project's protocols and they are not unified
        # after the fact: each number carries the rule it was registered with.
        "decision_rule_id": "bootstrap_ci_excludes_zero",
        "decision_rule": (
            "the percentile bootstrap 95% interval of the mean paired difference "
            "excludes zero"
        ),
        "sd_difference": sd,
        "effect_size": (mean / sd) if sd > 0 else None,
        "bootstrap_ci95": [lower, upper],
        "sign_lower_bound": bound,
        "decided": decided,
        "verdict": (
            "treatment better" if decided and mean > 0
            else "control better" if decided
            else "no decision"
        ),
        "per_seed": per_seed,
        "claim_boundary": (
            "Development-only paired ablation on the validation split. It compares "
            "two input representations and authorizes no test access and no model "
            "selection."
        ),
    }

    print(f"metric      : {arguments.scope} {arguments.metric}")
    print(f"arms        : {arguments.treatment} vs {arguments.control}")
    print(f"seeds       : {count} paired"
          + (f"  UNPAIRED {unpaired}" if unpaired else ""))
    print(f"control     : {control_mean:.6f}")
    print(f"treatment   : {report['treatment_mean']:.6f}")
    print(f"difference  : {mean:+.6f}  ({100 * report['relative_difference']:+.1f}%)"
          f"  sd {sd:.6f}" + (f"  d {report['effect_size']:.3f}" if report["effect_size"] else ""))
    print(f"bootstrap CI: [{lower:+.6f}, {upper:+.6f}]"
          f"   excludes zero: {decided}")
    print(f"favourable  : {favourable}/{count}   sign bound {bound:.4f}")
    print(f"VERDICT     : {report['verdict']}")
    print()
    for entry in per_seed:
        print(f"  seed {entry['seed']:>3d}  control {entry['control']:.6f}  "
              f"treatment {entry['treatment']:.6f}  diff {entry['difference']:+.6f}")

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
