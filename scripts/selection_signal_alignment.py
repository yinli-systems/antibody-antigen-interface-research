#!/usr/bin/env python3
"""Is a checkpoint-selection signal aligned with node generalisation?

    python3 scripts/selection_signal_alignment.py --run-root /path/to/arm \
        --target node_epitope_mcc --json alignment.json

A selection signal is good or bad only in relation to what it is used to choose
for. The size of its gradient says nothing about that: a term the trunk barely
optimises could still track generalisation perfectly, and a term with a large
gradient could track it not at all. So this measures the relation directly, on
the epoch trajectories that already exist.

Two quantities, per seed, over the epochs the run actually took:

rank correlation
    Spearman between the candidate signal and the node metric across epochs.
    Spearman rather than Pearson because selection only uses the ordering.
    Signals that are minimised (losses) are negated first, so that for every
    signal a *higher* value means *better*, and a positive correlation always
    means alignment.

selection regret
    ``M(e*_oracle) - M(e*_signal)``, where ``e*_signal`` is the epoch the rule
    would pick and ``e*_oracle`` the epoch with the best node metric. Zero means
    the rule picked as well as anything could. This is the quantity that matters
    in use, and it is reported paired across seeds so two rules can be compared
    on the same trajectories.

An epoch trajectory shorter than three points cannot support a correlation and
that seed is dropped with a note rather than contributing a degenerate value.
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

from aggregate_baseline_margin import (  # noqa: E402
    bootstrap_mean_interval,
    clopper_pearson_lower_bound,
)

# Signals the training loop records that could be used to stop early. Every one
# is a loss, so lower is better and all are negated to a higher-is-better sense.
MINIMISED_SIGNALS = ("total", "partner_rank", "edge", "epitope_bce")
MINIMUM_EPOCHS = 3


def spearman(first, second) -> float | None:
    """Rank correlation, with ties given average ranks."""
    import numpy

    def ranks(values):
        array = numpy.asarray(values, dtype=float)
        order = array.argsort()
        result = numpy.empty(len(array), dtype=float)
        result[order] = numpy.arange(len(array), dtype=float)
        # Average the ranks of tied values, or the correlation depends on the
        # order the sort happened to produce.
        _, inverse, counts = numpy.unique(array, return_inverse=True, return_counts=True)
        sums = numpy.zeros(len(counts))
        numpy.add.at(sums, inverse, result)
        return (sums / counts)[inverse]

    a, b = ranks(first), ranks(second)
    if a.std() == 0.0 or b.std() == 0.0:
        return None
    return float(numpy.corrcoef(a, b)[0, 1])


def read_trajectory(history_path: Path, target: str) -> dict | None:
    """One run's per-epoch node metric and candidate signals."""
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    epochs = payload.get("history")
    if not isinstance(epochs, list):
        return None
    metric, signals = [], {name: [] for name in MINIMISED_SIGNALS}
    for entry in epochs:
        validation = entry.get("validation", {})
        if not isinstance(validation.get(target), (int, float)):
            return None
        metric.append(float(validation[target]))
        for name in MINIMISED_SIGNALS:
            value = validation.get(name)
            signals[name].append(float(value) if isinstance(value, (int, float)) else None)
    kept = {name: values for name, values in signals.items() if all(v is not None for v in values)}
    return {"metric": metric, "signals": kept} if len(metric) >= MINIMUM_EPOCHS else None


def analyse(trajectory: dict) -> dict:
    metric = trajectory["metric"]
    oracle = max(metric)
    out = {"epochs": len(metric), "oracle": oracle,
           "oracle_epoch": metric.index(oracle) + 1, "signals": {}}
    for name, values in trajectory["signals"].items():
        # Negated so higher is better for every signal alike.
        oriented = [-value for value in values]
        chosen = oriented.index(max(oriented))
        out["signals"][name] = {
            "spearman": spearman(oriented, metric),
            "chosen_epoch": chosen + 1,
            "metric_at_choice": metric[chosen],
            "regret": oracle - metric[chosen],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, required=True,
                        help="directory of seed_*/train/history.json")
    parser.add_argument("--target", default="node_epitope_mcc",
                        help="the per-epoch node metric selection is judged against")
    parser.add_argument("--label", default="")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    per_seed, skipped = {}, []
    for directory in sorted(arguments.run_root.glob("seed_*"),
                            key=lambda p: int(p.name.split("_")[1])):
        trajectory = read_trajectory(directory / "train" / "history.json", arguments.target)
        if trajectory is None:
            skipped.append(directory.name)
            continue
        per_seed[int(directory.name.split("_")[1])] = analyse(trajectory)

    if not per_seed:
        print(f"no usable trajectories under {arguments.run_root} "
              f"carrying {arguments.target!r}", file=sys.stderr)
        if skipped:
            print(f"skipped: {', '.join(skipped)}", file=sys.stderr)
        return 2

    names = sorted(set().union(*(set(entry["signals"]) for entry in per_seed.values())))
    seeds = sorted(per_seed)
    summary = {}
    for name in names:
        correlations = [per_seed[s]["signals"][name]["spearman"] for s in seeds
                        if name in per_seed[s]["signals"]
                        and per_seed[s]["signals"][name]["spearman"] is not None]
        regrets = [per_seed[s]["signals"][name]["regret"] for s in seeds
                   if name in per_seed[s]["signals"]]
        entry = {"seed_count": len(regrets),
                 "mean_spearman": sum(correlations) / len(correlations) if correlations else None,
                 "mean_regret": sum(regrets) / len(regrets) if regrets else None}
        if len(regrets) >= 2:
            entry["regret_ci95"] = list(bootstrap_mean_interval(regrets))
        if len(correlations) >= 2:
            entry["spearman_ci95"] = list(bootstrap_mean_interval(correlations))
            entry["positive_correlation_seeds"] = sum(1 for v in correlations if v > 0)
            entry["positive_sign_lower_bound"] = clopper_pearson_lower_bound(
                entry["positive_correlation_seeds"], len(correlations))
        summary[name] = entry

    label = arguments.label or arguments.run_root.name
    print(f"arm      : {label}   target {arguments.target}   seeds {seeds}")
    if skipped:
        print(f"skipped  : {', '.join(skipped)} (no {arguments.target} in the trajectory)")
    print()
    header = f"{'signal':>14}  {'rho':>7}  {'rho CI95':>18}  {'+seeds':>7}  {'regret':>9}  {'regret CI95':>20}"
    print(header)
    print("-" * len(header))
    for name in names:
        entry = summary[name]
        rho = f"{entry['mean_spearman']:+.4f}" if entry["mean_spearman"] is not None else "n/a"
        rho_ci = (f"[{entry['spearman_ci95'][0]:+.3f}, {entry['spearman_ci95'][1]:+.3f}]"
                  if "spearman_ci95" in entry else "n/a")
        positive = (f"{entry['positive_correlation_seeds']}/{entry['seed_count']}"
                    if "positive_correlation_seeds" in entry else "n/a")
        regret_ci = (f"[{entry['regret_ci95'][0]:+.5f}, {entry['regret_ci95'][1]:+.5f}]"
                     if "regret_ci95" in entry else "n/a")
        print(f"{name:>14}  {rho:>7}  {rho_ci:>18}  {positive:>7}  "
              f"{entry['mean_regret']:>+9.5f}  {regret_ci:>20}")

    # Paired, because the rules are compared on the same trajectories.
    if "partner_rank" in summary and "total" in summary:
        paired = [s for s in seeds
                  if {"partner_rank", "total"} <= set(per_seed[s]["signals"])]
        if len(paired) >= 2:
            differences = [per_seed[s]["signals"]["partner_rank"]["regret"]
                           - per_seed[s]["signals"]["total"]["regret"] for s in paired]
            mean = sum(differences) / len(differences)
            lower, upper = bootstrap_mean_interval(differences)
            worse = sum(1 for value in differences if value > 0.0)
            bound = clopper_pearson_lower_bound(worse, len(differences))
            summary["partner_rank_minus_total_regret"] = {
                "seeds": paired, "mean": mean, "bootstrap_ci95": [lower, upper],
                "worse_seed_count": worse, "sign_lower_bound": bound,
                "decided": bound > 0.5,
            }
            print()
            print(f"regret(partner_rank) - regret(total), paired over {len(paired)} seeds")
            print(f"  mean {mean:+.5f}   CI [{lower:+.5f}, {upper:+.5f}]   "
                  f"partner_rank worse on {worse}/{len(paired)}   "
                  f"sign bound {bound:.4f}   decided: {bound > 0.5}")

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps({
            "format_version": "selection_signal_alignment_v1",
            "label": label, "target": arguments.target,
            "seeds": seeds, "skipped": skipped,
            "summary": summary,
            "per_seed": {str(s): per_seed[s] for s in seeds},
            "claim_boundary": (
                "Validation trajectories only. This measures whether a signal orders "
                "epochs the way the node metric does; it says nothing about test "
                "performance and reads no test split."
            ),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
