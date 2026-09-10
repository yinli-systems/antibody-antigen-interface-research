#!/usr/bin/env python3
"""The single registered reading of the AsEP epitope_ratio test split.

Registered in ``protocols/CHECKPOINT_SELECTION_CONFIRMATION_V1.md`` stage three.
That split has been read once before, for the original frozen configuration,
giving MCC 0.2584. This is the second and final reading, for the
selection-metric configuration at the ``d_model`` the mechanical rule chose.

    python3 scripts/ratio_test_confirmation.py --run-root /path/ratio_seltotal/d256 \
        --json confirmation.json

Every choice was fixed before the split was touched. The only quantity computed
from the test split is the metric at a threshold **selected on validation**, and
the only thing selected on the test split is nothing at all.

    threshold(seed) = argmax over the grid of  mean per-complex epitope MCC
                      on ratio VALIDATION for that seed
    figure(seed)    = the same quantity on ratio TEST at that threshold

The primary figure is the mean over seeds of the single-model result, because
the published rows this is placed beside are single models. The ten-seed
mean-score ensemble is computed too and reported separately and labelled, as
registered -- it is not the headline and may not be swapped for it.
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

from aggregate_baseline_margin import bootstrap_mean_interval  # noqa: E402
from aggregate_selection_metric_confirmation import (  # noqa: E402
    GRID_STEPS,
    ROLE,
    macro_mcc_curve,
    matthews,
)

# The first and only prior reading of this split, for the frozen configuration.
PRIOR_READING_MCC = 0.2584


def load_node_frame(directory: Path):
    """The epitope rows of one split's node predictions."""
    import pandas

    parquet = directory / "node_predictions.parquet"
    if not parquet.is_file():
        return None
    frame = pandas.read_parquet(parquet)
    return frame[frame["role"] == ROLE]


def metrics_at(frame, threshold: float) -> dict[str, float]:
    """Per-complex metrics at a fixed threshold, plus threshold-free ones."""
    import numpy

    from interaction_first.metrics import average_precision, roc_auc

    per_complex = []
    for _, group in frame.groupby("instance_id", sort=True):
        labels = group["label"].to_numpy().astype(bool)
        scores = group["score"].to_numpy().astype(float)
        predicted = scores >= threshold
        true_positive = int(numpy.logical_and(predicted, labels).sum())
        false_positive = int(predicted.sum()) - true_positive
        false_negative = int(labels.sum()) - true_positive
        true_negative = int((~labels).sum()) - false_positive
        precision = true_positive / (true_positive + false_positive) if predicted.any() else 0.0
        recall = true_positive / (true_positive + false_negative) if labels.any() else 0.0
        per_complex.append({
            "mcc": matthews(true_positive, false_positive, true_negative, false_negative),
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)
                   if precision + recall > 0 else 0.0),
            "auprc": average_precision(labels, scores),
            "auroc": roc_auc(labels, scores),
        })
    summary = {}
    for name in ("mcc", "precision", "recall", "f1", "auprc", "auroc"):
        values = [row[name] for row in per_complex if isinstance(row[name], float)]
        summary[name] = sum(values) / len(values) if values else float("nan")
    summary["complex_count"] = float(len(per_complex))
    return summary


def ensemble_frame(frames):
    """Mean score per residue across seeds, labels taken from the first frame."""
    import pandas

    keys = ["instance_id", "res_idx"]
    # The residue keys every seed must agree on, checked as sets before any
    # join. Equal row counts after an inner join do imply equal key sets, but
    # only by an argument the next reader has to reconstruct, and the failure it
    # would report -- a row count -- does not say which seed disagreed or about
    # what. This says both.
    key_sets = [set(map(tuple, frame[keys].itertuples(index=False, name=None)))
                for frame in frames]
    reference = key_sets[0]
    for index, observed in enumerate(key_sets[1:], start=1):
        if observed != reference:
            missing = len(reference - observed)
            extra = len(observed - reference)
            raise ValueError(
                f"seeds do not share a residue index: seed at position {index} is "
                f"missing {missing} of the first seed's residues and adds {extra} "
                f"of its own; refusing to average"
            )

    merged = None
    for index, frame in enumerate(frames):
        part = frame[keys + ["label", "score"]].rename(columns={"score": f"score_{index}"})
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part.drop(columns=["label"]), on=keys, how="inner")
    score_columns = [name for name in merged.columns if name.startswith("score_")]
    # The join is inner, so a seed that indexes residues differently would drop
    # rows instead of failing, and the ensemble would quietly be an average over
    # whatever the runs happened to have in common. Column count cannot catch
    # that -- every seed contributes a column even when it shares no rows -- so
    # the row count is what is checked.
    if len(score_columns) != len(frames) or any(len(merged) != len(f) for f in frames):
        raise ValueError(
            f"seeds do not share a residue index ({len(merged)} rows in common "
            f"against {[len(f) for f in frames]}); refusing to average"
        )
    merged["score"] = merged[score_columns].mean(axis=1)
    return pandas.DataFrame(merged[keys + ["label", "score"]])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, required=True,
                        help="directory of seed_*/{validation,test}")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    thresholds = [(index + 1) / (GRID_STEPS + 1) for index in range(GRID_STEPS)]
    per_seed, validation_frames, test_frames = {}, [], []
    for directory in sorted(arguments.run_root.glob("seed_*"),
                            key=lambda p: int(p.name.split("_")[1])):
        seed = int(directory.name.split("_")[1])
        validation = load_node_frame(directory / "validation")
        test = load_node_frame(directory / "test")
        if validation is None or test is None:
            print(f"seed {seed}: missing predictions, skipped", file=sys.stderr)
            continue
        _, curve = macro_mcc_curve(validation)
        best = max(range(len(curve)), key=lambda index: curve[index])
        threshold = thresholds[best]
        per_seed[seed] = {
            "threshold": threshold,
            "validation_mcc_at_threshold": curve[best],
            "test": metrics_at(test, threshold),
        }
        validation_frames.append(validation)
        test_frames.append(test)

    if len(per_seed) < 2:
        print("at least two seeds are required", file=sys.stderr)
        return 2

    seeds = sorted(per_seed)
    primary = {}
    for name in ("mcc", "precision", "recall", "f1", "auprc", "auroc"):
        values = [per_seed[seed]["test"][name] for seed in seeds]
        primary[name] = {"mean": sum(values) / len(values),
                         "ci95": list(bootstrap_mean_interval(values)),
                         "per_seed": values}

    # Registered as a secondary figure. Its threshold is selected the same way,
    # on the ensemble's own validation scores, so the two figures differ in the
    # model and not in the protocol applied to it.
    ensemble = None
    if len(test_frames) == len(seeds) >= 2:
        try:
            ensemble_validation = ensemble_frame(validation_frames)
            ensemble_test = ensemble_frame(test_frames)
            _, curve = macro_mcc_curve(ensemble_validation)
            best = max(range(len(curve)), key=lambda index: curve[index])
            ensemble = {"threshold": thresholds[best],
                        "validation_mcc_at_threshold": curve[best],
                        "test": metrics_at(ensemble_test, thresholds[best])}
        except ValueError as error:
            print(f"ensemble not formed: {error}", file=sys.stderr)

    print("AsEP epitope_ratio TEST -- the second and final registered reading")
    print(f"configuration: selection-metric total, d_model 256, seeds {seeds}")
    print()
    print("PRIMARY (single model, mean over seeds; threshold selected on ratio validation)")
    for name in ("mcc", "auprc", "auroc", "precision", "recall", "f1"):
        entry = primary[name]
        print(f"  {name:>10}  {entry['mean']:.4f}   "
              f"CI [{entry['ci95'][0]:.4f}, {entry['ci95'][1]:.4f}]")
    print()
    difference = primary["mcc"]["mean"] - PRIOR_READING_MCC
    print(f"vs the prior reading of this split (frozen configuration) MCC "
          f"{PRIOR_READING_MCC:.4f}: {difference:+.4f}")
    print()
    if ensemble:
        print("SECONDARY (ten-seed mean-score ensemble -- registered as secondary, "
              "not the headline)")
        for name in ("mcc", "auprc", "auroc", "precision", "recall", "f1"):
            print(f"  {name:>10}  {ensemble['test'][name]:.4f}")
        print()
    print(f"{'seed':>5}  {'thr':>6}  {'val mcc':>8}  {'test mcc':>9}  {'test auprc':>11}")
    for seed in seeds:
        entry = per_seed[seed]
        print(f"{seed:>5}  {entry['threshold']:>6.3f}  "
              f"{entry['validation_mcc_at_threshold']:>8.4f}  "
              f"{entry['test']['mcc']:>9.4f}  {entry['test']['auprc']:>11.4f}")

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps({
            "format_version": "ratio_test_confirmation_v1",
            "split": "epitope_ratio test",
            "reading": "second and final, per CHECKPOINT_SELECTION_CONFIRMATION_V1.md",
            "configuration": {"selection_metric": "total", "d_model": 256,
                              "layers": 1, "epochs": 12,
                              "representations": "ESM-2 + ESM-IF1", "seeds": seeds},
            "threshold_policy": ("selected per seed on ratio validation to maximise mean "
                                 "per-complex epitope MCC, applied unchanged to test"),
            "primary_single_model": primary,
            "secondary_ensemble": ensemble,
            "prior_reading_mcc": PRIOR_READING_MCC,
            "difference_from_prior_reading": difference,
            "per_seed": per_seed,
            "claim_boundary": (
                "One reading of one test split, taken once, with every choice fixed "
                "beforehand. No further reading of this split follows from any outcome. "
                "The epitope_group test split remains permanently frozen and was not "
                "touched."
            ),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
