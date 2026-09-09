#!/usr/bin/env python3
"""Report every pair metric for one or more evaluation runs, against baselines.

Consumes the ``pair_predictions.parquet`` and ``node_predictions.parquet`` that
``sabdab2_interaction_first.py evaluate`` writes, rebuilds each complex, and
scores the model against the predicted-marginal outer product and the prevalence
floor.

    python3 scripts/report_pair_evaluation.py --run bilinear=/path/a --run hadamard=/path/b

Several ``--run`` arguments make this an ablation table: identical metrics for
each arm, plus each arm's margin over the same baselines. A pair head that does
not clear the outer product has not learned partner specificity, whichever arm
it is, and that comparison is the point of the report rather than a footnote.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.pair_evaluation import (  # noqa: E402
    ComplexPrediction,
    PairEvaluationError,
    evaluate_pair_predictions,
)

HEADLINE = (
    "auprc",
    "auroc",
    "bedroc_alpha20",
    "mcc",
    "best_mcc",
    "f1",
    "best_f1",
    "balanced_accuracy",
    "precision_at_true_count",
    "ef_at_1pct",
    "cohens_d",
)


def load_run(directory: Path) -> list[ComplexPrediction]:
    """Rebuild per-complex predictions from one evaluation output directory."""
    import numpy
    import pandas

    pair_path = directory / "pair_predictions.parquet"
    node_path = directory / "node_predictions.parquet"
    if not pair_path.exists():
        raise PairEvaluationError(f"{pair_path} does not exist")
    pairs = pandas.read_parquet(pair_path)
    nodes = pandas.read_parquet(node_path) if node_path.exists() else None

    predictions: list[ComplexPrediction] = []
    for instance_id, frame in pairs.groupby("instance_id", sort=True):
        antigen = sorted(frame["ag_res_idx"].unique().tolist())
        antibody = sorted(frame["ab_res_idx"].unique().tolist())
        ag_index = {value: position for position, value in enumerate(antigen)}
        ab_index = {value: position for position, value in enumerate(antibody)}

        shape = (len(antigen), len(antibody))
        labels = numpy.zeros(shape, dtype=float)
        scores = numpy.zeros(shape, dtype=float)
        # Only the pairs the evaluation actually scored are candidates; the rest
        # are absent rather than negative.
        mask = numpy.zeros(shape, dtype=bool)
        rows = frame["ag_res_idx"].map(ag_index).to_numpy()
        columns = frame["ab_res_idx"].map(ab_index).to_numpy()
        labels[rows, columns] = frame["label"].to_numpy(dtype=float)
        scores[rows, columns] = frame["score"].to_numpy(dtype=float)
        mask[rows, columns] = True

        epitope = paratope = None
        if nodes is not None:
            local = nodes[nodes["instance_id"] == instance_id]
            epitope = _marginal(local, "epitope", antigen)
            paratope = _marginal(local, "paratope", antibody)

        predictions.append(
            ComplexPrediction(
                identifier=str(instance_id),
                pair_labels=labels,
                pair_scores=scores,
                candidate_mask=mask,
                epitope_probability=epitope,
                paratope_probability=paratope,
            )
        )
    if not predictions:
        raise PairEvaluationError(f"{pair_path} contains no complex")
    return predictions


def _marginal(frame, role: str, order: list[int]):
    """Align one role's per-residue scores to the pair grid's residue order."""
    import numpy

    subset = frame[frame["role"] == role]
    if subset.empty:
        return None
    lookup = dict(zip(subset["res_idx"].tolist(), subset["score"].tolist()))
    if any(index not in lookup for index in order):
        return None  # incomplete marginals: skip the baseline rather than guess
    return numpy.asarray([lookup[index] for index in order], dtype=float)


def _table(reports: dict[str, dict]) -> str:
    lines: list[str] = []
    add = lines.append
    names = list(reports)

    for scope in ("macro", "micro"):
        add(f"{scope.upper()} (over {'complexes' if scope == 'macro' else 'candidate pairs'})")
        add(f"  {'metric':24s} " + " ".join(f"{name:>14s}" for name in names))
        for metric in HEADLINE:
            row = []
            for name in names:
                value = reports[name][scope].get(metric)
                row.append(f"{value:14.6f}" if isinstance(value, float) else f"{'-':>14s}")
            add(f"  {metric:24s} " + " ".join(row))
        add("")

    add("BASELINES (identical inputs, so identical across arms)")
    first = reports[names[0]]
    for baseline, entry in first.get("baselines", {}).items():
        macro = entry.get("macro", {}).get("auprc")
        micro = entry.get("micro", {}).get("auprc")
        add(
            f"  {baseline:24s} macro auprc {macro:.6f}"
            if isinstance(macro, float)
            else f"  {baseline:24s} macro auprc -"
        )
        if isinstance(micro, float):
            add(f"  {'':24s} micro auprc {micro:.6f}")
    add("")

    add("MARGIN OVER BASELINE (positive means the pair head adds something)")
    for baseline in first.get("versus_baseline", {}):
        add(f"  vs {baseline}")
        for key in ("macro_auprc", "micro_auprc", "macro_mcc"):
            row = []
            for name in names:
                value = reports[name]["versus_baseline"][baseline].get(key)
                row.append(f"{value:+14.6f}" if isinstance(value, float) else f"{'-':>14s}")
            add(f"    {key:22s} " + " ".join(row))
        verdicts = [
            str(reports[name]["versus_baseline"][baseline].get("model_beats_baseline_macro_auprc"))
            for name in names
        ]
        add(f"    {'beats baseline':22s} " + " ".join(f"{v:>14s}" for v in verdicts))
        add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="evaluation output directory, named for the ablation table",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the full report here")
    arguments = parser.parse_args(argv)

    reports: dict[str, dict] = {}
    for entry in arguments.run:
        if "=" not in entry:
            print(f"--run must be NAME=DIR, got {entry!r}", file=sys.stderr)
            return 2
        name, _, directory = entry.partition("=")
        try:
            reports[name] = evaluate_pair_predictions(load_run(Path(directory)))
        except PairEvaluationError as error:
            print(f"{name}: {error}", file=sys.stderr)
            return 2

    print(_table(reports))
    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(reports, indent=2, sort_keys=True, default=float) + "\n",
            encoding="utf-8",
        )
        print(f"full report written to {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
