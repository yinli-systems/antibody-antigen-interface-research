"""Complete pair-level evaluation, reported against the baselines that matter.

A native residue-pair model is only interesting if it beats what you get for
free. Two free things matter here:

``prevalence``
    Ranking at random. Average precision equals the positive rate, so any
    AUPRC near the prevalence is worth nothing whatever its AUROC says.

``predicted-marginal outer product``
    Score a pair by ``P(epitope_i) * P(paratope_j)``, using the model's *own*
    marginal heads. This carries no partner-specific information -- it cannot
    distinguish which antigen residue pairs with which antibody residue, only
    which residues are generally interface-prone -- so a native pair head that
    does not beat it has not learned the thing the project exists to measure.
    The tracked MIPE replay records exactly that inversion: native pair macro
    AUPRC 0.00342 against 0.04045 for the outer product.

Reporting against baselines is not a stylistic preference. Deep sequence-based
interaction models drop to random once train/test sequence similarity is
minimised, while simple baselines keep working, which is why Bernett et al.
(*Briefings in Bioinformatics* 2024) conclude that "any improvements should be
reported relative to baseline methods in the future". This module makes that the
default rather than an extra step.

Metrics are computed per complex and aggregated two ways, because they answer
different questions and disagree under imbalance:

``macro``
    Mean over complexes. Every complex counts once, so large antigens do not
    dominate. This is the headline for a per-complex claim.

``micro``
    Pooled over every candidate pair. Large complexes dominate in proportion to
    their pair count. This is the headline for a per-pair claim.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from .metrics import average_precision, binary_metrics, ranking_at_k, roc_auc

DEFAULT_CUTOFFS = (1, 5, 10, 50, 100)
DEFAULT_FRACTIONS = (0.01, 0.05, 0.10)


class PairEvaluationError(ValueError):
    """Raised when predictions and labels cannot be evaluated together."""


def _numpy():
    try:
        import numpy
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise PairEvaluationError("NumPy is required") from error
    return numpy


@dataclasses.dataclass(frozen=True)
class ComplexPrediction:
    """One complex: pair labels and scores, plus the model's marginal heads.

    ``pair_labels`` and ``pair_scores`` are ``[antigen, antibody]``.
    ``candidate_mask`` selects the pairs that are scored at all; pairs outside it
    are excluded from every metric rather than counted as negatives.
    """

    identifier: str
    pair_labels: Any
    pair_scores: Any
    candidate_mask: Any | None = None
    epitope_probability: Any | None = None
    paratope_probability: Any | None = None

    def selected(self) -> tuple[Any, Any]:
        """Return the flattened labels and scores inside the candidate mask."""
        numpy = _numpy()
        labels = numpy.asarray(self.pair_labels)
        scores = numpy.asarray(self.pair_scores)
        if labels.shape != scores.shape:
            raise PairEvaluationError(
                f"{self.identifier}: labels {labels.shape} and scores {scores.shape} differ"
            )
        if labels.ndim != 2:
            raise PairEvaluationError(f"{self.identifier}: pair arrays must be two-dimensional")
        if self.candidate_mask is None:
            return labels.reshape(-1), scores.reshape(-1)
        mask = numpy.asarray(self.candidate_mask, dtype=bool)
        if mask.shape != labels.shape:
            raise PairEvaluationError(f"{self.identifier}: candidate mask shape differs")
        if not mask.any():
            raise PairEvaluationError(f"{self.identifier}: candidate mask selects no pair")
        return labels[mask], scores[mask]


def enrichment_factor(
    labels: Any,
    scores: Any,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> dict[str, float | None]:
    """Enrichment over random at the top ``x`` fraction of the ranking.

    ``EF@x = hit rate in the top x / prevalence``, so 1.0 is random and the
    ceiling is ``1 / x``. Built on the repository's tie-aware ``ranking_at_k``
    rather than a hard sort, because a contact map produces large tie groups and
    input order must not decide the metric.
    """
    numpy = _numpy()
    labels = numpy.asarray(labels).reshape(-1)
    scores = numpy.asarray(scores).reshape(-1)
    total = int(labels.sum())
    result: dict[str, float | None] = {}
    for fraction in fractions:
        name = f"ef_at_{int(round(fraction * 100))}pct"
        depth = max(1, int(round(labels.size * fraction)))
        if total == 0 or labels.size == 0 or depth > labels.size:
            result[name] = None
            continue
        precision = ranking_at_k(labels, scores, cutoffs=[depth])[f"precision_at_{depth}"]
        prevalence = total / labels.size
        result[name] = float(precision / prevalence) if prevalence else None
    return result


def score_separation_stats(labels: Any, scores: Any) -> dict[str, float | None]:
    """Positive/negative score separation, including Cohen's d."""
    numpy = _numpy()
    labels = numpy.asarray(labels).reshape(-1)
    scores = numpy.asarray(scores).reshape(-1)
    positive = scores[labels.astype(bool)]
    negative = scores[~labels.astype(bool)]
    empty = {
        "positive_score_mean": None,
        "negative_score_mean": None,
        "score_gap": None,
        "cohens_d": None,
    }
    if positive.size == 0 or negative.size == 0:
        return empty
    positive_mean = float(positive.mean())
    negative_mean = float(negative.mean())
    gap = positive_mean - negative_mean
    if positive.size < 2 or negative.size < 2:
        return {
            "positive_score_mean": positive_mean,
            "negative_score_mean": negative_mean,
            "score_gap": gap,
            "cohens_d": None,
        }
    positive_variance = float(positive.var(ddof=1))
    negative_variance = float(negative.var(ddof=1))
    pooled = (
        (positive.size - 1) * positive_variance + (negative.size - 1) * negative_variance
    ) / (positive.size + negative.size - 2)
    return {
        "positive_score_mean": positive_mean,
        "negative_score_mean": negative_mean,
        "score_gap": gap,
        "cohens_d": (gap / pooled**0.5) if pooled > 0.0 else None,
    }


def outer_product_scores(epitope_probability: Any, paratope_probability: Any) -> Any:
    """Score every pair by the product of the two marginal probabilities.

    The partner-agnostic baseline: it depends only on how interface-prone each
    residue is on its own side.
    """
    numpy = _numpy()
    epitope = numpy.asarray(epitope_probability, dtype=float).reshape(-1)
    paratope = numpy.asarray(paratope_probability, dtype=float).reshape(-1)
    if epitope.size == 0 or paratope.size == 0:
        raise PairEvaluationError("marginal probabilities must be non-empty")
    return numpy.outer(epitope, paratope)


def evaluate_single(
    labels: Any,
    scores: Any,
    *,
    threshold: float = 0.5,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> dict[str, Any]:
    """Compute every supported metric for one flattened label/score pair."""
    numpy = _numpy()
    labels = numpy.asarray(labels).reshape(-1)
    scores = numpy.asarray(scores).reshape(-1)
    positives = int(labels.sum())
    record: dict[str, Any] = {
        "candidate_count": int(labels.size),
        "positive_count": positives,
        "prevalence": (positives / labels.size) if labels.size else None,
        "auroc": roc_auc(labels, scores),
        "auprc": average_precision(labels, scores),
    }
    record.update(binary_metrics(labels, scores, threshold=threshold))
    record.update(score_separation_stats(labels, scores))
    record.update(enrichment_factor(labels, scores, fractions=fractions))
    usable = [cutoff for cutoff in cutoffs if cutoff <= labels.size]
    if usable and positives:
        ranking = ranking_at_k(labels, scores, cutoffs=usable)
        for cutoff in usable:
            record[f"precision_at_{cutoff}"] = ranking[f"precision_at_{cutoff}"]
            record[f"recall_at_{cutoff}"] = ranking[f"recall_at_{cutoff}"]
    if positives:
        # Precision at the true positive count: the natural operating point when
        # the number of contacts is known, and comparable across complex sizes.
        depth = min(positives, int(labels.size))
        record["precision_at_true_count"] = ranking_at_k(labels, scores, cutoffs=[depth])[
            f"precision_at_{depth}"
        ]
    return record


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean each numeric key over complexes that reported it."""
    keys: list[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    summary: dict[str, Any] = {}
    for key in keys:
        values = [
            float(record[key])
            for record in records
            if isinstance(record.get(key), (int, float)) and not isinstance(record.get(key), bool)
        ]
        summary[key] = (sum(values) / len(values)) if values else None
        summary[f"{key}_reported_by"] = len(values)
    return summary


def evaluate_pair_predictions(
    predictions: Sequence[ComplexPrediction],
    *,
    threshold: float = 0.5,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> dict[str, Any]:
    """Evaluate a model against every available baseline, macro and micro.

    Returns per-complex records, macro and micro aggregates for the model and
    for each baseline that could be computed, and the model-minus-baseline
    deltas on the headline metrics.
    """
    numpy = _numpy()
    if not predictions:
        raise PairEvaluationError("at least one complex is required")

    per_complex: list[dict[str, Any]] = []
    baseline_records: dict[str, list[dict[str, Any]]] = {"outer_product": [], "prevalence": []}
    pooled_labels, pooled_scores = [], []
    pooled_baselines: dict[str, list[Any]] = {"outer_product": [], "prevalence": []}

    for prediction in predictions:
        labels, scores = prediction.selected()
        record = evaluate_single(
            labels, scores, threshold=threshold, cutoffs=cutoffs, fractions=fractions
        )
        record["identifier"] = prediction.identifier
        per_complex.append(record)
        pooled_labels.append(labels)
        pooled_scores.append(scores)

        # Prevalence baseline: a constant score, so average precision equals the
        # positive rate exactly and AUROC is undefined.
        constant = numpy.zeros_like(labels, dtype=float)
        baseline_records["prevalence"].append(
            evaluate_single(labels, constant, threshold=threshold, cutoffs=cutoffs,
                            fractions=fractions)
        )
        pooled_baselines["prevalence"].append(constant)

        if (
            prediction.epitope_probability is not None
            and prediction.paratope_probability is not None
        ):
            grid = outer_product_scores(
                prediction.epitope_probability, prediction.paratope_probability
            )
            if grid.shape != numpy.asarray(prediction.pair_labels).shape:
                raise PairEvaluationError(
                    f"{prediction.identifier}: outer product {grid.shape} does not match labels"
                )
            outer = (
                grid.reshape(-1)
                if prediction.candidate_mask is None
                else grid[numpy.asarray(prediction.candidate_mask, dtype=bool)]
            )
            baseline_records["outer_product"].append(
                evaluate_single(labels, outer, threshold=threshold, cutoffs=cutoffs,
                                fractions=fractions)
            )
            pooled_baselines["outer_product"].append(outer)

    report: dict[str, Any] = {
        "complex_count": len(predictions),
        "per_complex": per_complex,
        "macro": _aggregate(per_complex),
        "micro": evaluate_single(
            numpy.concatenate(pooled_labels),
            numpy.concatenate(pooled_scores),
            threshold=threshold,
            cutoffs=cutoffs,
            fractions=fractions,
        ),
        "baselines": {},
        "versus_baseline": {},
    }

    for name, records in baseline_records.items():
        if not records:
            continue
        entry: dict[str, Any] = {
            "complex_count": len(records),
            "macro": _aggregate(records),
        }
        if len(records) == len(predictions):
            entry["micro"] = evaluate_single(
                numpy.concatenate(pooled_labels),
                numpy.concatenate(pooled_baselines[name]),
                threshold=threshold,
                cutoffs=cutoffs,
                fractions=fractions,
            )
        report["baselines"][name] = entry

        deltas: dict[str, Any] = {}
        for scope in ("macro", "micro"):
            if scope not in entry:
                continue
            for metric in ("auprc", "auroc", "mcc", "f1", "precision_at_true_count"):
                model_value = report[scope].get(metric)
                base_value = entry[scope].get(metric)
                if isinstance(model_value, float) and isinstance(base_value, float):
                    deltas[f"{scope}_{metric}"] = model_value - base_value
        # The question the project exists to answer, stated as a boolean.
        macro_gap = deltas.get("macro_auprc")
        deltas["model_beats_baseline_macro_auprc"] = (
            bool(macro_gap > 0.0) if isinstance(macro_gap, float) else None
        )
        report["versus_baseline"][name] = deltas

    return report
