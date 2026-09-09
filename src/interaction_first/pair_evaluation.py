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
    record.update(balanced_accuracy_metrics(labels, scores, threshold=threshold))
    record.update(best_threshold_metrics(labels, scores))
    record.update(score_separation_stats(labels, scores))
    record.update(enrichment_factor(labels, scores, fractions=fractions))
    # Early recognition: with millions of candidates and a few hundred contacts,
    # only the head of the ranking is ever inspected.
    record["bedroc_alpha20"] = bedroc_score(labels, scores, alpha=20.0)
    record["bedroc_alpha80"] = bedroc_score(labels, scores, alpha=80.0)
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


def balanced_accuracy_metrics(
    labels: Any, scores: Any, threshold: float = 0.5
) -> dict[str, float | None]:
    """Sensitivity, specificity and their mean at one threshold.

    The epitope literature reports balanced accuracy alongside MCC because plain
    accuracy is uninformative when almost every pair is negative.
    """
    numpy = _numpy()
    labels = numpy.asarray(labels).reshape(-1).astype(bool)
    predicted = numpy.asarray(scores).reshape(-1) >= threshold
    positives, negatives = int(labels.sum()), int((~labels).sum())
    sensitivity = float((predicted & labels).sum() / positives) if positives else None
    specificity = float((~predicted & ~labels).sum() / negatives) if negatives else None
    balanced = (
        (sensitivity + specificity) / 2.0
        if sensitivity is not None and specificity is not None
        else None
    )
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced,
    }


def best_threshold_metrics(labels: Any, scores: Any) -> dict[str, float | None]:
    """Best attainable F1 and MCC over all thresholds, and where they occur.

    Reporting F1 and MCC only at a fixed 0.5 conflates ranking quality with
    calibration: at contact-map prevalence almost nothing exceeds 0.5, so a
    model that ranks perfectly can still score zero. The gap between the fixed
    and the best value is itself a calibration signal.

    Computed in one sorted pass rather than by re-scoring at sampled
    thresholds. Every distinct score is considered, so the optimum is exact
    rather than grid-limited, and no metric is recomputed per candidate -- the
    sampled version re-derived AUROC and AUPRC at every step, which made it
    unusable on a full contact map.
    """
    numpy = _numpy()
    labels = numpy.asarray(labels).reshape(-1).astype(numpy.float64)
    scores = numpy.asarray(scores).reshape(-1).astype(numpy.float64)
    total = labels.size
    positives = float(labels.sum())
    if total == 0 or positives == 0 or positives == total:
        return {"best_f1": None, "best_f1_threshold": None,
                "best_mcc": None, "best_mcc_threshold": None}
    negatives = total - positives

    order = numpy.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    true_positive = numpy.cumsum(labels[order])
    predicted = numpy.arange(1, total + 1, dtype=numpy.float64)
    false_positive = predicted - true_positive
    false_negative = positives - true_positive
    true_negative = negatives - false_positive

    # Only the last index of each tie group is a realisable cut point.
    keep = numpy.ones(total, dtype=bool)
    keep[:-1] = ordered_scores[:-1] != ordered_scores[1:]

    with numpy.errstate(divide="ignore", invalid="ignore"):
        f1 = numpy.where(
            (2.0 * true_positive + false_positive + false_negative) > 0,
            2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative),
            0.0,
        )
        denominator = numpy.sqrt(
            (true_positive + false_positive)
            * (true_positive + false_negative)
            * (true_negative + false_positive)
            * (true_negative + false_negative)
        )
        mcc = numpy.where(
            denominator > 0,
            (true_positive * true_negative - false_positive * false_negative) / denominator,
            0.0,
        )

    f1 = numpy.where(keep, f1, -numpy.inf)
    mcc = numpy.where(keep, mcc, -numpy.inf)
    f1_at = int(numpy.argmax(f1))
    mcc_at = int(numpy.argmax(mcc))
    return {
        "best_f1": float(f1[f1_at]),
        "best_f1_threshold": float(ordered_scores[f1_at]),
        "best_mcc": float(mcc[mcc_at]),
        "best_mcc_threshold": float(ordered_scores[mcc_at]),
    }


def bedroc_score(labels: Any, scores: Any, alpha: float = 20.0) -> float | None:
    """Boltzmann-enhanced discrimination of ROC (Truchon & Bayly, 2007).

    An early-recognition metric: it weights a hit by ``exp(-alpha * rank / N)``,
    so what happens at the top of the ranking dominates. With 18 million
    candidate pairs per complex and a few hundred contacts, AUPRC is still
    largely a statement about the bulk of the list, while only the head is ever
    looked at. ``alpha = 20`` concentrates roughly 80% of the weight in the top
    8% of the ranking.

    Returns a value in ``[0, 1]``: 1 for every active ranked first, ~``Ra`` for
    a random ordering.
    """
    numpy = _numpy()
    labels = numpy.asarray(labels).reshape(-1)
    scores = numpy.asarray(scores).reshape(-1)
    total = int(labels.size)
    actives = int(labels.sum())
    if total == 0 or actives == 0 or actives == total or alpha <= 0.0:
        return None
    ratio = actives / total
    order = numpy.argsort(-scores, kind="stable")
    ranks = numpy.nonzero(labels[order] > 0)[0] + 1  # 1-based ranks of actives

    numerator = float(numpy.exp(-alpha * ranks / total).sum()) / actives
    denominator = (1.0 - float(numpy.exp(-alpha))) / (total * (float(numpy.exp(alpha / total)) - 1.0))
    rie = numerator / denominator

    import math

    scale = ratio * math.sinh(alpha / 2.0) / (
        math.cosh(alpha / 2.0) - math.cosh(alpha / 2.0 - alpha * ratio)
    )
    offset = 1.0 / (1.0 - math.exp(alpha * (1.0 - ratio)))
    value = rie * scale + offset
    # A perfect ranking evaluates to 1 + ~1e-14 in floating point. Clamp the
    # round-off, but refuse to hide a real excursion, which would mean the
    # formula or its inputs are wrong rather than imprecise.
    if not -1e-6 <= value <= 1.0 + 1e-6:
        raise PairEvaluationError(f"BEDROC evaluated to {value!r}, outside [0, 1]")
    return float(min(1.0, max(0.0, value)))


def discrimination_metrics(cognate_scores: Any, decoy_scores: Any) -> dict[str, float | None]:
    """Score cognate pairs against decoys as a binder/non-binder classifier.

    The specificity literature evaluates a scorer by how well it separates
    cognate complexes from shuffled non-cognate ones, reporting ROC-AUC and
    average precision over that pooled set. The repository measures partner
    retrieval by rank instead, which answers a related but different question:
    a scorer can rank the true partner first within each small candidate set and
    still assign overlapping absolute scores across complexes, which is what
    fails when the candidate set is the whole repertoire.
    """
    numpy = _numpy()
    cognate = numpy.asarray(cognate_scores, dtype=float).reshape(-1)
    decoy = numpy.asarray(decoy_scores, dtype=float).reshape(-1)
    if cognate.size == 0 or decoy.size == 0:
        raise PairEvaluationError("both cognate and decoy scores are required")
    labels = numpy.concatenate([numpy.ones(cognate.size), numpy.zeros(decoy.size)])
    pooled = numpy.concatenate([cognate, decoy])
    record = {
        "cognate_count": int(cognate.size),
        "decoy_count": int(decoy.size),
        "discrimination_auroc": roc_auc(labels, pooled),
        "discrimination_auprc": average_precision(labels, pooled),
        "mean_cognate_score": float(cognate.mean()),
        "mean_decoy_score": float(decoy.mean()),
    }
    record.update(score_separation_stats(labels, pooled))
    return record
