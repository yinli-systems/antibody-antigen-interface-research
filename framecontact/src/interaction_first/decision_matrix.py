"""Staged, single-variable decision engine for the FrameContact ablation.

The architecture questions this project has to answer are causal, not
descriptive: how should residue predictions be produced, what should the pair
head predict, and does the triangle constraint earn its place.  Answering them
requires comparisons in which exactly one registered factor changes.  The most
common way to get this wrong is to compare two runs that differ in two factors
at once -- for example a unary-head binary-contact run against a hybrid-head
distogram run -- and attribute the difference to the pair target.

This module makes that mistake impossible to make silently:

* every experiment arm is declared as a point in a factor space;
* a comparison is refused unless exactly one factor differs;
* fixed-threshold precision gains are separated from genuine ranking gains,
  because a model can always buy precision by predicting fewer positives;
* paired-seed deltas are reported with an exact sign test rather than a mean.

Metric paths were read from the live training records, not assumed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FACTORS = (
    "residue_aggregation",
    "pair_target",
    "triangle",
    "residue_contrastive",
    "pair_gauge",
    "fusion",
    "geometry",
)


@dataclass(frozen=True)
class ArmFactors:
    """One point in the ablation factor space."""

    residue_aggregation: str
    pair_target: str
    triangle: str = "off"
    residue_contrastive: str = "off"
    pair_gauge: str = "raw"
    fusion: str = "late"
    geometry: str = "full_pose"

    def as_mapping(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in FACTORS}


ARM_FACTORS: Mapping[str, ArmFactors] = {
    "framecontact_unary_only": ArmFactors("unary", "binary_contact"),
    "framecontact_contact_marginal": ArmFactors("contact_marginal", "binary_contact"),
    "framecontact_contact_hybrid": ArmFactors("gated_hybrid", "binary_contact"),
    "framecontact_contact_hybrid_gauge_audit": ArmFactors(
        "gated_hybrid", "binary_contact", pair_gauge="doubly_centered"
    ),
    "framecontact_gauge_marginal": ArmFactors(
        "gauge_marginal", "binary_contact", pair_gauge="doubly_centered"
    ),
    "framecontact_gauge_hybrid": ArmFactors(
        "gauge_hybrid", "binary_contact", pair_gauge="doubly_centered"
    ),
    "framecontact_gauge_hybrid_contrastive": ArmFactors(
        "gauge_hybrid",
        "binary_contact",
        residue_contrastive="on",
        pair_gauge="doubly_centered",
    ),
    "framecontact_distogram": ArmFactors("gated_hybrid", "distogram"),
    "framecontact_distogram_triangle": ArmFactors(
        "gated_hybrid", "distogram", triangle="on"
    ),
    "framecontact_gauge_hybrid_distogram": ArmFactors(
        "gauge_hybrid", "distogram", pair_gauge="doubly_centered"
    ),
    "framecontact_gauge_hybrid_distogram_triangle": ArmFactors(
        "gauge_hybrid", "distogram", triangle="on", pair_gauge="doubly_centered"
    ),
    "framecontact_geometry_ablated_contact_hybrid": ArmFactors(
        "gated_hybrid", "binary_contact", geometry="ablated"
    ),
    "framecontact_geometry_ablated_distogram": ArmFactors(
        "gated_hybrid", "distogram", geometry="ablated"
    ),
    "framecontact_interleaved": ArmFactors(
        "gated_hybrid", "binary_contact", fusion="interleaved"
    ),
}


# Routing the residue head through the partner-specific channel requires the
# centered decomposition, so for those arms the gauge is entailed by the
# aggregation rather than chosen independently. It stays a free factor for
# arms whose residue head marginalizes the raw contact map, where enabling the
# gauge is provably inert and therefore a genuine null control.
GAUGE_ROUTED_AGGREGATIONS = frozenset({"gauge_marginal", "gauge_hybrid"})


class ComparisonError(ValueError):
    """Raised when a requested comparison cannot answer a causal question."""


def free_factors(arm: str) -> dict[str, str]:
    """Factors that are independently choosable for this arm."""
    if arm not in ARM_FACTORS:
        raise ComparisonError(f"experiment {arm!r} is not a registered arm")
    return ARM_FACTORS[arm].as_mapping()


def factor_difference(left: str, right: str) -> tuple[str, ...]:
    """Names of the factors that differ between two registered arms."""
    left_factors = free_factors(left)
    right_factors = free_factors(right)
    entailed_gauge = any(
        factors["residue_aggregation"] in GAUGE_ROUTED_AGGREGATIONS
        for factors in (left_factors, right_factors)
    )
    return tuple(
        name
        for name in FACTORS
        if left_factors[name] != right_factors[name]
        and not (name == "pair_gauge" and entailed_gauge)
    )


def require_single_variable(left: str, right: str) -> str:
    """Return the single differing factor, or refuse the comparison."""
    difference = factor_difference(left, right)
    if len(difference) == 1:
        return difference[0]
    if not difference:
        raise ComparisonError(
            f"{left!r} and {right!r} occupy the same point in the factor space"
        )
    raise ComparisonError(
        f"{left!r} and {right!r} differ in {len(difference)} factors "
        f"({', '.join(difference)}); the difference cannot be attributed to any "
        "one of them. Compare arms that differ in exactly one factor."
    )


_METRIC_PATHS: Mapping[str, tuple[str, ...]] = {
    "partner_margin_worst_direction": ("validation", "selection_metric"),
    "contact_pooled_auprc": ("validation", "contact", "auprc"),
    "contact_mean_complex_auprc": ("validation", "contact", "mean_per_complex_auprc"),
    "contact_auroc": ("validation", "contact", "auroc"),
    "contact_brier": ("validation", "contact", "brier"),
    "contact_ece_15": ("validation", "contact", "ece_15"),
    "contact_precision": (
        "validation",
        "contact",
        "fixed_raw_logit_threshold_zero",
        "precision",
    ),
    "contact_recall": (
        "validation",
        "contact",
        "fixed_raw_logit_threshold_zero",
        "recall",
    ),
    "contact_f1": ("validation", "contact", "fixed_raw_logit_threshold_zero", "f1"),
    "contact_predicted_positive_ratio": (
        "validation",
        "contact",
        "fixed_raw_logit_threshold_zero",
        "predicted_positive_ratio",
    ),
    "contact_p_at_10": (
        "validation",
        "contact",
        "ranked_pair_macro",
        "10",
        "mean_precision",
    ),
    "contact_p_at_lmin_over_10": (
        "validation",
        "contact",
        "ranked_pair_macro",
        "lmin_over_10",
        "mean_precision",
    ),
    "contact_recall_at_lmin_over_10": (
        "validation",
        "contact",
        "ranked_pair_macro",
        "lmin_over_10",
        "mean_recall",
    ),
    "contact_p_at_lmin": (
        "validation",
        "contact",
        "ranked_pair_macro",
        "lmin",
        "mean_precision",
    ),
    "contact_recall_at_lmin": (
        "validation",
        "contact",
        "ranked_pair_macro",
        "lmin",
        "mean_recall",
    ),
    "partner_specific_variance_share": (
        "validation",
        "contact_gauge_attribution",
        "mean_partner_specific_share",
    ),
    "epitope_gate_applied": ("residue_contact_gate", "epitope_applied"),
    "paratope_gate_applied": ("residue_contact_gate", "paratope_applied"),
}

for _residue in ("epitope", "paratope"):
    _METRIC_PATHS[f"{_residue}_pooled_auprc"] = ("validation", _residue, "auprc")
    _METRIC_PATHS[f"{_residue}_mean_complex_auprc"] = (
        "validation",
        _residue,
        "mean_per_complex_auprc",
    )
    _METRIC_PATHS[f"{_residue}_brier"] = ("validation", _residue, "brier")
    _METRIC_PATHS[f"{_residue}_ece_15"] = ("validation", _residue, "ece_15")
    for _operating in ("precision", "recall", "f1", "predicted_positive_ratio"):
        _METRIC_PATHS[f"{_residue}_{_operating}"] = (
            "validation",
            _residue,
            "fixed_raw_logit_threshold_zero",
            _operating,
        )

METRIC_PATHS: Mapping[str, tuple[str, ...]] = dict(_METRIC_PATHS)

# Metrics where a smaller value is the better outcome.
LOWER_IS_BETTER = frozenset(
    {
        "contact_brier",
        "contact_ece_15",
        "epitope_brier",
        "epitope_ece_15",
        "paratope_brier",
        "paratope_ece_15",
    }
)

DIRECTION_KEYS = ("fixed_antibody_wrong_antigen", "fixed_antigen_wrong_antibody")


def _dig(record: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = record
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def extract_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    """Pull the decision metrics out of one epoch record."""
    metrics: dict[str, float] = {}
    for name, path in METRIC_PATHS.items():
        value = _dig(record, path)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                metrics[name] = numeric
    directions = _dig(record, ("validation", "partner_retrieval_by_direction"))
    if isinstance(directions, Mapping):
        for direction in DIRECTION_KEYS:
            entry = directions.get(direction)
            if isinstance(entry, Mapping):
                value = entry.get("cluster_weighted_cognate_compatibility_margin")
                if isinstance(value, (int, float)):
                    metrics[f"partner_margin_{direction}"] = float(value)
                accuracy = entry.get("top1_retrieval_accuracy")
                if isinstance(accuracy, (int, float)):
                    metrics[f"top1_retrieval_{direction}"] = float(accuracy)
    return metrics


@dataclass(frozen=True)
class SeedRun:
    """All epochs of one (experiment, seed) run."""

    experiment: str
    seed: int
    epochs: tuple[int, ...]
    metrics_by_epoch: Mapping[int, Mapping[str, float]]
    source: str
    completed: bool = False
    selected_epoch: int | None = None

    def at(self, epoch: int) -> Mapping[str, float] | None:
        return self.metrics_by_epoch.get(epoch)

    @property
    def last_epoch(self) -> int:
        return self.epochs[-1]


def parse_epoch_records(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Recover the per-epoch JSON records a training run prints to stdout."""
    records: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping) and "epoch" in payload and "validation" in payload:
            records.append(dict(payload))
    return records


def _seed_run_from_records(
    experiment: str,
    seed: int,
    records: Sequence[Mapping[str, Any]],
    *,
    source: str,
    completed: bool = False,
    selected_epoch: int | None = None,
) -> SeedRun:
    metrics_by_epoch: dict[int, Mapping[str, float]] = {}
    for record in records:
        epoch = int(record["epoch"])
        metrics_by_epoch[epoch] = extract_metrics(record)
    if not metrics_by_epoch:
        raise ComparisonError(f"no epoch records found for {experiment} seed {seed}")
    return SeedRun(
        experiment=experiment,
        seed=seed,
        epochs=tuple(sorted(metrics_by_epoch)),
        metrics_by_epoch=metrics_by_epoch,
        source=source,
        completed=completed,
        selected_epoch=selected_epoch,
    )


def load_live_log(path: Path, *, experiment: str, seed: int) -> SeedRun:
    """Read an in-flight run from its Slurm stdout log."""
    text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    return _seed_run_from_records(
        experiment, seed, parse_epoch_records(text), source=str(path)
    )


def load_completed_run(path: Path) -> SeedRun:
    """Read a finished run from its development result JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ComparisonError(f"{path} has no epoch history")
    selection = payload.get("selection", {})
    return _seed_run_from_records(
        str(payload["experiment"]),
        int(payload["configuration"]["seed"]),
        history,
        source=str(path),
        completed=True,
        selected_epoch=(
            int(selection["best_epoch"]) if "best_epoch" in selection else None
        ),
    )


def matched_epoch(runs: Sequence[SeedRun]) -> int:
    """Latest epoch every supplied run has reached.

    Comparing arms at different epochs confounds the factor under test with
    training progress, so the reporting layer always pins a common epoch.
    """
    if not runs:
        raise ComparisonError("no runs supplied")
    common = set(runs[0].epochs)
    for run in runs[1:]:
        common &= set(run.epochs)
    if not common:
        raise ComparisonError("supplied runs share no common epoch")
    return max(common)


def _binomial_tail(successes: int, trials: int) -> float:
    total = 0.0
    for count in range(trials + 1):
        if count <= successes:
            total += math.comb(trials, count) * 0.5**trials
    return total


def exact_sign_test(deltas: Sequence[float], *, tolerance: float = 0.0) -> dict[str, Any]:
    """Two-sided exact sign test over paired seed deltas.

    With five seeds no test is powerful, so the count of seeds that moved in
    each direction is reported alongside the p-value and should be read first.
    """
    positive = sum(1 for value in deltas if value > tolerance)
    negative = sum(1 for value in deltas if value < -tolerance)
    trials = positive + negative
    if trials == 0:
        return {
            "positive": positive,
            "negative": negative,
            "ties": len(deltas),
            "p_value": 1.0,
        }
    smaller = min(positive, negative)
    p_value = min(1.0, 2.0 * _binomial_tail(smaller, trials))
    return {
        "positive": positive,
        "negative": negative,
        "ties": len(deltas) - trials,
        "p_value": p_value,
    }


def bootstrap_mean(
    values: Sequence[float], *, replicates: int = 10000, seed: int = 20260802
) -> dict[str, float]:
    """Percentile bootstrap over seeds for a paired mean delta."""
    import numpy as np

    if not values:
        raise ComparisonError("bootstrap requires at least one value")
    generator = np.random.default_rng(seed)
    sample = np.asarray(values, dtype=float)
    draws = generator.choice(sample, size=(replicates, sample.size), replace=True)
    means = draws.mean(axis=1)
    return {
        "mean": float(sample.mean()),
        "lower_95": float(np.percentile(means, 2.5)),
        "upper_95": float(np.percentile(means, 97.5)),
    }


# Number of trailing epochs used to estimate how much a metric moves on its own.
EPOCH_NOISE_WINDOW = 5


def epoch_noise(
    runs: Mapping[int, SeedRun],
    metric: str,
    *,
    epoch: int,
    window: int = EPOCH_NOISE_WINDOW,
) -> float | None:
    """Typical epoch-to-epoch movement of a metric within a single run.

    Model selection on a metric is only meaningful if the differences between
    arms are larger than the distance the metric travels between neighbouring
    epochs of the same run.
    """

    if window < 2:
        raise ComparisonError("epoch noise needs a window of at least two epochs")
    deviations: list[float] = []
    for run in runs.values():
        recent = [value for value in run.epochs if epoch - window < value <= epoch]
        series = [
            run.metrics_by_epoch[value][metric]
            for value in recent
            if metric in run.metrics_by_epoch[value]
        ]
        if len(series) < 2:
            continue
        mean_value = sum(series) / len(series)
        variance = sum((value - mean_value) ** 2 for value in series) / (
            len(series) - 1
        )
        deviations.append(math.sqrt(variance))
    if not deviations:
        return None
    return sum(deviations) / len(deviations)


def delta_against_epoch_noise(
    reference_runs: Mapping[int, SeedRun],
    candidate_runs: Mapping[int, SeedRun],
    metric: str,
    *,
    epoch: int,
    mean_delta: float,
    window: int = EPOCH_NOISE_WINDOW,
) -> dict[str, Any]:
    """Is the between-arm difference bigger than within-arm epoch wobble?"""
    reference_noise = epoch_noise(reference_runs, metric, epoch=epoch, window=window)
    candidate_noise = epoch_noise(candidate_runs, metric, epoch=epoch, window=window)
    available = [value for value in (reference_noise, candidate_noise) if value is not None]
    if not available:
        return {
            "metric": metric,
            "window": window,
            "noise": None,
            "exceeds_noise": None,
        }
    noise = max(available)
    return {
        "metric": metric,
        "window": window,
        "reference_noise": reference_noise,
        "candidate_noise": candidate_noise,
        "noise": noise,
        "mean_delta": mean_delta,
        "ratio": (abs(mean_delta) / noise) if noise > 0 else None,
        "exceeds_noise": abs(mean_delta) > noise,
    }


@dataclass(frozen=True)
class PairedComparison:
    factor: str
    reference: str
    candidate: str
    epoch: int
    seeds: tuple[int, ...]
    deltas_by_metric: Mapping[str, Mapping[str, Any]]
    sparsity_verdict: Mapping[str, Any]


def compare_arms(
    reference_runs: Mapping[int, SeedRun],
    candidate_runs: Mapping[int, SeedRun],
    *,
    epoch: int | None = None,
    metrics: Sequence[str] | None = None,
) -> PairedComparison:
    """Paired per-seed comparison of two arms that differ in one factor."""
    reference_names = {run.experiment for run in reference_runs.values()}
    candidate_names = {run.experiment for run in candidate_runs.values()}
    if len(reference_names) != 1 or len(candidate_names) != 1:
        raise ComparisonError("each side of a comparison must be a single arm")
    reference = reference_names.pop()
    candidate = candidate_names.pop()
    factor = require_single_variable(reference, candidate)

    seeds = tuple(sorted(set(reference_runs) & set(candidate_runs)))
    if not seeds:
        raise ComparisonError(f"{reference} and {candidate} share no seed")
    all_runs = [reference_runs[seed] for seed in seeds]
    all_runs += [candidate_runs[seed] for seed in seeds]
    resolved_epoch = matched_epoch(all_runs) if epoch is None else int(epoch)

    metric_names = tuple(metrics) if metrics else tuple(METRIC_PATHS)
    deltas_by_metric: dict[str, Mapping[str, Any]] = {}
    for name in metric_names:
        per_seed: dict[int, float] = {}
        for seed in seeds:
            left = reference_runs[seed].at(resolved_epoch)
            right = candidate_runs[seed].at(resolved_epoch)
            if left is None or right is None:
                continue
            if name not in left or name not in right:
                continue
            per_seed[seed] = right[name] - left[name]
        if not per_seed:
            continue
        values = list(per_seed.values())
        entry: dict[str, Any] = {
            "per_seed": per_seed,
            "mean_delta": sum(values) / len(values),
            "sign_test": exact_sign_test(values),
            "lower_is_better": name in LOWER_IS_BETTER,
        }
        try:
            entry["bootstrap"] = bootstrap_mean(values)
        except ImportError:  # pragma: no cover - numpy is a hard dependency here
            entry["bootstrap"] = None
        deltas_by_metric[name] = entry

    return PairedComparison(
        factor=factor,
        reference=reference,
        candidate=candidate,
        epoch=resolved_epoch,
        seeds=seeds,
        deltas_by_metric=deltas_by_metric,
        sparsity_verdict=classify_precision_change(deltas_by_metric),
    )


def classify_precision_change(
    deltas_by_metric: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Separate a real ranking improvement from a shift in prediction sparsity.

    Contact labels are extremely imbalanced, so a model can raise
    fixed-threshold precision simply by predicting fewer positives, at the cost
    of recall and with no change in how well it ranks pairs.  A precision gain
    only counts when a threshold-free metric (AUPRC) or a budget-matched metric
    (precision at a fixed K) also improves.
    """

    def mean(name: str) -> float | None:
        entry = deltas_by_metric.get(name)
        return None if entry is None else float(entry["mean_delta"])

    precision = mean("contact_precision")
    recall = mean("contact_recall")
    predicted_positive_ratio = mean("contact_predicted_positive_ratio")
    auprc = mean("contact_pooled_auprc")
    budget_matched = mean("contact_p_at_lmin_over_10")

    ranking_improved = any(
        value is not None and value > tolerance for value in (auprc, budget_matched)
    )
    ranking_degraded = any(
        value is not None and value < -tolerance for value in (auprc, budget_matched)
    )
    sparser = (
        predicted_positive_ratio is not None
        and predicted_positive_ratio < -tolerance
    )
    precision_up = precision is not None and precision > tolerance
    recall_down = recall is not None and recall < -tolerance

    if ranking_improved and not ranking_degraded:
        verdict = "ranking_gain"
        explanation = (
            "A threshold-free or budget-matched contact metric improved, so the "
            "model ranks pairs better rather than only predicting fewer of them."
        )
    elif ranking_degraded:
        verdict = "ranking_regression"
        explanation = (
            "Threshold-free or budget-matched contact ranking got worse, whatever "
            "the fixed-threshold precision did."
        )
    elif precision_up and sparser:
        verdict = "sparsity_drift"
        explanation = (
            "Fixed-threshold precision rose while the predicted-positive ratio "
            "fell and no threshold-free or budget-matched metric improved. This "
            "is a change in operating point, not in contact ranking, and must "
            "not be reported as a better model."
        )
    else:
        verdict = "inconclusive"
        explanation = (
            "No contact metric moved beyond tolerance in either direction."
        )

    return {
        "verdict": verdict,
        "explanation": explanation,
        "mean_contact_precision_delta": precision,
        "mean_contact_recall_delta": recall,
        "mean_predicted_positive_ratio_delta": predicted_positive_ratio,
        "mean_contact_pooled_auprc_delta": auprc,
        "mean_budget_matched_precision_delta": budget_matched,
        "tolerance": tolerance,
    }


@dataclass(frozen=True)
class DecisionPhase:
    key: str
    question: str
    factor: str
    baseline: str
    candidates: tuple[str, ...]
    primary_metric: str
    guard_metrics: tuple[str, ...]
    caveats: tuple[str, ...] = ()


DECISION_CHAIN: tuple[DecisionPhase, ...] = (
    DecisionPhase(
        key="phase_1_residue_aggregation",
        question=(
            "How should epitope and paratope be produced: from the single-chain "
            "head alone, from the contact map alone, or from a gated combination "
            "-- and does routing through the identifiable partner-specific "
            "channel change the answer?"
        ),
        factor="residue_aggregation",
        baseline="framecontact_unary_only",
        candidates=(
            "framecontact_contact_marginal",
            "framecontact_contact_hybrid",
            "framecontact_gauge_marginal",
            "framecontact_gauge_hybrid",
        ),
        primary_metric="partner_margin_worst_direction",
        guard_metrics=(
            "epitope_pooled_auprc",
            "paratope_pooled_auprc",
            "contact_pooled_auprc",
        ),
    ),
    DecisionPhase(
        key="phase_1b_residue_partner_supervision",
        question=(
            "Does supervising partner specificity at labelled interface residues, "
            "rather than through one score per complex, produce partner-dependent "
            "residue predictions?"
        ),
        factor="residue_contrastive",
        baseline="framecontact_gauge_hybrid",
        candidates=("framecontact_gauge_hybrid_contrastive",),
        primary_metric="partner_margin_worst_direction",
        guard_metrics=("epitope_pooled_auprc", "paratope_pooled_auprc"),
    ),
    DecisionPhase(
        key="phase_2_pair_target",
        question=(
            "Holding the residue head fixed, is a single distogram a better pair "
            "target than a binary contact head?"
        ),
        factor="pair_target",
        baseline="framecontact_contact_hybrid",
        candidates=("framecontact_distogram",),
        primary_metric="contact_pooled_auprc",
        guard_metrics=(
            "contact_p_at_lmin_over_10",
            "partner_margin_worst_direction",
            "contact_recall",
        ),
        caveats=(
            "Under specificity_score_source='interaction_only' the partner margin "
            "is not comparable across pair heads. The binary head's interaction "
            "term is an additive component of the contact logits, while the "
            "distogram head's is a nonlinear logsumexp collapse of per-bin "
            "interaction terms and is not a component of anything. Read the "
            "partner margin for this phase only from arms scored on the "
            "gauge-fixed channel, which is defined identically for both heads.",
        ),
    ),
    DecisionPhase(
        key="phase_3_triangle",
        question=(
            "Holding the pair target fixed, does the localized triangle "
            "constraint raise precision at a fixed budget while retaining recall?"
        ),
        factor="triangle",
        baseline="framecontact_distogram",
        candidates=("framecontact_distogram_triangle",),
        primary_metric="contact_p_at_lmin_over_10",
        guard_metrics=("contact_recall", "partner_margin_worst_direction"),
    ),
    DecisionPhase(
        key="phase_4_fusion_depth",
        question=(
            "Only after the head, pair target, and geometry constraint are "
            "settled: late or every-layer interleaved cross-attention?"
        ),
        factor="fusion",
        baseline="framecontact_contact_hybrid",
        candidates=("framecontact_interleaved",),
        primary_metric="partner_margin_worst_direction",
        guard_metrics=("contact_pooled_auprc", "epitope_pooled_auprc"),
    ),
)

# Recall must not collapse in exchange for precision; the triangle phase was
# pre-registered with a 95% retention floor.
TRIANGLE_RECALL_RETENTION_FLOOR = 0.95


def recall_retention(
    reference_runs: Mapping[int, SeedRun],
    candidate_runs: Mapping[int, SeedRun],
    *,
    epoch: int,
    metric: str = "contact_recall",
) -> dict[str, Any]:
    """Fraction of baseline recall the candidate keeps, per seed."""
    retentions: dict[int, float] = {}
    for seed in sorted(set(reference_runs) & set(candidate_runs)):
        left = reference_runs[seed].at(epoch)
        right = candidate_runs[seed].at(epoch)
        if left is None or right is None:
            continue
        baseline = left.get(metric)
        if baseline is None or right.get(metric) is None or baseline <= 0:
            continue
        retentions[seed] = right[metric] / baseline
    if not retentions:
        return {"per_seed": {}, "minimum": None, "mean": None, "meets_floor": None}
    values = list(retentions.values())
    minimum = min(values)
    return {
        "per_seed": retentions,
        "minimum": minimum,
        "mean": sum(values) / len(values),
        "floor": TRIANGLE_RECALL_RETENTION_FLOOR,
        "meets_floor": minimum >= TRIANGLE_RECALL_RETENTION_FLOOR,
    }


def summarize_arm(
    runs: Mapping[int, SeedRun], *, epoch: int, metrics: Sequence[str] | None = None
) -> dict[str, Any]:
    """Mean and per-seed values of one arm at a pinned epoch."""
    metric_names = tuple(metrics) if metrics else tuple(METRIC_PATHS)
    rows: dict[str, Any] = {}
    for name in metric_names:
        per_seed = {}
        for seed, run in sorted(runs.items()):
            values = run.at(epoch)
            if values is not None and name in values:
                per_seed[seed] = values[name]
        if per_seed:
            rows[name] = {
                "per_seed": per_seed,
                "mean": sum(per_seed.values()) / len(per_seed),
            }
    experiments = {run.experiment for run in runs.values()}
    if len(experiments) != 1:
        raise ComparisonError("summarize_arm expects a single arm")
    experiment = experiments.pop()
    return {
        "experiment": experiment,
        "factors": ARM_FACTORS[experiment].as_mapping()
        if experiment in ARM_FACTORS
        else None,
        "epoch": epoch,
        "seeds": sorted(runs),
        "metrics": rows,
    }


MAIN_TABLE_METRICS: tuple[str, ...] = (
    "contact_pooled_auprc",
    "contact_p_at_lmin_over_10",
    "contact_precision",
    "contact_recall",
    "contact_predicted_positive_ratio",
    "epitope_pooled_auprc",
    "epitope_precision",
    "epitope_recall",
    "paratope_pooled_auprc",
    "paratope_precision",
    "paratope_recall",
    "partner_margin_worst_direction",
    "partner_specific_variance_share",
    "epitope_gate_applied",
    "contact_brier",
    "contact_ece_15",
)


def build_main_table(
    runs_by_arm: Mapping[str, Mapping[int, SeedRun]],
    *,
    epoch: int | None = None,
    metrics: Sequence[str] = MAIN_TABLE_METRICS,
) -> dict[str, Any]:
    """The single table the ablation is meant to produce."""
    flattened = [run for runs in runs_by_arm.values() for run in runs.values()]
    resolved_epoch = matched_epoch(flattened) if epoch is None else int(epoch)
    return {
        "format_version": "framecontact_decision_main_table_v1",
        "epoch": resolved_epoch,
        "metrics": list(metrics),
        "rows": [
            summarize_arm(runs, epoch=resolved_epoch, metrics=metrics)
            for _, runs in sorted(runs_by_arm.items())
        ],
    }


def evaluate_phase(
    phase: DecisionPhase,
    runs_by_arm: Mapping[str, Mapping[int, SeedRun]],
    *,
    epoch: int | None = None,
) -> dict[str, Any]:
    """Run one stage of the decision chain and state what it decided."""
    if phase.baseline not in runs_by_arm:
        return {
            "phase": phase.key,
            "question": phase.question,
            "status": "unavailable",
            "reason": f"baseline arm {phase.baseline} has no runs",
        }
    available = [name for name in phase.candidates if name in runs_by_arm]
    if not available:
        return {
            "phase": phase.key,
            "question": phase.question,
            "status": "unavailable",
            "reason": (
                "no candidate arm has runs yet; expected any of "
                + ", ".join(phase.candidates)
            ),
        }

    comparisons = []
    for candidate in available:
        comparison = compare_arms(
            runs_by_arm[phase.baseline], runs_by_arm[candidate], epoch=epoch
        )
        entry: dict[str, Any] = {
            "candidate": candidate,
            "factor": comparison.factor,
            "epoch": comparison.epoch,
            "seeds": list(comparison.seeds),
            "primary_metric": phase.primary_metric,
            "primary": comparison.deltas_by_metric.get(phase.primary_metric),
            "guards": {
                name: comparison.deltas_by_metric.get(name)
                for name in phase.guard_metrics
            },
            "sparsity_verdict": comparison.sparsity_verdict,
        }
        primary = comparison.deltas_by_metric.get(phase.primary_metric)
        entry["epoch_stability"] = (
            None
            if primary is None
            else delta_against_epoch_noise(
                runs_by_arm[phase.baseline],
                runs_by_arm[candidate],
                phase.primary_metric,
                epoch=comparison.epoch,
                mean_delta=float(primary["mean_delta"]),
            )
        )
        if phase.factor == "triangle":
            entry["recall_retention"] = recall_retention(
                runs_by_arm[phase.baseline],
                runs_by_arm[candidate],
                epoch=comparison.epoch,
            )
        comparisons.append(entry)

    incomplete = [
        run.experiment
        for name in [phase.baseline, *available]
        for run in runs_by_arm[name].values()
        if not run.completed
    ]
    return {
        "phase": phase.key,
        "question": phase.question,
        "factor": phase.factor,
        "baseline": phase.baseline,
        "status": "in_flight" if incomplete else "complete",
        "claim_boundary": (
            "Development validation only at a pinned epoch. No frozen test was "
            "opened. In-flight runs may still change their selected checkpoint."
            if incomplete
            else "Development validation at validation-selected checkpoints."
        ),
        "caveats": list(phase.caveats),
        "missing_candidates": [
            name for name in phase.candidates if name not in runs_by_arm
        ],
        "comparisons": comparisons,
        "decision": _phase_decision(phase, comparisons),
    }


def _phase_decision(
    phase: DecisionPhase, comparisons: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """State the winner only when the evidence actually supports one."""
    ranked = []
    for entry in comparisons:
        primary = entry.get("primary")
        if primary is None:
            continue
        mean_delta = float(primary["mean_delta"])
        sign = primary["sign_test"]
        consistent = sign["negative"] == 0 and sign["positive"] >= 1
        stability = entry.get("epoch_stability") or {}
        ranked.append(
            {
                "candidate": entry["candidate"],
                "mean_delta": mean_delta,
                "seeds_positive": sign["positive"],
                "seeds_negative": sign["negative"],
                "p_value": sign["p_value"],
                "consistent": consistent,
                "exceeds_epoch_noise": stability.get("exceeds_noise"),
                "epoch_noise": stability.get("noise"),
            }
        )
    if not ranked:
        return {
            "resolved": False,
            "statement": (
                f"The primary metric {phase.primary_metric} is not yet available "
                "for any candidate."
            ),
        }
    ranked.sort(key=lambda item: item["mean_delta"], reverse=True)
    best = ranked[0]
    if best["mean_delta"] <= 0:
        return {
            "resolved": True,
            "winner": phase.baseline,
            "ranking": ranked,
            "statement": (
                f"No candidate improved {phase.primary_metric} over "
                f"{phase.baseline}; the baseline stands."
            ),
        }
    if not best["consistent"]:
        return {
            "resolved": False,
            "leader": best["candidate"],
            "ranking": ranked,
            "statement": (
                f"{best['candidate']} leads on {phase.primary_metric} by "
                f"{best['mean_delta']:.6f} but is not consistent across seeds "
                f"({best['seeds_positive']} up, {best['seeds_negative']} down). "
                "This is not yet a decision."
            ),
        }
    if best["exceeds_epoch_noise"] is False:
        return {
            "resolved": False,
            "leader": best["candidate"],
            "ranking": ranked,
            "statement": (
                f"{best['candidate']} leads on {phase.primary_metric} by "
                f"{best['mean_delta']:.6f}, but the same metric moves by "
                f"{best['epoch_noise']:.6f} between neighbouring epochs of a "
                "single run. The difference is inside the epoch noise band, so "
                "the pinned epoch, not the factor, could explain it."
            ),
        }
    return {
        "resolved": True,
        "winner": best["candidate"],
        "ranking": ranked,
        "statement": (
            f"{best['candidate']} improves {phase.primary_metric} by "
            f"{best['mean_delta']:.6f} in every seed that moved "
            f"({best['seeds_positive']}/{best['seeds_positive'] + best['seeds_negative']}, "
            f"exact sign-test p={best['p_value']:.4f})."
        ),
    }


def trajectory(
    runs: Mapping[int, SeedRun], metric: str, *, minimum_seeds: int = 1
) -> dict[str, Any]:
    """Per-epoch mean of one metric across seeds, for the trajectory figures."""
    values_by_epoch: dict[int, list[float]] = {}
    for run in runs.values():
        for epoch in run.epochs:
            entry = run.at(epoch)
            if entry is not None and metric in entry:
                values_by_epoch.setdefault(epoch, []).append(entry[metric])
    epochs = sorted(
        epoch
        for epoch, values in values_by_epoch.items()
        if len(values) >= minimum_seeds
    )
    return {
        "metric": metric,
        "epochs": epochs,
        "mean": [
            sum(values_by_epoch[epoch]) / len(values_by_epoch[epoch])
            for epoch in epochs
        ],
        "seed_count": [len(values_by_epoch[epoch]) for epoch in epochs],
    }


__all__ = [
    "ARM_FACTORS",
    "DECISION_CHAIN",
    "FACTORS",
    "EPOCH_NOISE_WINDOW",
    "GAUGE_ROUTED_AGGREGATIONS",
    "delta_against_epoch_noise",
    "epoch_noise",
    "free_factors",
    "LOWER_IS_BETTER",
    "MAIN_TABLE_METRICS",
    "METRIC_PATHS",
    "ArmFactors",
    "ComparisonError",
    "DecisionPhase",
    "PairedComparison",
    "SeedRun",
    "bootstrap_mean",
    "build_main_table",
    "classify_precision_change",
    "compare_arms",
    "evaluate_phase",
    "exact_sign_test",
    "extract_metrics",
    "factor_difference",
    "load_completed_run",
    "load_live_log",
    "matched_epoch",
    "parse_epoch_records",
    "recall_retention",
    "require_single_variable",
    "summarize_arm",
    "trajectory",
]
