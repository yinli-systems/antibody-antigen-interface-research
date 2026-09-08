"""Power and decision-rule analysis for the intra-chain geometry superiority gate.

The geometry ablation recorded a positive development signal that did not clear
its preregistered gate. The tracked evidence answers whether the effect was
detected; it does not answer whether the experiment could have detected it. No
tracked artifact in this repository reports statistical power, a minimum
detectable effect, or a decision rule other than a difference-of-means test on a
single fit.

This module supplies those missing quantities from the already-audited summary
statistics. It computes nothing new about the model: every input is read from a
committed receipt, and every output is a property of the *design*, not a revised
performance claim.

It provides three things.

Exact sign-criterion bounds
    A Clopper-Pearson one-sided lower bound on P(A > B), the probability that the
    candidate outperforms the reference across the randomness that was actually
    resampled. A result is significant under this criterion when the lower bound
    exceeds 0.5. Because the bound is exact, it also answers how many
    all-positive replicates are needed before the criterion can *ever* be met --
    a question that must be settled before running, not after.

Power and minimum detectable effect
    One-sided one-sample t power from the observed between-replicate standard
    deviation, so a failed gate can be read as either "no effect" or "no power".

Multiplicity family diagnostics
    Holm assumes arbitrary dependence and is correspondingly conservative. When
    a family mixes a composite endpoint with the components that determine it,
    the correction charges for hypotheses that carry no independent information.
    ``linear_dependence_residual`` measures exactly that, so the choice of family
    is reported rather than assumed.

References for the decision rule and the reporting requirement:
Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks",
MLSys 2021; Kapoor et al., "REFORMS: Consensus-based Recommendations for
Machine-learning-based Science", Science Advances 2024, items 7b and 7c.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

SIGN_CRITERION_THRESHOLD = 0.5
DEFAULT_ALPHA = 0.05
DEFAULT_TARGET_POWER = 0.80


class GeometryPowerError(ValueError):
    """Raised when an input cannot support the requested statistic."""


def _scipy_stats():
    try:
        from scipy import stats
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise GeometryPowerError("SciPy is required for the power analysis") from error
    return stats


def clopper_pearson_lower_bound(successes: int, trials: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Return the exact one-sided lower confidence bound on a binomial rate.

    Uses the Beta quantile form of the Clopper-Pearson interval, which is exact
    rather than asymptotic. A normal approximation is invalid at the replicate
    counts this analysis deals with, and a percentile bootstrap of an
    all-positive sample degenerates to a point mass at 1.0.
    """
    if trials <= 0:
        raise GeometryPowerError("trials must be positive")
    if not 0 <= successes <= trials:
        raise GeometryPowerError("successes must lie in [0, trials]")
    if not 0.0 < alpha < 1.0:
        raise GeometryPowerError("alpha must lie in (0, 1)")
    if successes == 0:
        return 0.0
    stats = _scipy_stats()
    return float(stats.beta.ppf(alpha, successes, trials - successes + 1))


def minimum_unanimous_replicates(
    alpha: float = DEFAULT_ALPHA,
    threshold: float = SIGN_CRITERION_THRESHOLD,
) -> int:
    """Smallest replicate count whose unanimous result can clear ``threshold``.

    With ``k == n`` the exact lower bound is ``alpha ** (1 / n)``. Below this
    count the sign criterion cannot be met by any outcome, so a gate written
    against it is unreachable however large the true effect is.
    """
    if not 0.0 < alpha < 1.0:
        raise GeometryPowerError("alpha must lie in (0, 1)")
    if not 0.0 < threshold < 1.0:
        raise GeometryPowerError("threshold must lie in (0, 1)")
    return int(math.floor(math.log(alpha) / math.log(threshold))) + 1


@dataclasses.dataclass(frozen=True)
class ReplicateSummary:
    """Distribution of a paired difference across independent replicates."""

    differences: tuple[float, ...]
    mean: float
    sample_sd: float
    positive_count: int
    count: int
    effect_size: float | None

    @classmethod
    def from_differences(cls, differences: Sequence[float]) -> ReplicateSummary:
        values = tuple(float(value) for value in differences)
        if len(values) < 2:
            raise GeometryPowerError("at least two replicates are required")
        count = len(values)
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        sample_sd = math.sqrt(variance)
        return cls(
            differences=values,
            mean=mean,
            sample_sd=sample_sd,
            positive_count=sum(1 for value in values if value > 0.0),
            count=count,
            effect_size=(mean / sample_sd) if sample_sd > 0.0 else None,
        )


def one_sided_t_power(effect_size: float, count: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Power of a one-sided one-sample t-test at a standardised effect size."""
    if count < 2:
        raise GeometryPowerError("count must be at least two")
    stats = _scipy_stats()
    degrees = count - 1
    critical = stats.t.ppf(1.0 - alpha, degrees)
    noncentrality = effect_size * math.sqrt(count)
    return float(stats.nct.sf(critical, degrees, noncentrality))


def replicates_for_power(
    effect_size: float,
    target_power: float = DEFAULT_TARGET_POWER,
    alpha: float = DEFAULT_ALPHA,
    maximum: int = 500,
) -> int | None:
    """Smallest replicate count reaching ``target_power`` at ``effect_size``."""
    if effect_size <= 0.0:
        return None
    for count in range(2, maximum + 1):
        if one_sided_t_power(effect_size, count, alpha) >= target_power:
            return count
    return None


def minimum_detectable_effect(
    count: int,
    target_power: float = DEFAULT_TARGET_POWER,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Standardised effect a design of ``count`` replicates can detect.

    An observed effect below this value is uninformative: the design would have
    missed it more often than not even if it were real.
    """
    if count < 2:
        raise GeometryPowerError("count must be at least two")
    low, high = 0.0, 50.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if one_sided_t_power(middle, count, alpha) < target_power:
            low = middle
        else:
            high = middle
    return high


def holm_adjusted(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjustment over a named family, order-stable."""
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, (total - index) * value)
        adjusted[name] = min(1.0, running)
    return {name: adjusted[name] for name in p_values}


def linear_dependence_residual(
    composite: Mapping[str, float],
    components: Sequence[Mapping[str, float]],
) -> float:
    """Largest deviation from ``composite == mean(components)`` across keys.

    A residual at floating-point scale proves the composite carries no
    information the components do not already determine, so treating it as an
    independent hypothesis in a multiplicity family charges for a test that
    cannot vary on its own.
    """
    if not components:
        raise GeometryPowerError("at least one component is required")
    shared = set(composite)
    for component in components:
        shared &= set(component)
    if not shared:
        raise GeometryPowerError("composite and components share no keys")
    worst = 0.0
    for key in sorted(shared):
        predicted = sum(component[key] for component in components) / len(components)
        worst = max(worst, abs(predicted - composite[key]))
    return worst
