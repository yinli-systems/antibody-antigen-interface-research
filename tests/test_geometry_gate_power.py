"""Tests for the geometry-gate power and decision-rule analysis."""

from __future__ import annotations

import math

import pytest

from interaction_first.geometry_gate_power import (
    GeometryPowerError,
    ReplicateSummary,
    clopper_pearson_lower_bound,
    holm_adjusted,
    linear_dependence_residual,
    minimum_detectable_effect,
    minimum_unanimous_replicates,
    one_sided_t_power,
    replicates_for_power,
)

pytest.importorskip("scipy")


def test_clopper_pearson_matches_the_closed_form_for_a_unanimous_sample() -> None:
    """With k == n the exact lower bound is alpha ** (1 / n)."""
    for trials in (3, 5, 10):
        assert clopper_pearson_lower_bound(trials, trials) == pytest.approx(
            0.05 ** (1 / trials), rel=1e-9
        )


def test_clopper_pearson_is_zero_without_a_single_success() -> None:
    assert clopper_pearson_lower_bound(0, 7) == 0.0


def test_clopper_pearson_rejects_impossible_counts() -> None:
    with pytest.raises(GeometryPowerError):
        clopper_pearson_lower_bound(8, 7)
    with pytest.raises(GeometryPowerError):
        clopper_pearson_lower_bound(1, 0)


def test_three_unanimous_replicates_cannot_clear_the_sign_criterion() -> None:
    """The governing fact behind the geometry gate: 3/3 is not evidence at 0.05."""
    assert clopper_pearson_lower_bound(3, 3) < 0.5
    assert clopper_pearson_lower_bound(5, 5) > 0.5
    assert minimum_unanimous_replicates() == 5


def test_minimum_unanimous_replicates_tracks_alpha() -> None:
    assert minimum_unanimous_replicates(alpha=0.10) == 4
    assert minimum_unanimous_replicates(alpha=0.01) == 7


def test_replicate_summary_reports_mean_sd_and_effect_size() -> None:
    summary = ReplicateSummary.from_differences([0.017724, 0.008886, 0.000886])

    assert summary.count == 3
    assert summary.positive_count == 3
    assert summary.mean == pytest.approx(0.009165, abs=1e-6)
    assert summary.effect_size == pytest.approx(summary.mean / summary.sample_sd)


def test_replicate_summary_needs_at_least_two_values() -> None:
    with pytest.raises(GeometryPowerError):
        ReplicateSummary.from_differences([0.01])


def test_zero_variance_yields_no_effect_size_rather_than_infinity() -> None:
    summary = ReplicateSummary.from_differences([0.02, 0.02, 0.02])

    assert summary.sample_sd == 0.0
    assert summary.effect_size is None


def test_power_rises_with_replicates_and_effect() -> None:
    assert one_sided_t_power(1.0, 3) < one_sided_t_power(1.0, 10)
    assert one_sided_t_power(0.5, 10) < one_sided_t_power(1.5, 10)
    assert 0.0 < one_sided_t_power(1.088, 3) < 1.0


def test_minimum_detectable_effect_is_the_inverse_of_the_power_curve() -> None:
    for count in (3, 5, 10):
        effect = minimum_detectable_effect(count)
        assert one_sided_t_power(effect, count) == pytest.approx(0.80, abs=1e-3)


def test_minimum_detectable_effect_shrinks_as_replicates_grow() -> None:
    effects = [minimum_detectable_effect(count) for count in (3, 5, 8, 20)]

    assert effects == sorted(effects, reverse=True)


def test_five_replicates_cannot_detect_the_observed_joint_effect() -> None:
    """The preregistered five-seed design is itself underpowered for this effect."""
    observed = ReplicateSummary.from_differences([0.017724, 0.008886, 0.000886])

    assert observed.effect_size is not None
    assert observed.effect_size < minimum_detectable_effect(5)
    assert replicates_for_power(observed.effect_size) == 7


def test_replicates_for_power_returns_none_for_a_nonpositive_effect() -> None:
    assert replicates_for_power(0.0) is None
    assert replicates_for_power(-1.0) is None


def test_holm_is_step_down_and_never_exceeds_bonferroni() -> None:
    raw = {"a": 0.0247975, "b": 0.0274973, "c": 0.1951805}

    adjusted = holm_adjusted(raw)

    assert adjusted["a"] == pytest.approx(3 * 0.0247975)
    assert adjusted["b"] == pytest.approx(3 * 0.0247975)  # step-down carries the running max
    assert adjusted["c"] == pytest.approx(0.1951805)
    for name, value in raw.items():
        assert adjusted[name] <= min(1.0, len(raw) * value) + 1e-12


def test_holm_is_monotone_and_capped_at_one() -> None:
    adjusted = holm_adjusted({"a": 0.4, "b": 0.5, "c": 0.9})

    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"] == 1.0


def test_holm_of_an_empty_family_is_empty() -> None:
    assert holm_adjusted({}) == {}


def test_linear_dependence_residual_detects_a_mean_composite() -> None:
    components = [{"m": 0.66, "n": 0.30}, {"m": 0.32, "n": 0.10}]
    composite = {"m": 0.49, "n": 0.20}

    assert linear_dependence_residual(composite, components) == pytest.approx(0.0, abs=1e-15)


def test_linear_dependence_residual_is_large_for_an_independent_endpoint() -> None:
    components = [{"m": 0.6}, {"m": 0.3}]
    composite = {"m": 0.9}

    assert linear_dependence_residual(composite, components) == pytest.approx(0.45)


def test_linear_dependence_residual_requires_shared_keys() -> None:
    with pytest.raises(GeometryPowerError):
        linear_dependence_residual({"a": 1.0}, [{"b": 1.0}])
    with pytest.raises(GeometryPowerError):
        linear_dependence_residual({"a": 1.0}, [])


def test_power_and_effect_helpers_reject_degenerate_counts() -> None:
    with pytest.raises(GeometryPowerError):
        one_sided_t_power(1.0, 1)
    with pytest.raises(GeometryPowerError):
        minimum_detectable_effect(1)


def test_power_is_finite_for_a_very_large_effect() -> None:
    assert math.isfinite(one_sided_t_power(50.0, 3))
    assert one_sided_t_power(50.0, 3) == pytest.approx(1.0, abs=1e-6)
