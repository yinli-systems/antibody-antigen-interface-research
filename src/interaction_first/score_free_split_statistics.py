"""Independent score-free distribution and split-shape audit primitives.

The module accepts only prospectively allowed sequence/topology summaries and
positive-pair incidence. It has no fields for contacts, interface labels,
affinity, embeddings, predictions, losses, or model scores. Split construction
and the independent audit must still derive these inputs separately from their
own authenticated train-only sources.

All discrete comparisons use exact ``Fraction`` arithmetic. Log-degree SMDs
use a pinned 50-digit decimal context because logarithms are irrational; the
comparison is fail-closed when a sample variance is undefined. This module does
not audit homology edges and cannot authorize training by itself.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from math import comb
from typing import Iterable, Mapping, Sequence


_DECIMAL_PRECISION = 50
_AXES = frozenset({"antibody", "antigen"})


class ScoreFreeAuditError(ValueError):
    """Raised when score-free audit inputs or invariants are invalid."""


@dataclass(frozen=True)
class DistributionThresholds:
    """Hard representativeness limits for one audit scope."""

    maximum_smd: Decimal
    maximum_tv: Fraction
    maximum_ks: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.maximum_smd, Decimal):
            raise ScoreFreeAuditError("maximum_smd must be a Decimal")
        if not isinstance(self.maximum_tv, Fraction):
            raise ScoreFreeAuditError("maximum_tv must be a Fraction")
        if not isinstance(self.maximum_ks, Fraction):
            raise ScoreFreeAuditError("maximum_ks must be a Fraction")
        if not Decimal(0) <= self.maximum_smd <= Decimal(1):
            raise ScoreFreeAuditError("maximum_smd must be in [0, 1]")
        if not Fraction(0) <= self.maximum_tv <= Fraction(1):
            raise ScoreFreeAuditError("maximum_tv must be in [0, 1]")
        if not Fraction(0) <= self.maximum_ks <= Fraction(1):
            raise ScoreFreeAuditError("maximum_ks must be in [0, 1]")


OUTER_DISTRIBUTION_THRESHOLDS = DistributionThresholds(
    maximum_smd=Decimal("0.35"),
    maximum_tv=Fraction(1, 5),
    maximum_ks=Fraction(1, 5),
)
POOLED_OUTER_DISTRIBUTION_THRESHOLDS = DistributionThresholds(
    maximum_smd=Decimal("0.25"),
    maximum_tv=Fraction(3, 20),
    maximum_ks=Fraction(1, 5),
)
INNER_DISTRIBUTION_THRESHOLDS = DistributionThresholds(
    maximum_smd=Decimal("0.40"),
    maximum_tv=Fraction(1, 4),
    maximum_ks=Fraction(1, 4),
)


@dataclass(frozen=True)
class FoldShapeThresholds:
    """Exact pair, endpoint, component, retention, and quarantine gates."""

    minimum_validation_pairs: int
    minimum_validation_endpoints_per_axis: int
    minimum_validation_components_per_axis: int
    validation_pair_lower_fraction: Fraction
    validation_pair_upper_fraction: Fraction
    minimum_training_pair_fraction: Fraction
    minimum_training_endpoint_fraction: Fraction
    maximum_quarantine_pair_fraction: Fraction

    def __post_init__(self) -> None:
        for name in (
            "minimum_validation_pairs",
            "minimum_validation_endpoints_per_axis",
            "minimum_validation_components_per_axis",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ScoreFreeAuditError(f"{name} must be a non-negative integer")
        for name in (
            "validation_pair_lower_fraction",
            "validation_pair_upper_fraction",
            "minimum_training_pair_fraction",
            "minimum_training_endpoint_fraction",
            "maximum_quarantine_pair_fraction",
        ):
            value = getattr(self, name)
            if not isinstance(value, Fraction) or not Fraction(0) <= value <= Fraction(1):
                raise ScoreFreeAuditError(f"{name} must be a Fraction in [0, 1]")
        if self.validation_pair_lower_fraction > self.validation_pair_upper_fraction:
            raise ScoreFreeAuditError("validation pair lower fraction exceeds upper")


OUTER_FOLD_SHAPE_THRESHOLDS = FoldShapeThresholds(
    minimum_validation_pairs=120,
    minimum_validation_endpoints_per_axis=60,
    minimum_validation_components_per_axis=25,
    validation_pair_lower_fraction=Fraction(7, 100),
    validation_pair_upper_fraction=Fraction(9, 100),
    minimum_training_pair_fraction=Fraction(3, 5),
    minimum_training_endpoint_fraction=Fraction(3, 5),
    maximum_quarantine_pair_fraction=Fraction(33, 100),
)
INNER_FOLD_SHAPE_THRESHOLDS = FoldShapeThresholds(
    minimum_validation_pairs=80,
    minimum_validation_endpoints_per_axis=40,
    minimum_validation_components_per_axis=20,
    validation_pair_lower_fraction=Fraction(7, 100),
    validation_pair_upper_fraction=Fraction(9, 100),
    minimum_training_pair_fraction=Fraction(11, 20),
    minimum_training_endpoint_fraction=Fraction(11, 20),
    maximum_quarantine_pair_fraction=Fraction(19, 50),
)


@dataclass(frozen=True)
class CrossFoldThresholds:
    """Locked exact relative-range limits across one three-fold level."""

    maximum_validation_pair_relative_range: Fraction
    maximum_other_relative_range: Fraction

    def __post_init__(self) -> None:
        for name in (
            "maximum_validation_pair_relative_range",
            "maximum_other_relative_range",
        ):
            value = getattr(self, name)
            if not isinstance(value, Fraction) or not Fraction(0) <= value <= Fraction(1):
                raise ScoreFreeAuditError(f"{name} must be a Fraction in [0, 1]")


OUTER_CROSS_FOLD_THRESHOLDS = CrossFoldThresholds(
    maximum_validation_pair_relative_range=Fraction(3, 20),
    maximum_other_relative_range=Fraction(1, 5),
)
INNER_CROSS_FOLD_THRESHOLDS = CrossFoldThresholds(
    maximum_validation_pair_relative_range=Fraction(1, 5),
    maximum_other_relative_range=Fraction(1, 4),
)


@dataclass(frozen=True)
class FeatureSpec:
    """One prospectively permitted score-free covariate."""

    name: str
    optional_when_vl_missing: bool = False
    categorical: bool = False
    gate_smd: bool = True
    gate_ks: bool = True
    gate_tv: bool = True


ANTIBODY_FEATURE_SPECS = (
    FeatureSpec("vh_length"),
    FeatureSpec("vl_length", optional_when_vl_missing=True),
    FeatureSpec("concatenated_antibody_length"),
    FeatureSpec("log1p_vh_direct_degree"),
    FeatureSpec("log1p_vl_direct_degree", optional_when_vl_missing=True),
    FeatureSpec("log1p_concatenated_direct_degree"),
    FeatureSpec("log1p_union_direct_degree"),
    FeatureSpec("log1p_pair_incidence_degree"),
    FeatureSpec("component_size"),
    FeatureSpec(
        "missing_vl",
        categorical=True,
        gate_smd=False,
        gate_ks=False,
    ),
)
ANTIGEN_FEATURE_SPECS = (
    FeatureSpec("antigen_length"),
    FeatureSpec("log1p_antigen_direct_degree"),
    FeatureSpec("log1p_pair_incidence_degree"),
    FeatureSpec("component_size"),
)
_SPECS_BY_AXIS = {
    "antibody": ANTIBODY_FEATURE_SPECS,
    "antigen": ANTIGEN_FEATURE_SPECS,
}


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ScoreFreeAuditError(f"{name} must be a non-empty string")
    if value != value.strip() or not value.isascii():
        raise ScoreFreeAuditError(f"{name} must be exact ASCII without edge whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ScoreFreeAuditError(f"{name} cannot contain control characters")


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScoreFreeAuditError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: int, name: str) -> int:
    result = _non_negative_integer(value, name)
    if result == 0:
        raise ScoreFreeAuditError(f"{name} must be positive")
    return result


def _log1p_decimal(value: int) -> Decimal:
    _non_negative_integer(value, "degree")
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        return +(Decimal(value) + Decimal(1)).ln()


@dataclass(frozen=True)
class AntibodyScoreFreeInputs:
    """Allowed raw summaries for one antibody entity."""

    entity_id: str
    component_id: str
    vh_length: int
    vl_length: int | None
    concatenated_antibody_length: int
    vh_direct_degree: int
    vl_direct_degree: int | None
    concatenated_direct_degree: int
    union_direct_degree: int
    pair_incidence_degree: int
    component_size: int

    def __post_init__(self) -> None:
        _validate_identifier(self.entity_id, "entity_id")
        _validate_identifier(self.component_id, "component_id")
        vh_length = _positive_integer(self.vh_length, "vh_length")
        if self.vl_length is None:
            if self.vl_direct_degree is not None:
                raise ScoreFreeAuditError(
                    "vl_direct_degree must be missing when vl_length is missing"
                )
            vl_length = 0
        else:
            vl_length = _positive_integer(self.vl_length, "vl_length")
            if self.vl_direct_degree is None:
                raise ScoreFreeAuditError(
                    "vl_direct_degree is required when vl_length is present"
                )
            _non_negative_integer(self.vl_direct_degree, "vl_direct_degree")
        if self.concatenated_antibody_length != vh_length + vl_length:
            raise ScoreFreeAuditError(
                "concatenated_antibody_length must equal vh_length plus vl_length"
            )
        for name in (
            "vh_direct_degree",
            "concatenated_direct_degree",
            "union_direct_degree",
        ):
            _non_negative_integer(getattr(self, name), name)
        _positive_integer(self.pair_incidence_degree, "pair_incidence_degree")
        component_size = _positive_integer(self.component_size, "component_size")
        view_degrees = (
            self.vh_direct_degree,
            0 if self.vl_direct_degree is None else self.vl_direct_degree,
            self.concatenated_direct_degree,
        )
        if self.union_direct_degree < max(view_degrees):
            raise ScoreFreeAuditError(
                "union_direct_degree cannot be smaller than a constituent view"
            )
        if self.union_direct_degree >= component_size:
            raise ScoreFreeAuditError(
                "union_direct_degree must be smaller than component_size"
            )


@dataclass(frozen=True)
class AntigenScoreFreeInputs:
    """Allowed raw summaries for one antigen entity."""

    entity_id: str
    component_id: str
    antigen_length: int
    antigen_direct_degree: int
    pair_incidence_degree: int
    component_size: int

    def __post_init__(self) -> None:
        _validate_identifier(self.entity_id, "entity_id")
        _validate_identifier(self.component_id, "component_id")
        _positive_integer(self.antigen_length, "antigen_length")
        direct_degree = _non_negative_integer(
            self.antigen_direct_degree, "antigen_direct_degree"
        )
        _positive_integer(self.pair_incidence_degree, "pair_incidence_degree")
        component_size = _positive_integer(self.component_size, "component_size")
        if direct_degree >= component_size:
            raise ScoreFreeAuditError(
                "antigen_direct_degree must be smaller than component_size"
            )


@dataclass(frozen=True)
class EntityProfile:
    """Validated immutable values for one entity and one allowed feature schema."""

    entity_id: str
    component_id: str
    axis: str
    values: tuple[tuple[str, Decimal | None], ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.entity_id, "entity_id")
        _validate_identifier(self.component_id, "component_id")
        if self.axis not in _AXES:
            raise ScoreFreeAuditError("axis must be 'antibody' or 'antigen'")
        if not isinstance(self.values, tuple):
            raise ScoreFreeAuditError("profile values must be an immutable tuple")
        expected = tuple(spec.name for spec in _SPECS_BY_AXIS[self.axis])
        observed = tuple(name for name, _ in self.values)
        if observed != expected:
            raise ScoreFreeAuditError(
                f"{self.axis} profile must use the exact allowed feature schema"
            )
        for name, value in self.values:
            if value is not None and not isinstance(value, Decimal):
                raise ScoreFreeAuditError(f"{name} must be Decimal or None")
            if value is not None and not value.is_finite():
                raise ScoreFreeAuditError(f"{name} must be finite")
            if value is not None and value < 0:
                raise ScoreFreeAuditError(f"{name} must be non-negative")
        if self.axis == "antibody":
            value_by_name = dict(self.values)
            missing_vl = value_by_name["missing_vl"]
            if missing_vl not in {Decimal(0), Decimal(1)}:
                raise ScoreFreeAuditError("missing_vl must be exactly 0 or 1")
            optional_missing = (
                value_by_name["vl_length"] is None,
                value_by_name["log1p_vl_direct_degree"] is None,
            )
            if optional_missing != (missing_vl == 1, missing_vl == 1):
                raise ScoreFreeAuditError(
                    "VL length, VL degree, and missing_vl must agree"
                )

    def value(self, feature_name: str) -> Decimal | None:
        """Return one value after exact-schema validation at construction."""
        for name, value in self.values:
            if name == feature_name:
                return value
        raise ScoreFreeAuditError(f"Unknown feature {feature_name!r}")


def antibody_profile(inputs: AntibodyScoreFreeInputs) -> EntityProfile:
    """Transform permitted raw antibody counts into the frozen feature schema."""
    missing_vl = inputs.vl_length is None
    return EntityProfile(
        entity_id=inputs.entity_id,
        component_id=inputs.component_id,
        axis="antibody",
        values=(
            ("vh_length", Decimal(inputs.vh_length)),
            ("vl_length", None if missing_vl else Decimal(inputs.vl_length)),
            (
                "concatenated_antibody_length",
                Decimal(inputs.concatenated_antibody_length),
            ),
            ("log1p_vh_direct_degree", _log1p_decimal(inputs.vh_direct_degree)),
            (
                "log1p_vl_direct_degree",
                None
                if inputs.vl_direct_degree is None
                else _log1p_decimal(inputs.vl_direct_degree),
            ),
            (
                "log1p_concatenated_direct_degree",
                _log1p_decimal(inputs.concatenated_direct_degree),
            ),
            (
                "log1p_union_direct_degree",
                _log1p_decimal(inputs.union_direct_degree),
            ),
            (
                "log1p_pair_incidence_degree",
                _log1p_decimal(inputs.pair_incidence_degree),
            ),
            ("component_size", Decimal(inputs.component_size)),
            ("missing_vl", Decimal(int(missing_vl))),
        ),
    )


def antigen_profile(inputs: AntigenScoreFreeInputs) -> EntityProfile:
    """Transform permitted raw antigen counts into the frozen feature schema."""
    return EntityProfile(
        entity_id=inputs.entity_id,
        component_id=inputs.component_id,
        axis="antigen",
        values=(
            ("antigen_length", Decimal(inputs.antigen_length)),
            (
                "log1p_antigen_direct_degree",
                _log1p_decimal(inputs.antigen_direct_degree),
            ),
            (
                "log1p_pair_incidence_degree",
                _log1p_decimal(inputs.pair_incidence_degree),
            ),
            ("component_size", Decimal(inputs.component_size)),
        ),
    )


@dataclass(frozen=True)
class ScoreFreePair:
    """One observed, indivisible positive pair without outcome labels."""

    pair_id: str
    antibody_id: str
    antigen_id: str

    def __post_init__(self) -> None:
        _validate_identifier(self.pair_id, "pair_id")
        _validate_identifier(self.antibody_id, "antibody_id")
        _validate_identifier(self.antigen_id, "antigen_id")


@dataclass(frozen=True)
class QuantileScheme:
    """Fixed type-1, tie-preserving bins built from one universe."""

    feature_name: str
    cuts: tuple[Decimal, ...]
    optional_missing_bin: bool
    categorical_binary: bool
    informative: bool

    def bin_index(self, value: Decimal | None) -> int:
        """Map equality to the lower closed bin and preserve missingness."""
        if self.categorical_binary:
            if value not in {Decimal(0), Decimal(1)}:
                raise ScoreFreeAuditError("binary categorical value must be 0 or 1")
            return int(value)
        if value is None:
            if not self.optional_missing_bin:
                raise ScoreFreeAuditError(
                    f"Unexpected missing value for {self.feature_name}"
                )
            return 0
        offset = 1 if self.optional_missing_bin else 0
        return offset + bisect_left(self.cuts, value)

    @property
    def bin_count(self) -> int:
        if self.categorical_binary:
            return 2
        return len(self.cuts) + 1 + int(self.optional_missing_bin)


def _build_quantile_scheme(
    profiles: Sequence[EntityProfile],
    spec: FeatureSpec,
) -> QuantileScheme:
    if spec.categorical:
        observed = {profile.value(spec.name) for profile in profiles}
        if not observed <= {Decimal(0), Decimal(1)}:
            raise ScoreFreeAuditError(f"{spec.name} must be binary")
        return QuantileScheme(
            feature_name=spec.name,
            cuts=(),
            optional_missing_bin=False,
            categorical_binary=True,
            informative=len(observed) >= 2,
        )
    values = sorted(
        value
        for profile in profiles
        if (value := profile.value(spec.name)) is not None
    )
    if not values:
        raise ScoreFreeAuditError(f"Universe has no values for {spec.name}")
    tentative = []
    for numerator in range(1, 5):
        one_based = (numerator * len(values) + 4) // 5
        tentative.append(values[one_based - 1])
    maximum = values[-1]
    cuts = tuple(sorted({value for value in tentative if value < maximum}))
    provisional = QuantileScheme(
        feature_name=spec.name,
        cuts=cuts,
        optional_missing_bin=spec.optional_when_vl_missing,
        categorical_binary=False,
        informative=False,
    )
    observed_bins = {
        provisional.bin_index(profile.value(spec.name)) for profile in profiles
    }
    return QuantileScheme(
        feature_name=spec.name,
        cuts=cuts,
        optional_missing_bin=spec.optional_when_vl_missing,
        categorical_binary=False,
        informative=len(observed_bins) >= 2,
    )


def build_quantile_schemes(
    profiles: Sequence[EntityProfile],
) -> tuple[QuantileScheme, ...]:
    """Build the locked full-universe schemes for exactly one entity axis."""
    checked = _validate_profile_collection(profiles, expected_axis=None)
    specs = _SPECS_BY_AXIS[checked[0].axis]
    return tuple(_build_quantile_scheme(checked, spec) for spec in specs)


def _validate_profile_collection(
    profiles: Sequence[EntityProfile],
    expected_axis: str | None,
) -> tuple[EntityProfile, ...]:
    if isinstance(profiles, (str, bytes)) or not isinstance(profiles, Sequence):
        raise ScoreFreeAuditError("profiles must be a non-string sequence")
    result = tuple(profiles)
    if not result:
        raise ScoreFreeAuditError("profile collection cannot be empty")
    if not all(isinstance(profile, EntityProfile) for profile in result):
        raise ScoreFreeAuditError("profile collection contains an invalid item")
    axes = {profile.axis for profile in result}
    if len(axes) != 1:
        raise ScoreFreeAuditError("profile collection cannot mix entity axes")
    if expected_axis is not None and axes != {expected_axis}:
        raise ScoreFreeAuditError(f"expected {expected_axis} profiles")
    identifiers = [profile.entity_id for profile in result]
    if len(set(identifiers)) != len(identifiers):
        raise ScoreFreeAuditError("profile entity identifiers must be unique")
    observed_component_counts = Counter(profile.component_id for profile in result)
    declared_component_sizes: dict[str, set[Decimal]] = {}
    for profile in result:
        declared_component_sizes.setdefault(profile.component_id, set()).add(
            profile.value("component_size")
        )
    for component_id, declared_values in declared_component_sizes.items():
        if len(declared_values) != 1:
            raise ScoreFreeAuditError(
                f"Component {component_id!r} has inconsistent declared sizes"
            )
        declared = next(iter(declared_values))
        if declared is None or declared != declared.to_integral_value():
            raise ScoreFreeAuditError("component_size must be an exact integer")
        if declared < observed_component_counts[component_id]:
            raise ScoreFreeAuditError(
                f"Component {component_id!r} is smaller than its observed membership"
            )
    return tuple(sorted(result, key=lambda profile: profile.entity_id))


def _fraction_mean(values: Sequence[Decimal]) -> Fraction:
    exact = tuple(Fraction(value) for value in values)
    return sum(exact, Fraction(0)) / len(exact)


def _fraction_sample_variance(
    values: Sequence[Decimal],
    mean: Fraction,
) -> Fraction:
    exact = tuple(Fraction(value) for value in values)
    return sum(((value - mean) ** 2 for value in exact), Fraction(0)) / (
        len(exact) - 1
    )


def _fraction_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _standardized_mean_difference(
    left: Sequence[Decimal],
    right: Sequence[Decimal],
) -> tuple[Decimal | None, str | None]:
    if len(left) < 2 or len(right) < 2:
        return None, "smd_requires_at_least_two_nonmissing_values_per_group"
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        left_mean = _fraction_mean(left)
        right_mean = _fraction_mean(right)
        pooled = (
            _fraction_sample_variance(left, left_mean)
            + _fraction_sample_variance(right, right_mean)
        ) / 2
        difference = abs(left_mean - right_mean)
        if pooled == 0:
            if difference == 0:
                return Decimal(0), None
            return None, "smd_zero_pooled_variance_with_unequal_means"
        return +(
            _fraction_to_decimal(difference)
            / _fraction_to_decimal(pooled).sqrt()
        ), None


def _exact_ks(left: Sequence[Decimal], right: Sequence[Decimal]) -> Fraction:
    if not left or not right:
        raise ScoreFreeAuditError("KS requires non-empty nonmissing groups")
    left_counts = Counter(left)
    right_counts = Counter(right)
    left_seen = 0
    right_seen = 0
    result = Fraction(0)
    for value in sorted(set(left_counts) | set(right_counts)):
        left_seen += left_counts[value]
        right_seen += right_counts[value]
        difference = abs(
            Fraction(left_seen, len(left)) - Fraction(right_seen, len(right))
        )
        result = max(result, difference)
    return result


def _exact_tv(
    left: Sequence[EntityProfile],
    right: Sequence[EntityProfile],
    scheme: QuantileScheme,
) -> Fraction:
    left_counts = Counter(
        scheme.bin_index(profile.value(scheme.feature_name)) for profile in left
    )
    right_counts = Counter(
        scheme.bin_index(profile.value(scheme.feature_name)) for profile in right
    )
    total = Fraction(0)
    for bin_index in range(scheme.bin_count):
        total += abs(
            Fraction(left_counts[bin_index], len(left))
            - Fraction(right_counts[bin_index], len(right))
        )
    return total / 2


def _fraction_payload(value: Fraction | None) -> list[int] | None:
    if value is None:
        return None
    return [value.numerator, value.denominator]


def _distribution_thresholds_payload(
    value: DistributionThresholds,
) -> Mapping[str, object]:
    return {
        "maximum_smd_decimal": str(value.maximum_smd),
        "maximum_tv_exact": _fraction_payload(value.maximum_tv),
        "maximum_ks_exact": _fraction_payload(value.maximum_ks),
    }


def _shape_thresholds_payload(value: FoldShapeThresholds) -> Mapping[str, object]:
    return {
        "minimum_validation_pairs": value.minimum_validation_pairs,
        "minimum_validation_endpoints_per_axis": (
            value.minimum_validation_endpoints_per_axis
        ),
        "minimum_validation_components_per_axis": (
            value.minimum_validation_components_per_axis
        ),
        "validation_pair_lower_fraction": _fraction_payload(
            value.validation_pair_lower_fraction
        ),
        "validation_pair_upper_fraction": _fraction_payload(
            value.validation_pair_upper_fraction
        ),
        "minimum_training_pair_fraction": _fraction_payload(
            value.minimum_training_pair_fraction
        ),
        "minimum_training_endpoint_fraction": _fraction_payload(
            value.minimum_training_endpoint_fraction
        ),
        "maximum_quarantine_pair_fraction": _fraction_payload(
            value.maximum_quarantine_pair_fraction
        ),
    }


@dataclass(frozen=True)
class FeatureComparison:
    """One feature's exact/decimal comparison and fail-closed decision."""

    feature_name: str
    left_count: int
    right_count: int
    nonmissing_left_count: int
    nonmissing_right_count: int
    smd: Decimal | None
    tv: Fraction
    ks: Fraction | None
    quantile_cuts: tuple[Decimal, ...]
    optional_missing_bin: bool
    categorical_binary: bool
    histogram_informative: bool
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "feature_name": self.feature_name,
            "left_count": self.left_count,
            "right_count": self.right_count,
            "nonmissing_left_count": self.nonmissing_left_count,
            "nonmissing_right_count": self.nonmissing_right_count,
            "smd_decimal": None if self.smd is None else str(self.smd),
            "tv_exact": _fraction_payload(self.tv),
            "ks_exact": _fraction_payload(self.ks),
            "quantile_cuts_decimal": [str(value) for value in self.quantile_cuts],
            "optional_missing_bin": self.optional_missing_bin,
            "categorical_binary": self.categorical_binary,
            "histogram_informative": self.histogram_informative,
            "failures": list(self.failures),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RepresentativenessComparison:
    """All locked covariate comparisons for one axis and reference set."""

    axis: str
    reference_name: str
    features: tuple[FeatureComparison, ...]

    @property
    def passed(self) -> bool:
        return all(feature.passed for feature in self.features)

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "axis": self.axis,
            "reference_name": self.reference_name,
            "features": [feature.canonical_payload() for feature in self.features],
            "passed": self.passed,
        }


def compare_representativeness(
    left_profiles: Sequence[EntityProfile],
    right_profiles: Sequence[EntityProfile],
    *,
    universe_profiles: Sequence[EntityProfile],
    thresholds: DistributionThresholds,
    reference_name: str,
) -> RepresentativenessComparison:
    """Compare two effective endpoint sets using universe-fixed quintile bins."""
    if not isinstance(thresholds, DistributionThresholds):
        raise ScoreFreeAuditError("thresholds must be DistributionThresholds")
    _validate_identifier(reference_name, "reference_name")
    universe = _validate_profile_collection(universe_profiles, expected_axis=None)
    axis = universe[0].axis
    left = _validate_profile_collection(left_profiles, expected_axis=axis)
    right = _validate_profile_collection(right_profiles, expected_axis=axis)
    universe_ids = {profile.entity_id for profile in universe}
    for group_name, group in (("left", left), ("right", right)):
        unknown = sorted(
            profile.entity_id for profile in group if profile.entity_id not in universe_ids
        )
        if unknown:
            raise ScoreFreeAuditError(
                f"{group_name} profiles are outside the quantile universe: {unknown[0]}"
            )
    schemes = {
        scheme.feature_name: scheme for scheme in build_quantile_schemes(universe)
    }
    comparisons = []
    for spec in _SPECS_BY_AXIS[axis]:
        scheme = schemes[spec.name]
        left_values = [
            value
            for profile in left
            if (value := profile.value(spec.name)) is not None
        ]
        right_values = [
            value
            for profile in right
            if (value := profile.value(spec.name)) is not None
        ]
        failures: list[str] = []
        smd: Decimal | None = None
        ks: Fraction | None = None
        if spec.gate_smd:
            smd, smd_failure = _standardized_mean_difference(
                left_values, right_values
            )
            if smd_failure is not None:
                failures.append(smd_failure)
            elif smd is not None and smd > thresholds.maximum_smd:
                failures.append("smd_exceeds_locked_threshold")
        if spec.gate_ks:
            if not left_values or not right_values:
                failures.append("ks_requires_nonmissing_values_in_both_groups")
            else:
                ks = _exact_ks(left_values, right_values)
                if ks > thresholds.maximum_ks:
                    failures.append("ks_exceeds_locked_threshold")
        tv = _exact_tv(left, right, scheme)
        if spec.gate_tv and scheme.informative and tv > thresholds.maximum_tv:
            failures.append("tv_exceeds_locked_threshold")
        comparisons.append(
            FeatureComparison(
                feature_name=spec.name,
                left_count=len(left),
                right_count=len(right),
                nonmissing_left_count=len(left_values),
                nonmissing_right_count=len(right_values),
                smd=smd,
                tv=tv,
                ks=ks,
                quantile_cuts=scheme.cuts,
                optional_missing_bin=scheme.optional_missing_bin,
                categorical_binary=scheme.categorical_binary,
                histogram_informative=scheme.informative,
                failures=tuple(failures),
            )
        )
    return RepresentativenessComparison(
        axis=axis,
        reference_name=reference_name,
        features=tuple(comparisons),
    )


def exact_finite_library_chance_mrr(
    library_size: int,
    positive_count: int,
) -> Fraction:
    """Return the exact expected reciprocal rank under a random permutation."""
    library_size = _positive_integer(library_size, "library_size")
    positive_count = _positive_integer(positive_count, "positive_count")
    if positive_count > library_size:
        raise ScoreFreeAuditError("positive_count cannot exceed library_size")
    denominator = comb(library_size, positive_count)
    return sum(
        Fraction(comb(library_size - rank, positive_count - 1), denominator * rank)
        for rank in range(1, library_size - positive_count + 2)
    )


def exact_relative_range(values: Sequence[int | Fraction]) -> Fraction:
    """Return ``(max-min)/mean`` exactly for a non-empty non-negative vector."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ScoreFreeAuditError("relative-range values must be a sequence")
    exact = tuple(Fraction(value) for value in values)
    if not exact:
        raise ScoreFreeAuditError("relative-range values cannot be empty")
    if any(value < 0 for value in exact):
        raise ScoreFreeAuditError("relative-range values must be non-negative")
    total = sum(exact, Fraction(0))
    if total == 0:
        return Fraction(0)
    return Fraction(len(exact)) * (max(exact) - min(exact)) / total


@dataclass(frozen=True)
class FoldPanelShape:
    """Exact score-free panel statistics for one fold."""

    fold_id: str
    universe_pair_count: int
    universe_antibody_count: int
    universe_antigen_count: int
    validation_pair_count: int
    validation_antibody_count: int
    validation_antigen_count: int
    validation_antibody_component_count: int
    validation_antigen_component_count: int
    training_pair_count: int
    training_antibody_count: int
    training_antigen_count: int
    quarantine_pair_count: int
    antibody_singleton_fraction: Fraction
    antigen_singleton_fraction: Fraction
    bipartite_density: Fraction
    antibody_to_antigen_chance_mrr: Fraction
    antigen_to_antibody_chance_mrr: Fraction
    bidirectional_chance_mrr: Fraction
    shape_failures: tuple[str, ...]

    @property
    def shape_passed(self) -> bool:
        return not self.shape_failures


@dataclass(frozen=True)
class FoldScoreFreeAudit:
    """One fold's shape and two-reference representativeness audit."""

    panel: FoldPanelShape
    antibody_vs_universe: RepresentativenessComparison
    antibody_vs_training: RepresentativenessComparison
    antigen_vs_universe: RepresentativenessComparison
    antigen_vs_training: RepresentativenessComparison
    shape_thresholds: FoldShapeThresholds
    distribution_thresholds: DistributionThresholds
    universe_pair_ids: tuple[str, ...]
    validation_pair_ids: tuple[str, ...]
    training_pair_ids: tuple[str, ...]
    validation_antibody_ids: tuple[str, ...]
    validation_antigen_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.shape_thresholds, FoldShapeThresholds):
            raise ScoreFreeAuditError("shape_thresholds has an invalid type")
        if not isinstance(self.distribution_thresholds, DistributionThresholds):
            raise ScoreFreeAuditError("distribution_thresholds has an invalid type")
        for name in (
            "universe_pair_ids",
            "validation_pair_ids",
            "training_pair_ids",
            "validation_antibody_ids",
            "validation_antigen_ids",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or values != tuple(sorted(set(values))):
                raise ScoreFreeAuditError(f"{name} must be a sorted unique tuple")
        universe = frozenset(self.universe_pair_ids)
        validation = frozenset(self.validation_pair_ids)
        training = frozenset(self.training_pair_ids)
        if not validation <= universe or not training <= universe:
            raise ScoreFreeAuditError("selected pair IDs must be inside the universe")
        if validation & training:
            raise ScoreFreeAuditError("audit memberships overlap")
        if len(universe) != self.panel.universe_pair_count:
            raise ScoreFreeAuditError("universe membership count disagrees with panel")
        if len(validation) != self.panel.validation_pair_count:
            raise ScoreFreeAuditError("validation membership count disagrees with panel")
        if len(training) != self.panel.training_pair_count:
            raise ScoreFreeAuditError("training membership count disagrees with panel")
        if len(self.validation_antibody_ids) != self.panel.validation_antibody_count:
            raise ScoreFreeAuditError("validation antibody count disagrees with panel")
        if len(self.validation_antigen_ids) != self.panel.validation_antigen_count:
            raise ScoreFreeAuditError("validation antigen count disagrees with panel")

    @property
    def passed(self) -> bool:
        return self.panel.shape_passed and all(
            comparison.passed
            for comparison in (
                self.antibody_vs_universe,
                self.antibody_vs_training,
                self.antigen_vs_universe,
                self.antigen_vs_training,
            )
        )

    def canonical_payload(self) -> Mapping[str, object]:
        panel = self.panel
        return {
            "schema_version": "score_free_fold_audit_v1",
            "fold_id": panel.fold_id,
            "shape_thresholds": _shape_thresholds_payload(self.shape_thresholds),
            "distribution_thresholds": _distribution_thresholds_payload(
                self.distribution_thresholds
            ),
            "panel": {
                "universe_pair_count": panel.universe_pair_count,
                "universe_antibody_count": panel.universe_antibody_count,
                "universe_antigen_count": panel.universe_antigen_count,
                "validation_pair_count": panel.validation_pair_count,
                "validation_antibody_count": panel.validation_antibody_count,
                "validation_antigen_count": panel.validation_antigen_count,
                "validation_antibody_component_count": (
                    panel.validation_antibody_component_count
                ),
                "validation_antigen_component_count": (
                    panel.validation_antigen_component_count
                ),
                "training_pair_count": panel.training_pair_count,
                "training_antibody_count": panel.training_antibody_count,
                "training_antigen_count": panel.training_antigen_count,
                "quarantine_pair_count": panel.quarantine_pair_count,
                "antibody_singleton_fraction": _fraction_payload(
                    panel.antibody_singleton_fraction
                ),
                "antigen_singleton_fraction": _fraction_payload(
                    panel.antigen_singleton_fraction
                ),
                "bipartite_density": _fraction_payload(panel.bipartite_density),
                "antibody_to_antigen_chance_mrr": _fraction_payload(
                    panel.antibody_to_antigen_chance_mrr
                ),
                "antigen_to_antibody_chance_mrr": _fraction_payload(
                    panel.antigen_to_antibody_chance_mrr
                ),
                "bidirectional_chance_mrr": _fraction_payload(
                    panel.bidirectional_chance_mrr
                ),
                "shape_failures": list(panel.shape_failures),
                "shape_passed": panel.shape_passed,
            },
            "representativeness": {
                "antibody_vs_universe": self.antibody_vs_universe.canonical_payload(),
                "antibody_vs_training": self.antibody_vs_training.canonical_payload(),
                "antigen_vs_universe": self.antigen_vs_universe.canonical_payload(),
                "antigen_vs_training": self.antigen_vs_training.canonical_payload(),
            },
            "universe_pair_ids": list(self.universe_pair_ids),
            "validation_pair_ids": list(self.validation_pair_ids),
            "training_pair_ids": list(self.training_pair_ids),
            "validation_antibody_ids": list(self.validation_antibody_ids),
            "validation_antigen_ids": list(self.validation_antigen_ids),
            "passed": self.passed,
        }

    @property
    def audit_sha256(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def _validated_pair_inventory(
    pairs: Sequence[ScoreFreePair],
) -> tuple[ScoreFreePair, ...]:
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise ScoreFreeAuditError("pairs must be a non-string sequence")
    result = tuple(pairs)
    if not result or not all(isinstance(pair, ScoreFreePair) for pair in result):
        raise ScoreFreeAuditError("pairs must contain ScoreFreePair values")
    pair_ids = [pair.pair_id for pair in result]
    endpoint_pairs = [(pair.antibody_id, pair.antigen_id) for pair in result]
    if len(set(pair_ids)) != len(pair_ids):
        raise ScoreFreeAuditError("pair identifiers must be unique")
    if len(set(endpoint_pairs)) != len(endpoint_pairs):
        raise ScoreFreeAuditError("exact endpoint pairs must already be collapsed")
    return tuple(sorted(result, key=lambda pair: pair.pair_id))


def _validate_selected_ids(
    values: Iterable[str],
    *,
    name: str,
    allowed: frozenset[str],
) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise ScoreFreeAuditError(f"{name} cannot be a string")
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ScoreFreeAuditError(f"{name} cannot contain duplicates")
    for value in materialized:
        _validate_identifier(value, name)
    unknown = sorted(set(materialized) - allowed)
    if unknown:
        raise ScoreFreeAuditError(f"{name} contains unknown pair ID {unknown[0]!r}")
    if not materialized:
        raise ScoreFreeAuditError(f"{name} cannot be empty")
    return frozenset(materialized)


def _profile_map(
    profiles: Sequence[EntityProfile],
    *,
    axis: str,
    required_ids: frozenset[str],
) -> dict[str, EntityProfile]:
    checked = _validate_profile_collection(profiles, expected_axis=axis)
    result = {profile.entity_id: profile for profile in checked}
    if frozenset(result) != required_ids:
        missing = sorted(required_ids - result.keys())
        extra = sorted(result.keys() - required_ids)
        detail = missing[0] if missing else extra[0]
        raise ScoreFreeAuditError(
            f"{axis} profiles must exactly cover universe endpoints: {detail}"
        )
    return result


def _panel_shape(
    *,
    fold_id: str,
    pairs: tuple[ScoreFreePair, ...],
    validation_ids: frozenset[str],
    training_ids: frozenset[str],
    antibody_profiles: Mapping[str, EntityProfile],
    antigen_profiles: Mapping[str, EntityProfile],
    thresholds: FoldShapeThresholds,
) -> FoldPanelShape:
    validation_pairs = [pair for pair in pairs if pair.pair_id in validation_ids]
    training_pairs = [pair for pair in pairs if pair.pair_id in training_ids]
    universe_antibodies = {pair.antibody_id for pair in pairs}
    universe_antigens = {pair.antigen_id for pair in pairs}
    validation_antibodies = {pair.antibody_id for pair in validation_pairs}
    validation_antigens = {pair.antigen_id for pair in validation_pairs}
    training_antibodies = {pair.antibody_id for pair in training_pairs}
    training_antigens = {pair.antigen_id for pair in training_pairs}
    antibody_degrees = Counter(pair.antibody_id for pair in validation_pairs)
    antigen_degrees = Counter(pair.antigen_id for pair in validation_pairs)
    antibody_singleton = Fraction(
        sum(degree == 1 for degree in antibody_degrees.values()),
        len(validation_antibodies),
    )
    antigen_singleton = Fraction(
        sum(degree == 1 for degree in antigen_degrees.values()),
        len(validation_antigens),
    )
    density = Fraction(
        len(validation_pairs),
        len(validation_antibodies) * len(validation_antigens),
    )
    antibody_chance = sum(
        (
            exact_finite_library_chance_mrr(len(validation_antigens), degree)
            for degree in antibody_degrees.values()
        ),
        Fraction(0),
    ) / len(validation_antibodies)
    antigen_chance = sum(
        (
            exact_finite_library_chance_mrr(len(validation_antibodies), degree)
            for degree in antigen_degrees.values()
        ),
        Fraction(0),
    ) / len(validation_antigens)
    pair_count = len(pairs)
    lower = max(
        thresholds.minimum_validation_pairs,
        _ceil_fraction(thresholds.validation_pair_lower_fraction * pair_count),
    )
    upper = _floor_fraction(
        thresholds.validation_pair_upper_fraction * pair_count
    )
    failures = []
    if not lower <= len(validation_pairs) <= upper:
        failures.append("validation_pair_count_outside_locked_interval")
    if len(validation_antibodies) < thresholds.minimum_validation_endpoints_per_axis:
        failures.append("validation_antibody_count_below_locked_minimum")
    if len(validation_antigens) < thresholds.minimum_validation_endpoints_per_axis:
        failures.append("validation_antigen_count_below_locked_minimum")
    antibody_component_count = len(
        {antibody_profiles[item].component_id for item in validation_antibodies}
    )
    antigen_component_count = len(
        {antigen_profiles[item].component_id for item in validation_antigens}
    )
    if antibody_component_count < thresholds.minimum_validation_components_per_axis:
        failures.append("validation_antibody_component_count_below_locked_minimum")
    if antigen_component_count < thresholds.minimum_validation_components_per_axis:
        failures.append("validation_antigen_component_count_below_locked_minimum")
    if len(training_pairs) < _ceil_fraction(
        thresholds.minimum_training_pair_fraction * pair_count
    ):
        failures.append("training_pair_retention_below_locked_minimum")
    if len(training_antibodies) < _ceil_fraction(
        thresholds.minimum_training_endpoint_fraction * len(universe_antibodies)
    ):
        failures.append("training_antibody_retention_below_locked_minimum")
    if len(training_antigens) < _ceil_fraction(
        thresholds.minimum_training_endpoint_fraction * len(universe_antigens)
    ):
        failures.append("training_antigen_retention_below_locked_minimum")
    quarantine_count = pair_count - len(validation_pairs) - len(training_pairs)
    if quarantine_count > _floor_fraction(
        thresholds.maximum_quarantine_pair_fraction * pair_count
    ):
        failures.append("quarantine_pair_count_above_locked_maximum")
    return FoldPanelShape(
        fold_id=fold_id,
        universe_pair_count=pair_count,
        universe_antibody_count=len(universe_antibodies),
        universe_antigen_count=len(universe_antigens),
        validation_pair_count=len(validation_pairs),
        validation_antibody_count=len(validation_antibodies),
        validation_antigen_count=len(validation_antigens),
        validation_antibody_component_count=antibody_component_count,
        validation_antigen_component_count=antigen_component_count,
        training_pair_count=len(training_pairs),
        training_antibody_count=len(training_antibodies),
        training_antigen_count=len(training_antigens),
        quarantine_pair_count=quarantine_count,
        antibody_singleton_fraction=antibody_singleton,
        antigen_singleton_fraction=antigen_singleton,
        bipartite_density=density,
        antibody_to_antigen_chance_mrr=antibody_chance,
        antigen_to_antibody_chance_mrr=antigen_chance,
        bidirectional_chance_mrr=(antibody_chance + antigen_chance) / 2,
        shape_failures=tuple(failures),
    )


def audit_score_free_fold(
    *,
    fold_id: str,
    pairs: Sequence[ScoreFreePair],
    antibody_profiles: Sequence[EntityProfile],
    antigen_profiles: Sequence[EntityProfile],
    validation_pair_ids: Iterable[str],
    training_pair_ids: Iterable[str],
    shape_thresholds: FoldShapeThresholds,
    distribution_thresholds: DistributionThresholds,
) -> FoldScoreFreeAudit:
    """Audit one already-scoped outer or inner universe without model scores."""
    _validate_identifier(fold_id, "fold_id")
    if not isinstance(shape_thresholds, FoldShapeThresholds):
        raise ScoreFreeAuditError("shape_thresholds must be FoldShapeThresholds")
    if not isinstance(distribution_thresholds, DistributionThresholds):
        raise ScoreFreeAuditError(
            "distribution_thresholds must be DistributionThresholds"
        )
    inventory = _validated_pair_inventory(pairs)
    all_pair_ids = frozenset(pair.pair_id for pair in inventory)
    validation_ids = _validate_selected_ids(
        validation_pair_ids,
        name="validation_pair_ids",
        allowed=all_pair_ids,
    )
    training_ids = _validate_selected_ids(
        training_pair_ids,
        name="training_pair_ids",
        allowed=all_pair_ids,
    )
    overlap = validation_ids & training_ids
    if overlap:
        raise ScoreFreeAuditError(
            f"training and validation overlap at pair ID {min(overlap)!r}"
        )
    universe_antibodies = frozenset(pair.antibody_id for pair in inventory)
    universe_antigens = frozenset(pair.antigen_id for pair in inventory)
    antibody_by_id = _profile_map(
        antibody_profiles,
        axis="antibody",
        required_ids=universe_antibodies,
    )
    antigen_by_id = _profile_map(
        antigen_profiles,
        axis="antigen",
        required_ids=universe_antigens,
    )
    antibody_incidence = Counter(pair.antibody_id for pair in inventory)
    antigen_incidence = Counter(pair.antigen_id for pair in inventory)
    for entity_id, degree in antibody_incidence.items():
        if antibody_by_id[entity_id].value(
            "log1p_pair_incidence_degree"
        ) != _log1p_decimal(degree):
            raise ScoreFreeAuditError(
                f"Antibody {entity_id!r} pair-incidence degree is inconsistent"
            )
    for entity_id, degree in antigen_incidence.items():
        if antigen_by_id[entity_id].value(
            "log1p_pair_incidence_degree"
        ) != _log1p_decimal(degree):
            raise ScoreFreeAuditError(
                f"Antigen {entity_id!r} pair-incidence degree is inconsistent"
            )
    validation_pairs = [
        pair for pair in inventory if pair.pair_id in validation_ids
    ]
    training_pairs = [pair for pair in inventory if pair.pair_id in training_ids]
    validation_antibodies = tuple(
        antibody_by_id[item]
        for item in sorted({pair.antibody_id for pair in validation_pairs})
    )
    validation_antigens = tuple(
        antigen_by_id[item]
        for item in sorted({pair.antigen_id for pair in validation_pairs})
    )
    training_antibodies = tuple(
        antibody_by_id[item]
        for item in sorted({pair.antibody_id for pair in training_pairs})
    )
    training_antigens = tuple(
        antigen_by_id[item]
        for item in sorted({pair.antigen_id for pair in training_pairs})
    )
    all_antibodies = tuple(antibody_by_id.values())
    all_antigens = tuple(antigen_by_id.values())
    panel = _panel_shape(
        fold_id=fold_id,
        pairs=inventory,
        validation_ids=validation_ids,
        training_ids=training_ids,
        antibody_profiles=antibody_by_id,
        antigen_profiles=antigen_by_id,
        thresholds=shape_thresholds,
    )
    return FoldScoreFreeAudit(
        panel=panel,
        antibody_vs_universe=compare_representativeness(
            validation_antibodies,
            all_antibodies,
            universe_profiles=all_antibodies,
            thresholds=distribution_thresholds,
            reference_name="universe",
        ),
        antibody_vs_training=compare_representativeness(
            validation_antibodies,
            training_antibodies,
            universe_profiles=all_antibodies,
            thresholds=distribution_thresholds,
            reference_name="training",
        ),
        antigen_vs_universe=compare_representativeness(
            validation_antigens,
            all_antigens,
            universe_profiles=all_antigens,
            thresholds=distribution_thresholds,
            reference_name="universe",
        ),
        antigen_vs_training=compare_representativeness(
            validation_antigens,
            training_antigens,
            universe_profiles=all_antigens,
            thresholds=distribution_thresholds,
            reference_name="training",
        ),
        shape_thresholds=shape_thresholds,
        distribution_thresholds=distribution_thresholds,
        universe_pair_ids=tuple(pair.pair_id for pair in inventory),
        validation_pair_ids=tuple(sorted(validation_ids)),
        training_pair_ids=tuple(sorted(training_ids)),
        validation_antibody_ids=tuple(
            profile.entity_id for profile in validation_antibodies
        ),
        validation_antigen_ids=tuple(
            profile.entity_id for profile in validation_antigens
        ),
    )


@dataclass(frozen=True)
class CrossFoldBalanceAudit:
    """Exact relative-range results across exactly three fold panels."""

    thresholds: CrossFoldThresholds
    relative_ranges: tuple[tuple[str, Fraction], ...]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "schema_version": "score_free_cross_fold_balance_v1",
            "thresholds": {
                "maximum_validation_pair_relative_range": _fraction_payload(
                    self.thresholds.maximum_validation_pair_relative_range
                ),
                "maximum_other_relative_range": _fraction_payload(
                    self.thresholds.maximum_other_relative_range
                ),
            },
            "relative_ranges": {
                name: _fraction_payload(value)
                for name, value in self.relative_ranges
            },
            "failures": list(self.failures),
            "passed": self.passed,
        }


def audit_cross_fold_balance(
    audits: Sequence[FoldScoreFreeAudit],
    *,
    thresholds: CrossFoldThresholds,
) -> CrossFoldBalanceAudit:
    """Audit the prospectively locked cross-fold balance metrics exactly."""
    if not isinstance(thresholds, CrossFoldThresholds):
        raise ScoreFreeAuditError("thresholds must be CrossFoldThresholds")
    if isinstance(audits, (str, bytes)) or not isinstance(audits, Sequence):
        raise ScoreFreeAuditError("audits must be a sequence")
    values = tuple(audits)
    if len(values) != 3 or not all(
        isinstance(value, FoldScoreFreeAudit) for value in values
    ):
        raise ScoreFreeAuditError("cross-fold balance requires exactly three audits")
    fold_ids = [value.panel.fold_id for value in values]
    if len(set(fold_ids)) != 3:
        raise ScoreFreeAuditError("cross-fold audit IDs must be unique")
    panels = [value.panel for value in sorted(values, key=lambda value: value.panel.fold_id)]
    universe_shapes = {
        (
            panel.universe_pair_count,
            panel.universe_antibody_count,
            panel.universe_antigen_count,
        )
        for panel in panels
    }
    universe_memberships = {value.universe_pair_ids for value in values}
    if len(universe_shapes) != 1 or len(universe_memberships) != 1:
        raise ScoreFreeAuditError(
            "cross-fold audits must share one exact universe membership"
        )
    metric_values: dict[str, Sequence[int | Fraction]] = {
        "validation_pair_count": [panel.validation_pair_count for panel in panels],
        "validation_antibody_count": [
            panel.validation_antibody_count for panel in panels
        ],
        "validation_antigen_count": [
            panel.validation_antigen_count for panel in panels
        ],
        "training_pair_retention": [
            Fraction(panel.training_pair_count, panel.universe_pair_count)
            for panel in panels
        ],
        "quarantine_pair_count": [panel.quarantine_pair_count for panel in panels],
        "antibody_singleton_fraction": [
            panel.antibody_singleton_fraction for panel in panels
        ],
        "antigen_singleton_fraction": [
            panel.antigen_singleton_fraction for panel in panels
        ],
        "bipartite_density": [panel.bipartite_density for panel in panels],
        "bidirectional_chance_mrr": [
            panel.bidirectional_chance_mrr for panel in panels
        ],
    }
    ranges = tuple(
        (name, exact_relative_range(metric_values[name]))
        for name in (
            "validation_pair_count",
            "validation_antibody_count",
            "validation_antigen_count",
            "training_pair_retention",
            "quarantine_pair_count",
            "antibody_singleton_fraction",
            "antigen_singleton_fraction",
            "bipartite_density",
            "bidirectional_chance_mrr",
        )
    )
    failures = []
    for name, value in ranges:
        maximum = (
            thresholds.maximum_validation_pair_relative_range
            if name == "validation_pair_count"
            else thresholds.maximum_other_relative_range
        )
        if value > maximum:
            failures.append(f"{name}_relative_range_exceeds_locked_threshold")
    return CrossFoldBalanceAudit(
        thresholds=thresholds,
        relative_ranges=ranges,
        failures=tuple(failures),
    )


@dataclass(frozen=True)
class PooledRepresentativenessAudit:
    """Pooled entity-disjoint outer validation versus full-pool comparisons."""

    thresholds: DistributionThresholds
    antibody_vs_universe: RepresentativenessComparison
    antigen_vs_universe: RepresentativenessComparison
    pooled_antibody_ids: tuple[str, ...]
    pooled_antigen_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.antibody_vs_universe.passed and self.antigen_vs_universe.passed

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "schema_version": "score_free_pooled_outer_audit_v1",
            "thresholds": _distribution_thresholds_payload(self.thresholds),
            "antibody_vs_universe": (
                self.antibody_vs_universe.canonical_payload()
            ),
            "antigen_vs_universe": self.antigen_vs_universe.canonical_payload(),
            "pooled_antibody_ids": list(self.pooled_antibody_ids),
            "pooled_antigen_ids": list(self.pooled_antigen_ids),
            "passed": self.passed,
        }

    @property
    def audit_sha256(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


def audit_pooled_outer_representativeness(
    audits: Sequence[FoldScoreFreeAudit],
    *,
    antibody_profiles: Sequence[EntityProfile],
    antigen_profiles: Sequence[EntityProfile],
    thresholds: DistributionThresholds = POOLED_OUTER_DISTRIBUTION_THRESHOLDS,
) -> PooledRepresentativenessAudit:
    """Require disjoint outer endpoints before computing the pooled hard gate."""
    if not isinstance(thresholds, DistributionThresholds):
        raise ScoreFreeAuditError("thresholds must be DistributionThresholds")
    if len(audits) != 3:
        raise ScoreFreeAuditError("pooled outer audit requires exactly three folds")
    if len({audit.panel.fold_id for audit in audits}) != 3:
        raise ScoreFreeAuditError("pooled outer fold IDs must be unique")
    if len({audit.universe_pair_ids for audit in audits}) != 1:
        raise ScoreFreeAuditError(
            "pooled outer folds must share one exact universe membership"
        )
    full_antibodies = _validate_profile_collection(
        antibody_profiles, expected_axis="antibody"
    )
    full_antigens = _validate_profile_collection(
        antigen_profiles, expected_axis="antigen"
    )
    antibody_by_id = {profile.entity_id: profile for profile in full_antibodies}
    antigen_by_id = {profile.entity_id: profile for profile in full_antigens}
    seen_antibodies: set[str] = set()
    seen_antigens: set[str] = set()
    for audit in audits:
        antibody_overlap = seen_antibodies & set(audit.validation_antibody_ids)
        antigen_overlap = seen_antigens & set(audit.validation_antigen_ids)
        if antibody_overlap:
            raise ScoreFreeAuditError(
                "outer validation antibody endpoints are not entity-disjoint"
            )
        if antigen_overlap:
            raise ScoreFreeAuditError(
                "outer validation antigen endpoints are not entity-disjoint"
            )
        seen_antibodies.update(audit.validation_antibody_ids)
        seen_antigens.update(audit.validation_antigen_ids)
    unknown_antibodies = seen_antibodies - antibody_by_id.keys()
    unknown_antigens = seen_antigens - antigen_by_id.keys()
    if unknown_antibodies or unknown_antigens:
        raise ScoreFreeAuditError("pooled audit references an unknown endpoint")
    pooled_antibodies = tuple(antibody_by_id[item] for item in sorted(seen_antibodies))
    pooled_antigens = tuple(antigen_by_id[item] for item in sorted(seen_antigens))
    return PooledRepresentativenessAudit(
        thresholds=thresholds,
        antibody_vs_universe=compare_representativeness(
            pooled_antibodies,
            full_antibodies,
            universe_profiles=full_antibodies,
            thresholds=thresholds,
            reference_name="full_authority_pool",
        ),
        antigen_vs_universe=compare_representativeness(
            pooled_antigens,
            full_antigens,
            universe_profiles=full_antigens,
            thresholds=thresholds,
            reference_name="full_authority_pool",
        ),
        pooled_antibody_ids=tuple(sorted(seen_antibodies)),
        pooled_antigen_ids=tuple(sorted(seen_antigens)),
    )
