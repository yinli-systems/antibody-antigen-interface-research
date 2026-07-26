"""Adversarial tests for the independent score-free split-statistics oracle."""

from __future__ import annotations

import itertools
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

from interaction_first.score_free_split_statistics import (
    AntibodyScoreFreeInputs,
    AntigenScoreFreeInputs,
    CrossFoldThresholds,
    DistributionThresholds,
    EntityProfile,
    FoldShapeThresholds,
    OUTER_CROSS_FOLD_THRESHOLDS,
    POOLED_OUTER_DISTRIBUTION_THRESHOLDS,
    ScoreFreeAuditError,
    ScoreFreePair,
    antibody_profile,
    antigen_profile,
    audit_cross_fold_balance,
    audit_pooled_outer_representativeness,
    audit_score_free_fold,
    build_quantile_schemes,
    compare_representativeness,
    exact_finite_library_chance_mrr,
    exact_relative_range,
)


ZERO_DISTRIBUTION_THRESHOLDS = DistributionThresholds(
    maximum_smd=Decimal(0),
    maximum_tv=Fraction(0),
    maximum_ks=Fraction(0),
)
SYNTHETIC_SHAPE_THRESHOLDS = FoldShapeThresholds(
    minimum_validation_pairs=4,
    minimum_validation_endpoints_per_axis=4,
    minimum_validation_components_per_axis=4,
    validation_pair_lower_fraction=Fraction(1, 3),
    validation_pair_upper_fraction=Fraction(1, 3),
    minimum_training_pair_fraction=Fraction(2, 3),
    minimum_training_endpoint_fraction=Fraction(2, 3),
    maximum_quarantine_pair_fraction=Fraction(0),
)


def _antibody(index: int, *, pair_degree: int = 1) -> EntityProfile:
    pattern = index % 4
    vl_length = None if pattern % 2 else 50 + pattern
    return antibody_profile(
        AntibodyScoreFreeInputs(
            entity_id=f"a{index:02d}",
            component_id=f"ac{index:02d}",
            vh_length=100 + 5 * pattern,
            vl_length=vl_length,
            concatenated_antibody_length=(
                100 + 5 * pattern + (0 if vl_length is None else vl_length)
            ),
            vh_direct_degree=pattern,
            vl_direct_degree=None if vl_length is None else pattern,
            concatenated_direct_degree=pattern,
            union_direct_degree=pattern,
            pair_incidence_degree=pair_degree,
            component_size=4,
        )
    )


def _antigen(index: int, *, pair_degree: int = 1) -> EntityProfile:
    pattern = index % 4
    return antigen_profile(
        AntigenScoreFreeInputs(
            entity_id=f"g{index:02d}",
            component_id=f"gc{index:02d}",
            antigen_length=200 + 10 * pattern,
            antigen_direct_degree=pattern,
            pair_incidence_degree=pair_degree,
            component_size=4,
        )
    )


def _fixture() -> tuple[
    tuple[ScoreFreePair, ...],
    tuple[EntityProfile, ...],
    tuple[EntityProfile, ...],
]:
    pairs = tuple(
        ScoreFreePair(f"p{index:02d}", f"a{index:02d}", f"g{index:02d}")
        for index in range(12)
    )
    return (
        pairs,
        tuple(_antibody(index) for index in range(12)),
        tuple(_antigen(index) for index in range(12)),
    )


def _audit(fold_index: int):
    pairs, antibodies, antigens = _fixture()
    validation = tuple(
        f"p{index:02d}" for index in range(4 * fold_index, 4 * fold_index + 4)
    )
    training = tuple(pair.pair_id for pair in pairs if pair.pair_id not in validation)
    return audit_score_free_fold(
        fold_id=f"outer_{fold_index}",
        pairs=pairs,
        antibody_profiles=antibodies,
        antigen_profiles=antigens,
        validation_pair_ids=validation,
        training_pair_ids=training,
        shape_thresholds=SYNTHETIC_SHAPE_THRESHOLDS,
        distribution_thresholds=ZERO_DISTRIBUTION_THRESHOLDS,
    )


def test_typed_inputs_bind_missing_vl_and_concatenated_length() -> None:
    with pytest.raises(ScoreFreeAuditError, match="must be missing"):
        AntibodyScoreFreeInputs(
            entity_id="a",
            component_id="a",
            vh_length=100,
            vl_length=None,
            concatenated_antibody_length=100,
            vh_direct_degree=0,
            vl_direct_degree=0,
            concatenated_direct_degree=0,
            union_direct_degree=0,
            pair_incidence_degree=1,
            component_size=1,
        )


def test_direct_degrees_must_be_coherent_with_union_and_component_size() -> None:
    common = dict(
        entity_id="a",
        component_id="a",
        vh_length=100,
        vl_length=50,
        concatenated_antibody_length=150,
        vh_direct_degree=2,
        vl_direct_degree=0,
        concatenated_direct_degree=0,
        pair_incidence_degree=1,
        component_size=3,
    )
    with pytest.raises(ScoreFreeAuditError, match="constituent view"):
        AntibodyScoreFreeInputs(**common, union_direct_degree=1)
    with pytest.raises(ScoreFreeAuditError, match="smaller than component_size"):
        AntibodyScoreFreeInputs(**common, union_direct_degree=3)
    with pytest.raises(ScoreFreeAuditError, match="smaller than component_size"):
        AntigenScoreFreeInputs(
            entity_id="g",
            component_id="g",
            antigen_length=200,
            antigen_direct_degree=2,
            pair_incidence_degree=1,
            component_size=2,
        )
    with pytest.raises(ScoreFreeAuditError, match="must equal"):
        AntibodyScoreFreeInputs(
            entity_id="a",
            component_id="a",
            vh_length=100,
            vl_length=50,
            concatenated_antibody_length=149,
            vh_direct_degree=0,
            vl_direct_degree=0,
            concatenated_direct_degree=0,
            union_direct_degree=0,
            pair_incidence_degree=1,
            component_size=1,
        )


def test_profile_schema_rejects_outcome_or_model_derived_field_injection() -> None:
    profile = _antibody(0)
    with pytest.raises(ScoreFreeAuditError, match="exact allowed feature schema"):
        replace(
            profile,
            values=profile.values + (("epitope_size", Decimal(7)),),
        )
    with pytest.raises(ScoreFreeAuditError, match="exact allowed feature schema"):
        replace(
            profile,
            values=profile.values + (("model_score", Decimal("0.99")),),
        )


def test_type1_quintiles_merge_ties_and_keep_cut_equality_in_lower_bin() -> None:
    profiles = tuple(
        antigen_profile(
            AntigenScoreFreeInputs(
                entity_id=f"g{index}",
                component_id=f"c{index}",
                antigen_length=value,
                antigen_direct_degree=0,
                pair_incidence_degree=1,
                component_size=1,
            )
        )
        for index, value in enumerate((1, 1, 2, 3, 4))
    )
    scheme = next(
        item
        for item in build_quantile_schemes(profiles)
        if item.feature_name == "antigen_length"
    )
    assert scheme.cuts == (Decimal(1), Decimal(2), Decimal(3))
    assert scheme.bin_index(Decimal(1)) == 0
    assert scheme.bin_index(Decimal(2)) == 1
    assert scheme.bin_index(Decimal(4)) == 3
    assert build_quantile_schemes(tuple(reversed(profiles))) == (
        build_quantile_schemes(profiles)
    )


def test_component_membership_cannot_declare_inconsistent_sizes() -> None:
    first = antigen_profile(
        AntigenScoreFreeInputs(
            entity_id="g0",
            component_id="component",
            antigen_length=100,
            antigen_direct_degree=0,
            pair_incidence_degree=1,
            component_size=2,
        )
    )
    second = antigen_profile(
        AntigenScoreFreeInputs(
            entity_id="g1",
            component_id="component",
            antigen_length=100,
            antigen_direct_degree=0,
            pair_incidence_degree=1,
            component_size=3,
        )
    )
    with pytest.raises(ScoreFreeAuditError, match="inconsistent declared sizes"):
        build_quantile_schemes((first, second))


def test_optional_vl_histogram_has_an_explicit_missing_bin() -> None:
    profiles = tuple(_antibody(index) for index in range(4))
    scheme = next(
        item
        for item in build_quantile_schemes(profiles)
        if item.feature_name == "vl_length"
    )
    assert scheme.optional_missing_bin
    assert scheme.bin_index(None) == 0
    assert scheme.bin_index(Decimal(50)) >= 1


def test_smd_tv_and_ks_pass_at_exact_locked_boundaries() -> None:
    profiles = tuple(
        antigen_profile(
            AntigenScoreFreeInputs(
                entity_id=f"g{index}",
                component_id=f"c{index}",
                antigen_length=value,
                antigen_direct_degree=0,
                pair_incidence_degree=1,
                component_size=1,
            )
        )
        for index, value in enumerate((10, 10, 10, 20))
    )
    comparison = compare_representativeness(
        profiles[:2],
        profiles[2:],
        universe_profiles=profiles,
        thresholds=DistributionThresholds(
            maximum_smd=Decimal(1),
            maximum_tv=Fraction(1, 2),
            maximum_ks=Fraction(1, 2),
        ),
        reference_name="boundary_fixture",
    )
    length = next(
        item for item in comparison.features if item.feature_name == "antigen_length"
    )
    assert length.smd == Decimal(1)
    assert length.tv == Fraction(1, 2)
    assert length.ks == Fraction(1, 2)
    assert length.passed
    assert comparison.passed


def test_each_distribution_threshold_fails_closed_above_boundary() -> None:
    profiles = tuple(
        antigen_profile(
            AntigenScoreFreeInputs(
                entity_id=f"g{index}",
                component_id=f"c{index}",
                antigen_length=value,
                antigen_direct_degree=0,
                pair_incidence_degree=1,
                component_size=1,
            )
        )
        for index, value in enumerate((10, 10, 10, 20))
    )
    comparison = compare_representativeness(
        profiles[:2],
        profiles[2:],
        universe_profiles=profiles,
        thresholds=DistributionThresholds(
            maximum_smd=Decimal("0.99"),
            maximum_tv=Fraction(49, 100),
            maximum_ks=Fraction(49, 100),
        ),
        reference_name="strict_boundary_fixture",
    )
    length = next(
        item for item in comparison.features if item.feature_name == "antigen_length"
    )
    assert set(length.failures) == {
        "smd_exceeds_locked_threshold",
        "tv_exceeds_locked_threshold",
        "ks_exceeds_locked_threshold",
    }
    assert not comparison.passed


def test_smd_with_a_single_observation_is_never_silently_accepted() -> None:
    profiles = tuple(_antigen(index) for index in range(4))
    comparison = compare_representativeness(
        profiles[:1],
        profiles[1:],
        universe_profiles=profiles,
        thresholds=DistributionThresholds(
            maximum_smd=Decimal(1),
            maximum_tv=Fraction(1),
            maximum_ks=Fraction(1),
        ),
        reference_name="small_group",
    )
    assert any(
        "smd_requires_at_least_two_nonmissing_values_per_group" in item.failures
        for item in comparison.features
    )
    assert not comparison.passed


def test_exact_chance_mrr_matches_complete_rank_enumeration() -> None:
    observed = exact_finite_library_chance_mrr(5, 2)
    best_ranks = [
        min(positions) for positions in itertools.combinations(range(1, 6), 2)
    ]
    expected = sum((Fraction(1, rank) for rank in best_ranks), Fraction(0)) / len(
        best_ranks
    )
    assert observed == expected
    assert exact_finite_library_chance_mrr(4, 1) == Fraction(25, 48)
    with pytest.raises(ScoreFreeAuditError, match="cannot exceed"):
        exact_finite_library_chance_mrr(4, 5)


def test_relative_range_uses_exact_rational_arithmetic() -> None:
    assert exact_relative_range([1, 2, 3]) == Fraction(1)
    assert exact_relative_range([Fraction(1, 3)] * 3) == 0
    with pytest.raises(ScoreFreeAuditError, match="non-negative"):
        exact_relative_range([1, -1, 2])


def test_complete_fold_audit_passes_balanced_score_free_fixture() -> None:
    audit = _audit(0)
    assert audit.passed
    assert audit.panel.shape_passed
    assert audit.panel.validation_pair_count == 4
    assert audit.panel.training_pair_count == 8
    assert audit.panel.quarantine_pair_count == 0
    assert audit.panel.antibody_singleton_fraction == 1
    assert audit.panel.antigen_singleton_fraction == 1
    assert audit.panel.bipartite_density == Fraction(1, 4)
    assert audit.panel.bidirectional_chance_mrr == Fraction(25, 48)
    assert len(audit.audit_sha256) == 64


def test_fold_audit_is_byte_canonical_under_input_reordering() -> None:
    pairs, antibodies, antigens = _fixture()
    first = _audit(0)
    reordered = audit_score_free_fold(
        fold_id="outer_0",
        pairs=tuple(reversed(pairs)),
        antibody_profiles=tuple(reversed(antibodies)),
        antigen_profiles=tuple(reversed(antigens)),
        validation_pair_ids=tuple(reversed(first.validation_pair_ids)),
        training_pair_ids=tuple(
            reversed(
                [pair.pair_id for pair in pairs if pair.pair_id not in first.validation_pair_ids]
            )
        ),
        shape_thresholds=SYNTHETIC_SHAPE_THRESHOLDS,
        distribution_thresholds=ZERO_DISTRIBUTION_THRESHOLDS,
    )
    assert first.canonical_payload() == reordered.canonical_payload()
    assert first.audit_sha256 == reordered.audit_sha256


def test_fold_audit_digest_binds_thresholds_memberships_and_quantile_cuts() -> None:
    audit = _audit(0)
    changed_threshold = replace(
        audit,
        distribution_thresholds=replace(
            audit.distribution_thresholds,
            maximum_tv=Fraction(1, 100),
        ),
    )
    feature = audit.antigen_vs_universe.features[0]
    changed_feature = replace(
        feature,
        quantile_cuts=feature.quantile_cuts + (Decimal(999),),
    )
    changed_comparison = replace(
        audit.antigen_vs_universe,
        features=(changed_feature,) + audit.antigen_vs_universe.features[1:],
    )
    changed_cuts = replace(audit, antigen_vs_universe=changed_comparison)
    assert changed_threshold.audit_sha256 != audit.audit_sha256
    assert changed_cuts.audit_sha256 != audit.audit_sha256
    with pytest.raises(ScoreFreeAuditError, match="sorted unique tuple"):
        replace(
            audit,
            training_pair_ids=tuple(reversed(audit.training_pair_ids)),
        )


def test_pair_overlap_unknown_ids_and_duplicate_ids_fail_closed() -> None:
    pairs, antibodies, antigens = _fixture()
    common = dict(
        fold_id="outer_0",
        pairs=pairs,
        antibody_profiles=antibodies,
        antigen_profiles=antigens,
        shape_thresholds=SYNTHETIC_SHAPE_THRESHOLDS,
        distribution_thresholds=ZERO_DISTRIBUTION_THRESHOLDS,
    )
    with pytest.raises(ScoreFreeAuditError, match="overlap"):
        audit_score_free_fold(
            **common,
            validation_pair_ids=("p00",),
            training_pair_ids=("p00",),
        )
    with pytest.raises(ScoreFreeAuditError, match="unknown pair ID"):
        audit_score_free_fold(
            **common,
            validation_pair_ids=("unknown",),
            training_pair_ids=("p00",),
        )
    with pytest.raises(ScoreFreeAuditError, match="duplicates"):
        audit_score_free_fold(
            **common,
            validation_pair_ids=("p00", "p00"),
            training_pair_ids=("p01",),
        )
    duplicate_endpoint_pairs = pairs + (
        ScoreFreePair("new-pair-id", pairs[0].antibody_id, pairs[0].antigen_id),
    )
    with pytest.raises(ScoreFreeAuditError, match="already be collapsed"):
        audit_score_free_fold(
            **{**common, "pairs": duplicate_endpoint_pairs},
            validation_pair_ids=("p00",),
            training_pair_ids=("p01",),
        )


def test_pair_incidence_summary_tampering_is_detected() -> None:
    pairs, antibodies, antigens = _fixture()
    original = antibodies[0]
    values = tuple(
        (
            name,
            Decimal(2).ln()
            if name == "log1p_pair_incidence_degree"
            else value,
        )
        for name, value in original.values
    )
    tampered = (replace(original, values=values),) + antibodies[1:]
    with pytest.raises(ScoreFreeAuditError, match="incidence degree is inconsistent"):
        audit_score_free_fold(
            fold_id="outer_0",
            pairs=pairs,
            antibody_profiles=tampered,
            antigen_profiles=antigens,
            validation_pair_ids=("p00", "p01", "p02", "p03"),
            training_pair_ids=tuple(f"p{index:02d}" for index in range(4, 12)),
            shape_thresholds=SYNTHETIC_SHAPE_THRESHOLDS,
            distribution_thresholds=ZERO_DISTRIBUTION_THRESHOLDS,
        )


def test_shape_gate_failure_is_separate_from_distribution_pass() -> None:
    pairs, antibodies, antigens = _fixture()
    audit = audit_score_free_fold(
        fold_id="outer_0",
        pairs=pairs,
        antibody_profiles=antibodies,
        antigen_profiles=antigens,
        validation_pair_ids=("p00", "p01", "p02", "p03"),
        training_pair_ids=tuple(f"p{index:02d}" for index in range(4, 12)),
        shape_thresholds=replace(
            SYNTHETIC_SHAPE_THRESHOLDS,
            minimum_validation_pairs=5,
        ),
        distribution_thresholds=ZERO_DISTRIBUTION_THRESHOLDS,
    )
    assert not audit.panel.shape_passed
    assert audit.antibody_vs_universe.passed
    assert audit.antigen_vs_training.passed
    assert not audit.passed


def test_cross_fold_ranges_pass_equal_panels_and_reject_imbalance() -> None:
    audits = tuple(_audit(index) for index in range(3))
    balanced = audit_cross_fold_balance(
        audits,
        thresholds=OUTER_CROSS_FOLD_THRESHOLDS,
    )
    assert balanced.passed
    assert all(value == 0 for _, value in balanced.relative_ranges)
    pairs, antibodies, antigens = _fixture()
    imbalanced_validation = ("p00", "p08", "p09", "p10", "p11")
    changed_audit = audit_score_free_fold(
        fold_id="outer_2_imbalanced",
        pairs=pairs,
        antibody_profiles=antibodies,
        antigen_profiles=antigens,
        validation_pair_ids=imbalanced_validation,
        training_pair_ids=tuple(
            pair.pair_id for pair in pairs if pair.pair_id not in imbalanced_validation
        ),
        shape_thresholds=FoldShapeThresholds(
            minimum_validation_pairs=1,
            minimum_validation_endpoints_per_axis=1,
            minimum_validation_components_per_axis=1,
            validation_pair_lower_fraction=Fraction(0),
            validation_pair_upper_fraction=Fraction(1),
            minimum_training_pair_fraction=Fraction(0),
            minimum_training_endpoint_fraction=Fraction(0),
            maximum_quarantine_pair_fraction=Fraction(1),
        ),
        distribution_thresholds=DistributionThresholds(
            maximum_smd=Decimal(1),
            maximum_tv=Fraction(1),
            maximum_ks=Fraction(1),
        ),
    )
    imbalanced = audit_cross_fold_balance(
        (audits[0], audits[1], changed_audit),
        thresholds=CrossFoldThresholds(
            maximum_validation_pair_relative_range=Fraction(3, 20),
            maximum_other_relative_range=Fraction(1, 5),
        ),
    )
    assert not imbalanced.passed
    assert (
        "validation_pair_count_relative_range_exceeds_locked_threshold"
        in imbalanced.failures
    )


def test_cross_fold_balance_rejects_mixed_universe_memberships() -> None:
    audits = tuple(_audit(index) for index in range(3))
    replacement_universe = tuple(
        sorted(set(audits[0].universe_pair_ids) - {"p11"} | {"replacement"})
    )
    replacement_training = tuple(
        sorted(set(audits[0].training_pair_ids) - {"p11"} | {"replacement"})
    )
    tampered = replace(
        audits[0],
        universe_pair_ids=replacement_universe,
        training_pair_ids=replacement_training,
    )
    with pytest.raises(ScoreFreeAuditError, match="one exact universe"):
        audit_cross_fold_balance(
            (tampered, audits[1], audits[2]),
            thresholds=OUTER_CROSS_FOLD_THRESHOLDS,
        )


def test_pooled_outer_requires_entity_disjoint_folds_and_uses_full_pool() -> None:
    pairs, antibodies, antigens = _fixture()
    audits = tuple(_audit(index) for index in range(3))
    pooled = audit_pooled_outer_representativeness(
        audits,
        antibody_profiles=antibodies,
        antigen_profiles=antigens,
        thresholds=POOLED_OUTER_DISTRIBUTION_THRESHOLDS,
    )
    assert pooled.passed
    assert len(pooled.audit_sha256) == 64
    changed_thresholds = replace(
        pooled,
        thresholds=replace(pooled.thresholds, maximum_tv=Fraction(1, 10)),
    )
    assert changed_thresholds.audit_sha256 != pooled.audit_sha256
    assert len(pooled.pooled_antibody_ids) == 12
    assert len(pooled.pooled_antigen_ids) == 12
    duplicate_with_new_id = replace(
        audits[0], panel=replace(audits[0].panel, fold_id="outer_duplicate")
    )
    with pytest.raises(ScoreFreeAuditError, match="not entity-disjoint"):
        audit_pooled_outer_representativeness(
            (audits[0], duplicate_with_new_id, audits[1]),
            antibody_profiles=antibodies,
            antigen_profiles=antigens,
        )
