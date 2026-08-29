"""Contract tests for the staged single-variable decision engine."""

from __future__ import annotations

import json

import pytest

from interaction_first.decision_matrix import (
    ComparisonError,
    SeedRun,
    build_main_table,
    classify_precision_change,
    compare_arms,
    epoch_noise,
    evaluate_phase,
    exact_sign_test,
    extract_metrics,
    factor_difference,
    load_completed_run,
    load_live_log,
    matched_epoch,
    parse_epoch_records,
    recall_retention,
    require_single_variable,
    trajectory,
)
from interaction_first.decision_matrix import DECISION_CHAIN


def _record(
    epoch: int,
    *,
    margin: float = 0.001,
    contact_auprc: float = 0.05,
    contact_precision: float = 0.1,
    contact_recall: float = 0.2,
    predicted_positive_ratio: float = 0.01,
    budget_precision: float = 0.1,
    epitope_auprc: float = 0.4,
    paratope_auprc: float = 0.7,
    gauge_share: float | None = None,
    epitope_gate: float | None = None,
) -> dict:
    contact = {
        "auprc": contact_auprc,
        "auroc": 0.8,
        "brier": 0.02,
        "ece_15": 0.01,
        "mean_per_complex_auprc": contact_auprc + 0.01,
        "fixed_raw_logit_threshold_zero": {
            "precision": contact_precision,
            "recall": contact_recall,
            "f1": 0.15,
            "predicted_positive_ratio": predicted_positive_ratio,
        },
        "ranked_pair_macro": {
            "10": {"mean_precision": 0.02, "mean_recall": 0.005},
            "lmin_over_10": {
                "mean_precision": budget_precision,
                "mean_recall": 0.05,
            },
            "lmin": {"mean_precision": 0.03, "mean_recall": 0.3},
        },
    }

    def residue(auprc: float) -> dict:
        return {
            "auprc": auprc,
            "mean_per_complex_auprc": auprc,
            "brier": 0.05,
            "ece_15": 0.02,
            "fixed_raw_logit_threshold_zero": {
                "precision": 0.3,
                "recall": 0.4,
                "f1": 0.34,
                "predicted_positive_ratio": 0.2,
            },
        }

    validation = {
        "selection_metric": margin,
        "contact": contact,
        "epitope": residue(epitope_auprc),
        "paratope": residue(paratope_auprc),
        "partner_retrieval_by_direction": {
            "fixed_antibody_wrong_antigen": {
                "cluster_weighted_cognate_compatibility_margin": margin,
                "top1_retrieval_accuracy": 0.4,
            },
            "fixed_antigen_wrong_antibody": {
                "cluster_weighted_cognate_compatibility_margin": margin + 0.0005,
                "top1_retrieval_accuracy": 0.5,
            },
        },
    }
    if gauge_share is not None:
        validation["contact_gauge_attribution"] = {
            "mean_partner_specific_share": gauge_share
        }
    record = {"epoch": epoch, "train": {"loss": 1.0}, "validation": validation}
    if epitope_gate is not None:
        record["residue_contact_gate"] = {
            "epitope_applied": epitope_gate,
            "paratope_applied": epitope_gate,
        }
    return record


def _run(experiment: str, seed: int, records, *, completed: bool = False) -> SeedRun:
    metrics = {int(record["epoch"]): extract_metrics(record) for record in records}
    return SeedRun(
        experiment=experiment,
        seed=seed,
        epochs=tuple(sorted(metrics)),
        metrics_by_epoch=metrics,
        source="synthetic",
        completed=completed,
    )


def test_extract_metrics_reads_the_live_record_schema() -> None:
    metrics = extract_metrics(_record(3, gauge_share=0.42, epitope_gate=0.31))

    assert metrics["partner_margin_worst_direction"] == pytest.approx(0.001)
    assert metrics["contact_pooled_auprc"] == pytest.approx(0.05)
    assert metrics["contact_precision"] == pytest.approx(0.1)
    assert metrics["contact_predicted_positive_ratio"] == pytest.approx(0.01)
    assert metrics["contact_p_at_lmin_over_10"] == pytest.approx(0.1)
    assert metrics["partner_specific_variance_share"] == pytest.approx(0.42)
    assert metrics["epitope_gate_applied"] == pytest.approx(0.31)
    assert metrics["partner_margin_fixed_antigen_wrong_antibody"] == pytest.approx(
        0.0015
    )


def test_comparisons_that_change_two_factors_are_refused() -> None:
    """This is exactly the reporting mistake the engine exists to prevent."""

    difference = factor_difference(
        "framecontact_unary_only", "framecontact_distogram"
    )
    assert set(difference) == {"residue_aggregation", "pair_target"}

    with pytest.raises(ComparisonError, match="differ in 2 factors"):
        require_single_variable("framecontact_unary_only", "framecontact_distogram")


def test_single_variable_comparisons_name_the_changed_factor() -> None:
    assert (
        require_single_variable(
            "framecontact_contact_hybrid", "framecontact_distogram"
        )
        == "pair_target"
    )
    assert (
        require_single_variable(
            "framecontact_distogram", "framecontact_distogram_triangle"
        )
        == "triangle"
    )
    assert (
        require_single_variable(
            "framecontact_gauge_hybrid", "framecontact_gauge_hybrid_contrastive"
        )
        == "residue_contrastive"
    )
    assert (
        require_single_variable(
            "framecontact_unary_only", "framecontact_contact_marginal"
        )
        == "residue_aggregation"
    )


def test_gauge_routing_is_entailed_by_the_residue_aggregation() -> None:
    """A gauge-routed head cannot exist without the centered channel.

    Treating the gauge as an independent factor there would block every
    phase-one comparison for a difference the arm cannot avoid.
    """

    assert factor_difference(
        "framecontact_unary_only", "framecontact_gauge_hybrid"
    ) == ("residue_aggregation",)
    assert factor_difference(
        "framecontact_contact_hybrid", "framecontact_gauge_hybrid"
    ) == ("residue_aggregation",)


def test_the_gauge_remains_a_free_factor_for_raw_marginal_heads() -> None:
    """The audit arm is a null control: same head, gauge only observed."""

    assert (
        require_single_variable(
            "framecontact_contact_hybrid",
            "framecontact_contact_hybrid_gauge_audit",
        )
        == "pair_gauge"
    )


def test_identical_arms_are_refused() -> None:
    with pytest.raises(ComparisonError, match="same point"):
        require_single_variable(
            "framecontact_contact_hybrid", "framecontact_contact_hybrid"
        )


def test_precision_gain_without_ranking_gain_is_called_sparsity_drift() -> None:
    """The failure mode the live distogram branch is already showing."""

    deltas = {
        "contact_precision": {"mean_delta": 0.012},
        "contact_recall": {"mean_delta": -0.030},
        "contact_predicted_positive_ratio": {"mean_delta": -0.004},
        "contact_pooled_auprc": {"mean_delta": 0.00002},
        "contact_p_at_lmin_over_10": {"mean_delta": 0.00001},
    }

    verdict = classify_precision_change(deltas)

    assert verdict["verdict"] == "sparsity_drift"
    assert "operating point" in verdict["explanation"]


def test_budget_matched_improvement_is_called_a_ranking_gain() -> None:
    deltas = {
        "contact_precision": {"mean_delta": 0.012},
        "contact_recall": {"mean_delta": 0.004},
        "contact_predicted_positive_ratio": {"mean_delta": 0.0},
        "contact_pooled_auprc": {"mean_delta": 0.008},
        "contact_p_at_lmin_over_10": {"mean_delta": 0.011},
    }

    assert classify_precision_change(deltas)["verdict"] == "ranking_gain"


def test_ranking_regression_is_reported_even_when_precision_rises() -> None:
    deltas = {
        "contact_precision": {"mean_delta": 0.02},
        "contact_recall": {"mean_delta": -0.05},
        "contact_predicted_positive_ratio": {"mean_delta": -0.01},
        "contact_pooled_auprc": {"mean_delta": -0.006},
        "contact_p_at_lmin_over_10": {"mean_delta": -0.004},
    }

    assert classify_precision_change(deltas)["verdict"] == "ranking_regression"


def test_exact_sign_test_matches_the_binomial() -> None:
    assert exact_sign_test([1.0, 1.0, 1.0, 1.0, 1.0])["p_value"] == pytest.approx(
        2 * 0.5**5
    )
    assert exact_sign_test([1.0, -1.0])["p_value"] == pytest.approx(1.0)
    assert exact_sign_test([0.0, 0.0])["ties"] == 2


def test_compare_arms_pins_a_common_epoch_and_pairs_seeds() -> None:
    baseline = {
        seed: _run(
            "framecontact_contact_hybrid",
            seed,
            [_record(epoch, margin=0.001) for epoch in (1, 2, 3)],
        )
        for seed in (17, 42)
    }
    candidate = {
        seed: _run(
            "framecontact_distogram",
            seed,
            [_record(epoch, margin=0.002) for epoch in (1, 2)],
        )
        for seed in (17, 42)
    }

    comparison = compare_arms(baseline, candidate)

    assert comparison.factor == "pair_target"
    assert comparison.epoch == 2
    assert comparison.seeds == (17, 42)
    primary = comparison.deltas_by_metric["partner_margin_worst_direction"]
    assert primary["mean_delta"] == pytest.approx(0.001)
    assert primary["sign_test"]["positive"] == 2


def test_matched_epoch_requires_overlap() -> None:
    left = _run("framecontact_distogram", 17, [_record(1)])
    right = _run("framecontact_distogram_triangle", 17, [_record(5)])

    with pytest.raises(ComparisonError, match="no common epoch"):
        matched_epoch([left, right])


def test_recall_retention_flags_a_collapse() -> None:
    baseline = {17: _run("framecontact_distogram", 17, [_record(1, contact_recall=0.4)])}
    candidate = {
        17: _run(
            "framecontact_distogram_triangle", 17, [_record(1, contact_recall=0.2)]
        )
    }

    retention = recall_retention(baseline, candidate, epoch=1)

    assert retention["minimum"] == pytest.approx(0.5)
    assert retention["meets_floor"] is False


def test_phase_decision_refuses_an_inconsistent_leader() -> None:
    baseline = {
        seed: _run("framecontact_unary_only", seed, [_record(1, margin=0.001)])
        for seed in (17, 42, 73)
    }
    # Two seeds up, one seed sharply down: a positive mean that is not a decision.
    candidate = {
        17: _run("framecontact_gauge_hybrid", 17, [_record(1, margin=0.004)]),
        42: _run("framecontact_gauge_hybrid", 42, [_record(1, margin=0.004)]),
        73: _run("framecontact_gauge_hybrid", 73, [_record(1, margin=0.0005)]),
    }
    phase = next(
        item for item in DECISION_CHAIN if item.key == "phase_1_residue_aggregation"
    )

    result = evaluate_phase(phase, {**{"framecontact_unary_only": baseline},
                                    "framecontact_gauge_hybrid": candidate})

    assert result["decision"]["resolved"] is False
    assert "not consistent across seeds" in result["decision"]["statement"]


def test_a_delta_inside_the_epoch_noise_band_is_not_a_decision() -> None:
    """The live partner margin swings sign between neighbouring epochs.

    A consistent per-seed delta is still meaningless if the metric wanders
    further than that on its own between epochs.
    """

    wobble = (0.001, 0.030, -0.020, 0.025, -0.015)
    baseline = {
        seed: _run(
            "framecontact_unary_only",
            seed,
            [_record(epoch, margin=value) for epoch, value in enumerate(wobble, 1)],
        )
        for seed in (17, 42, 73)
    }
    candidate = {
        seed: _run(
            "framecontact_gauge_hybrid",
            seed,
            [
                _record(epoch, margin=value + 0.0005)
                for epoch, value in enumerate(wobble, 1)
            ],
        )
        for seed in (17, 42, 73)
    }
    phase = next(
        item for item in DECISION_CHAIN if item.key == "phase_1_residue_aggregation"
    )

    result = evaluate_phase(
        phase,
        {
            "framecontact_unary_only": baseline,
            "framecontact_gauge_hybrid": candidate,
        },
    )

    comparison = result["comparisons"][0]
    assert comparison["epoch_stability"]["exceeds_noise"] is False
    assert result["decision"]["resolved"] is False
    assert "epoch noise band" in result["decision"]["statement"]


def test_epoch_noise_needs_a_usable_window() -> None:
    runs = {17: _run("framecontact_unary_only", 17, [_record(1, margin=0.001)])}

    assert epoch_noise(runs, "partner_margin_worst_direction", epoch=1) is None
    with pytest.raises(ComparisonError, match="at least two epochs"):
        epoch_noise(runs, "partner_margin_worst_direction", epoch=1, window=1)


def test_phase_decision_resolves_a_consistent_winner() -> None:
    # A flat baseline gives a small noise band, so a real gap can be resolved.
    baseline = {
        seed: _run(
            "framecontact_unary_only",
            seed,
            [_record(epoch, margin=0.001) for epoch in (1, 2, 3)],
        )
        for seed in (17, 42, 73)
    }
    candidate = {
        seed: _run(
            "framecontact_gauge_hybrid",
            seed,
            [_record(epoch, margin=0.004) for epoch in (1, 2, 3)],
        )
        for seed in (17, 42, 73)
    }
    phase = next(
        item for item in DECISION_CHAIN if item.key == "phase_1_residue_aggregation"
    )

    result = evaluate_phase(
        phase,
        {
            "framecontact_unary_only": baseline,
            "framecontact_gauge_hybrid": candidate,
        },
    )

    assert result["decision"]["resolved"] is True
    assert result["decision"]["winner"] == "framecontact_gauge_hybrid"
    assert result["status"] == "in_flight"


def test_phase_reports_unavailability_instead_of_guessing() -> None:
    phase = next(item for item in DECISION_CHAIN if item.key == "phase_3_triangle")

    result = evaluate_phase(
        phase,
        {
            "framecontact_distogram": {
                17: _run("framecontact_distogram", 17, [_record(1)])
            }
        },
    )

    assert result["status"] == "unavailable"
    assert "no candidate arm" in result["reason"]


def test_main_table_covers_every_arm_at_one_epoch() -> None:
    runs = {
        "framecontact_unary_only": {
            17: _run("framecontact_unary_only", 17, [_record(1), _record(2)])
        },
        "framecontact_contact_hybrid": {
            17: _run("framecontact_contact_hybrid", 17, [_record(1), _record(2)])
        },
    }

    table = build_main_table(runs)

    assert table["epoch"] == 2
    assert [row["experiment"] for row in table["rows"]] == [
        "framecontact_contact_hybrid",
        "framecontact_unary_only",
    ]
    assert table["rows"][0]["factors"]["pair_target"] == "binary_contact"


def test_trajectory_averages_across_seeds() -> None:
    runs = {
        17: _run("framecontact_distogram", 17, [_record(1, margin=0.001)]),
        42: _run("framecontact_distogram", 42, [_record(1, margin=0.003)]),
    }

    series = trajectory(runs, "partner_margin_worst_direction")

    assert series["epochs"] == [1]
    assert series["mean"] == [pytest.approx(0.002)]
    assert series["seed_count"] == [2]


def test_live_log_parsing_ignores_non_json_noise(tmp_path) -> None:
    log = tmp_path / "fc-res-head-1_0.out"
    log.write_text(
        "\n".join(
            [
                "loading dataset",
                json.dumps(_record(1)),
                "warning: something happened",
                "{not json}",
                json.dumps(_record(2)),
            ]
        ),
        encoding="utf-8",
    )

    run = load_live_log(log, experiment="framecontact_unary_only", seed=17)

    assert run.epochs == (1, 2)
    assert run.completed is False


def test_parse_epoch_records_rejects_unrelated_json() -> None:
    assert parse_epoch_records([json.dumps({"unrelated": True})]) == []


def test_completed_run_reports_its_selected_epoch(tmp_path) -> None:
    result = tmp_path / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    result.write_text(
        json.dumps(
            {
                "experiment": "framecontact_contact_hybrid",
                "configuration": {"seed": 42},
                "selection": {"best_epoch": 2},
                "history": [_record(1), _record(2), _record(3)],
            }
        ),
        encoding="utf-8",
    )

    run = load_completed_run(result)

    assert run.completed is True
    assert run.selected_epoch == 2
    assert run.experiment == "framecontact_contact_hybrid"
    assert run.seed == 42
