import copy
from fractions import Fraction
import json
from pathlib import Path

import pytest

from antibody_pretraining.corpus_plan import (
    CorpusPlanError,
    allocate_capped_sqrt,
    build_corpus_plan,
    validate_corpus_plan,
)
from antibody_pretraining.independent_corpus_audit import (
    IndependentCorpusAuditError,
    audit_corpus_plan,
    canonical_json_bytes,
)
import hashlib


AUTHORITY_DIGEST = "a" * 64
ROOT = Path(__file__).resolve().parents[1]


def _entry(
    study: str,
    organism: str,
    chain: str,
    count: int,
    suffix: str,
) -> dict:
    relative = f"{study}/csv/{suffix}_{chain}.csv.gz"
    return {
        "canonical_url": f"https://example.invalid/{relative}",
        "metadata": {"Chain": chain, "Organism": organism},
        "relative_unit": relative,
        "study": study,
        "unique_sequences": count,
    }


def _fixture() -> tuple[dict, dict]:
    units = [
        _entry("hm_val", "human", "Heavy", 60, "a"),
        _entry("hm_val", "human", "Light", 50, "b"),
        _entry("nh_val", "mouse", "Heavy", 80, "c"),
        _entry("nl_val", "mouse", "Light", 70, "d"),
    ]
    for index in range(5):
        units.extend(
            [
                _entry(f"h{index}", "human", "Heavy", 100, f"hh{index}"),
                _entry(f"h{index}", "human", "Light", 80, f"hl{index}"),
                _entry(f"n{index}", "mouse", "Heavy", 70, f"nh{index}"),
                _entry(f"n{index}", "mouse", "Light", 60, f"nl{index}"),
            ]
        )
    manifest = {
        "data_units": units,
        "manifest_sha256": AUTHORITY_DIGEST,
        "summary": {},
    }
    spec = {
        "authority": {
            "expected_data_unit_count": len(units),
            "expected_study_count": 13,
            "expected_unique_sequence_count": sum(
                unit["unique_sequences"] for unit in units
            ),
            "manifest_file_sha256": "b" * 64,
            "manifest_payload_sha256": AUTHORITY_DIGEST,
            "source_class": "unpaired",
        },
        "format_version": "oas_unpaired_corpus_plan_spec_v1",
        "namespace": "fixture_v1",
        "excluded_authority_studies": [],
        "record_selection": {
            "exact_duplicate_policy": "canonical_sequence_digest_then_authority_order",
            "partial_unit_filter": "sha256_integer_below_exact_fraction",
            "validation_train_duplicate_policy": "remove_from_training",
            "validation_units_are_never_training_units": True,
        },
        "study_allocation": {
            "maximum_share_denominator": 2,
            "maximum_share_numerator": 1,
            "minimum_study_cap_records": 1,
            "stratum_maximum_share_overrides": {},
            "weight": "square_root_available_records",
        },
        "test_access": {
            "downstream_heldout_accessed": False,
            "downstream_metrics_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "training_expected_prefilter_records": {
            "human_heavy": 200,
            "human_light": 160,
            "nonhuman_heavy": 140,
            "nonhuman_light": 120,
        },
        "validation": {
            "cohorts": [
                {
                    "count": 1,
                    "minimum_stratum_records": {
                        "human_heavy": 1,
                        "human_light": 1,
                    },
                    "name": "human_mixed",
                },
                {
                    "count": 1,
                    "minimum_stratum_records": {"nonhuman_heavy": 1},
                    "name": "nonhuman_heavy",
                },
                {
                    "count": 1,
                    "minimum_stratum_records": {"nonhuman_light": 1},
                    "name": "nonhuman_light",
                },
            ],
            "maximum_eligible_study_records": 120,
            "minimum_eligible_study_records": 50,
            "sample_records_per_study": 20,
            "selection": "cohort_constrained_sha256_order",
            "unit_and_record_selection_are_loss_independent": True,
        },
    }
    return manifest, spec


def test_capped_square_root_allocation_is_exact_and_deterministic() -> None:
    capacities = {"large": 1000, "medium": 400, "small": 100}
    first = allocate_capped_sqrt(
        capacities,
        300,
        maximum_share=Fraction(1, 2),
        minimum_cap=0,
        namespace="test",
    )
    second = allocate_capped_sqrt(
        dict(reversed(list(capacities.items()))),
        300,
        maximum_share=Fraction(1, 2),
        minimum_cap=0,
        namespace="test",
    )

    assert first == second
    assert sum(first.values()) == 300
    assert max(first.values()) <= 150


def test_capped_allocation_fails_when_caps_make_target_infeasible() -> None:
    with pytest.raises(CorpusPlanError, match="cap makes the allocation infeasible"):
        allocate_capped_sqrt(
            {"a": 100, "b": 100},
            100,
            maximum_share=Fraction(1, 4),
            minimum_cap=0,
            namespace="test",
        )


def test_plan_is_deterministic_and_keeps_whole_studies_isolated() -> None:
    manifest, spec = _fixture()
    first = build_corpus_plan(
        manifest,
        spec,
        spec_file_sha256="c" * 64,
        planner_module_sha256="e" * 64,
        planner_source_sha256="d" * 64,
    )
    second = build_corpus_plan(
        copy.deepcopy(manifest),
        copy.deepcopy(spec),
        spec_file_sha256="c" * 64,
        planner_module_sha256="e" * 64,
        planner_source_sha256="d" * 64,
    )

    assert first == second
    validation_studies = {entry["study"] for entry in first["validation_studies"]}
    training_studies = {
        unit["study"]
        for unit in first["selected_data_units"]
        if unit["split"] == "train"
    }
    assert validation_studies.isdisjoint(training_studies)
    assert sum(first["training_expected_prefilter_records"].values()) == 620
    validate_corpus_plan(first)


def test_plan_digest_and_split_tampering_fail_closed() -> None:
    manifest, spec = _fixture()
    plan = build_corpus_plan(
        manifest,
        spec,
        spec_file_sha256="c" * 64,
        planner_module_sha256="e" * 64,
        planner_source_sha256="d" * 64,
    )
    plan["selected_data_units"][0]["split"] = "validation"

    with pytest.raises(CorpusPlanError, match="payload digest differs"):
        validate_corpus_plan(plan)


def test_plan_rejects_authority_count_drift() -> None:
    manifest, spec = _fixture()
    spec["authority"]["expected_unique_sequence_count"] += 1

    with pytest.raises(CorpusPlanError, match="Authority count differs"):
        build_corpus_plan(
            manifest,
            spec,
            spec_file_sha256="c" * 64,
            planner_module_sha256="e" * 64,
            planner_source_sha256="d" * 64,
        )


def test_independent_auditor_rebuilds_the_complete_plan() -> None:
    manifest, spec = _fixture()
    plan = build_corpus_plan(
        manifest,
        spec,
        spec_file_sha256="c" * 64,
        planner_module_sha256="e" * 64,
        planner_source_sha256="d" * 64,
    )
    receipt = audit_corpus_plan(
        manifest,
        plan,
        spec,
        manifest_file_sha256="b" * 64,
        plan_file_sha256=hashlib.sha256(canonical_json_bytes(plan)).hexdigest(),
        spec_file_sha256="c" * 64,
        auditor_module_sha256="f" * 64,
        auditor_source_sha256="1" * 64,
        planner_module_sha256="e" * 64,
        planner_source_sha256="d" * 64,
    )

    assert receipt["audit_result"] == "passed"
    assert receipt["selected_data_unit_count"] == len(plan["selected_data_units"])
    assert receipt["training_expected_prefilter_records"] == 620


def test_independent_auditor_rejects_membership_reanchoring() -> None:
    manifest, spec = _fixture()
    plan = build_corpus_plan(
        manifest,
        spec,
        spec_file_sha256="c" * 64,
        planner_module_sha256="e" * 64,
        planner_source_sha256="d" * 64,
    )
    plan["selected_data_units"][0]["relative_unit"] = "tampered.csv.gz"
    unsigned = dict(plan)
    del unsigned["plan_sha256"]
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()

    with pytest.raises(IndependentCorpusAuditError, match="membership differs"):
        audit_corpus_plan(
            manifest,
            plan,
            spec,
            manifest_file_sha256="b" * 64,
            plan_file_sha256="2" * 64,
            spec_file_sha256="c" * 64,
            auditor_module_sha256="f" * 64,
            auditor_source_sha256="1" * 64,
            planner_module_sha256="e" * 64,
            planner_source_sha256="d" * 64,
        )


def test_committed_real_authority_audit_receipt_is_self_anchored() -> None:
    path = ROOT / "results/pretraining/oas_unpaired_corpus_plan_250m_v1_audit.json"
    receipt = json.loads(path.read_text(encoding="ascii"))
    unsigned = dict(receipt)
    recorded = unsigned.pop("receipt_sha256")

    assert receipt["audit_result"] == "passed"
    assert receipt["training_expected_prefilter_records"] == 250_000_000
    assert receipt["selected_data_unit_count"] == 6_373
    assert receipt["test_access"] == {
        "downstream_heldout_accessed": False,
        "downstream_metrics_used": False,
        "sabdab2_heldout_accessed": False,
    }
    assert hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() == recorded
