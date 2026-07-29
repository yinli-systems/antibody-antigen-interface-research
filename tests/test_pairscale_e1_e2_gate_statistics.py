"""Tests for the fail-closed PAIRScale E1/E2 gate statistics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from interaction_first.development_authority import canonical_json_bytes
from interaction_first.development_holdout import (
    REQUIRED_IMPLEMENTATION_SOURCES,
    ScoreFreeSubset,
    build_component_holdout,
    build_holdout_receipt,
)
from interaction_first.development_subset import AuthorizedInstance
from interaction_first.pairscale_e1_e2_gate_statistics import (
    PairscaleE1E2GateStatisticsError,
    analyze_e1_e2_screening,
)


SEEDS = (17, 42, 73)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _write_fixture(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    instances = tuple(
        AuthorizedInstance(f"i{index:02d}", f"a{index:02d}", f"g{index:02d}")
        for index in range(12)
    )
    subset = ScoreFreeSubset(
        instances=instances,
        subset_sha256="1" * 64,
        policy_sha256="2" * 64,
        permission_name="development_only_nonconfirmatory",
        source_split="train",
    )
    assignment = build_component_holdout(
        subset,
        holdout_pair_count=10,
        seed=29,
    )
    receipt = build_holdout_receipt(
        assignment,
        implementation_source_sha256={
            name: _digest(index + 3)
            for index, name in enumerate(sorted(REQUIRED_IMPLEMENTATION_SOURCES))
        },
    )
    assignment_bytes = canonical_json_bytes(assignment)
    receipt_bytes = canonical_json_bytes(receipt)
    assignment_path = root / "assignment.json"
    receipt_path = root / "assignment_receipt.json"
    assignment_path.write_bytes(assignment_bytes)
    receipt_path.write_bytes(receipt_bytes)
    holdout_rows = list(assignment["holdout"]["instances"])

    common_provenance = {
        "assignment_content_sha256": assignment["assignment_content_sha256"],
        "assignment_file_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
        "assignment_receipt_file_sha256": hashlib.sha256(
            receipt_bytes
        ).hexdigest(),
        "cache_manifest_sha256": _digest(101),
        "cache_receipt_file_sha256": _digest(102),
        "controller_sha256": _digest(103),
        "protocol_file_sha256": _digest(104),
        "scheduler_script_sha256": _digest(105),
        "source_model.py": _digest(106),
    }
    reports: dict[int, dict[str, object]] = {}
    report_paths: dict[int, Path] = {}
    report_sha256: dict[int, str] = {}
    for seed in SEEDS:
        per_complex = [
            {
                "antibody_id": row["antibody_id"],
                "antigen_id": row["antigen_id"],
                "ap_lift": (0.5 - 0.1) / 0.9,
                "auprc": 0.5,
                "instance_id": row["instance_id"],
                "precision_at_true_contact_count": 0.6,
                "prevalence": 0.1,
                "unary_additive_auprc": 0.3,
            }
            for row in holdout_rows
        ]
        matched = [1.0 + seed / 10_000.0] * len(per_complex)
        decoys = [[0.2, 0.2, 0.2, 0.2] for _ in per_complex]
        panel = {
            "decoy_count_per_direction": 4,
            "unknown_cross_pair_semantics": (
                "unlabeled_ranking_control_not_nonbinder"
            ),
        }
        for direction in ("antibody_to_antigen", "antigen_to_antibody"):
            panel[direction] = {
                source: {
                    "decoy_scores": decoys,
                    "matched_scores": matched,
                }
                for source in ("raw_residual", "centered_residual")
            }
        antibody_ids = sorted(row["antibody_id"] for row in holdout_rows)
        antigen_ids = sorted(row["antigen_id"] for row in holdout_rows)
        exhaustive = {
            "antibody_ids": antibody_ids,
            "antigen_ids": antigen_ids,
            "candidate_pair_count": len(antibody_ids) * len(antigen_ids),
            "observed_cognate_pair_count": len(holdout_rows),
            "unknown_cross_pair_semantics": (
                "unlabeled_retrieval_control_not_nonbinder"
            ),
        }
        for source in ("raw_residual", "centered_residual"):
            exhaustive[source] = {
                "evaluation_contract": (
                    "prospective_pairformer_exhaustive_retrieval_v2"
                ),
                "negative_candidate_subsampling_performed": False,
                "antibody_to_antigen": {
                    "candidate_count_per_query": len(antigen_ids),
                    "query_count": len(antibody_ids),
                    "recall_at_1": 1.0,
                },
                "antigen_to_antibody": {
                    "candidate_count_per_query": len(antibody_ids),
                    "query_count": len(antigen_ids),
                    "recall_at_1": 1.0,
                },
            }
        evaluation = {
            "exhaustive_bidirectional_retrieval": exhaustive,
            "format_version": "pairscale_e1_e2_development_evaluation_v1",
            "model_selection_authorized": False,
            "partner_controls": panel,
            "per_complex": per_complex,
            "threshold_selection_performed": False,
            "unknown_cross_pairs_assigned_binary_labels": False,
        }
        report = {
            "checkpoint_promotion_authorized": False,
            "completed_epoch": 260,
            "confirmatory_claim_authorized": False,
            "development_holdout_evaluation": evaluation,
            "development_holdout_evaluation_performed": True,
            "development_holdout_is_untouched_final_test": False,
            "early_stopping_used": False,
            "fixed_epoch_rule_satisfied": True,
            "format_version": "pairscale_e1_e2_screen_report_v1",
            "frozen_checkpoint": {
                "checkpoint_sha256": _digest(seed),
                "epoch": 260,
                "provenance_hashes": {
                    **common_provenance,
                    "decoy_assignment_sha256": _digest(seed + 1_000),
                },
                "receipt_file_sha256": _digest(seed + 2_000),
            },
            "model_selection_authorized": False,
            "new_final_test_accessed": False,
            "scaling_authorized": False,
            "screening_seed": seed,
        }
        path = root / f"seed_{seed}_report.json"
        raw = canonical_json_bytes(report)
        path.write_bytes(raw)
        reports[seed] = report
        report_paths[seed] = path
        report_sha256[seed] = hashlib.sha256(raw).hexdigest()

    protocol_path = (
        Path(__file__).resolve().parents[1]
        / "protocols"
        / "pairscale_e1_e2_gate_audit_v1.json"
    )
    return {
        "assignment_content_sha256": assignment["assignment_content_sha256"],
        "assignment_file_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
        "assignment_path": assignment_path,
        "assignment_receipt_path": receipt_path,
        "assignment_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "protocol_path": protocol_path,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "report_paths": report_paths,
        "report_sha256": report_sha256,
        "reports": reports,
    }


def _analyze(fixture: dict[str, object]) -> dict[str, object]:
    return analyze_e1_e2_screening(
        assignment_path=fixture["assignment_path"],
        assignment_receipt_path=fixture["assignment_receipt_path"],
        expected_assignment_file_sha256=fixture["assignment_file_sha256"],
        expected_assignment_content_sha256=fixture["assignment_content_sha256"],
        expected_assignment_receipt_sha256=fixture[
            "assignment_receipt_sha256"
        ],
        protocol_path=fixture["protocol_path"],
        expected_protocol_sha256=fixture["protocol_sha256"],
        report_paths_by_seed=fixture["report_paths"],
        expected_report_sha256_by_seed=fixture["report_sha256"],
        created_at_utc="2026-07-28T06:00:00Z",
    )


def _rewrite_report(fixture: dict[str, object], seed: int) -> None:
    report = fixture["reports"][seed]
    raw = canonical_json_bytes(report)
    fixture["report_paths"][seed].write_bytes(raw)
    fixture["report_sha256"][seed] = hashlib.sha256(raw).hexdigest()


def test_screening_passes_extension_but_never_passes_final_gates(
    tmp_path: Path,
) -> None:
    result = _analyze(_write_fixture(tmp_path))
    assert result["independent_component_count"] == 10
    assert result["five_seed_extension"]["eligible"] is True
    assert all(result["five_seed_extension"]["checks"].values())
    assert result["retrieval"]["homolog_matched_candidate_pool_available"] is False
    assert result["contact"]["pair_minus_unary"]["mean_difference"] == pytest.approx(
        0.2
    )
    for gate in ("S", "C", "R"):
        assert result["gates"][gate]["passed"] is False
        assert result["gates"][gate]["scaling_authorized"] is False


def test_screening_fails_closed_when_one_specificity_cell_is_negative(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    for seed in SEEDS:
        report = fixture["reports"][seed]
        panel = report["development_holdout_evaluation"]["partner_controls"]
        for direction in ("antibody_to_antigen", "antigen_to_antibody"):
            cell = panel[direction]["centered_residual"]
            cell["matched_scores"] = [-1.0] * len(cell["matched_scores"])
        _rewrite_report(fixture, seed)
    result = _analyze(fixture)
    cell = result["specificity"]["cells"][
        "antibody_to_antigen.centered_residual"
    ]
    assert cell["component_bootstrap"]["lower"] < 0.0
    assert cell["positive_seed_count"] == 0
    assert result["five_seed_extension"]["eligible"] is False


def test_screening_rejects_report_hash_and_source_anchor_tampering(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    fixture["report_paths"][17].write_text("{}\n")
    with pytest.raises(PairscaleE1E2GateStatisticsError, match="SHA-256"):
        _analyze(fixture)

    fixture = _write_fixture(tmp_path / "anchor")
    report = fixture["reports"][42]
    report["frozen_checkpoint"]["provenance_hashes"][
        "source_model.py"
    ] = _digest(999)
    _rewrite_report(fixture, 42)
    with pytest.raises(
        PairscaleE1E2GateStatisticsError,
        match="anchors differ",
    ):
        _analyze(fixture)


def test_screening_rejects_missing_seed_and_protocol_tampering(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    del fixture["report_paths"][73]
    with pytest.raises(PairscaleE1E2GateStatisticsError, match="screening seeds"):
        _analyze(fixture)

    fixture = _write_fixture(tmp_path / "protocol")
    fixture["protocol_sha256"] = "f" * 64
    with pytest.raises(PairscaleE1E2GateStatisticsError, match="protocol SHA-256"):
        _analyze(fixture)
