"""Tests for the PAIRScale E1/E2 posthoc contact audit boundary."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from interaction_first.pairscale_e1_e2_posthoc_audit import (
    PairscaleE1E2PosthocAuditError,
    validate_frozen_report,
    validate_posthoc_source_boundary,
    verify_v2_reproduces_v1,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _v1_evaluation() -> dict[str, object]:
    return {
        "format_version": "pairscale_e1_e2_development_evaluation_v1",
        "macro_pair_auprc": 0.4,
        "macro_unary_additive_auprc": 0.2,
        "macro_interaction_residual_auprc": 0.3,
        "partner_controls": {"fixed": True},
        "per_complex": [
            {
                "instance_id": "example-1",
                "auprc": 0.4,
                "unary_additive_auprc": 0.2,
                "interaction_residual_auprc": 0.3,
            }
        ],
    }


def _v2_evaluation() -> dict[str, object]:
    result = copy.deepcopy(_v1_evaluation())
    result["format_version"] = "pairscale_e1_e2_development_evaluation_v2"
    result.update(
        {
            "macro_interaction_residual_ap_lift": 0.25,
            "macro_interaction_residual_precision_at_true_contact_count": 0.35,
            "macro_unary_additive_ap_lift": 0.15,
            "macro_unary_additive_precision_at_true_contact_count": 0.25,
        }
    )
    result["per_complex"][0].update(
        {
            "interaction_residual_ap_lift": 0.25,
            "interaction_residual_precision_at_true_contact_count": 0.35,
            "interaction_residual_precision_at_true_contact_count_tie_policy": (
                "inclusive_expected_precision"
            ),
            "unary_additive_ap_lift": 0.15,
            "unary_additive_precision_at_true_contact_count": 0.25,
            "unary_additive_precision_at_true_contact_count_tie_policy": (
                "inclusive_expected_precision"
            ),
        }
    )
    return result


def _report() -> dict[str, object]:
    evaluation = _v1_evaluation()
    return {
        "format_version": "pairscale_e1_e2_screen_report_v1",
        "screening_seed": 17,
        "completed_epoch": 260,
        "fixed_epoch_rule_satisfied": True,
        "early_stopping_used": False,
        "development_holdout_evaluation_performed": True,
        "development_holdout_is_untouched_final_test": False,
        "new_final_test_accessed": False,
        "model_selection_authorized": False,
        "checkpoint_promotion_authorized": False,
        "confirmatory_claim_authorized": False,
        "scaling_authorized": False,
        "development_holdout_evaluation": evaluation,
        "frozen_checkpoint": {
            "checkpoint_sha256": _digest(1),
            "receipt_file_sha256": _digest(2),
            "generation": "epoch-0260-step-00049920",
            "epoch": 260,
            "provenance_hashes": {"source_model.py": _digest(3)},
        },
    }


def test_v2_must_exactly_reproduce_every_v1_field() -> None:
    result = verify_v2_reproduces_v1(_v1_evaluation(), _v2_evaluation())
    assert result["reference_v1_canonical_sha256"] == (
        result["reproduced_v1_canonical_sha256"]
    )
    altered = _v2_evaluation()
    altered["macro_pair_auprc"] = 0.4000001
    with pytest.raises(PairscaleE1E2PosthocAuditError, match="reproduce"):
        verify_v2_reproduces_v1(_v1_evaluation(), altered)


def test_v2_rejects_missing_new_metric() -> None:
    altered = _v2_evaluation()
    del altered["per_complex"][0]["unary_additive_ap_lift"]
    with pytest.raises(PairscaleE1E2PosthocAuditError, match="missing"):
        verify_v2_reproduces_v1(_v1_evaluation(), altered)


def test_frozen_report_must_bind_epoch_and_authorization_boundary() -> None:
    checkpoint = validate_frozen_report(
        _report(),
        seed=17,
        expected_holdout_count=1,
    )
    assert checkpoint["epoch"] == 260
    altered = _report()
    altered["scaling_authorized"] = True
    with pytest.raises(PairscaleE1E2PosthocAuditError, match="scaling"):
        validate_frozen_report(altered, seed=17, expected_holdout_count=1)


def test_only_evaluator_source_may_differ() -> None:
    frozen = {
        "source_model.py": _digest(1),
        "source_pairscale_e1_e2_evaluation.py": _digest(2),
    }
    current = {
        "model.py": _digest(1),
        "pairscale_e1_e2_evaluation.py": _digest(3),
    }
    result = validate_posthoc_source_boundary(frozen, current)
    assert result["all_other_executed_sources_exact"] is True
    altered = dict(current)
    altered["model.py"] = _digest(4)
    with pytest.raises(PairscaleE1E2PosthocAuditError, match="boundary"):
        validate_posthoc_source_boundary(frozen, altered)


def test_controller_authenticates_checkpoint_before_repeated_holdout_access() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_pairscale_e1_e2_checkpoint.py"
    ).read_text(encoding="utf-8")
    report_validation = source.index("validate_frozen_report(")
    checkpoint_restore = source.index("restore_checkpoint_generation(")
    access_receipt = source.index("POSTHOC_HOLDOUT_ACCESS_RECEIPT.json")
    holdout_load = source.index("holdout_examples = _load_examples")
    evaluation = source.index("evaluation = evaluate_e1_e2(")
    reproduction = source.index("reproduction = verify_v2_reproduces_v1(")
    assert (
        report_validation
        < checkpoint_restore
        < access_receipt
        < holdout_load
        < evaluation
        < reproduction
    )
    assert '"scaling_authorized": False' in source
    assert '"new_final_test_accessed": False' in source
    assert '"posthoc_implementation_source_sha256"' in source
