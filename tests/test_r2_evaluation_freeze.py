"""Adversarial tests for one-shot R2 outer and external-test freezes."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from interaction_first.r2_evaluation_freeze import (
    CheckpointBinding,
    EXTERNAL_RESULT_LABEL,
    EXTERNAL_TEST_PHASE,
    ExternalFreezeReceipt,
    FROZEN_SEEDS,
    FrozenEvaluationRegistry,
    OUTER_PHASE,
    R2EvaluationFreezeError,
    canonical_json_bytes,
    create_one_shot_attempt_ledger,
    verify_external_freeze,
)


HASHES = tuple(f"{index:064x}" for index in range(1, 64))


def _checkpoints(candidate_id: str = "R2") -> tuple[CheckpointBinding, ...]:
    epoch = 58 if candidate_id == "D0" else 40
    return tuple(
        CheckpointBinding(
            seed=seed,
            final_epoch=epoch,
            checkpoint_sha256=HASHES[index],
            checkpoint_receipt_sha256=HASHES[index + 8],
            source_run_id=f"frozen-refit-seed-{seed}",
        )
        for index, seed in enumerate(FROZEN_SEEDS)
    )


def _registry(
    phase: str = OUTER_PHASE,
    **updates: object,
) -> FrozenEvaluationRegistry:
    values: dict[str, object] = {
        "phase": phase,
        "protocol_file_sha256": HASHES[20],
        "execution_plan_sha256": HASHES[21],
        "gate0_decision_sha256": HASHES[22],
        "split_assignment_sha256": HASHES[23],
        "inner_selection_receipt_sha256": HASHES[24],
        "source_manifest_sha256": HASHES[25],
        "runtime_manifest_sha256": HASHES[26],
        "candidate_domain_sha256": HASHES[27],
        "metric_registry_sha256": HASHES[28],
        "calibration_registry_sha256": HASHES[29],
        "threshold_registry_sha256": HASHES[30],
        "analysis_registry_sha256": HASHES[31],
        "candidate_id": "R2",
        "checkpoint_rule": "fixed_final_epoch_40",
        "checkpoints": _checkpoints(),
        "frozen_at_utc": "2026-07-29T16:00:00Z",
        "outer_fold": 0,
    }
    if phase == EXTERNAL_TEST_PHASE:
        values.update(
            {
                "outer_fold": None,
                "completed_outer_analysis_receipt_sha256": HASHES[32],
                "prior_external_test_schema_access": True,
                "result_label": EXTERNAL_RESULT_LABEL,
            }
        )
    values.update(updates)
    return FrozenEvaluationRegistry(**values)  # type: ignore[arg-type]


def _receipt(registry: FrozenEvaluationRegistry) -> ExternalFreezeReceipt:
    return ExternalFreezeReceipt(
        phase=registry.phase,
        registry_sha256=registry.registry_sha256,
        controller_id="independent.freeze-controller",
        frozen_at_utc="2026-07-29T16:01:00Z",
    )


@pytest.mark.parametrize("phase", (OUTER_PHASE, EXTERNAL_TEST_PHASE))
def test_external_digest_anchor_verifies_exact_registry(phase: str) -> None:
    registry = _registry(phase)
    receipt = _receipt(registry)
    verify_external_freeze(
        registry,
        receipt,
        expected_registry_sha256=registry.registry_sha256,
        expected_receipt_sha256=receipt.receipt_sha256,
        required_phase=phase,
    )
    payload = registry.canonical_payload()
    assert payload["checkpoint_selection_from_outer_or_test"] is False
    assert payload["threshold_selection_from_outer_or_test"] is False
    assert payload["calibration_selection_from_outer_or_test"] is False
    assert payload["attempt_limit"] == 1
    assert json.loads(canonical_json_bytes(payload)) == payload


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("validation_best", "checkpoint rule"),
        ("missing_seed", "exactly five"),
        ("duplicate_checkpoint", "distinct checkpoint"),
        ("outer_result", "cannot consume completed outer"),
        ("external_fold", "cannot name one outer"),
        ("hide_schema_access", "must be disclosed"),
        ("wrong_label", "must disclose"),
    ),
)
def test_registry_mutations_fail_closed(mutation: str, message: str) -> None:
    if mutation in {"external_fold", "hide_schema_access", "wrong_label"}:
        registry = _registry(EXTERNAL_TEST_PHASE)
    else:
        registry = _registry()
    changes: dict[str, object]
    if mutation == "validation_best":
        changes = {"checkpoint_rule": "validation_best"}
    elif mutation == "missing_seed":
        changes = {"checkpoints": registry.checkpoints[:-1]}
    elif mutation == "duplicate_checkpoint":
        duplicate = replace(
            registry.checkpoints[-1],
            checkpoint_sha256=registry.checkpoints[0].checkpoint_sha256,
        )
        changes = {"checkpoints": (*registry.checkpoints[:-1], duplicate)}
    elif mutation == "outer_result":
        changes = {"completed_outer_analysis_receipt_sha256": HASHES[33]}
    elif mutation == "external_fold":
        changes = {"outer_fold": 1}
    elif mutation == "hide_schema_access":
        changes = {"prior_external_test_schema_access": False}
    elif mutation == "wrong_label":
        changes = {"result_label": "pristine_blind_test"}
    else:
        raise AssertionError(f"Unhandled mutation {mutation}")
    with pytest.raises(R2EvaluationFreezeError, match=message):
        replace(registry, **changes)


def test_freeze_receipt_and_out_of_band_digests_cannot_be_reanchored() -> None:
    registry = _registry()
    receipt = _receipt(registry)
    with pytest.raises(R2EvaluationFreezeError, match="registry digest"):
        verify_external_freeze(
            registry,
            receipt,
            expected_registry_sha256=HASHES[40],
            expected_receipt_sha256=receipt.receipt_sha256,
            required_phase=OUTER_PHASE,
        )
    with pytest.raises(R2EvaluationFreezeError, match="receipt digest"):
        verify_external_freeze(
            registry,
            receipt,
            expected_registry_sha256=registry.registry_sha256,
            expected_receipt_sha256=HASHES[41],
            required_phase=OUTER_PHASE,
        )
    with pytest.raises(R2EvaluationFreezeError, match="does not bind"):
        verify_external_freeze(
            registry,
            replace(receipt, registry_sha256=HASHES[42]),
            expected_registry_sha256=registry.registry_sha256,
            expected_receipt_sha256=replace(
                receipt,
                registry_sha256=HASHES[42],
            ).receipt_sha256,
            required_phase=OUTER_PHASE,
        )


def test_one_shot_ledger_is_create_only_and_precedes_data_access(
    tmp_path,
) -> None:
    registry = _registry(EXTERNAL_TEST_PHASE)
    receipt = _receipt(registry)
    path = tmp_path / "attempt.json"
    payload = create_one_shot_attempt_ledger(
        path,
        registry=registry,
        receipt=receipt,
        expected_registry_sha256=registry.registry_sha256,
        expected_receipt_sha256=receipt.receipt_sha256,
        started_at_utc="2026-07-29T16:02:00Z",
    )
    assert path.read_bytes() == canonical_json_bytes(payload)
    assert payload["attempt_number"] == 1
    assert payload["evaluation_data_read"] is False
    assert payload["inference_started"] is False
    assert payload["performance_scored"] is False
    with pytest.raises(FileExistsError, match="already exists"):
        create_one_shot_attempt_ledger(
            path,
            registry=registry,
            receipt=receipt,
            expected_registry_sha256=registry.registry_sha256,
            expected_receipt_sha256=receipt.receipt_sha256,
            started_at_utc="2026-07-29T16:03:00Z",
        )


def test_symlink_and_symbolic_parent_are_rejected(tmp_path) -> None:
    registry = _registry()
    receipt = _receipt(registry)
    target = tmp_path / "real.json"
    symbolic_target = tmp_path / "attempt.json"
    symbolic_target.symlink_to(target)
    with pytest.raises(FileExistsError, match="already exists"):
        create_one_shot_attempt_ledger(
            symbolic_target,
            registry=registry,
            receipt=receipt,
            expected_registry_sha256=registry.registry_sha256,
            expected_receipt_sha256=receipt.receipt_sha256,
            started_at_utc="2026-07-29T16:02:00Z",
        )

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    symbolic_parent = tmp_path / "symbolic-parent"
    symbolic_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(R2EvaluationFreezeError, match="regular directory"):
        create_one_shot_attempt_ledger(
            symbolic_parent / "attempt.json",
            registry=registry,
            receipt=receipt,
            expected_registry_sha256=registry.registry_sha256,
            expected_receipt_sha256=receipt.receipt_sha256,
            started_at_utc="2026-07-29T16:02:00Z",
        )


def test_invalid_calendar_timestamp_is_rejected() -> None:
    registry = _registry()
    with pytest.raises(R2EvaluationFreezeError, match="valid UTC calendar"):
        replace(registry, frozen_at_utc="2026-99-99T16:00:00Z")
