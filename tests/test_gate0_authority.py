from __future__ import annotations

import json
from dataclasses import replace

import pytest

from interaction_first.gate0_authority import (
    DigestBinding,
    ExpectedArtifact,
    ExpectedReceipt,
    Gate0Decision,
    Gate0Error,
    Gate0Policy,
    GateName,
    GateReceipt,
    ReceiptStatus,
    decide_gate0,
    gate0_decision_from_json,
    gate0_decision_from_payload,
    verify_gate0_decision,
)


PROTOCOL_SHA = "a" * 64


def _receipt(
    gate: GateName,
    *,
    status: ReceiptStatus = ReceiptStatus.PASS,
    producer: str = "producer",
    auditor: str | None = "independent-auditor",
    artifact_sha: str | None = None,
    test_access: bool = False,
) -> GateReceipt:
    return GateReceipt(
        gate=gate,
        status=status,
        producer_id=producer,
        auditor_id=auditor,
        protocol_sha256=PROTOCOL_SHA,
        artifacts=(
            DigestBinding(
                name=f"{gate.value}_manifest",
                sha256=artifact_sha or (str(list(GateName).index(gate) + 1) * 64),
            ),
        ),
        assertion=f"{gate.value} passed its frozen checks",
        issued_at_utc="2026-07-25T15:00:00Z",
        sealed_or_test_data_accessed=test_access,
    )


def _receipts() -> list[GateReceipt]:
    return [_receipt(gate) for gate in GateName]


def _policy(
    receipts: list[GateReceipt] | None = None,
    *,
    expected_artifacts: tuple[ExpectedArtifact, ...] | None = None,
    authorized_permissions: tuple[str, ...] = (
        "contact_training",
        "model_selection",
    ),
) -> Gate0Policy:
    values = _receipts() if receipts is None else receipts
    artifact_by_key = {
        (receipt.gate, artifact.name): ExpectedArtifact(
            gate=receipt.gate,
            name=artifact.name,
            sha256=artifact.sha256,
        )
        for receipt in values
        for artifact in receipt.artifacts
    }
    if expected_artifacts is not None:
        artifact_by_key.update(
            {
                (artifact.gate, artifact.name): artifact
                for artifact in expected_artifacts
            }
        )
    return Gate0Policy(
        protocol_sha256=PROTOCOL_SHA,
        expected_artifacts=tuple(artifact_by_key.values()),
        expected_receipts=tuple(
            ExpectedReceipt(gate=receipt.gate, sha256=receipt.receipt_sha256)
            for receipt in values
        ),
        authorized_permissions=authorized_permissions,
    )


def test_complete_independent_receipts_authorize_only_gate0_permissions() -> None:
    policy = _policy(
        expected_artifacts=(
            ExpectedArtifact(
                gate=GateName.JOINT_NESTED_SPLIT,
                name="joint_nested_split_manifest",
                sha256="2" * 64,
            ),
        ),
    )
    decision = decide_gate0(
        reversed(_receipts()),
        policy=policy,
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )

    assert decision.authorized
    assert decision.contact_training_authorized
    assert decision.model_selection_authorized
    assert not decision.outer_evaluation_authorized
    assert not decision.external_test_inference_authorized
    assert decision.blocking_reasons == ()
    assert [name for name, _ in decision.receipt_sha256_by_gate] == sorted(
        gate.value for gate in GateName
    )

    verify_gate0_decision(
        decision,
        expected_policy_sha256=policy.policy_sha256,
        expected_decision_sha256=decision.decision_sha256,
        required_permission="contact_training",
    )
    with pytest.raises(Gate0Error, match="does not authorize"):
        verify_gate0_decision(
            decision,
            expected_policy_sha256=policy.policy_sha256,
            expected_decision_sha256=decision.decision_sha256,
            required_permission="outer_evaluation",
        )


def test_receipt_policy_and_decision_hash_domains_are_versioned() -> None:
    receipt = _receipts()[0]
    policy = _policy()
    decision = decide_gate0(
        _receipts(),
        policy=policy,
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    assert (
        receipt.canonical_payload()["schema_version"]
        == "pairformer_gate0_receipt_v1"
    )
    assert (
        policy.canonical_payload()["schema_version"]
        == "pairformer_gate0_policy_v1"
    )
    assert (
        decision.canonical_payload()["schema_version"]
        == "pairformer_gate0_decision_v1"
    )


@pytest.mark.parametrize("status", [ReceiptStatus.PENDING, ReceiptStatus.FAIL])
def test_nonpassing_receipt_blocks_every_permission(status: ReceiptStatus) -> None:
    receipts = _receipts()
    receipts[0] = replace(receipts[0], status=status)
    decision = decide_gate0(
        receipts,
        policy=_policy(),
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )

    assert not decision.authorized
    assert not decision.contact_training_authorized
    assert not decision.model_selection_authorized
    assert not decision.outer_evaluation_authorized
    assert not decision.external_test_inference_authorized
    assert any(f"status_{status.value}" in reason for reason in decision.blocking_reasons)


def test_missing_receipt_blocks_and_does_not_invent_receipt_hash() -> None:
    decision = decide_gate0(
        _receipts()[:-1],
        policy=_policy(),
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    assert not decision.authorized
    assert "immutable_manifests:missing_receipt" in decision.blocking_reasons
    assert len(decision.receipt_sha256_by_gate) == len(GateName) - 1


def test_same_producer_and_auditor_is_not_independent() -> None:
    receipts = _receipts()
    index = list(GateName).index(GateName.JOINT_NESTED_SPLIT)
    receipts[index] = _receipt(
        GateName.JOINT_NESTED_SPLIT,
        producer="same-identity",
        auditor="same-identity",
    )
    decision = decide_gate0(
        receipts,
        policy=_policy(),
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    assert not decision.authorized
    assert (
        "joint_nested_split:producer_auditor_not_independent"
        in decision.blocking_reasons
    )


def test_any_sealed_or_test_access_blocks_gate0() -> None:
    receipts = _receipts()
    index = list(GateName).index(GateName.RUNTIME_LOCK)
    receipts[index] = _receipt(GateName.RUNTIME_LOCK, test_access=True)
    decision = decide_gate0(
        receipts,
        policy=_policy(),
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    assert not decision.authorized
    assert "runtime_lock:sealed_or_test_access" in decision.blocking_reasons


def test_expected_artifact_digest_must_match() -> None:
    policy = _policy(
        expected_artifacts=(
            ExpectedArtifact(
                gate=GateName.JOINT_NESTED_SPLIT,
                name="joint_nested_split_manifest",
                sha256="f" * 64,
            ),
        ),
    )
    decision = decide_gate0(
        _receipts(),
        policy=policy,
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    assert not decision.authorized
    assert (
        "joint_nested_split:artifact_digest_mismatch:"
        "joint_nested_split_manifest"
    ) in decision.blocking_reasons


def test_complete_receipt_is_anchored_not_only_its_artifacts() -> None:
    original = _receipts()
    policy = _policy(original)
    changed = list(original)
    changed[0] = replace(
        changed[0],
        assertion="a different self-declared assertion",
    )
    decision = decide_gate0(
        changed,
        policy=policy,
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    assert not decision.authorized
    assert (
        "data_license_provenance:receipt_digest_mismatch"
        in decision.blocking_reasons
    )


def test_receipt_order_does_not_change_decision_digest() -> None:
    policy = _policy()
    first = decide_gate0(
        _receipts(),
        policy=policy,
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    second = decide_gate0(
        reversed(_receipts()),
        policy=policy,
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    assert first == second
    assert first.decision_sha256 == second.decision_sha256


def test_duplicate_receipt_is_rejected() -> None:
    receipts = _receipts()
    with pytest.raises(Gate0Error, match="duplicate receipt"):
        decide_gate0(
            [*receipts, receipts[0]],
            policy=_policy(),
            evaluated_at_utc="2026-07-25T15:01:00Z",
        )


def test_malformed_digest_and_mutable_artifacts_are_rejected() -> None:
    with pytest.raises(Gate0Error, match="SHA-256"):
        DigestBinding(name="bad", sha256="ABC")
    with pytest.raises(Gate0Error, match="immutable tuple"):
        GateReceipt(
            gate=GateName.RUNTIME_LOCK,
            status=ReceiptStatus.PASS,
            producer_id="producer",
            auditor_id=None,
            protocol_sha256=PROTOCOL_SHA,
            artifacts=[DigestBinding(name="runtime", sha256="1" * 64)],  # type: ignore[arg-type]
            assertion="runtime passed",
            issued_at_utc="2026-07-25T15:00:00Z",
        )


def test_launcher_verification_fails_closed_on_digest_or_permission() -> None:
    policy = _policy()
    decision = decide_gate0(
        _receipts()[:-1],
        policy=policy,
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    with pytest.raises(Gate0Error, match="does not authorize"):
        verify_gate0_decision(
            decision,
            expected_policy_sha256=policy.policy_sha256,
            expected_decision_sha256=decision.decision_sha256,
            required_permission="contact_training",
        )
    with pytest.raises(Gate0Error, match="decision digest"):
        verify_gate0_decision(
            decision,
            expected_policy_sha256=policy.policy_sha256,
            expected_decision_sha256="f" * 64,
            required_permission="contact_training",
        )


def test_gate0_policy_cannot_grant_outer_or_external_scoring() -> None:
    with pytest.raises(Gate0Error, match="later model-freeze authority"):
        _policy(
            authorized_permissions=("contact_training", "outer_evaluation"),
        )


def test_timestamp_validation_rejects_calendar_nonsense() -> None:
    with pytest.raises(Gate0Error, match="valid canonical UTC"):
        decide_gate0(
            _receipts(),
            policy=_policy(),
            evaluated_at_utc="2026-99-99T15:01:00Z",
        )


def test_policy_sequences_must_be_deeply_immutable_and_typed() -> None:
    with pytest.raises(Gate0Error, match="immutable tuple"):
        Gate0Policy(
            protocol_sha256=PROTOCOL_SHA,
            required_gates=list(GateName),  # type: ignore[arg-type]
            expected_receipts=tuple(
                ExpectedReceipt(gate=item.gate, sha256=item.receipt_sha256)
                for item in _receipts()
            ),
        )
    with pytest.raises(Gate0Error, match="expected_artifacts"):
        Gate0Policy(protocol_sha256=PROTOCOL_SHA)
    complete_artifacts = _policy().expected_artifacts
    with pytest.raises(Gate0Error, match="expected_receipts"):
        Gate0Policy(
            protocol_sha256=PROTOCOL_SHA,
            expected_artifacts=complete_artifacts,
        )
    with pytest.raises(Gate0Error, match="ExpectedArtifact"):
        Gate0Policy(
            protocol_sha256=PROTOCOL_SHA,
            expected_artifacts=("not-an-artifact",),  # type: ignore[arg-type]
            expected_receipts=tuple(
                ExpectedReceipt(gate=item.gate, sha256=item.receipt_sha256)
                for item in _receipts()
            ),
        )


def test_untrusted_decision_payload_round_trips_strictly() -> None:
    policy = _policy()
    decision = decide_gate0(
        _receipts(),
        policy=policy,
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    parsed = gate0_decision_from_payload(decision.canonical_payload())
    assert parsed == decision
    assert parsed.decision_sha256 == decision.decision_sha256


def test_untrusted_decision_payload_rejects_unknown_fields_and_string_bools() -> None:
    policy = _policy()
    decision = decide_gate0(
        _receipts(),
        policy=policy,
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    extra = dict(decision.canonical_payload())
    extra["silent_override"] = True
    with pytest.raises(Gate0Error, match="locked schema"):
        gate0_decision_from_payload(extra)

    string_bool = dict(decision.canonical_payload())
    string_bool["permissions"] = {
        **string_bool["permissions"],  # type: ignore[dict-item]
        "contact_training": "true",
    }
    with pytest.raises(Gate0Error, match="must be a boolean"):
        gate0_decision_from_payload(string_bool)


def test_self_issued_replacement_cannot_replace_externally_anchored_receipt() -> None:
    original = _receipts()
    policy = _policy(original)
    changed = list(original)
    index = list(GateName).index(GateName.JOINT_NESTED_SPLIT)
    changed[index] = replace(
        changed[index],
        producer_id="self-issued-producer",
        auditor_id="self-issued-auditor",
    )

    decision = decide_gate0(
        changed,
        policy=policy,
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    assert not decision.authorized
    assert "joint_nested_split:receipt_digest_mismatch" in decision.blocking_reasons


def test_locked_policy_cannot_omit_gate_or_downgrade_independent_audit() -> None:
    policy = _policy()
    with pytest.raises(Gate0Error, match="complete locked Gate 0 set"):
        replace(
            policy,
            required_gates=tuple(
                gate for gate in GateName if gate is not GateName.RUNTIME_LOCK
            ),
        )
    with pytest.raises(Gate0Error, match="locked audit set"):
        replace(
            policy,
            independent_audit_gates=tuple(
                gate
                for gate in policy.independent_audit_gates
                if gate is not GateName.FOLD_CACHE_EQUALITY
            ),
        )


def test_every_gate_requires_external_artifact_and_receipt_anchors() -> None:
    policy = _policy()
    with pytest.raises(Gate0Error, match="expected_artifacts"):
        replace(
            policy,
            expected_artifacts=tuple(
                item
                for item in policy.expected_artifacts
                if item.gate is not GateName.RUNTIME_LOCK
            ),
        )
    with pytest.raises(Gate0Error, match="expected_receipts"):
        replace(
            policy,
            expected_receipts=tuple(
                item
                for item in policy.expected_receipts
                if item.gate is not GateName.RUNTIME_LOCK
            ),
        )


@pytest.mark.parametrize(
    ("producer", "auditor"),
    [
        ("Producer", "independent-auditor"),
        ("producer", "Independent-Auditor"),
        ("prod\u200ducer", "independent-auditor"),
        ("pr\u043educer", "independent-auditor"),
    ],
)
def test_authority_ids_reject_case_and_unicode_spoofing(
    producer: str,
    auditor: str,
) -> None:
    with pytest.raises(Gate0Error, match="canonical lowercase ASCII"):
        _receipt(
            GateName.JOINT_NESTED_SPLIT,
            producer=producer,
            auditor=auditor,
        )


def test_auditor_cannot_also_produce_a_different_gate_receipt() -> None:
    receipts = _receipts()
    split_index = list(GateName).index(GateName.JOINT_NESTED_SPLIT)
    runtime_index = list(GateName).index(GateName.RUNTIME_LOCK)
    receipts[split_index] = replace(
        receipts[split_index],
        auditor_id="cross-role-authority",
    )
    receipts[runtime_index] = replace(
        receipts[runtime_index],
        producer_id="cross-role-authority",
    )
    decision = decide_gate0(
        receipts,
        policy=_policy(receipts),
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    assert not decision.authorized
    assert (
        "joint_nested_split:auditor_also_receipt_producer"
        in decision.blocking_reasons
    )


def test_required_independent_auditor_cannot_be_omitted() -> None:
    receipts = _receipts()
    index = list(GateName).index(GateName.INDEPENDENT_LEAKAGE_AUDIT)
    receipts[index] = replace(receipts[index], auditor_id=None)
    decision = decide_gate0(
        receipts,
        policy=_policy(receipts),
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    assert not decision.authorized
    assert (
        "independent_leakage_audit:independent_auditor_missing"
        in decision.blocking_reasons
    )


def test_policy_rejects_duplicate_gate_artifact_and_receipt_anchors() -> None:
    policy = _policy()
    with pytest.raises(Gate0Error, match="required_gates contains duplicates"):
        replace(
            policy,
            required_gates=(*policy.required_gates, policy.required_gates[0]),
        )
    with pytest.raises(Gate0Error, match="duplicate gate/name"):
        replace(
            policy,
            expected_artifacts=(
                *policy.expected_artifacts,
                policy.expected_artifacts[0],
            ),
        )
    with pytest.raises(Gate0Error, match="duplicate gates"):
        replace(
            policy,
            expected_receipts=(
                *policy.expected_receipts,
                policy.expected_receipts[0],
            ),
        )


def test_unknown_gate_and_nonreceipt_inputs_are_rejected() -> None:
    with pytest.raises(Gate0Error, match="GateName"):
        GateReceipt(
            gate="unknown_gate",  # type: ignore[arg-type]
            status=ReceiptStatus.PASS,
            producer_id="producer",
            auditor_id=None,
            protocol_sha256=PROTOCOL_SHA,
            artifacts=(DigestBinding(name="manifest", sha256="1" * 64),),
            assertion="unknown gate",
            issued_at_utc="2026-07-25T15:00:00Z",
        )
    with pytest.raises(Gate0Error, match="only GateReceipt"):
        decide_gate0(
            [*_receipts(), object()],  # type: ignore[list-item]
            policy=_policy(),
            evaluated_at_utc="2026-07-25T15:01:00Z",
        )
    with pytest.raises(Gate0Error, match="validated Gate0Policy"):
        decide_gate0(
            _receipts(),
            policy=object(),  # type: ignore[arg-type]
            evaluated_at_utc="2026-07-25T15:01:00Z",
        )


@pytest.mark.parametrize("missing", [True, False])
def test_actual_artifact_set_must_exactly_match_external_anchor(
    missing: bool,
) -> None:
    original = _receipts()
    anchored_policy = _policy(original)
    changed = list(original)
    index = list(GateName).index(GateName.JOINT_NESTED_SPLIT)
    if missing:
        changed[index] = replace(changed[index], artifacts=())
        expected_reason = (
            "joint_nested_split:expected_artifact_missing:"
            "joint_nested_split_manifest"
        )
    else:
        changed[index] = replace(
            changed[index],
            artifacts=(
                *changed[index].artifacts,
                DigestBinding(name="unanchored_extra", sha256="f" * 64),
            ),
        )
        expected_reason = "joint_nested_split:unexpected_artifact:unanchored_extra"
    policy = Gate0Policy(
        protocol_sha256=PROTOCOL_SHA,
        expected_artifacts=anchored_policy.expected_artifacts,
        expected_receipts=tuple(
            ExpectedReceipt(gate=item.gate, sha256=item.receipt_sha256)
            for item in changed
        ),
    )

    decision = decide_gate0(
        changed,
        policy=policy,
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    assert not decision.authorized
    assert expected_reason in decision.blocking_reasons


def test_protocol_and_launcher_policy_digests_are_bound_exactly() -> None:
    receipts = _receipts()
    receipts[0] = replace(receipts[0], protocol_sha256="b" * 64)
    policy = _policy()
    decision = decide_gate0(
        receipts,
        policy=policy,
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    assert not decision.authorized
    assert (
        "data_license_provenance:protocol_digest_mismatch"
        in decision.blocking_reasons
    )
    with pytest.raises(Gate0Error, match="policy digest"):
        verify_gate0_decision(
            decision,
            expected_policy_sha256="f" * 64,
            expected_decision_sha256=decision.decision_sha256,
            required_permission="contact_training",
        )


def test_future_receipt_blocks_even_when_every_digest_is_anchored() -> None:
    receipts = _receipts()
    receipts[0] = replace(
        receipts[0],
        issued_at_utc="2026-07-25T15:02:00Z",
    )
    decision = decide_gate0(
        receipts,
        policy=_policy(receipts),
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    assert not decision.authorized
    assert (
        "data_license_provenance:issued_after_evaluation"
        in decision.blocking_reasons
    )


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-07-25T15:00:00.000Z",
        "2026-07-25T15:00:00+00:00",
        "2026-07-25 15:00:00Z",
        "2026-07-25T15:00Z",
        "2026-7-25T15:00:00Z",
    ],
)
def test_timestamps_have_one_canonical_utc_spelling(timestamp: str) -> None:
    with pytest.raises(Gate0Error, match="canonical UTC"):
        replace(_receipts()[0], issued_at_utc=timestamp)


def test_boolean_and_text_fields_are_strictly_typed() -> None:
    with pytest.raises(Gate0Error, match="explicit boolean"):
        replace(
            _receipts()[0],
            sealed_or_test_data_accessed=0,  # type: ignore[arg-type]
        )
    with pytest.raises(Gate0Error, match="normalized nonempty text"):
        replace(_receipts()[0], assertion="passed\nforged")
    with pytest.raises(Gate0Error, match="canonical lowercase ASCII"):
        replace(_receipts()[0], producer_id=7)  # type: ignore[arg-type]

    decision = decide_gate0(
        _receipts(),
        policy=_policy(),
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    with pytest.raises(Gate0Error, match="explicit boolean"):
        replace(decision, authorized=1)  # type: ignore[arg-type]


def test_decision_nested_structures_are_deeply_immutable() -> None:
    decision = decide_gate0(
        _receipts(),
        policy=_policy(),
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    mutable_binding = [
        decision.receipt_sha256_by_gate[0][0],
        decision.receipt_sha256_by_gate[0][1],
    ]
    with pytest.raises(Gate0Error, match="immutable"):
        replace(
            decision,
            receipt_sha256_by_gate=(
                mutable_binding,  # type: ignore[arg-type]
                *decision.receipt_sha256_by_gate[1:],
            ),
        )
    with pytest.raises(Gate0Error, match="immutable tuple"):
        replace(
            decision,
            blocking_reasons=[],  # type: ignore[arg-type]
        )


def test_authorized_decision_must_bind_all_gates_and_never_post_gate0_scope() -> None:
    decision = decide_gate0(
        _receipts(),
        policy=_policy(),
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    with pytest.raises(Gate0Error, match="bind every locked"):
        replace(
            decision,
            receipt_sha256_by_gate=decision.receipt_sha256_by_gate[:-1],
        )
    with pytest.raises(Gate0Error, match="cannot grant outer"):
        replace(decision, outer_evaluation_authorized=True)
    with pytest.raises(Gate0Error, match="cannot grant outer"):
        replace(decision, external_test_inference_authorized=True)


def test_launcher_cannot_use_permission_not_granted_by_policy() -> None:
    policy = _policy(authorized_permissions=("contact_training",))
    decision = decide_gate0(
        _receipts(),
        policy=policy,
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    assert decision.authorized
    assert decision.contact_training_authorized
    assert not decision.model_selection_authorized
    with pytest.raises(Gate0Error, match="does not authorize model_selection"):
        verify_gate0_decision(
            decision,
            expected_policy_sha256=policy.policy_sha256,
            expected_decision_sha256=decision.decision_sha256,
            required_permission="model_selection",
        )
    with pytest.raises(Gate0Error, match="unknown Gate 0 permission"):
        verify_gate0_decision(
            decision,
            expected_policy_sha256=policy.policy_sha256,
            expected_decision_sha256=decision.decision_sha256,
            required_permission="contact_training\u0000",
        )


@pytest.mark.parametrize(
    "permission",
    ["outer_evaluation", "external_test_inference"],
)
def test_policy_never_grants_post_gate0_permission(permission: str) -> None:
    with pytest.raises(Gate0Error, match="later model-freeze authority"):
        _policy(authorized_permissions=("contact_training", permission))


def test_raw_json_parser_round_trips_and_rejects_duplicate_keys() -> None:
    policy = _policy()
    decision = decide_gate0(
        _receipts(),
        policy=policy,
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    encoded = json.dumps(
        decision.canonical_payload(),
        separators=(",", ":"),
        ensure_ascii=True,
    )
    assert gate0_decision_from_json(encoded) == decision
    assert gate0_decision_from_json(encoded.encode("utf-8")) == decision

    duplicate_top_level = encoded.replace(
        '"authorized":true',
        '"authorized":true,"authorized":false',
        1,
    )
    with pytest.raises(Gate0Error, match="duplicate object key 'authorized'"):
        gate0_decision_from_json(duplicate_top_level)

    duplicate_nested = encoded.replace(
        '"contact_training":true',
        '"contact_training":true,"contact_training":false',
        1,
    )
    with pytest.raises(Gate0Error, match="duplicate object key 'contact_training'"):
        gate0_decision_from_json(duplicate_nested)


def test_raw_json_parser_rejects_encoding_and_nonfinite_ambiguity() -> None:
    with pytest.raises(Gate0Error, match="UTF-8"):
        gate0_decision_from_json('{"authorized":true}'.encode("utf-16"))
    with pytest.raises(Gate0Error, match="unsupported constant"):
        gate0_decision_from_json('{"value":NaN}')


def test_canonical_decision_digest_is_independent_of_json_layout() -> None:
    decision = decide_gate0(
        _receipts(),
        policy=_policy(),
        evaluated_at_utc="2026-07-25T15:01:00Z",
    )
    compact = json.dumps(decision.canonical_payload(), separators=(",", ":"))
    reordered = json.dumps(
        decision.canonical_payload(),
        sort_keys=True,
        indent=2,
    )
    parsed_compact = gate0_decision_from_json(compact)
    parsed_reordered = gate0_decision_from_json(reordered)
    assert parsed_compact.decision_sha256 == decision.decision_sha256
    assert parsed_reordered.decision_sha256 == decision.decision_sha256
