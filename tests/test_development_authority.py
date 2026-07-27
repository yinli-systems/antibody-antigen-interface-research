from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from interaction_first.development_authority import (
    AuthorityCounts,
    DevelopmentAuthorityError,
    DevelopmentPolicy,
    DevelopmentPurpose,
    DevelopmentRequest,
    canonical_json_bytes,
    decide_development_authority,
    development_policy_from_json,
    membership_sha256,
    verify_development_decision,
)


def _authority_ids() -> tuple[str, ...]:
    return tuple(f"instance-{index:04d}" for index in range(8))


def _policy() -> DevelopmentPolicy:
    return DevelopmentPolicy(
        protocol_sha256="1" * 64,
        authority_manifest_sha256="2" * 64,
        cache_authorization_manifest_sha256="3" * 64,
        cache_manifest_sha256="4" * 64,
        authority_membership_sha256=membership_sha256(_authority_ids()),
        counts=AuthorityCounts(
            source_instances=12,
            exact_pairs=8,
            antibody_entities=7,
            antigen_entities=6,
        ),
    )


def _request(**changes: object) -> DevelopmentRequest:
    values: dict[str, object] = {
        "run_id": "pairscale-overfit-32",
        "purpose": DevelopmentPurpose.OVERFIT_SANITY,
        "subset_instance_ids": _authority_ids()[:4],
        "subset_manifest_sha256": "5" * 64,
        "requested_at_utc": "2026-07-27T16:30:00Z",
    }
    values.update(changes)
    return DevelopmentRequest(**values)  # type: ignore[arg-type]


def _decision(
    request: DevelopmentRequest | None = None,
    *,
    expected_policy_sha256: str | None = None,
    authority_ids: tuple[str, ...] | None = None,
):
    policy = _policy()
    return decide_development_authority(
        policy=policy,
        request=request or _request(),
        authority_instance_ids=authority_ids or _authority_ids(),
        expected_policy_sha256=(
            expected_policy_sha256 or policy.policy_sha256
        ),
        evaluated_at_utc="2026-07-27T16:31:00Z",
    )


def test_minimal_train_only_request_authorizes_only_development() -> None:
    policy = _policy()
    request = _request()
    decision = _decision(request)

    assert decision.authorized
    assert decision.blocking_reasons == ()
    assert decision.canonical_payload()["test_data_access_authorized"] is False
    assert decision.canonical_payload()["model_selection_authorized"] is False
    assert decision.canonical_payload()["confirmatory_claim_authorized"] is False
    verify_development_decision(
        decision,
        expected_policy_sha256=policy.policy_sha256,
        expected_decision_sha256=decision.decision_sha256,
        expected_run_id=request.run_id,
        expected_purpose=request.purpose,
        expected_subset_manifest_sha256=request.subset_manifest_sha256,
    )
    with pytest.raises(DevelopmentAuthorityError, match="cannot grant"):
        verify_development_decision(
            decision,
            expected_policy_sha256=policy.policy_sha256,
            expected_decision_sha256=decision.decision_sha256,
            expected_run_id=request.run_id,
            expected_purpose=request.purpose,
            expected_subset_manifest_sha256=request.subset_manifest_sha256,
            required_permission="contact_training",
        )


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("sealed_data_accessed", "sealed_data_accessed"),
        ("validation_data_accessed", "validation_data_accessed"),
        ("test_data_accessed", "test_data_accessed"),
        ("model_selection_requested", "model_selection_requested"),
        ("checkpoint_promotion_requested", "checkpoint_promotion_requested"),
        ("confirmatory_claim_requested", "confirmatory_claim_requested"),
    ],
)
def test_each_forbidden_request_flag_blocks(field: str, reason: str) -> None:
    decision = _decision(_request(**{field: True}))
    assert not decision.authorized
    assert decision.blocking_reasons == (reason,)


def test_nontrain_split_and_unknown_instance_block() -> None:
    request = _request(
        source_split="validation",
        subset_instance_ids=("instance-0000", "unknown-instance"),
    )
    decision = _decision(request)
    assert not decision.authorized
    assert decision.blocking_reasons == (
        "source_split_not_train",
        "subset_contains_unknown_instance",
    )


def test_duplicate_and_ambiguous_identifiers_fail_before_decision() -> None:
    with pytest.raises(DevelopmentAuthorityError, match="duplicates"):
        _request(subset_instance_ids=("instance-0000", "instance-0000"))
    with pytest.raises(DevelopmentAuthorityError, match="canonical identifier"):
        _request(subset_instance_ids=("instance-0000", "test\ninstance"))
    with pytest.raises(DevelopmentAuthorityError, match="duplicates"):
        _decision(authority_ids=(*_authority_ids(), _authority_ids()[0]))


def test_policy_reanchor_and_authority_tamper_block() -> None:
    decision = _decision(expected_policy_sha256="f" * 64)
    assert not decision.authorized
    assert decision.blocking_reasons == ("policy_digest_mismatch",)

    changed_authority = ("instance-changed",) + _authority_ids()[1:]
    decision = _decision(authority_ids=changed_authority)
    assert not decision.authorized
    assert decision.blocking_reasons == (
        "authority_membership_digest_mismatch",
        "subset_contains_unknown_instance",
    )


def test_decision_anchor_tamper_is_rejected() -> None:
    policy = _policy()
    request = _request()
    decision = _decision(request)
    with pytest.raises(DevelopmentAuthorityError, match="decision digest mismatch"):
        verify_development_decision(
            decision,
            expected_policy_sha256=policy.policy_sha256,
            expected_decision_sha256="e" * 64,
            expected_run_id=request.run_id,
            expected_purpose=request.purpose,
            expected_subset_manifest_sha256=request.subset_manifest_sha256,
        )
    changed = replace(decision, run_id="different-run")
    with pytest.raises(DevelopmentAuthorityError, match="decision digest mismatch"):
        verify_development_decision(
            changed,
            expected_policy_sha256=policy.policy_sha256,
            expected_decision_sha256=decision.decision_sha256,
            expected_run_id=request.run_id,
            expected_purpose=request.purpose,
            expected_subset_manifest_sha256=request.subset_manifest_sha256,
        )


def test_policy_json_round_trip_and_forbidden_field_injection() -> None:
    policy = _policy()
    restored = development_policy_from_json(
        canonical_json_bytes(policy.canonical_payload())
    )
    assert restored == policy
    assert restored.policy_sha256 == policy.policy_sha256

    injected = dict(policy.canonical_payload())
    injected["contact_density"] = 0.5
    with pytest.raises(DevelopmentAuthorityError, match="extra"):
        development_policy_from_json(canonical_json_bytes(injected))


def test_policy_json_rejects_duplicate_keys_unknown_purpose_and_true_access() -> None:
    with pytest.raises(DevelopmentAuthorityError, match="duplicate key"):
        development_policy_from_json(b'{"schema_version":"a","schema_version":"b"}')

    payload = json.loads(canonical_json_bytes(_policy().canonical_payload()))
    payload["allowed_purposes"].append("paper_metric_tuning")
    with pytest.raises(DevelopmentAuthorityError, match="unknown development purpose"):
        development_policy_from_json(canonical_json_bytes(payload))

    payload = json.loads(canonical_json_bytes(_policy().canonical_payload()))
    payload["test_data_access_authorized"] = True
    with pytest.raises(DevelopmentAuthorityError, match="exactly false"):
        development_policy_from_json(canonical_json_bytes(payload))


def test_order_independence_and_all_purposes_are_authorized() -> None:
    policy = _policy()
    forward = _decision()
    reverse = decide_development_authority(
        policy=policy,
        request=_request(subset_instance_ids=tuple(reversed(_authority_ids()[:4]))),
        authority_instance_ids=reversed(_authority_ids()),
        expected_policy_sha256=policy.policy_sha256,
        evaluated_at_utc="2026-07-27T16:31:00Z",
    )
    assert reverse.authorized
    assert reverse.subset_membership_sha256 == forward.subset_membership_sha256

    for purpose in DevelopmentPurpose:
        assert _decision(_request(purpose=purpose)).authorized


def test_membership_digest_reproduces_frozen_lf_framing() -> None:
    values = ("instance-c", "instance-a", "pdb_00001g6v_K_+")
    expected = hashlib.sha256(
        b"instance-a\ninstance-c\npdb_00001g6v_K_+\n"
    ).hexdigest()
    assert membership_sha256(values) == expected


def test_run_identifier_cannot_be_reinterpreted_as_a_path() -> None:
    with pytest.raises(DevelopmentAuthorityError, match="run identifier"):
        _request(run_id="../outside")


def test_repository_policy_binds_protocol_and_locked_train_inventory() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    protocol_path = (
        repository_root
        / "protocols"
        / "PAIRSCALE_DEVELOPMENT_ONLY_NONCONFIRMATORY_PROTOCOL_V1.md"
    )
    policy_path = (
        repository_root
        / "protocols"
        / "pairscale_development_only_nonconfirmatory_policy_v1.json"
    )
    policy = development_policy_from_json(policy_path.read_bytes())

    assert policy.protocol_sha256 == hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    assert policy.authority_manifest_sha256 == (
        "e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b"
    )
    assert policy.authority_membership_sha256 == (
        "e3465e558ab8231bb6280d038d07856846ab31608e6e8d3867da3348e85c0a4d"
    )
    assert policy.counts == AuthorityCounts(
        source_instances=3229,
        exact_pairs=1830,
        antibody_entities=1383,
        antigen_entities=1172,
    )
