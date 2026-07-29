"""Adversarial tests for prospective fail-closed dataset admission."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path

import pytest

from interaction_first.data_admission import (
    ADMISSION_RECEIPT_FORMAT,
    CANONICAL_JSON_POLICY,
    DUPLICATE_AUDIT_RECEIPT_FORMAT,
    EXACT_DUPLICATE_METHOD,
    LINEAGE_RECEIPT_FORMAT,
    MANIFEST_ENTRY_ORDER_POLICY,
    NEAR_DUPLICATE_METHOD,
    NEAR_DUPLICATE_THRESHOLD,
    SHA_MANIFEST_RECEIPT_FORMAT,
    SOURCE_REGISTRY_ENTRY_FORMAT,
    DataAdmissionError,
    make_content_addressed_receipt,
    make_source_registry_entry,
    validate_dataset_admission,
)


_ADMISSION_POLICY_SHA256 = hashlib.sha256(
    b"admission-policy-v1"
).hexdigest()
_VALIDATION_CUTOFF = "2026-07-25T01:05:00Z"


def _digest(label: str) -> str:
    """Return a deterministic synthetic digest without touching a dataset."""
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _cas(digest: str) -> str:
    """Return the content-addressed URI used by synthetic receipts."""
    return f"cas://sha256/{digest}"


def _rewrap(receipt: dict[str, object]) -> dict[str, object]:
    """Recompute a receipt after an intentional test-fixture mutation."""
    return make_content_addressed_receipt(
        str(receipt["receipt_format"]),
        receipt["payload"],
    )


def _manifest_receipt(
    dataset_id: str,
    stage: str,
    count: int,
) -> dict[str, object]:
    """Build one synthetic raw or normalized manifest receipt."""
    manifest_hash = _digest(f"{dataset_id}:{stage}:manifest")
    return make_content_addressed_receipt(
        SHA_MANIFEST_RECEIPT_FORMAT,
        {
            "dataset_id": dataset_id,
            "stage": stage,
            "manifest_sha256": manifest_hash,
            "manifest_uri": _cas(manifest_hash),
            "entry_count": count,
            "total_size_bytes": count * 113,
            "producer_code_sha256": _digest(f"{stage}:producer"),
            "canonicalization_policy": CANONICAL_JSON_POLICY,
            "entry_order_policy": MANIFEST_ENTRY_ORDER_POLICY,
            "created_at_utc": "2026-07-25T01:02:03Z",
        },
    )


def _lineage_receipt(
    dataset_id: str,
    stage_manifest_hashes: dict[str, str],
    stage_counts: dict[str, int],
) -> dict[str, object]:
    """Build complete synthetic lineage across every admission stage."""
    lineage_hash = _digest(f"{dataset_id}:lineage")
    return make_content_addressed_receipt(
        LINEAGE_RECEIPT_FORMAT,
        {
            "dataset_id": dataset_id,
            **{
                f"{stage}_manifest_sha256": stage_manifest_hashes[stage]
                for stage in (
                    "raw",
                    "normalized",
                    "qc_passed",
                    "deduplicated",
                    "admitted",
                )
            },
            "lineage_manifest_sha256": lineage_hash,
            "lineage_manifest_uri": _cas(lineage_hash),
            **{
                f"{stage}_row_count": stage_counts[stage]
                for stage in (
                    "normalized",
                    "qc_passed",
                    "deduplicated",
                    "admitted",
                )
            },
            **{
                f"{stage}_rows_with_lineage_count": stage_counts[stage]
                for stage in (
                    "normalized",
                    "qc_passed",
                    "deduplicated",
                    "admitted",
                )
            },
            "lineage_edge_count": sum(
                stage_counts[stage]
                for stage in (
                    "normalized",
                    "qc_passed",
                    "deduplicated",
                    "admitted",
                )
            )
            + 2,
            **{
                f"missing_{stage}_row_count": 0
                for stage in (
                    "normalized",
                    "qc_passed",
                    "deduplicated",
                    "admitted",
                )
            },
            "orphan_source_reference_count": 0,
            **{
                f"duplicate_{stage}_row_id_count": 0
                for stage in (
                    "raw",
                    "normalized",
                    "qc_passed",
                    "deduplicated",
                    "admitted",
                )
            },
            "transform_code_sha256": _digest("normalizer"),
            "created_at_utc": "2026-07-25T01:03:00Z",
        },
    )


def _duplicate_receipt(
    dataset_id: str,
    normalized_manifest_hash: str,
    axis: str,
    *,
    candidate_count: int,
) -> dict[str, object]:
    """Build a resolved exact and near-duplicate audit receipt."""
    registry_hash = _digest(f"{axis}:registry")
    resolution_hash = _digest(f"{axis}:resolutions")
    candidate_entity_hash = _digest(f"{axis}:candidate-entities")
    duplicate_count = 2 if candidate_count else 0
    return make_content_addressed_receipt(
        DUPLICATE_AUDIT_RECEIPT_FORMAT,
        {
            "dataset_id": dataset_id,
            "axis": axis,
            "cross_source": True,
            "candidate_manifest_sha256": normalized_manifest_hash,
            "candidate_entity_manifest_sha256": candidate_entity_hash,
            "candidate_entity_manifest_uri": _cas(candidate_entity_hash),
            "candidate_entity_count": candidate_count,
            "comparison_registry_sha256": registry_hash,
            "comparison_registry_uri": _cas(registry_hash),
            "comparison_source_count": 3,
            "comparison_count": 100 if candidate_count else 0,
            "exact_duplicate_method": EXACT_DUPLICATE_METHOD,
            "near_duplicate_method": NEAR_DUPLICATE_METHOD,
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "exact_duplicate_count": 1 if candidate_count else 0,
            "near_duplicate_count_excluding_exact": 1 if candidate_count else 0,
            "resolved_duplicate_count": duplicate_count,
            "unresolved_duplicate_count": 0,
            "resolution_action_counts": {
                "quarantined": 0,
                "merged": 0,
                "linked_to_split_component": duplicate_count,
            },
            "resolution_manifest_sha256": resolution_hash,
            "resolution_manifest_uri": _cas(resolution_hash),
            "auditor_code_sha256": _digest("duplicate-auditor"),
            "created_at_utc": "2026-07-25T01:04:00Z",
        },
    )


def _valid_admission() -> dict[str, object]:
    """Build a valid measured-interaction admission using synthetic metadata."""
    dataset_id = "AVIDa_fixture_v1"
    raw = _manifest_receipt(dataset_id, "raw", 10)
    normalized = _manifest_receipt(dataset_id, "normalized", 9)
    qc_passed = _manifest_receipt(dataset_id, "qc_passed", 9)
    deduplicated = _manifest_receipt(dataset_id, "deduplicated", 8)
    admitted = _manifest_receipt(dataset_id, "admitted", 8)
    manifest_receipts = {
        "raw": raw,
        "normalized": normalized,
        "qc_passed": qc_passed,
        "deduplicated": deduplicated,
        "admitted": admitted,
    }
    stage_manifest_hashes = {
        stage: stage_receipt["payload"]["manifest_sha256"]
        for stage, stage_receipt in manifest_receipts.items()
    }
    stage_counts = {
        "raw": 10,
        "normalized": 9,
        "qc_passed": 9,
        "deduplicated": 8,
        "admitted": 8,
    }
    license_hash = _digest("CC-BY-4.0-text")
    payload = {
        "dataset_id": dataset_id,
        "source": {
            "source_id": "avida",
            "source_name": "AVIDa fixture",
            "landing_page_url": "https://example.org/datasets/avida/v1",
            "revision": "v1.0.0",
            "revision_type": "versioned_release",
            "acquired_at_utc": "2026-07-25T01:00:00Z",
            "semantic_type": "measured_interaction",
            "record_unit": "interaction_pair",
            "molecular_axes": ["antibody", "antigen"],
        },
        "counts": {
            "raw_records": 10,
            "normalized_records": 9,
            "qc_passed_records": 9,
            "deduplicated_records": 8,
            "admitted_records": 8,
            "experimental_label_records": 8,
            "predicted_structure_records": 2,
            "synthetic_decoy_records": 0,
            "generated_or_predicted_records": 0,
            "residue_tokens": 1600,
            "antibody_entities": 7,
            "antigen_entities": 4,
        },
        "label_semantics": {
            "experimental_positive_records": 5,
            "experimental_nonbinder_records": 3,
            "unknown_pair_records": 0,
            "generated_negative_records": 0,
            "unknown_pairs_used_as_negative": False,
        },
        "raw_manifest_receipt": raw,
        "normalized_manifest_receipt": normalized,
        "qc_passed_manifest_receipt": qc_passed,
        "deduplicated_manifest_receipt": deduplicated,
        "admitted_manifest_receipt": admitted,
        "lineage_receipt": _lineage_receipt(
            dataset_id,
            stage_manifest_hashes,
            stage_counts,
        ),
        "license": {
            "license_name": "Creative Commons Attribution 4.0",
            "spdx_expression": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
            "license_text_sha256": license_hash,
            "license_text_uri": _cas(license_hash),
            "raw_redistribution_allowed": True,
            "normalized_redistribution_allowed": True,
            "commercial_use_allowed": True,
            "derivatives_allowed": True,
            "attribution_required": True,
            "share_alike_required": False,
        },
        "routing": {
            "track": "general_research",
            "commercial_use_requested": False,
            "redistribute_raw_requested": True,
            "redistribute_normalized_requested": True,
            "propagate_noncommercial_restriction": False,
            "propagate_share_alike_restriction": False,
            "propagate_redistribution_restrictions": False,
            "propagate_attribution_requirement": True,
        },
        "duplicate_audit_receipts": {
            "antibody": _duplicate_receipt(
                dataset_id,
                stage_manifest_hashes["admitted"],
                "antibody",
                candidate_count=7,
            ),
            "antigen": _duplicate_receipt(
                dataset_id,
                stage_manifest_hashes["admitted"],
                "antigen",
                candidate_count=4,
            ),
        },
        "admission_policy_sha256": _ADMISSION_POLICY_SHA256,
        "source_registry_entry_sha256": "0" * 64,
        "created_at_utc": "2026-07-25T01:05:00Z",
    }
    registry = _source_registry_for_payload(payload)
    payload["source_registry_entry_sha256"] = registry["receipt_sha256"]
    return make_content_addressed_receipt(ADMISSION_RECEIPT_FORMAT, payload)


def _source_registry_for_payload(
    payload: dict[str, object],
) -> dict[str, object]:
    """Build the synthetic caller-owned registry entry for one payload."""
    return make_source_registry_entry(
        dataset_id=payload["dataset_id"],
        source=payload["source"],
        license_value=payload["license"],
        admission_policy_sha256=payload["admission_policy_sha256"],
        comparison_registry_sha256_by_axis={
            axis: payload["duplicate_audit_receipts"][axis]["payload"][
                "comparison_registry_sha256"
            ]
            for axis in ("antibody", "antigen")
        },
        created_at_utc="2026-07-25T00:59:00Z",
    )


def _validate(receipt: dict[str, object]):
    """Validate a fixture with independently supplied frozen caller inputs."""
    try:
        registry = _source_registry_for_payload(receipt["payload"])
    except (DataAdmissionError, KeyError, TypeError):
        base = _valid_admission()
        registry = _source_registry_for_payload(base["payload"])
    return validate_dataset_admission(
        receipt,
        expected_admission_policy_sha256=_ADMISSION_POLICY_SHA256,
        source_registry_entry=registry,
        expected_source_registry_entry_sha256=registry["receipt_sha256"],
        not_after_utc=_VALIDATION_CUTOFF,
    )


def _mutate_payload(
    receipt: dict[str, object],
    mutation: object,
) -> dict[str, object]:
    """Deep-copy, mutate, and rehash an admission payload."""
    changed = deepcopy(receipt)
    mutation(changed["payload"])
    try:
        registry = _source_registry_for_payload(changed["payload"])
    except (DataAdmissionError, KeyError, TypeError):
        pass
    else:
        changed["payload"]["source_registry_entry_sha256"] = registry[
            "receipt_sha256"
        ]
    return _rewrap(changed)


def test_valid_admission_returns_only_immutable_metadata() -> None:
    """A valid measured source returns a frozen decision without data access."""
    decision = _validate(_valid_admission())

    assert decision.dataset_id == "AVIDa_fixture_v1"
    assert decision.semantic_type == "measured_interaction"
    assert decision.molecular_axes == ("antibody", "antigen")
    assert decision.routing_track == "general_research"
    assert decision.contamination_policy == "none"
    assert not decision.noncommercial_contaminated
    assert decision.counts.admitted_records == 8
    assert len(decision.duplicate_audit_receipt_sha256) == 2


def test_receipt_canonicalization_sorts_keys_but_preserves_array_order() -> None:
    """Document the exact object-key and input-array ordering convention."""
    first = make_content_addressed_receipt(
        "fixture_receipt_v1",
        {"b": 2, "a": 1, "rows": ["a", "b"]},
    )
    reordered_keys = make_content_addressed_receipt(
        "fixture_receipt_v1",
        {"rows": ["a", "b"], "a": 1, "b": 2},
    )
    reordered_rows = make_content_addressed_receipt(
        "fixture_receipt_v1",
        {"b": 2, "a": 1, "rows": ["b", "a"]},
    )

    assert first["receipt_sha256"] == reordered_keys["receipt_sha256"]
    assert first["receipt_sha256"] != reordered_rows["receipt_sha256"]


def test_tampered_nested_receipt_is_rejected_before_scientific_validation() -> None:
    """Changing a bound payload without its receipt hash fails closed."""
    admission = deepcopy(_valid_admission())
    raw = admission["payload"]["raw_manifest_receipt"]
    raw["payload"]["entry_count"] = 9
    admission = _rewrap(admission)

    with pytest.raises(DataAdmissionError, match="receipt SHA-256"):
        _validate(admission)


@pytest.mark.parametrize("revision", ["main", "latest", "HEAD"])
def test_mutable_source_revisions_are_rejected(revision: str) -> None:
    """Branches and rolling aliases are not pinned revisions."""
    admission = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["source"].update(revision=revision),
    )

    with pytest.raises(DataAdmissionError, match="pinned versioned release"):
        _validate(admission)


def test_missing_revision_is_rejected() -> None:
    """A source cannot pass without an explicit pinned revision."""
    admission = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["source"].pop("revision"),
    )

    with pytest.raises(DataAdmissionError, match="missing=.*revision"):
        _validate(admission)


def test_missing_manifest_hash_is_rejected() -> None:
    """Raw and normalized content hashes are mandatory."""

    def mutation(payload: dict[str, object]) -> None:
        raw = deepcopy(payload["raw_manifest_receipt"])
        raw["payload"].pop("manifest_sha256")
        payload["raw_manifest_receipt"] = _rewrap(raw)

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="manifest_sha256"):
        _validate(admission)


def test_acquisition_time_must_be_explicit_utc() -> None:
    """Local or timezone-naive acquisition times are not pinned timestamps."""
    admission = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["source"].update(
            acquired_at_utc="2026-07-25T01:00:00"
        ),
    )

    with pytest.raises(DataAdmissionError, match="exact UTC form"):
        _validate(admission)


def test_renaming_a_source_cannot_bypass_the_external_registry_contract() -> None:
    """Source semantics are registry-bound and never inferred from display names."""
    original = _valid_admission()
    registry = _source_registry_for_payload(original["payload"])
    renamed = deepcopy(original)
    renamed["payload"]["source"]["source_name"] = "Innocuous renamed source"
    renamed = _rewrap(renamed)

    with pytest.raises(DataAdmissionError, match="source declarations differ"):
        validate_dataset_admission(
            renamed,
            expected_admission_policy_sha256=_ADMISSION_POLICY_SHA256,
            source_registry_entry=registry,
            expected_source_registry_entry_sha256=registry["receipt_sha256"],
            not_after_utc=_VALIDATION_CUTOFF,
        )


def test_generated_artifacts_cannot_be_called_experimental() -> None:
    """Predicted or generated sources must report zero experimental labels."""
    admission = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["source"].update(
            semantic_type="predicted_or_synthetic"
        ),
    )

    with pytest.raises(DataAdmissionError, match="cannot be called experimental"):
        _validate(admission)


def test_unknown_pairs_cannot_be_used_as_nonbinders() -> None:
    """Unobserved pairs remain unknown and cannot silently become negatives."""

    def mutation(payload: dict[str, object]) -> None:
        payload["label_semantics"]["unknown_pair_records"] = 1
        payload["label_semantics"]["unknown_pairs_used_as_negative"] = True

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="Unknown.*unlabeled"):
        _validate(admission)


def test_missing_normalized_row_lineage_is_rejected() -> None:
    """Every normalized row requires at least one bound source-row edge."""

    def mutation(payload: dict[str, object]) -> None:
        lineage = deepcopy(payload["lineage_receipt"])
        lineage["payload"]["normalized_rows_with_lineage_count"] = 8
        lineage["payload"]["missing_normalized_row_count"] = 1
        payload["lineage_receipt"] = _rewrap(lineage)

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="Every normalized row"):
        _validate(admission)


def test_mutable_or_globbed_manifest_uri_is_rejected() -> None:
    """Ordinary filesystem paths and glob locators are never content addresses."""

    def mutation(payload: dict[str, object]) -> None:
        raw = deepcopy(payload["raw_manifest_receipt"])
        raw["payload"]["manifest_uri"] = str(Path("/tmp") / "raw-*.jsonl")
        payload["raw_manifest_receipt"] = _rewrap(raw)

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="mutable paths and globs"):
        _validate(admission)


def test_pipeline_count_increase_is_rejected() -> None:
    """QC, deduplication, and admission cannot invent source records."""
    admission = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["counts"].update(qc_passed_records=11),
    )

    with pytest.raises(DataAdmissionError, match="raw >= normalized"):
        _validate(admission)


def test_manifest_entry_count_must_match_audited_count() -> None:
    """Manifest counts cannot disagree with the pipeline count ledger."""

    def mutation(payload: dict[str, object]) -> None:
        normalized = deepcopy(payload["normalized_manifest_receipt"])
        normalized["payload"]["entry_count"] = 8
        payload["normalized_manifest_receipt"] = _rewrap(normalized)

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="entry_count does not match"):
        _validate(admission)


def test_missing_molecular_axis_duplicate_receipt_is_rejected() -> None:
    """Both antibody and antigen exact/near audit receipts are mandatory."""
    admission = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["duplicate_audit_receipts"].pop("antigen"),
    )

    with pytest.raises(DataAdmissionError, match="missing=.*antigen"):
        _validate(admission)


def test_unresolved_cross_source_duplicates_are_rejected() -> None:
    """Duplicates must be quarantined, merged, or linked to split components."""

    def mutation(payload: dict[str, object]) -> None:
        audit = deepcopy(payload["duplicate_audit_receipts"]["antibody"])
        audit["payload"]["resolved_duplicate_count"] = 1
        audit["payload"]["unresolved_duplicate_count"] = 1
        audit["payload"]["resolution_action_counts"][
            "linked_to_split_component"
        ] = 1
        payload["duplicate_audit_receipts"]["antibody"] = _rewrap(audit)

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="unresolved"):
        _validate(admission)


def test_unparseable_or_missing_license_is_rejected() -> None:
    """License terms must be present, hashed, and parseable."""
    invalid = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["license"].update(spdx_expression="unknown"),
    )
    with pytest.raises(DataAdmissionError, match="Unlicensed or unresolved"):
        _validate(invalid)

    missing = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["license"].pop("license_text_sha256"),
    )
    with pytest.raises(DataAdmissionError, match="license_text_sha256"):
        _validate(missing)


def test_incompatible_redistribution_request_is_rejected() -> None:
    """A permissive route cannot override a no-redistribution source term."""

    def mutation(payload: dict[str, object]) -> None:
        payload["license"]["normalized_redistribution_allowed"] = False

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="redistribution was requested"):
        _validate(admission)


def test_noncommercial_source_must_use_isolated_contamination_route() -> None:
    """NC data cannot contaminate a nominally general or commercial track."""
    admission = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["license"].update(commercial_use_allowed=False),
    )

    with pytest.raises(DataAdmissionError, match="Routing does not conservatively"):
        _validate(admission)


def test_nc_sa_expression_cannot_be_paired_with_permissive_flags() -> None:
    """Obvious SPDX NC/SA terms cannot be hidden behind permissive booleans."""
    admission = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["license"].update(
            spdx_expression="CC-BY-NC-SA-4.0"
        ),
    )

    with pytest.raises(DataAdmissionError, match="NC restriction"):
        _validate(admission)


def test_noncommercial_share_alike_restricted_route_is_admitted_and_marked() -> None:
    """A correctly isolated NC/SA/no-redistribution source stays traceable."""

    def mutation(payload: dict[str, object]) -> None:
        payload["license"].update(
            spdx_expression="CC-BY-NC-SA-4.0",
            commercial_use_allowed=False,
            share_alike_required=True,
            raw_redistribution_allowed=False,
            normalized_redistribution_allowed=False,
        )
        payload["routing"].update(
            track="noncommercial_isolated",
            redistribute_raw_requested=False,
            redistribute_normalized_requested=False,
            propagate_noncommercial_restriction=True,
            propagate_share_alike_restriction=True,
            propagate_redistribution_restrictions=True,
        )

    admission = _mutate_payload(_valid_admission(), mutation)
    decision = _validate(admission)

    assert decision.noncommercial_contaminated
    assert decision.routing_track == "noncommercial_isolated"
    assert decision.contamination_policy == (
        "noncommercial+share_alike+redistribution_restricted"
    )


def test_share_alike_requirement_cannot_be_dropped() -> None:
    """SA terms propagate even when commercial use and redistribution are allowed."""
    admission = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["license"].update(share_alike_required=True),
    )

    with pytest.raises(DataAdmissionError, match="share_alike=True"):
        _validate(admission)


def test_nonimmutable_receipt_is_rejected() -> None:
    """A mutable declaration cannot authorize any prospective data."""
    admission = _valid_admission()
    admission["immutable"] = False

    with pytest.raises(DataAdmissionError, match="immutable=true"):
        _validate(admission)


def test_caller_frozen_policy_cannot_be_replaced_by_a_self_declared_hash() -> None:
    """Rehashing a payload under an attacker-selected policy never authorizes it."""
    admission = _mutate_payload(
        _valid_admission(),
        lambda payload: payload.update(
            admission_policy_sha256=_digest("attacker-policy")
        ),
    )

    with pytest.raises(DataAdmissionError, match="caller-frozen admission policy"):
        _validate(admission)


def test_self_generated_source_registry_cannot_replace_the_frozen_entry() -> None:
    """A payload and matching attacker registry still fail the caller digest check."""
    original = _valid_admission()
    trusted_registry = _source_registry_for_payload(original["payload"])
    attacker = _mutate_payload(
        original,
        lambda payload: payload["source"].update(
            source_id="renamed-source",
            source_name="Renamed source",
        ),
    )
    attacker_registry = _source_registry_for_payload(attacker["payload"])

    with pytest.raises(DataAdmissionError, match="caller-frozen source registry"):
        validate_dataset_admission(
            attacker,
            expected_admission_policy_sha256=_ADMISSION_POLICY_SHA256,
            source_registry_entry=attacker_registry,
            expected_source_registry_entry_sha256=trusted_registry[
                "receipt_sha256"
            ],
            not_after_utc=_VALIDATION_CUTOFF,
        )


def test_numeric_string_near_duplicate_threshold_is_rejected() -> None:
    """JSON strings never coerce into scientific threshold values."""

    def mutation(payload: dict[str, object]) -> None:
        audit = deepcopy(payload["duplicate_audit_receipts"]["antibody"])
        audit["payload"]["near_duplicate_threshold"] = "0.8"
        payload["duplicate_audit_receipts"]["antibody"] = _rewrap(audit)

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="finite scalar"):
        _validate(admission)


def test_duplicate_candidate_count_must_match_the_axis_entity_ledger() -> None:
    """A nominal cross-source audit cannot cover only a subset of one axis."""

    def mutation(payload: dict[str, object]) -> None:
        audit = deepcopy(payload["duplicate_audit_receipts"]["antibody"])
        audit["payload"]["candidate_entity_count"] = 6
        payload["duplicate_audit_receipts"]["antibody"] = _rewrap(audit)

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="audited antibody entity count"):
        _validate(admission)


def test_present_axis_requires_an_actual_cross_source_comparison() -> None:
    """A receipt with zero comparisons is not a duplicate audit."""

    def mutation(payload: dict[str, object]) -> None:
        audit = deepcopy(payload["duplicate_audit_receipts"]["antibody"])
        audit["payload"].update(
            comparison_count=0,
            exact_duplicate_count=0,
            near_duplicate_count_excluding_exact=0,
            resolved_duplicate_count=0,
            unresolved_duplicate_count=0,
            resolution_action_counts={
                "quarantined": 0,
                "merged": 0,
                "linked_to_split_component": 0,
            },
        )
        payload["duplicate_audit_receipts"]["antibody"] = _rewrap(audit)

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="cover every candidate"):
        _validate(admission)


def test_axis_specific_candidate_manifests_cannot_be_reused_cross_axis() -> None:
    """Antibody and antigen audit domains require distinct bound artifacts."""

    def mutation(payload: dict[str, object]) -> None:
        antibody = payload["duplicate_audit_receipts"]["antibody"]["payload"]
        antigen = deepcopy(payload["duplicate_audit_receipts"]["antigen"])
        digest = antibody["candidate_entity_manifest_sha256"]
        antigen["payload"]["candidate_entity_manifest_sha256"] = digest
        antigen["payload"]["candidate_entity_manifest_uri"] = _cas(digest)
        payload["duplicate_audit_receipts"]["antigen"] = _rewrap(antigen)

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="incompatible artifact roles"):
        _validate(admission)


def test_unknown_pairs_cannot_be_hidden_inside_a_measured_source() -> None:
    """Every admitted measured-interaction row requires experimental evidence."""

    def mutation(payload: dict[str, object]) -> None:
        payload["counts"]["experimental_label_records"] = 7
        payload["label_semantics"].update(
            experimental_positive_records=4,
            experimental_nonbinder_records=3,
            unknown_pair_records=1,
        )

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="all be experimentally measured"):
        _validate(admission)


def test_attribution_requirement_must_propagate_to_the_route() -> None:
    """A route cannot silently discard a source attribution obligation."""
    admission = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["routing"].update(
            propagate_attribution_requirement=False
        ),
    )

    with pytest.raises(DataAdmissionError, match="attribution=True"):
        _validate(admission)


def test_future_or_out_of_order_audit_timestamp_is_rejected() -> None:
    """Rehashed receipts cannot claim an audit completed after admission."""

    def mutation(payload: dict[str, object]) -> None:
        audit = deepcopy(payload["duplicate_audit_receipts"]["antigen"])
        audit["payload"]["created_at_utc"] = "2026-07-25T01:06:00Z"
        payload["duplicate_audit_receipts"]["antigen"] = _rewrap(audit)

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="must precede admission"):
        _validate(admission)


def test_caller_cutoff_rejects_a_future_admission() -> None:
    """The admission timestamp is bounded by caller-owned wall-clock evidence."""
    admission = _valid_admission()
    registry = _source_registry_for_payload(admission["payload"])

    with pytest.raises(DataAdmissionError, match="caller cutoff"):
        validate_dataset_admission(
            admission,
            expected_admission_policy_sha256=_ADMISSION_POLICY_SHA256,
            source_registry_entry=registry,
            expected_source_registry_entry_sha256=registry["receipt_sha256"],
            not_after_utc="2026-07-25T01:04:59Z",
        )


@pytest.mark.parametrize(
    "ambiguous_value",
    [
        -0.0,
        2**53,
        "e\u0301",
    ],
)
def test_nonportable_canonical_json_values_are_rejected(
    ambiguous_value: object,
) -> None:
    """Canonical receipts reject negative zero, unsafe integers, and non-NFC."""
    with pytest.raises(DataAdmissionError):
        make_content_addressed_receipt(
            "fixture_receipt_v1",
            {"value": ambiguous_value},
        )


def test_nonstring_json_object_keys_and_dict_subclasses_are_rejected() -> None:
    """Python-only mapping behavior cannot alter a supposedly canonical receipt."""

    class CustomDict(dict):
        pass

    with pytest.raises(DataAdmissionError, match="object keys must be strings"):
        make_content_addressed_receipt(
            "fixture_receipt_v1",
            {"nested": {1: "ambiguous"}},
        )
    with pytest.raises(DataAdmissionError, match="JSON object"):
        make_content_addressed_receipt(
            "fixture_receipt_v1",
            CustomDict({"value": 1}),
        )


def test_encoded_url_path_traversal_is_rejected() -> None:
    """Percent encoding cannot smuggle dot segments into a reference URL."""
    admission = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["license"].update(
            license_url="https://example.org/licenses/a/%2e%2e/secret"
        ),
    )

    with pytest.raises(DataAdmissionError, match="path traversal"):
        _validate(admission)


def test_manifest_digest_cannot_be_reused_as_a_different_artifact_role() -> None:
    """CAS digests remain bound to one semantic artifact role."""

    def mutation(payload: dict[str, object]) -> None:
        raw_hash = payload["raw_manifest_receipt"]["payload"]["manifest_sha256"]
        normalized = deepcopy(payload["normalized_manifest_receipt"])
        normalized["payload"]["manifest_sha256"] = raw_hash
        normalized["payload"]["manifest_uri"] = _cas(raw_hash)
        payload["normalized_manifest_receipt"] = _rewrap(normalized)
        lineage = deepcopy(payload["lineage_receipt"])
        lineage["payload"]["normalized_manifest_sha256"] = raw_hash
        payload["lineage_receipt"] = _rewrap(lineage)

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="incompatible artifact roles"):
        _validate(admission)


def test_duplicate_auditor_code_must_be_independent_of_producer_code() -> None:
    """Reusing producer code cannot masquerade as an independent audit."""

    def mutation(payload: dict[str, object]) -> None:
        producer_hash = payload["raw_manifest_receipt"]["payload"][
            "producer_code_sha256"
        ]
        audit = deepcopy(payload["duplicate_audit_receipts"]["antibody"])
        audit["payload"]["auditor_code_sha256"] = producer_hash
        payload["duplicate_audit_receipts"]["antibody"] = _rewrap(audit)

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="independent"):
        _validate(admission)


def test_success_decision_has_no_mutable_nested_state() -> None:
    """The admitted result and its nested count ledger are both frozen."""
    decision = _validate(_valid_admission())

    with pytest.raises(FrozenInstanceError):
        decision.counts.raw_records = 11
    assert isinstance(decision.duplicate_audit_receipt_sha256, tuple)
    assert all(
        isinstance(item, tuple)
        for item in decision.duplicate_audit_receipt_sha256
    )


def test_source_registry_format_is_domain_separated() -> None:
    """An ordinary receipt cannot be substituted for a source-registry entry."""
    admission = _valid_admission()
    registry = _source_registry_for_payload(admission["payload"])
    assert registry["receipt_format"] == SOURCE_REGISTRY_ENTRY_FORMAT
    wrong = make_content_addressed_receipt(
        "fixture_receipt_v1",
        registry["payload"],
    )

    with pytest.raises(DataAdmissionError, match="receipt_format"):
        validate_dataset_admission(
            admission,
            expected_admission_policy_sha256=_ADMISSION_POLICY_SHA256,
            source_registry_entry=wrong,
            expected_source_registry_entry_sha256=registry["receipt_sha256"],
            not_after_utc=_VALIDATION_CUTOFF,
        )


def test_canonical_doi_revision_is_supported_but_snapshot_alias_is_not() -> None:
    """Revision validation is namespace-specific rather than path-like coercion."""
    doi_admission = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["source"].update(
            revision_type="doi_version",
            revision="10.1234/release.v2",
        ),
    )
    assert _validate(doi_admission).source_revision == "10.1234/release.v2"

    snapshot_alias = _mutate_payload(
        _valid_admission(),
        lambda payload: payload["source"].update(
            revision_type="immutable_snapshot",
            revision="snapshot-v2",
        ),
    )
    with pytest.raises(DataAdmissionError, match="lowercase SHA-256"):
        _validate(snapshot_alias)


def test_every_pipeline_stage_requires_a_bound_manifest() -> None:
    """QC, deduplication, and final admission counts cannot remain hashless."""
    admission = _mutate_payload(
        _valid_admission(),
        lambda payload: payload.pop("admitted_manifest_receipt"),
    )

    with pytest.raises(DataAdmissionError, match="admitted_manifest_receipt"):
        _validate(admission)


def test_late_stage_lineage_gap_is_rejected() -> None:
    """Lineage coverage extends through final admission, not only normalization."""

    def mutation(payload: dict[str, object]) -> None:
        lineage = deepcopy(payload["lineage_receipt"])
        lineage["payload"]["admitted_rows_with_lineage_count"] = 7
        lineage["payload"]["missing_admitted_row_count"] = 1
        payload["lineage_receipt"] = _rewrap(lineage)

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="Every admitted row"):
        _validate(admission)


def test_admitted_manifest_count_must_match_the_final_count_ledger() -> None:
    """A final manifest cannot silently omit one admitted record."""

    def mutation(payload: dict[str, object]) -> None:
        admitted = deepcopy(payload["admitted_manifest_receipt"])
        admitted["payload"]["entry_count"] = 7
        payload["admitted_manifest_receipt"] = _rewrap(admitted)

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="audited admitted count"):
        _validate(admission)


def test_all_predicted_source_rows_must_be_generation_accounted() -> None:
    """A predicted source cannot leave most admitted rows semantically untyped."""

    def mutation(payload: dict[str, object]) -> None:
        payload["source"].update(
            semantic_type="predicted_or_synthetic",
            record_unit="structure",
        )
        payload["counts"].update(
            experimental_label_records=0,
            predicted_structure_records=1,
            generated_or_predicted_records=1,
        )
        payload["label_semantics"].update(
            experimental_positive_records=0,
            experimental_nonbinder_records=0,
        )

    admission = _mutate_payload(_valid_admission(), mutation)

    with pytest.raises(DataAdmissionError, match="Every predicted_or_synthetic"):
        _validate(admission)


def test_license_rights_cannot_drift_from_the_frozen_source_registry() -> None:
    """Rehashing license flags does not change the reviewed registry contract."""
    original = _valid_admission()
    trusted_registry = _source_registry_for_payload(original["payload"])
    changed = deepcopy(original)
    changed["payload"]["license"]["commercial_use_allowed"] = False
    changed = _rewrap(changed)

    with pytest.raises(DataAdmissionError, match="license declarations differ"):
        validate_dataset_admission(
            changed,
            expected_admission_policy_sha256=_ADMISSION_POLICY_SHA256,
            source_registry_entry=trusted_registry,
            expected_source_registry_entry_sha256=trusted_registry[
                "receipt_sha256"
            ],
            not_after_utc=_VALIDATION_CUTOFF,
        )


def test_duplicate_audit_registry_cannot_drift_from_the_frozen_contract() -> None:
    """Cross-source audit domains are external contract inputs, not self-reports."""
    original = _valid_admission()
    trusted_registry = _source_registry_for_payload(original["payload"])
    changed = deepcopy(original)
    audit = deepcopy(
        changed["payload"]["duplicate_audit_receipts"]["antibody"]
    )
    replacement = _digest("attacker-comparison-registry")
    audit["payload"]["comparison_registry_sha256"] = replacement
    audit["payload"]["comparison_registry_uri"] = _cas(replacement)
    changed["payload"]["duplicate_audit_receipts"]["antibody"] = _rewrap(audit)
    changed = _rewrap(changed)

    with pytest.raises(DataAdmissionError, match="caller-frozen.*contract"):
        validate_dataset_admission(
            changed,
            expected_admission_policy_sha256=_ADMISSION_POLICY_SHA256,
            source_registry_entry=trusted_registry,
            expected_source_registry_entry_sha256=trusted_registry[
                "receipt_sha256"
            ],
            not_after_utc=_VALIDATION_CUTOFF,
        )
