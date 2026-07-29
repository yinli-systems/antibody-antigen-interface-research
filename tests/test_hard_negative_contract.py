"""Adversarial tests for train-only hard-negative semantics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from interaction_first.hard_negative_contract import (
    BindingEvidence,
    ControlFamily,
    ControlRecord,
    GenerationBinding,
    HardNegativeManifest,
    PartnerAxis,
    ResidueDomainSupport,
    SourceBinding,
    TrainOnlyAllowlist,
    TrainOnlyEvidenceRegistry,
    TrainOnlySplitBinding,
    manifest_from_json_bytes,
    mutation_relation_binding_sha256,
    verify_manifest_document,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _rehash(document: dict[str, object]) -> None:
    payload = {
        key: value
        for key, value in document.items()
        if key != "manifest_sha256"
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    document["manifest_sha256"] = hashlib.sha256(
        b"interaction_first:hard_negative_contract:v2\x00"
        b"manifest\x00"
        + canonical
    ).hexdigest()


def _source(label: str = "source") -> SourceBinding:
    return SourceBinding(
        source_id="licensed-structural-source",
        revision=f"release-{label}-2026-07-01",
        license_id="CC-BY-4.0",
        license_receipt_sha256=_sha(f"{label}-license"),
        provenance_sha256=_sha(f"{label}-provenance"),
        source_content_sha256=_sha(f"{label}-content"),
        dataset_admission_receipt_sha256=_sha(f"{label}-admission"),
    )


def _split(
    *,
    outer_fold: int = 0,
    inner_fold: int = 1,
) -> TrainOnlySplitBinding:
    return TrainOnlySplitBinding(
        split_manifest_sha256=_sha("split-manifest"),
        train_membership_receipt_sha256=_sha("membership"),
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )


def _generation(label: str = "base") -> GenerationBinding:
    return GenerationBinding(
        strategy_id="matched-controls",
        strategy_version=f"1.0-{label}",
        implementation_sha256=_sha(f"{label}-implementation"),
        configuration_sha256=_sha(f"{label}-configuration"),
        generation_run_sha256=_sha(f"{label}-run"),
        deterministic_seed=17,
    )


def _allowlist() -> TrainOnlyAllowlist:
    return TrainOnlyAllowlist(
        allowlist_id="outer0-inner1-train",
        split_manifest_sha256=_sha("split-manifest"),
        train_membership_receipt_sha256=_sha("membership"),
        outer_fold=0,
        inner_fold=1,
        antibody_entity_ids=("ABM", "AB1", "AB0"),
        antigen_entity_ids=("AG1", "AG0"),
    )


def _support(
    antibody: str,
    antigen: str,
    *,
    label: str | None = None,
) -> ResidueDomainSupport:
    pair_label = label or f"{antibody}-{antigen}"
    return ResidueDomainSupport(
        antibody_domain_size=4,
        antigen_domain_size=5,
        antibody_valid_count=4,
        antigen_valid_count=5,
        candidate_pair_count=20,
        antibody_mask_sha256=_sha("shared-antibody-mask"),
        antigen_mask_sha256=_sha("shared-antigen-mask"),
        pair_mask_sha256=_sha("shared-pair-mask"),
        antibody_residue_ids_sha256=_sha(f"{antibody}-residues"),
        antigen_residue_ids_sha256=_sha(f"{antigen}-residues"),
        canonical_domain_manifest_sha256=_sha(f"{pair_label}-domain"),
        domain_selection_policy_sha256=_sha("shared-domain-policy"),
        candidate_mask_policy_sha256=_sha("shared-candidate-policy"),
    )


def _record(
    record_id: str,
    family: ControlFamily,
    evidence: BindingEvidence,
    antibody: str,
    antigen: str,
    *,
    parent: str | None = "cognate",
    patch: str = "native-patch",
    pose: str = "native-pose",
    patch_definition_sha256: str | None = None,
    pose_transform_sha256: str | None = None,
    support: ResidueDomainSupport | None = None,
    source: SourceBinding | None = None,
    split: TrainOnlySplitBinding | None = None,
    generation: GenerationBinding | None = None,
    allowlist: TrainOnlyAllowlist | None = None,
    mutation_axis: PartnerAxis | None = None,
    mutation_spec_sha256: str | None = None,
    paired_assay_receipt_sha256: str | None = None,
    assay_protocol_sha256: str | None = None,
    measurement_receipt_sha256: str | None = None,
    sample_content_sha256: str | None = None,
    match_group_id: str = "group-0",
) -> ControlRecord:
    allowlist = allowlist or _allowlist()
    measured = evidence is not BindingEvidence.UNKNOWN_UNMEASURED
    return ControlRecord(
        record_id=record_id,
        match_group_id=match_group_id,
        parent_cognate_id=parent,
        control_family=family,
        binding_evidence=evidence,
        antibody_entity_id=antibody,
        antigen_entity_id=antigen,
        patch_id=patch,
        pose_id=pose,
        patch_definition_sha256=(
            patch_definition_sha256 or _sha(f"{patch}-definition")
        ),
        pose_transform_sha256=(
            pose_transform_sha256 or _sha(f"{pose}-transform")
        ),
        source=source or _source(),
        split=split or _split(),
        generation=generation or _generation(),
        train_allowlist_sha256=allowlist.allowlist_sha256,
        sample_content_sha256=(
            sample_content_sha256 or _sha(f"{record_id}-sample")
        ),
        construction_provenance_sha256=_sha(f"{record_id}-construction"),
        support=support or _support(antibody, antigen),
        label_provenance_sha256=(
            _sha(f"{record_id}-label") if measured else None
        ),
        assay_protocol_sha256=(
            assay_protocol_sha256 or _sha("shared-assay")
            if measured
            else None
        ),
        measurement_receipt_sha256=(
            measurement_receipt_sha256 or _sha(f"{record_id}-measurement")
            if measured
            else None
        ),
        mutation_axis=mutation_axis,
        mutation_spec_sha256=mutation_spec_sha256,
        paired_assay_receipt_sha256=paired_assay_receipt_sha256,
    )


def _cognate() -> ControlRecord:
    return _record(
        "cognate",
        ControlFamily.COGNATE,
        BindingEvidence.MEASURED_BINDER,
        "AB0",
        "AG0",
        parent=None,
    )


def _all_family_records() -> tuple[ControlRecord, ...]:
    return (
        _cognate(),
        _record(
            "measured-negative",
            ControlFamily.MEASURED_NON_BINDER,
            BindingEvidence.MEASURED_NON_BINDER,
            "ABM",
            "AG1",
        ),
        _record(
            "unmeasured",
            ControlFamily.UNKNOWN_UNMEASURED_DECOY,
            BindingEvidence.UNKNOWN_UNMEASURED,
            "AB1",
            "AG1",
        ),
        _record(
            "wrong-antibody",
            ControlFamily.SAME_ANTIGEN_WRONG_ANTIBODY,
            BindingEvidence.UNKNOWN_UNMEASURED,
            "AB1",
            "AG0",
        ),
        _record(
            "wrong-antigen",
            ControlFamily.SAME_ANTIBODY_WRONG_ANTIGEN,
            BindingEvidence.MEASURED_NON_BINDER,
            "AB0",
            "AG1",
        ),
        _record(
            "mutation-flip",
            ControlFamily.MUTATION_FLIP,
            BindingEvidence.MEASURED_NON_BINDER,
            "ABM",
            "AG0",
            mutation_axis=PartnerAxis.ANTIBODY,
            mutation_spec_sha256=_sha("AB0-to-ABM-mutation"),
            paired_assay_receipt_sha256=_sha("paired-assay"),
        ),
        _record(
            "wrong-patch",
            ControlFamily.WRONG_PATCH,
            BindingEvidence.UNKNOWN_UNMEASURED,
            "AB0",
            "AG0",
            patch="decoy-patch",
            support=_support("AB0", "AG0"),
        ),
        _record(
            "wrong-pose",
            ControlFamily.WRONG_POSE,
            BindingEvidence.UNKNOWN_UNMEASURED,
            "AB0",
            "AG0",
            pose="decoy-pose",
            support=_support("AB0", "AG0"),
        ),
    )


def _evidence_registry(
    records: tuple[ControlRecord, ...],
    *,
    allowlist: TrainOnlyAllowlist | None = None,
) -> TrainOnlyEvidenceRegistry:
    measured = tuple(
        binding
        for record in records
        if (binding := record.measured_evidence_binding_sha256) is not None
    )
    cognates = {
        record.match_group_id: record
        for record in records
        if record.control_family is ControlFamily.COGNATE
    }
    mutation_relations = tuple(
        mutation_relation_binding_sha256(
            cognates[record.match_group_id],
            record,
        )
        for record in records
        if record.control_family is ControlFamily.MUTATION_FLIP
    )
    split = records[0].split
    allowlist = allowlist or _allowlist()
    return TrainOnlyEvidenceRegistry(
        split_manifest_sha256=split.split_manifest_sha256,
        train_membership_receipt_sha256=(
            split.train_membership_receipt_sha256
        ),
        outer_fold=split.outer_fold,
        inner_fold=split.inner_fold,
        train_allowlist_sha256=allowlist.allowlist_sha256,
        authority_receipt_sha256=_sha("external-evidence-authority"),
        measured_evidence_binding_sha256s=measured,
        mutation_relation_binding_sha256s=mutation_relations,
    )


def _manifest(
    records: tuple[ControlRecord, ...],
    *,
    allowlist: TrainOnlyAllowlist | None = None,
    evidence_registry: TrainOnlyEvidenceRegistry | None = None,
) -> HardNegativeManifest:
    allowlist = allowlist or _allowlist()
    return HardNegativeManifest(
        allowlist=allowlist,
        records=records,
        evidence_registry=evidence_registry
        or _evidence_registry(records, allowlist=allowlist),
    )


def test_all_control_families_have_explicit_supervision_semantics() -> None:
    manifest = _manifest(_all_family_records())
    records = {record.record_id: record for record in manifest.records}

    assert records["cognate"].binary_target == 1
    assert records["measured-negative"].binary_target == 0
    assert records["wrong-antigen"].binary_target == 0
    assert records["mutation-flip"].binary_target == 0
    assert records["unmeasured"].binary_target is None
    assert records["wrong-antibody"].binary_target is None
    assert records["wrong-patch"].binary_target is None
    assert records["wrong-pose"].binary_target is None
    assert records["unmeasured"].supervision_mode == "ranking_control_only"


def test_unknown_decoy_cannot_be_forced_to_binary_zero() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ControlRecord(
            **{
                **_record(
                    "unknown",
                    ControlFamily.UNKNOWN_UNMEASURED_DECOY,
                    BindingEvidence.UNKNOWN_UNMEASURED,
                    "AB1",
                    "AG1",
                ).__dict__,
                "binary_target": 0,
            }
        )

    with pytest.raises(ValueError, match="cannot carry measured-label"):
        _record(
            "unknown-with-label",
            ControlFamily.UNKNOWN_UNMEASURED_DECOY,
            BindingEvidence.UNKNOWN_UNMEASURED,
            "AB1",
            "AG1",
        ).__class__(
            **{
                **_record(
                    "unknown-with-label",
                    ControlFamily.UNKNOWN_UNMEASURED_DECOY,
                    BindingEvidence.UNKNOWN_UNMEASURED,
                    "AB1",
                    "AG1",
                ).__dict__,
                "label_provenance_sha256": _sha("forged-negative-label"),
            }
        )


def test_self_consistent_digest_cannot_authorize_forged_unknown_negative() -> None:
    manifest = _manifest(_all_family_records())
    forged = manifest.to_document()
    unknown = next(
        record
        for record in forged["records"]
        if record["record_id"] == "unmeasured"
    )
    unknown["binary_target"] = 0
    unknown["supervision_mode"] = "binary_negative"
    _rehash(forged)

    with pytest.raises(ValueError, match="canonical semantic representation"):
        verify_manifest_document(
            forged,
            expected_manifest_sha256=forged["manifest_sha256"],
        )


def test_measured_non_binder_requires_assay_and_label_provenance() -> None:
    valid = _record(
        "negative",
        ControlFamily.MEASURED_NON_BINDER,
        BindingEvidence.MEASURED_NON_BINDER,
        "AB1",
        "AG1",
    )
    assert valid.binary_target == 0

    for missing_field in (
        "label_provenance_sha256",
        "assay_protocol_sha256",
        "measurement_receipt_sha256",
    ):
        with pytest.raises(ValueError, match="Measured binding evidence requires"):
            replace(valid, **{missing_field: None})


def test_mutation_flip_requires_paired_assay_and_matching_protocol() -> None:
    mutation = _record(
        "mutation",
        ControlFamily.MUTATION_FLIP,
        BindingEvidence.MEASURED_NON_BINDER,
        "ABM",
        "AG0",
        mutation_axis=PartnerAxis.ANTIBODY,
        mutation_spec_sha256=_sha("mutation"),
        paired_assay_receipt_sha256=_sha("paired-assay"),
    )
    with pytest.raises(ValueError, match="paired-assay"):
        replace(mutation, paired_assay_receipt_sha256=None)

    mismatched_protocol = replace(
        mutation,
        assay_protocol_sha256=_sha("different-assay"),
    )
    with pytest.raises(ValueError, match="cognate assay protocol"):
        _manifest((_cognate(), mismatched_protocol))


@pytest.mark.parametrize(
    "changed_field,replacement_value,error",
    [
        ("source", _source("other"), "cross source"),
        ("split", _split(outer_fold=2), "split"),
        ("generation", _generation("other"), "generation strategy"),
    ],
)
def test_controls_cannot_cross_context_boundaries(
    changed_field: str,
    replacement_value: object,
    error: str,
) -> None:
    control = _record(
        "control",
        ControlFamily.UNKNOWN_UNMEASURED_DECOY,
        BindingEvidence.UNKNOWN_UNMEASURED,
        "AB1",
        "AG1",
    )
    control = replace(control, **{changed_field: replacement_value})
    with pytest.raises(ValueError, match=error):
        _manifest((_cognate(), control))


@pytest.mark.parametrize(
    "support",
    [
        replace(_support("AB1", "AG1"), pair_mask_sha256=_sha("cropped-mask")),
        replace(
            _support("AB1", "AG1"),
            antigen_valid_count=4,
            candidate_pair_count=16,
        ),
        replace(
            _support("AB1", "AG1"),
            antibody_domain_size=5,
        ),
    ],
)
def test_different_mask_or_domain_support_is_rejected(
    support: ResidueDomainSupport,
) -> None:
    control = _record(
        "control",
        ControlFamily.UNKNOWN_UNMEASURED_DECOY,
        BindingEvidence.UNKNOWN_UNMEASURED,
        "AB1",
        "AG1",
        support=support,
    )
    with pytest.raises(ValueError, match="mask/support"):
        _manifest((_cognate(), control))


def test_wrong_partner_axis_semantics_fail_closed() -> None:
    wrong_antibody = _record(
        "wrong-antibody",
        ControlFamily.SAME_ANTIGEN_WRONG_ANTIBODY,
        BindingEvidence.UNKNOWN_UNMEASURED,
        "AB1",
        "AG1",
    )
    with pytest.raises(ValueError, match="preserve only the antigen"):
        _manifest((_cognate(), wrong_antibody))

    wrong_antigen = _record(
        "wrong-antigen",
        ControlFamily.SAME_ANTIBODY_WRONG_ANTIGEN,
        BindingEvidence.UNKNOWN_UNMEASURED,
        "AB1",
        "AG1",
    )
    with pytest.raises(ValueError, match="preserve only the antibody"):
        _manifest((_cognate(), wrong_antigen))


def test_wrong_patch_and_pose_cannot_crop_or_change_partners() -> None:
    changed_partner = _record(
        "bad-patch",
        ControlFamily.WRONG_PATCH,
        BindingEvidence.UNKNOWN_UNMEASURED,
        "AB1",
        "AG0",
        patch="decoy-patch",
    )
    with pytest.raises(ValueError, match="change only the patch"):
        _manifest((_cognate(), changed_partner))

    changed_patch_and_pose = _record(
        "bad-pose",
        ControlFamily.WRONG_POSE,
        BindingEvidence.UNKNOWN_UNMEASURED,
        "AB0",
        "AG0",
        patch="decoy-patch",
        pose="decoy-pose",
        support=_support("AB0", "AG0"),
    )
    with pytest.raises(ValueError, match="change only the pose"):
        _manifest((_cognate(), changed_patch_and_pose))


def test_mutation_flip_must_change_exactly_declared_axis() -> None:
    mutation = _record(
        "bad-mutation",
        ControlFamily.MUTATION_FLIP,
        BindingEvidence.MEASURED_NON_BINDER,
        "ABM",
        "AG1",
        mutation_axis=PartnerAxis.ANTIBODY,
        mutation_spec_sha256=_sha("mutation"),
        paired_assay_receipt_sha256=_sha("paired-assay"),
    )
    with pytest.raises(ValueError, match="change only the antibody"):
        _manifest((_cognate(), mutation))


def test_mutation_flip_cannot_rename_an_unchanged_residue_domain() -> None:
    mutation = _record(
        "no-op-mutation",
        ControlFamily.MUTATION_FLIP,
        BindingEvidence.MEASURED_NON_BINDER,
        "ABM",
        "AG0",
        support=replace(
            _support("ABM", "AG0"),
            antibody_residue_ids_sha256=_sha("AB0-residues"),
        ),
        mutation_axis=PartnerAxis.ANTIBODY,
        mutation_spec_sha256=_sha("mutation"),
        paired_assay_receipt_sha256=_sha("paired-assay"),
    )
    with pytest.raises(ValueError, match="change only the antibody"):
        _manifest((_cognate(), mutation))


def test_generic_pair_control_cannot_disguise_a_geometry_decoy() -> None:
    control = _record(
        "disguised-pose",
        ControlFamily.UNKNOWN_UNMEASURED_DECOY,
        BindingEvidence.UNKNOWN_UNMEASURED,
        "AB0",
        "AG0",
        pose="different-pose",
        support=_support("AB0", "AG0"),
    )
    with pytest.raises(ValueError, match="use wrong-patch or wrong-pose"):
        _manifest((_cognate(), control))


def test_same_pair_control_requires_exact_domain_manifest() -> None:
    control = _record(
        "wrong-pose-domain-substitution",
        ControlFamily.WRONG_POSE,
        BindingEvidence.UNKNOWN_UNMEASURED,
        "AB0",
        "AG0",
        pose="different-pose",
        support=replace(
            _support("AB0", "AG0"),
            canonical_domain_manifest_sha256=_sha("substituted-domain"),
        ),
    )
    with pytest.raises(ValueError, match="changes residue identities"):
        _manifest((_cognate(), control))


def test_non_train_roles_are_rejected_before_manifest_construction() -> None:
    with pytest.raises(ValueError, match="outer-train"):
        TrainOnlySplitBinding(
            split_manifest_sha256=_sha("split"),
            train_membership_receipt_sha256=_sha("membership"),
            outer_fold=0,
            inner_fold=0,
            outer_role="held_out",
            inner_role="train",
        )


def test_entities_outside_exact_train_allowlist_are_rejected() -> None:
    control = _record(
        "outside",
        ControlFamily.UNKNOWN_UNMEASURED_DECOY,
        BindingEvidence.UNKNOWN_UNMEASURED,
        "AB1",
        "AG1",
    )
    restricted_allowlist = replace(
        _allowlist(),
        antibody_entity_ids=("AB0",),
    )
    rebound = replace(
        control,
        train_allowlist_sha256=restricted_allowlist.allowlist_sha256,
    )
    cognate = replace(
        _cognate(),
        train_allowlist_sha256=restricted_allowlist.allowlist_sha256,
    )
    with pytest.raises(ValueError, match="outside the train-only allowlist"):
        _manifest((cognate, rebound), allowlist=restricted_allowlist)


def test_manifest_is_order_independent_immutable_and_hash_verified() -> None:
    records = _all_family_records()
    first = _manifest(records)
    second = _manifest(tuple(reversed(records)))

    assert first.records == second.records
    assert first.canonical_bytes == second.canonical_bytes
    assert first.manifest_sha256 == second.manifest_sha256
    assert (
        verify_manifest_document(
            first.to_document(),
            expected_manifest_sha256=first.manifest_sha256,
        )
        == first.manifest_sha256
    )
    with pytest.raises(FrozenInstanceError):
        first.records = ()

    tampered = first.to_document()
    tampered["records"][0]["supervision_mode"] = "binary_negative"
    with pytest.raises(ValueError, match="verification failed"):
        verify_manifest_document(
            tampered,
            expected_manifest_sha256=first.manifest_sha256,
        )


def test_source_binding_rejects_mutable_revision_and_missing_license() -> None:
    with pytest.raises(ValueError, match="immutable"):
        replace(_source(), revision="latest")
    with pytest.raises(ValueError, match="explicit license"):
        replace(_source(), license_id="unknown")


def test_duplicate_constructed_control_identity_is_rejected() -> None:
    first = _record(
        "first",
        ControlFamily.UNKNOWN_UNMEASURED_DECOY,
        BindingEvidence.UNKNOWN_UNMEASURED,
        "AB1",
        "AG1",
    )
    second = replace(first, record_id="second")
    with pytest.raises(ValueError, match="duplicates a constructed identity"):
        _manifest((_cognate(), first, second))


def test_invalid_digest_and_impossible_pair_support_are_rejected() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(_source(), source_content_sha256="ABC")
    with pytest.raises(ValueError, match="Cartesian support"):
        replace(_support("AB0", "AG0"), candidate_pair_count=21)


def test_externally_anchored_digest_rejects_semantic_self_hash_forgery() -> None:
    original_records = _all_family_records()
    original = _manifest(original_records)
    forged_records = tuple(
        replace(
            record,
            control_family=ControlFamily.MEASURED_NON_BINDER,
            binding_evidence=BindingEvidence.MEASURED_NON_BINDER,
            label_provenance_sha256=_sha("forged-label"),
            assay_protocol_sha256=_sha("forged-assay"),
            measurement_receipt_sha256=_sha("forged-measurement"),
        )
        if record.record_id == "unmeasured"
        else record
        for record in original_records
    )
    forged = _manifest(forged_records)
    assert forged.manifest_sha256 != original.manifest_sha256

    with pytest.raises(ValueError, match="external authority"):
        verify_manifest_document(
            forged.to_document(),
            expected_manifest_sha256=original.manifest_sha256,
        )


def test_measured_evidence_registry_binds_the_exact_pair_and_support() -> None:
    cognate = _cognate()
    negative = _record(
        "measured-negative",
        ControlFamily.MEASURED_NON_BINDER,
        BindingEvidence.MEASURED_NON_BINDER,
        "ABM",
        "AG1",
    )
    records = (cognate, negative)
    registry = _evidence_registry(records)
    changed = replace(
        negative,
        antibody_entity_id="AB1",
        support=_support("AB1", "AG1"),
    )

    with pytest.raises(ValueError, match="Measured evidence does not exactly"):
        _manifest((cognate, changed), evidence_registry=registry)


def test_evidence_registry_cannot_cross_allowlist_boundaries() -> None:
    records = (_cognate(),) + (_all_family_records()[1],)
    registry = _evidence_registry(records)
    alternate = replace(_allowlist(), allowlist_id="alternate-train-allowlist")
    rebound = tuple(
        replace(
            record,
            train_allowlist_sha256=alternate.allowlist_sha256,
        )
        for record in records
    )

    with pytest.raises(ValueError, match="different train-only allowlist"):
        _manifest(
            rebound,
            allowlist=alternate,
            evidence_registry=registry,
        )


def test_measurement_receipt_cannot_be_reused_for_a_different_record() -> None:
    cognate = _cognate()
    negative = _record(
        "receipt-reuse",
        ControlFamily.MEASURED_NON_BINDER,
        BindingEvidence.MEASURED_NON_BINDER,
        "AB1",
        "AG1",
        measurement_receipt_sha256=cognate.measurement_receipt_sha256,
    )

    with pytest.raises(ValueError, match="cannot authorize multiple"):
        _manifest((cognate, negative))


@pytest.mark.parametrize(
    "policy_field",
    [
        "domain_selection_policy_sha256",
        "candidate_mask_policy_sha256",
    ],
)
def test_equal_shape_and_mask_cannot_hide_a_different_selection_policy(
    policy_field: str,
) -> None:
    support = replace(
        _support("AB1", "AG1"),
        **{policy_field: _sha(f"label-aware-{policy_field}")},
    )
    control = _record(
        "policy-leak",
        ControlFamily.UNKNOWN_UNMEASURED_DECOY,
        BindingEvidence.UNKNOWN_UNMEASURED,
        "AB1",
        "AG1",
        support=support,
    )

    with pytest.raises(ValueError, match="mask/support"):
        _manifest((_cognate(), control))


def test_patch_and_pose_ids_cannot_rename_unchanged_geometry() -> None:
    cognate = _cognate()
    renamed_patch = _record(
        "renamed-patch",
        ControlFamily.WRONG_PATCH,
        BindingEvidence.UNKNOWN_UNMEASURED,
        "AB0",
        "AG0",
        patch="renamed-patch",
        patch_definition_sha256=cognate.patch_definition_sha256,
        support=_support("AB0", "AG0"),
    )
    with pytest.raises(ValueError, match="patch definition"):
        _manifest((cognate, renamed_patch))

    renamed_pose = _record(
        "renamed-pose",
        ControlFamily.WRONG_POSE,
        BindingEvidence.UNKNOWN_UNMEASURED,
        "AB0",
        "AG0",
        pose="renamed-pose",
        pose_transform_sha256=cognate.pose_transform_sha256,
        support=_support("AB0", "AG0"),
    )
    with pytest.raises(ValueError, match="pose transform"):
        _manifest((cognate, renamed_pose))


def test_mutation_relation_registry_binds_the_exact_mutation_spec() -> None:
    cognate = _cognate()
    mutation = _record(
        "mutation",
        ControlFamily.MUTATION_FLIP,
        BindingEvidence.MEASURED_NON_BINDER,
        "ABM",
        "AG0",
        mutation_axis=PartnerAxis.ANTIBODY,
        mutation_spec_sha256=_sha("original-mutation"),
        paired_assay_receipt_sha256=_sha("paired-assay"),
    )
    records = (cognate, mutation)
    registry = _evidence_registry(records)
    changed = replace(
        mutation,
        mutation_spec_sha256=_sha("different-mutation"),
    )

    with pytest.raises(ValueError, match="Mutation relations do not exactly"):
        _manifest((cognate, changed), evidence_registry=registry)


def test_strict_json_parser_rejects_ambiguous_or_noncanonical_bytes() -> None:
    manifest = _manifest(_all_family_records())
    canonical = json.dumps(
        manifest.to_document(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    parsed = manifest_from_json_bytes(
        canonical,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    assert parsed.manifest_sha256 == manifest.manifest_sha256

    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        manifest_from_json_bytes(
            b'{"duplicate":1,"duplicate":2}',
            expected_manifest_sha256=_sha("untrusted"),
        )
    with pytest.raises(ValueError, match="canonical serialization"):
        manifest_from_json_bytes(
            b" " + canonical,
            expected_manifest_sha256=manifest.manifest_sha256,
        )


def test_unicode_and_integer_canonicalization_fail_closed() -> None:
    with pytest.raises(ValueError, match="NFC-normalized"):
        replace(_source(), source_id="source-e\u0301")
    with pytest.raises(ValueError, match="interoperable JSON integer"):
        replace(_split(), outer_fold=1 << 53)


def test_registry_copies_and_freezes_nested_digest_sequences() -> None:
    record = _cognate()
    measured = [record.measured_evidence_binding_sha256]
    registry = TrainOnlyEvidenceRegistry(
        split_manifest_sha256=record.split.split_manifest_sha256,
        train_membership_receipt_sha256=(
            record.split.train_membership_receipt_sha256
        ),
        outer_fold=record.split.outer_fold,
        inner_fold=record.split.inner_fold,
        train_allowlist_sha256=_allowlist().allowlist_sha256,
        authority_receipt_sha256=_sha("authority"),
        measured_evidence_binding_sha256s=measured,
    )
    measured.append(_sha("late-mutation"))

    assert registry.measured_evidence_binding_sha256s == (
        record.measured_evidence_binding_sha256,
    )
    with pytest.raises(FrozenInstanceError):
        registry.authority_receipt_sha256 = _sha("changed")
