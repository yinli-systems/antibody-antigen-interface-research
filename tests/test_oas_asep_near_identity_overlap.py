"""Test fail-closed dual-track OAS-to-AsEP near-identity reconciliation."""

from __future__ import annotations

import pytest

from antibody_pretraining.downstream_overlap import (
    DevelopmentChainReference,
    sequence_sha256,
)
from antibody_pretraining.near_identity_overlap import (
    IMGT_ENGINE,
    MMSEQS_ENGINE,
    NearIdentityHit,
    NearIdentityOverlapError,
    NearIdentityThresholds,
    REQUIRED_SENSITIVITY_CONTROLS,
    SearchTrackEvidence,
    normalize_fraction,
    parse_mmseqs_tsv,
    reconcile_near_identity_evidence,
    reference_key,
)


def _reference(instance: str, chain: str = "heavy", role: str = "train"):
    sequence = ("A" if chain == "heavy" else "C") * 100
    return DevelopmentChainReference(
        chain=chain,
        instance_id=instance,
        role=role,
        sequence_length=len(sequence),
        sequence_sha256=sequence_sha256(sequence),
    )


def _controls(*, passed: bool = True) -> dict[str, bool]:
    return {name: passed for name in REQUIRED_SENSITIVITY_CONTROLS}


def _track(
    engine: str,
    references: tuple[DevelopmentChainReference, ...],
    *,
    hits: tuple[NearIdentityHit, ...] = (),
    completed: bool = True,
    query_keys: frozenset[str] | None = None,
    target_splits: frozenset[str] = frozenset({"train", "validation"}),
    truncated: bool = False,
    unrepresentable: frozenset[str] = frozenset(),
    controls: dict[str, bool] | None = None,
) -> SearchTrackEvidence:
    return SearchTrackEvidence(
        engine=engine,
        completed=completed,
        query_reference_keys=(
            frozenset(reference_key(item) for item in references)
            if query_keys is None
            else query_keys
        ),
        target_splits_completed=target_splits,
        maximum_sequences_truncated=truncated,
        unrepresentable_query_keys=unrepresentable,
        sensitivity_controls=_controls() if controls is None else controls,
        hits=hits,
    )


def _hit(
    engine: str,
    reference: DevelopmentChainReference,
    *,
    identity: float = 0.90,
    query_coverage: float = 0.90,
    target_coverage: float = 0.90,
) -> NearIdentityHit:
    return NearIdentityHit(
        engine=engine,
        query_reference_key=reference_key(reference),
        target_id=f"oas-{engine}",
        target_split="train",
        identity=identity,
        query_coverage=query_coverage,
        target_coverage=target_coverage,
    )


def test_fraction_normalization_and_frozen_mmseqs_parser() -> None:
    reference = _reference("complex-1")
    query = reference_key(reference)
    rows = parse_mmseqs_tsv(
        [f"{query}\toas-1\t90\t0.91\t92\n"],
        target_split_by_id={"oas-1": "train"},
    )
    assert rows[0].identity == pytest.approx(0.90)
    assert rows[0].query_coverage == pytest.approx(0.91)
    assert rows[0].target_coverage == pytest.approx(0.92)
    assert normalize_fraction(100) == 1.0
    with pytest.raises(NearIdentityOverlapError, match="five columns"):
        parse_mmseqs_tsv(["too\tfew\tcolumns\n"], target_split_by_id={})
    with pytest.raises(NearIdentityOverlapError, match="absent from its manifest"):
        parse_mmseqs_tsv(
            [f"{query}\tunknown\t0.9\t0.9\t0.9\n"],
            target_split_by_id={},
        )


def test_inclusive_90_90_90_union_contaminates_entire_instance() -> None:
    heavy = _reference("complex-1", "heavy")
    light = _reference("complex-1", "light")
    clean = _reference("complex-2", "heavy", "val")
    references = (heavy, light, clean)
    mmseqs = _track(
        MMSEQS_ENGINE,
        references,
        hits=(_hit(MMSEQS_ENGINE, heavy),),
    )
    imgt = _track(
        IMGT_ENGINE,
        references,
        hits=(_hit(IMGT_ENGINE, heavy),),
    )
    result = reconcile_near_identity_evidence(
        references=references,
        exact_overlap_instance_ids=(),
        tracks=(mmseqs, imgt),
    )
    assert result["exactly_or_near_identity_contaminated_instance_ids"] == [
        "complex-1"
    ]
    assert result["decontaminated_development_instance_ids"] == ["complex-2"]
    assert result["transfer_gate"]["downstream_transfer_authorized"] is True
    assert result["transfer_gate"]["authorized_instance_ids"] == ["complex-2"]
    assert result["transfer_gate"]["official_or_frozen_test_accessed"] is False


def test_below_any_threshold_is_not_a_qualified_hit() -> None:
    reference = _reference("complex")
    for metric in ("identity", "query_coverage", "target_coverage"):
        values = {
            "identity": 0.90,
            "query_coverage": 0.90,
            "target_coverage": 0.90,
        }
        values[metric] = 0.899999
        hit = _hit(MMSEQS_ENGINE, reference, **values)
        assert hit.qualifies(NearIdentityThresholds()) is False


def test_missing_query_and_failed_controls_fail_closed() -> None:
    first = _reference("complex-1")
    second = _reference("complex-2")
    references = (first, second)
    incomplete = _track(
        MMSEQS_ENGINE,
        references,
        query_keys=frozenset({reference_key(first)}),
    )
    controls = _controls()
    controls["internal_indel_above_threshold"] = False
    failed_control = _track(IMGT_ENGINE, references, controls=controls)
    result = reconcile_near_identity_evidence(
        references=references,
        exact_overlap_instance_ids=(),
        tracks=(incomplete, failed_control),
    )
    assert result["completion_failures_by_engine"][MMSEQS_ENGINE] == [
        "query_coverage_incomplete"
    ]
    assert result["completion_failures_by_engine"][IMGT_ENGINE] == [
        "sensitivity_control_failed"
    ]
    assert result["transfer_gate"]["downstream_transfer_authorized"] is False
    assert result["transfer_gate"]["authorized_instance_ids"] == []


def test_engine_disagreement_fails_closed_even_after_union_decontamination() -> None:
    contaminated = _reference("complex-1")
    clean = _reference("complex-2")
    references = (contaminated, clean)
    mmseqs = _track(
        MMSEQS_ENGINE,
        references,
        hits=(_hit(MMSEQS_ENGINE, contaminated),),
    )
    imgt = _track(IMGT_ENGINE, references)
    result = reconcile_near_identity_evidence(
        references=references,
        exact_overlap_instance_ids=(),
        tracks=(mmseqs, imgt),
    )
    assert result["decontaminated_development_instance_ids"] == ["complex-2"]
    assert result["counts"]["engine_disagreement_chain_references"] == 1
    assert result["transfer_gate"]["dual_track_evidence_completed"] is True
    assert result["transfer_gate"]["engine_agreement_passed"] is False
    assert result["transfer_gate"]["downstream_transfer_authorized"] is False


def test_unrepresentable_or_truncated_search_fails_closed() -> None:
    reference = _reference("complex")
    key = reference_key(reference)
    mmseqs = _track(MMSEQS_ENGINE, (reference,), truncated=True)
    imgt = _track(
        IMGT_ENGINE,
        (reference,),
        unrepresentable=frozenset({key}),
    )
    result = reconcile_near_identity_evidence(
        references=(reference,),
        exact_overlap_instance_ids=(),
        tracks=(mmseqs, imgt),
    )
    assert result["completion_failures_by_engine"][MMSEQS_ENGINE] == [
        "hit_list_truncated"
    ]
    assert result["completion_failures_by_engine"][IMGT_ENGINE] == [
        "unrepresentable_query"
    ]
    assert result["transfer_gate"]["downstream_transfer_authorized"] is False


def test_test_role_and_unknown_exact_instance_are_rejected() -> None:
    invalid = _reference("official", role="train")
    invalid = DevelopmentChainReference(
        chain=invalid.chain,
        instance_id=invalid.instance_id,
        role="test",
        sequence_length=invalid.sequence_length,
        sequence_sha256=invalid.sequence_sha256,
    )
    with pytest.raises(NearIdentityOverlapError, match="data role"):
        reconcile_near_identity_evidence(
            references=(invalid,), exact_overlap_instance_ids=(), tracks=()
        )
    valid = _reference("development")
    with pytest.raises(NearIdentityOverlapError, match="unknown instance"):
        reconcile_near_identity_evidence(
            references=(valid,), exact_overlap_instance_ids=("official",), tracks=()
        )
