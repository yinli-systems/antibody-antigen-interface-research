"""Tests for PAIRBench v0 evidence semantics and input tracks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interaction_first.pairbench_schema import (
    PairbenchSchemaError,
    authorize_input_track,
    binding_target_for_evidence,
    load_pairbench_contract,
    validate_pair_record,
)


def _contract_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "pairbench_v0"
        / "content_contract.json"
    )


def _pair_record(evidence: str, target: int | None) -> dict[str, object]:
    return {
        "antibody_id": "ab_sha256:abc",
        "antigen_id": "ag_sha256:def",
        "assay_id": "assay:one" if evidence.startswith("measured_") else None,
        "binary_binding_target": target,
        "evidence_semantics": evidence,
        "pair_id": "pair_sha256:ghi",
        "source_id": "source.one",
        "source_record_id": "row:1",
    }


def test_contract_is_complete_and_hashable() -> None:
    contract, digest = load_pairbench_contract(_contract_path())
    assert len(digest) == 64
    assert contract["schema_status"] == "development_not_frozen"


@pytest.mark.parametrize(
    ("evidence", "target"),
    [
        ("measured_binder", 1),
        ("measured_nonbinder", 0),
        ("unknown_cross_pair", None),
        ("synthetic_decoy", None),
        ("observed_structural_positive", None),
        ("predicted_complex_candidate", None),
    ],
)
def test_pair_evidence_has_exact_binary_semantics(
    evidence: str,
    target: int | None,
) -> None:
    assert binding_target_for_evidence(evidence) == target
    validate_pair_record(_pair_record(evidence, target))


def test_unknown_cross_pair_cannot_be_promoted_to_nonbinder() -> None:
    with pytest.raises(PairbenchSchemaError, match="contradicts"):
        validate_pair_record(_pair_record("unknown_cross_pair", 0))


def test_input_tracks_reject_complex_features_before_track_c() -> None:
    contract, _ = load_pairbench_contract(_contract_path())
    authorize_input_track(
        contract,
        track="S",
        feature_groups=["antibody_sequence", "antigen_sequence"],
    )
    with pytest.raises(PairbenchSchemaError, match="unauthorized"):
        authorize_input_track(
            contract,
            track="M",
            feature_groups=["antibody_sequence", "cross_chain_distance"],
        )
    authorize_input_track(
        contract,
        track="C",
        feature_groups=["antibody_sequence", "cross_chain_distance"],
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("unknown_negative", "binary role differs"),
        ("unsorted_fields", "sorted and unique"),
        ("primary_key_not_required", "primary key is not required"),
        ("complex_feature_in_m", "allowed feature groups differ"),
        ("frozen_status", "status is not fail-closed"),
    ],
)
def test_contract_mutations_fail_closed(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    payload = json.loads(_contract_path().read_text(encoding="utf-8"))
    if mutation == "unknown_negative":
        payload["evidence_semantics"]["unknown_cross_pair"][
            "binary_binding_role"
        ] = "negative"
    elif mutation == "unsorted_fields":
        payload["tables"]["pairs.parquet"]["required_fields"].reverse()
    elif mutation == "primary_key_not_required":
        payload["tables"]["pairs.parquet"]["primary_key"] = ["missing_id"]
    elif mutation == "complex_feature_in_m":
        payload["input_tracks"]["M"]["allowed_feature_groups"].append(
            "complex_candidate_coordinates"
        )
        payload["input_tracks"]["M"]["allowed_feature_groups"].sort()
    elif mutation == "frozen_status":
        payload["schema_status"] = "frozen"
    else:
        raise AssertionError("unknown test mutation")
    path = tmp_path / f"{mutation}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PairbenchSchemaError, match=error):
        load_pairbench_contract(path)
