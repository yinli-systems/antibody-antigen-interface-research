"""Tests for fail-closed AVIDa-hIL6 PAIRBench candidate generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from interaction_first.avida_hil6_candidate import (
    AvidaCandidateError,
    AvidaExpectations,
    build_avida_hil6_candidate,
)


def _canonical(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(root: Path, *, conflicting: bool = False) -> tuple[Path, Path]:
    raw_root = root / "raw" / "avida_hil6_2024_11"
    raw_root.mkdir(parents=True)
    pairs = (
        "VHH_sequence,label,Ag_label,subject_species,subject_name,subject_sex\n"
        "ACDE,1,IL6_WT,alpaca,wizzy,male\n"
        "FGHI,0,IL6_WT,alpaca,wizzy,male\n"
        "ACDE,0,IL6_A,alpaca,wizzy,male\n"
        + ("ACDE,0,IL6_WT,alpaca,wizzy,male\n" if conflicting else "KLMN,1,IL6_A,alpaca,wizzy,male\n")
    ).encode()
    antigens = b"Ag_label,Ag_sequence\nIL6_WT,PQRS\nIL6_A,TVWY\n"
    files = {
        ".authority/provider_metadata.json": b"{}\n",
        ".gitattributes": b"\n",
        "AVIDa-hIL6.csv": pairs,
        "README.md": b"fixture\n",
        "antigen_sequences.csv": antigens,
    }
    inventory = {
        "files": [],
        "format_version": "pairscale_raw_sha256_inventory_v1",
        "materialized_at_utc": "2026-07-27T15:00:00Z",
        "pinned_revision": "fixture-revision",
        "source_id": "avida_hil6_2024_11",
        "status": "materialized",
    }
    for name, payload in sorted(files.items()):
        path = raw_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        inventory["files"].append(
            {"path": name, "sha256": _sha(payload), "size_bytes": len(payload)}
        )
    registry_root = root / "registry"
    registry = registry_root / "avida_hil6_2024_11"
    registry.mkdir(parents=True)
    source = {
        "canonical_record_url": "https://example.org/avida",
        "confirmatory_evaluation_eligible": False,
        "declared_scale": {"count": 4, "source_claim_only": True, "unit": "pairs"},
        "display_name": "fixture",
        "duplicate_groups_status": "not_computed",
        "format_version": "pairscale_source_manifest_v1",
        "pinned_revision": "fixture-revision",
        "planned_roles": ["measured_binding_supervision"],
        "registered_at_utc": "2026-07-27T15:00:00Z",
        "sealed_evaluation_source": False,
        "source_class": "fixture",
        "source_id": "avida_hil6_2024_11",
        "training_policy": "eligible_after_materialization",
    }
    license_receipt = {
        "assets": [
            {
                "asset_name": "fixture",
                "asset_scope": "data",
                "attribution_required": True,
                "commercial_use_permitted": False,
                "declared_license_spdx": "CC-BY-NC-4.0",
                "evidence_note": "fixture",
                "evidence_revision": "fixture",
                "evidence_sha256": "0" * 64,
                "evidence_url": "https://example.org/license",
                "license_route": "noncommercial_only",
            }
        ],
        "format_version": "pairscale_license_receipt_v1",
        "legal_review_required_for_release": True,
        "receipt_status": "reviewed",
        "reviewed_at_utc": "2026-07-27T15:00:00Z",
        "source_id": "avida_hil6_2024_11",
    }
    normalization = {
        "format_version": "pairscale_normalization_receipt_v1",
        "input_raw_inventory_sha256": None,
        "normalized_at_utc": None,
        "normalizer_source_sha256": None,
        "outputs": [],
        "source_id": "avida_hil6_2024_11",
        "status": "not_run",
    }
    (registry / "source_manifest.json").write_bytes(_canonical(source))
    (registry / "license_receipt.json").write_bytes(_canonical(license_receipt))
    (registry / "normalization_receipt.json").write_bytes(_canonical(normalization))
    inventory_bytes = _canonical(inventory)
    (registry / "raw_sha256.json").write_bytes(inventory_bytes)
    (raw_root / "raw_sha256.json").write_bytes(inventory_bytes)
    return raw_root, registry_root


def _expectations(*, conflicting: bool = False) -> AvidaExpectations:
    return AvidaExpectations(
        rows=4,
        antibodies=2 if conflicting else 3,
        antigens=2,
        binders=1 if conflicting else 2,
        nonbinders=3 if conflicting else 2,
        exact_duplicate_groups=1 if conflicting else 0,
        exact_duplicate_records=1 if conflicting else 0,
        conflicting_pairs=1 if conflicting else 0,
        subject_species=("alpaca",),
        subject_names=("wizzy",),
        subject_sexes=("male",),
    )


def _contract() -> Path:
    return Path(__file__).resolve().parents[1] / "schemas/pairbench_v0/content_contract.json"


def test_candidate_preserves_construct_semantics_and_binary_evidence(tmp_path: Path) -> None:
    raw, registry = _fixture(tmp_path)
    output = tmp_path / "candidate"
    receipt = build_avida_hil6_candidate(
        raw_root=raw,
        registry_root=registry,
        contract_path=_contract(),
        output_root=output,
        expectations=_expectations(),
        created_at_utc="2026-07-27T16:00:00Z",
    )
    assert receipt["status"] == "candidate_complete_blocked"
    assert not receipt["registry_normalization_eligible"]
    antibodies = pq.read_table(output / "entities_antibody.parquet").to_pylist()
    assert all(row["vh_sequence"] is None for row in antibodies)
    assert all("untrimmed" in row["annotation_status"] for row in antibodies)
    pairs = pq.read_table(output / "pairs.parquet").to_pylist()
    assert {(row["evidence_semantics"], row["binary_binding_target"]) for row in pairs} == {
        ("measured_binder", 1),
        ("measured_nonbinder", 0),
    }
    assert pq.read_table(output / "duplicate_groups.parquet").num_rows == 0


def test_raw_tamper_fails_before_candidate_creation(tmp_path: Path) -> None:
    raw, registry = _fixture(tmp_path)
    with (raw / "AVIDa-hIL6.csv").open("ab") as handle:
        handle.write(b"tamper\n")
    output = tmp_path / "candidate"
    with pytest.raises(AvidaCandidateError, match="size differs"):
        build_avida_hil6_candidate(
            raw_root=raw,
            registry_root=registry,
            contract_path=_contract(),
            output_root=output,
            expectations=_expectations(),
        )
    assert not output.exists()


def test_conflicting_exact_pair_is_preserved_and_blocks_expectation(tmp_path: Path) -> None:
    raw, registry = _fixture(tmp_path, conflicting=True)
    output = tmp_path / "candidate"
    receipt = build_avida_hil6_candidate(
        raw_root=raw,
        registry_root=registry,
        contract_path=_contract(),
        output_root=output,
        expectations=_expectations(conflicting=True),
        created_at_utc="2026-07-27T16:00:00Z",
    )
    assert receipt["counts"]["conflicting_pairs"] == 1
    duplicate = pq.read_table(output / "duplicate_groups.parquet").to_pylist()[0]
    assert duplicate["labels"] == [1, 0]
