"""Tests for authenticated raw score-free authority derivation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from interaction_first.score_free_authority import (
    AUTHORITY_FORMAT_VERSION,
    EDGE_LEDGER_FORMAT_VERSION,
    AuthorityCounts,
    ScoreFreeAuthorityError,
    build_score_free_authority,
    canonical_json_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_score_free_authority.py"
SPEC = importlib.util.spec_from_file_location("build_score_free_authority", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def _frame(sequence: str) -> bytes:
    encoded = sequence.encode("ascii")
    return str(len(encoded)).encode("ascii") + b":" + encoded


def _antibody_id(vh: str, vl: str) -> str:
    return hashlib.sha256(b"antibody-v1\0" + _frame(vh) + _frame(vl)).hexdigest()


def _antigen_id(sequence: str) -> str:
    return hashlib.sha256(b"antigen-v1\0" + _frame(sequence)).hexdigest()


def _fixture() -> tuple[bytes, bytes, dict[str, bytes], AuthorityCounts]:
    rows = [
        ("row0", "AAAAA", "", "IIIII"),
        ("row1", "CCCCC", "DDD", "KKKKK"),
        ("row2", "EEEEE", "FFF", "LLLLL"),
        ("row3", "GGGGG", "HHH", "MMMMM"),
        ("row4", "AAAAA", "", "IIIII"),
    ]
    authority = canonical_json_bytes(
        {
            "format_version": AUTHORITY_FORMAT_VERSION,
            "authorized_split": "train",
            "instance_ids": [row[0] for row in rows],
        }
    )
    manifest = (
        "INSTANCE,vh_seq,vl_seq,ag_seq\n"
        + "".join(",".join(row) + "\n" for row in rows)
    ).encode("ascii")
    antibody_ids = [_antibody_id(vh, vl) for _, vh, vl, _ in rows[:4]]
    antigen_ids = [_antigen_id(antigen) for _, _, _, antigen in rows[:4]]
    raw_edges = {
        "vh": [sorted((antibody_ids[0], antibody_ids[1]))],
        "vl": [sorted((antibody_ids[2], antibody_ids[3]))],
        "concatenated_antibody": [
            sorted((antibody_ids[1], antibody_ids[2]))
        ],
        "antigen": [
            sorted((antigen_ids[0], antigen_ids[1])),
            sorted((antigen_ids[2], antigen_ids[3])),
        ],
    }
    ledgers = {
        view: canonical_json_bytes(
            {
                "format_version": EDGE_LEDGER_FORMAT_VERSION,
                "view": view,
                "edges": sorted(edges),
            }
        )
        for view, edges in raw_edges.items()
    }
    return (
        authority,
        manifest,
        ledgers,
        AuthorityCounts(5, 4, 4, 4),
    )


def test_raw_derivation_reproduces_lengths_degrees_components_and_pairs() -> None:
    authority, manifest, ledgers, counts = _fixture()
    bundle = build_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_counts=counts,
    )

    assert len(bundle.antibodies) == 4
    assert len(bundle.antigens) == 4
    assert len(bundle.pairs) == 4
    assert sum(len(pair.source_instance_ids) for pair in bundle.pairs) == 5
    assert sorted(value.union_direct_degree for value in bundle.antibodies) == [1, 1, 2, 2]
    assert {value.component_size for value in bundle.antibodies} == {4}
    assert {value.component_size for value in bundle.antigens} == {2}
    assert {value.pair_incidence_degree for value in bundle.antibodies} == {1}
    assert {value.pair_incidence_degree for value in bundle.antigens} == {1}
    missing = [value for value in bundle.antibodies if value.vl_length is None]
    assert len(missing) == 1
    assert missing[0].vl_direct_degree is None
    assert {value.axis for value in bundle.quantile_schemes} == {
        "antibody",
        "antigen",
    }
    payload = bundle.canonical_payload()
    assert payload["counts"] == {
        "source_rows": 5,
        "exact_pairs": 4,
        "antibody_entities": 4,
        "antigen_entities": 4,
    }
    assert payload["scientific_payload_sha256"] == bundle.scientific_payload_sha256


def test_authority_and_manifest_schemas_reject_forbidden_fields() -> None:
    authority, manifest, ledgers, counts = _fixture()
    contaminated_authority = canonical_json_bytes(
        {
            "format_version": AUTHORITY_FORMAT_VERSION,
            "authorized_split": "train",
            "instance_ids": ["row0"],
            "test_instance_ids": ["sealed"],
        }
    )
    with pytest.raises(ScoreFreeAuthorityError, match="forbidden or missing"):
        build_score_free_authority(
            authority_bytes=contaminated_authority,
            manifest_bytes=manifest,
            edge_ledger_bytes=ledgers,
            expected_counts=counts,
        )

    contaminated_manifest = manifest.replace(
        b"INSTANCE,vh_seq,vl_seq,ag_seq\n",
        b"INSTANCE,vh_seq,vl_seq,ag_seq,contact_density\n",
    )
    with pytest.raises(ScoreFreeAuthorityError, match="exactly INSTANCE"):
        build_score_free_authority(
            authority_bytes=authority,
            manifest_bytes=contaminated_manifest,
            edge_ledger_bytes=ledgers,
            expected_counts=counts,
        )


def test_inventory_counts_are_a_hard_external_anchor() -> None:
    authority, manifest, ledgers, _ = _fixture()
    with pytest.raises(ScoreFreeAuthorityError, match="authority inventory differs"):
        build_score_free_authority(
            authority_bytes=authority,
            manifest_bytes=manifest,
            edge_ledger_bytes=ledgers,
            expected_counts=AuthorityCounts(6, 4, 4, 4),
        )


def test_edge_ledgers_reject_unknown_unsorted_and_missing_vl_endpoints() -> None:
    authority, manifest, ledgers, counts = _fixture()
    vh = json.loads(ledgers["vh"])
    vh["edges"] = [["f" * 64, "0" * 64]]
    changed = {**ledgers, "vh": canonical_json_bytes(vh)}
    with pytest.raises(ScoreFreeAuthorityError, match="canonical and distinct"):
        build_score_free_authority(
            authority_bytes=authority,
            manifest_bytes=manifest,
            edge_ledger_bytes=changed,
            expected_counts=counts,
        )

    rows = manifest.decode("ascii").splitlines()
    missing_vl_id = _antibody_id("AAAAA", "")
    present_vl_id = _antibody_id("CCCCC", "DDD")
    assert rows[1].startswith("row0,")
    vl = json.loads(ledgers["vl"])
    vl["edges"] = [sorted((missing_vl_id, present_vl_id))]
    changed = {**ledgers, "vl": canonical_json_bytes(vl)}
    with pytest.raises(ScoreFreeAuthorityError, match="missing VL"):
        build_score_free_authority(
            authority_bytes=authority,
            manifest_bytes=manifest,
            edge_ledger_bytes=changed,
            expected_counts=counts,
        )


def test_every_required_view_and_only_required_views_must_be_bound() -> None:
    authority, manifest, ledgers, counts = _fixture()
    for changed in (
        {key: value for key, value in ledgers.items() if key != "vl"},
        {**ledgers, "model_scores": canonical_json_bytes({})},
    ):
        with pytest.raises(ScoreFreeAuthorityError, match="exactly four"):
            build_score_free_authority(
                authority_bytes=authority,
                manifest_bytes=manifest,
                edge_ledger_bytes=changed,
                expected_counts=counts,
            )


def test_byte_reordering_changes_input_binding_but_not_derived_inventory() -> None:
    authority, manifest, ledgers, counts = _fixture()
    original = build_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_counts=counts,
    )
    authority_payload = json.loads(authority)
    authority_payload["instance_ids"] = list(
        reversed(authority_payload["instance_ids"])
    )
    reordered = build_score_free_authority(
        authority_bytes=canonical_json_bytes(authority_payload),
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_counts=counts,
    )
    assert original.antibodies == reordered.antibodies
    assert original.antigens == reordered.antigens
    assert original.pairs == reordered.pairs
    assert original.input_sha256 != reordered.input_sha256
    assert original.scientific_payload_sha256 != reordered.scientific_payload_sha256


def test_verified_reader_consumes_anchored_bytes_and_rejects_symlinks() -> None:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        path = root / "input.json"
        payload = b"authenticated\n"
        path.write_bytes(payload)
        verified = SCRIPT.read_verified_regular_file(
            path,
            hashlib.sha256(payload).hexdigest(),
            "test input",
        )
        assert verified.payload == payload
        assert verified.reference["nofollow_same_descriptor_read"] is True

        link = root / "link.json"
        link.symlink_to(path)
        with pytest.raises(ValueError, match="symlink"):
            SCRIPT.read_verified_regular_file(
                link,
                hashlib.sha256(payload).hexdigest(),
                "linked input",
            )


def test_create_only_publication_never_replaces_existing_bytes() -> None:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        output = Path(directory) / "bundle.json"
        SCRIPT.create_only_json(output, {"value": 1})
        original = output.read_bytes()
        with pytest.raises(FileExistsError, match="already exists"):
            SCRIPT.create_only_json(output, {"value": 2})
        assert output.read_bytes() == original
        assert os.stat(output).st_nlink == 1
