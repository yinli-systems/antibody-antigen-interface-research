"""Tests for the independent raw-byte V6-A authority auditor."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
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
    build_score_free_authority,
    canonical_json_bytes,
    implementation_source_sha256 as primary_authority_source_sha256,
)
from interaction_first.score_free_independent_auditor import (
    IndependentScoreFreeAuditError,
    audit_cross_fold_candidate,
    audit_pooled_outer_candidate,
    audit_score_free_authority_candidate,
    audit_score_free_fold_candidate,
    authority_audit_markdown,
    reconstruct_score_free_authority,
)
from interaction_first.score_free_split_statistics import (
    INNER_DISTRIBUTION_THRESHOLDS,
    INNER_FOLD_SHAPE_THRESHOLDS,
    OUTER_CROSS_FOLD_THRESHOLDS,
    OUTER_DISTRIBUTION_THRESHOLDS,
    OUTER_FOLD_SHAPE_THRESHOLDS,
    antibody_profile,
    antigen_profile,
    audit_cross_fold_balance,
    audit_inner_score_free_fold,
    audit_pooled_outer_representativeness,
    audit_score_free_fold,
    implementation_source_sha256 as primary_statistics_source_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
AUDITOR_SOURCE = ROOT / "src" / "interaction_first" / "score_free_independent_auditor.py"
SCRIPT_PATH = ROOT / "scripts" / "audit_score_free_authority_independent.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_score_free_authority_independent",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def _primary_implementation_sha256() -> dict[str, str]:
    return {
        "score_free_authority": primary_authority_source_sha256(),
        "score_free_split_statistics": primary_statistics_source_sha256(),
    }


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
        ("row4", "NNNNN", "PPP", "QQQQQ"),
        ("row5", "RRRRR", "SSS", "TTTTT"),
        ("row6", "VVVVV", "WWW", "YYYYY"),
        ("row7", "AAAAA", "", "IIIII"),
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
    antibody_ids = [_antibody_id(vh, vl) for _, vh, vl, _ in rows[:7]]
    antigen_ids = [_antigen_id(antigen) for _, _, _, antigen in rows[:7]]
    edge_rows = {
        "vh": [sorted((antibody_ids[0], antibody_ids[1]))],
        "vl": [
            sorted((antibody_ids[1], antibody_ids[2])),
            sorted((antibody_ids[4], antibody_ids[5])),
        ],
        "concatenated_antibody": [
            sorted((antibody_ids[2], antibody_ids[3])),
            sorted((antibody_ids[5], antibody_ids[6])),
        ],
        "antigen": [
            sorted((antigen_ids[0], antigen_ids[1])),
            sorted((antigen_ids[1], antigen_ids[2])),
            sorted((antigen_ids[4], antigen_ids[5])),
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
        for view, edges in edge_rows.items()
    }
    return authority, manifest, ledgers, AuthorityCounts(8, 7, 7, 7)


def _primary_candidate() -> tuple[bytes, bytes, dict[str, bytes], AuthorityCounts, bytes]:
    authority, manifest, ledgers, counts = _fixture()
    bundle = build_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_counts=counts,
    )
    return authority, manifest, ledgers, counts, canonical_json_bytes(
        bundle.canonical_payload()
    )


def test_independent_reconstruction_exactly_matches_primary_payload() -> None:
    authority, manifest, ledgers, counts, candidate = _primary_candidate()
    independent = reconstruct_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_primary_implementation_sha256=_primary_implementation_sha256(),
        expected_counts={
            "source_rows": counts.source_rows,
            "exact_pairs": counts.exact_pairs,
            "antibody_entities": counts.antibody_entities,
            "antigen_entities": counts.antigen_entities,
        },
    )
    assert canonical_json_bytes(independent) == candidate
    receipt = audit_score_free_authority_candidate(
        candidate_bytes=candidate,
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_primary_implementation_sha256=_primary_implementation_sha256(),
        expected_counts=independent["counts"],
    )
    assert receipt["passed"] is True
    assert receipt["field_by_field_agreement"] is True
    assert receipt["training_authorized"] is False


def test_published_candidate_binds_primary_controller_source() -> None:
    authority, manifest, ledgers, counts, candidate = _primary_candidate()
    payload = json.loads(candidate)
    controller_sha256 = "c" * 64
    payload["publication_controller_sha256"] = controller_sha256
    payload["publication_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    receipt = audit_score_free_authority_candidate(
        candidate_bytes=canonical_json_bytes(payload),
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_primary_implementation_sha256=(
            _primary_implementation_sha256()
        ),
        expected_primary_controller_sha256=controller_sha256,
        expected_counts={
            "source_rows": counts.source_rows,
            "exact_pairs": counts.exact_pairs,
            "antibody_entities": counts.antibody_entities,
            "antigen_entities": counts.antigen_entities,
        },
    )
    assert receipt["primary_publication_controller_sha256"] == controller_sha256
    with pytest.raises(
        IndependentScoreFreeAuditError,
        match="controller source disagrees",
    ):
        audit_score_free_authority_candidate(
            candidate_bytes=canonical_json_bytes(payload),
            authority_bytes=authority,
            manifest_bytes=manifest,
            edge_ledger_bytes=ledgers,
            expected_primary_implementation_sha256=(
                _primary_implementation_sha256()
            ),
            expected_primary_controller_sha256="d" * 64,
            expected_counts={
                "source_rows": counts.source_rows,
                "exact_pairs": counts.exact_pairs,
                "antibody_entities": counts.antibody_entities,
                "antigen_entities": counts.antigen_entities,
            },
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "antibody_degree",
        "component_identity",
        "pair_incidence",
        "quantile_cut",
        "raw_input_hash",
    ],
)
def test_independent_auditor_rejects_each_candidate_tamper(mutation: str) -> None:
    authority, manifest, ledgers, counts, candidate = _primary_candidate()
    payload = json.loads(candidate)
    if mutation == "antibody_degree":
        payload["antibodies"][0]["union_direct_degree"] += 1
    elif mutation == "component_identity":
        payload["antigens"][0]["component_id"] = "0" * 64
    elif mutation == "pair_incidence":
        payload["pairs"][0]["antibody_id"] = payload["pairs"][1]["antibody_id"]
    elif mutation == "quantile_cut":
        payload["quantile_schemes"][0]["cuts_decimal"].append("999")
    else:
        payload["input_sha256"]["canonical_manifest"] = "0" * 64
    with pytest.raises(IndependentScoreFreeAuditError, match="disagreement"):
        audit_score_free_authority_candidate(
            candidate_bytes=canonical_json_bytes(payload),
            authority_bytes=authority,
            manifest_bytes=manifest,
            edge_ledger_bytes=ledgers,
            expected_primary_implementation_sha256=(
                _primary_implementation_sha256()
            ),
            expected_counts={
                "source_rows": counts.source_rows,
                "exact_pairs": counts.exact_pairs,
                "antibody_entities": counts.antibody_entities,
                "antigen_entities": counts.antigen_entities,
            },
        )


def test_independent_auditor_has_no_primary_or_solver_import() -> None:
    tree = ast.parse(AUDITOR_SOURCE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    forbidden = {
        "interaction_first.score_free_authority",
        "interaction_first.score_free_split_statistics",
        "interaction_first.nested_joint_milp",
    }
    assert imported.isdisjoint(forbidden)


def test_independent_fold_statistics_match_primary_decimal_fraction_payload() -> None:
    authority, manifest, ledgers, counts = _fixture()
    bundle = build_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_counts=counts,
    )
    pair_ids = tuple(pair.pair_id for pair in bundle.pairs)
    audit = audit_score_free_fold(
        fold_id="outer_0",
        pairs=tuple(pair.score_free_pair() for pair in bundle.pairs),
        antibody_profiles=tuple(antibody_profile(row) for row in bundle.antibodies),
        antigen_profiles=tuple(antigen_profile(row) for row in bundle.antigens),
        validation_pair_ids=pair_ids[:2],
        training_pair_ids=pair_ids[2:],
        shape_thresholds=OUTER_FOLD_SHAPE_THRESHOLDS,
        distribution_thresholds=OUTER_DISTRIBUTION_THRESHOLDS,
    )
    authority_payload = reconstruct_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_primary_implementation_sha256=_primary_implementation_sha256(),
        expected_counts={
            "source_rows": counts.source_rows,
            "exact_pairs": counts.exact_pairs,
            "antibody_entities": counts.antibody_entities,
            "antigen_entities": counts.antigen_entities,
        },
    )
    candidate = canonical_json_bytes(audit.canonical_payload())
    receipt = audit_score_free_fold_candidate(
        candidate_fold_bytes=candidate,
        authority_payload=authority_payload,
        scope="outer",
        expected_primary_source_sha256=primary_statistics_source_sha256(),
    )
    assert receipt["field_by_field_agreement"] is True
    assert receipt["candidate_gate_passed"] is audit.passed


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("panel", "bidirectional_chance_mrr"), [1, 1]),
        (
            (
                "representativeness",
                "antibody_vs_universe",
                "features",
                0,
                "smd_decimal",
            ),
            "123.45",
        ),
        (
            (
                "representativeness",
                "antigen_vs_training",
                "features",
                0,
                "tv_exact",
            ),
            [1, 1],
        ),
        (("input_bindings", "implementation_source_sha256"), "0" * 64),
    ],
)
def test_independent_fold_auditor_rejects_statistic_tampering(
    path: tuple[object, ...],
    value: object,
) -> None:
    authority, manifest, ledgers, counts = _fixture()
    bundle = build_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_counts=counts,
    )
    pair_ids = tuple(pair.pair_id for pair in bundle.pairs)
    audit = audit_score_free_fold(
        fold_id="outer_0",
        pairs=tuple(pair.score_free_pair() for pair in bundle.pairs),
        antibody_profiles=tuple(antibody_profile(row) for row in bundle.antibodies),
        antigen_profiles=tuple(antigen_profile(row) for row in bundle.antigens),
        validation_pair_ids=pair_ids[:2],
        training_pair_ids=pair_ids[2:],
        shape_thresholds=OUTER_FOLD_SHAPE_THRESHOLDS,
        distribution_thresholds=OUTER_DISTRIBUTION_THRESHOLDS,
    )
    payload = audit.canonical_payload()
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    authority_payload = reconstruct_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_primary_implementation_sha256=_primary_implementation_sha256(),
        expected_counts={
            "source_rows": counts.source_rows,
            "exact_pairs": counts.exact_pairs,
            "antibody_entities": counts.antibody_entities,
            "antigen_entities": counts.antigen_entities,
        },
    )
    with pytest.raises(IndependentScoreFreeAuditError, match="disagreement"):
        audit_score_free_fold_candidate(
            candidate_fold_bytes=canonical_json_bytes(payload),
            authority_payload=authority_payload,
            scope="outer",
            expected_primary_source_sha256=primary_statistics_source_sha256(),
        )


def test_independent_inner_fold_auditor_requires_outer_training_anchor() -> None:
    authority, manifest, ledgers, counts = _fixture()
    bundle = build_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_counts=counts,
    )
    complete_pairs = tuple(pair.score_free_pair() for pair in bundle.pairs)
    outer_training_ids = tuple(pair.pair_id for pair in complete_pairs[:5])
    inner_pairs = complete_pairs[:5]
    antibody_ids = {pair.antibody_id for pair in inner_pairs}
    antigen_ids = {pair.antigen_id for pair in inner_pairs}
    audit = audit_inner_score_free_fold(
        outer_fold_id="outer_0",
        inner_fold_id="outer_0_inner_0",
        complete_authority_pairs=complete_pairs,
        outer_training_pair_ids=outer_training_ids,
        inner_universe_pairs=inner_pairs,
        antibody_profiles=tuple(
            antibody_profile(row)
            for row in bundle.antibodies
            if row.entity_id in antibody_ids
        ),
        antigen_profiles=tuple(
            antigen_profile(row)
            for row in bundle.antigens
            if row.entity_id in antigen_ids
        ),
        validation_pair_ids=outer_training_ids[:2],
        training_pair_ids=outer_training_ids[2:],
        shape_thresholds=INNER_FOLD_SHAPE_THRESHOLDS,
        distribution_thresholds=INNER_DISTRIBUTION_THRESHOLDS,
    )
    authority_payload = reconstruct_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_primary_implementation_sha256=_primary_implementation_sha256(),
        expected_counts={
            "source_rows": counts.source_rows,
            "exact_pairs": counts.exact_pairs,
            "antibody_entities": counts.antibody_entities,
            "antigen_entities": counts.antigen_entities,
        },
    )
    receipt = audit_score_free_fold_candidate(
        candidate_fold_bytes=canonical_json_bytes(audit.canonical_payload()),
        authority_payload=authority_payload,
        scope="inner",
        expected_primary_source_sha256=primary_statistics_source_sha256(),
        expected_outer_fold_id="outer_0",
        expected_outer_training_pair_ids=outer_training_ids,
    )
    assert receipt["field_by_field_agreement"] is True
    with pytest.raises(IndependentScoreFreeAuditError, match="differs"):
        audit_score_free_fold_candidate(
            candidate_fold_bytes=canonical_json_bytes(audit.canonical_payload()),
            authority_payload=authority_payload,
            scope="inner",
            expected_primary_source_sha256=primary_statistics_source_sha256(),
            expected_outer_fold_id="outer_0",
            expected_outer_training_pair_ids=outer_training_ids[:-1],
        )


def test_small_exhaustive_assignments_match_both_statistics_implementations() -> None:
    authority, manifest, ledgers, counts = _fixture()
    bundle = build_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_counts=counts,
    )
    complete_pairs = tuple(pair.score_free_pair() for pair in bundle.pairs)
    pairs = complete_pairs[:4]
    antibody_ids = {pair.antibody_id for pair in pairs}
    antigen_ids = {pair.antigen_id for pair in pairs}
    antibodies = tuple(
        antibody_profile(row)
        for row in bundle.antibodies
        if row.entity_id in antibody_ids
    )
    antigens = tuple(
        antigen_profile(row)
        for row in bundle.antigens
        if row.entity_id in antigen_ids
    )
    authority_payload = reconstruct_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_primary_implementation_sha256=_primary_implementation_sha256(),
        expected_counts={
            "source_rows": counts.source_rows,
            "exact_pairs": counts.exact_pairs,
            "antibody_entities": counts.antibody_entities,
            "antigen_entities": counts.antigen_entities,
        },
    )
    compared = 0
    for roles in itertools.product(("training", "validation", "quarantine"), repeat=4):
        validation = tuple(
            pair.pair_id
            for pair, role in zip(pairs, roles)
            if role == "validation"
        )
        training = tuple(
            pair.pair_id
            for pair, role in zip(pairs, roles)
            if role == "training"
        )
        if not validation or not training:
            continue
        primary = audit_score_free_fold(
            fold_id="exhaustive_outer",
            pairs=pairs,
            antibody_profiles=antibodies,
            antigen_profiles=antigens,
            validation_pair_ids=validation,
            training_pair_ids=training,
            shape_thresholds=OUTER_FOLD_SHAPE_THRESHOLDS,
            distribution_thresholds=OUTER_DISTRIBUTION_THRESHOLDS,
        )
        receipt = audit_score_free_fold_candidate(
            candidate_fold_bytes=canonical_json_bytes(primary.canonical_payload()),
            authority_payload=authority_payload,
            scope="outer",
            expected_primary_source_sha256=primary_statistics_source_sha256(),
        )
        assert receipt["field_by_field_agreement"] is True
        compared += 1
    assert compared == 50


def test_independent_cross_fold_and_pooled_auditors_match_primary() -> None:
    authority, manifest, ledgers, counts = _fixture()
    bundle = build_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_counts=counts,
    )
    pairs = tuple(pair.score_free_pair() for pair in bundle.pairs)
    pair_ids = tuple(pair.pair_id for pair in pairs)
    antibodies = tuple(antibody_profile(row) for row in bundle.antibodies)
    antigens = tuple(antigen_profile(row) for row in bundle.antigens)
    audits = []
    for index in range(3):
        validation = pair_ids[2 * index : 2 * index + 2]
        audits.append(
            audit_score_free_fold(
                fold_id=f"outer_{index}",
                pairs=pairs,
                antibody_profiles=antibodies,
                antigen_profiles=antigens,
                validation_pair_ids=validation,
                training_pair_ids=tuple(
                    pair_id for pair_id in pair_ids if pair_id not in validation
                ),
                shape_thresholds=OUTER_FOLD_SHAPE_THRESHOLDS,
                distribution_thresholds=OUTER_DISTRIBUTION_THRESHOLDS,
            )
        )
    cross = audit_cross_fold_balance(
        audits,
        thresholds=OUTER_CROSS_FOLD_THRESHOLDS,
    )
    pooled = audit_pooled_outer_representativeness(
        audits,
        antibody_profiles=antibodies,
        antigen_profiles=antigens,
    )
    fold_payloads = [audit.canonical_payload() for audit in audits]
    cross_receipt = audit_cross_fold_candidate(
        candidate_bytes=canonical_json_bytes(cross.canonical_payload()),
        fold_payloads=fold_payloads,
        scope="outer",
        expected_primary_source_sha256=primary_statistics_source_sha256(),
    )
    assert cross_receipt["field_by_field_agreement"] is True
    authority_payload = reconstruct_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_primary_implementation_sha256=_primary_implementation_sha256(),
        expected_counts={
            "source_rows": counts.source_rows,
            "exact_pairs": counts.exact_pairs,
            "antibody_entities": counts.antibody_entities,
            "antigen_entities": counts.antigen_entities,
        },
    )
    pooled_receipt = audit_pooled_outer_candidate(
        candidate_bytes=canonical_json_bytes(pooled.canonical_payload()),
        authority_payload=authority_payload,
        fold_payloads=fold_payloads,
        expected_primary_source_sha256=primary_statistics_source_sha256(),
    )
    assert pooled_receipt["field_by_field_agreement"] is True

    tampered = pooled.canonical_payload()
    tampered["pooled_antibody_ids"] = tampered["pooled_antibody_ids"][:-1]
    with pytest.raises(IndependentScoreFreeAuditError, match="disagreement"):
        audit_pooled_outer_candidate(
            candidate_bytes=canonical_json_bytes(tampered),
            authority_payload=authority_payload,
            fold_payloads=fold_payloads,
            expected_primary_source_sha256=primary_statistics_source_sha256(),
        )


def test_machine_receipt_and_markdown_are_create_only_and_consistent() -> None:
    authority, manifest, ledgers, counts, candidate = _primary_candidate()
    receipt = audit_score_free_authority_candidate(
        candidate_bytes=candidate,
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_primary_implementation_sha256=_primary_implementation_sha256(),
        expected_counts={
            "source_rows": counts.source_rows,
            "exact_pairs": counts.exact_pairs,
            "antibody_entities": counts.antibody_entities,
            "antigen_entities": counts.antigen_entities,
        },
    )
    markdown = authority_audit_markdown(receipt)
    assert receipt["candidate_file_sha256"] in markdown
    assert "Training authorized: `false`" in markdown
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        output = Path(directory) / "receipt"
        SCRIPT.create_only_report_directory(output, receipt)
        assert json.loads((output / "audit_receipt.json").read_text()) == receipt
        assert (output / "audit_report.md").read_text() == markdown
        assert stat_mode(output) == 0o500
        with pytest.raises(FileExistsError, match="already exists"):
            SCRIPT.create_only_report_directory(output, receipt)


def stat_mode(path: Path) -> int:
    """Return only the POSIX permission bits for one test path."""
    return os.stat(path).st_mode & 0o777
