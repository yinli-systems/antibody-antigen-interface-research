"""Tests for the score-free component-disjoint development holdout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from interaction_first.development_authority import CLAIM_BOUNDARY, canonical_json_bytes
from interaction_first.development_holdout import (
    DevelopmentHoldoutError,
    REQUIRED_IMPLEMENTATION_SOURCES,
    ScoreFreeSubset,
    build_component_holdout,
    build_holdout_receipt,
    connected_components,
    parse_component_holdout_assignment,
    parse_score_free_subset,
    select_exact_holdout_components,
)
from interaction_first.development_subset import AuthorizedInstance


def _instances() -> tuple[AuthorizedInstance, ...]:
    rows = [
        ("i00", "a0", "g0"),
        ("i01", "a1", "g0"),
        ("i02", "a2", "g2"),
        ("i03", "a3", "g3"),
        ("i04", "a3", "g4"),
        ("i05", "a5", "g5"),
        ("i06", "a6", "g6"),
        ("i07", "a7", "g7"),
    ]
    return tuple(AuthorizedInstance(*row) for row in rows)


def _source_payload() -> dict[str, object]:
    instances = _instances()
    from interaction_first.development_authority import membership_sha256

    return {
        "format_version": "pairscale_development_subset_v1",
        "permission_name": "development_only_nonconfirmatory",
        "purpose": "overfit_sanity",
        "claim_boundary": CLAIM_BOUNDARY,
        "source_split": "train",
        "parent_authorization_sha256": "1" * 64,
        "policy_sha256": "2" * 64,
        "selection": {"algorithm": "fixture"},
        "authorization_manifest_sha256": "3" * 64,
        "instance_count": len(instances),
        "instance_membership_sha256": membership_sha256(
            value.instance_id for value in instances
        ),
        "antibody_count": len({value.antibody_id for value in instances}),
        "antigen_count": len({value.antigen_id for value in instances}),
        "positive_pair_count": len(instances),
        "excluded_pair_count": 0,
        "instances": [value.canonical_payload() for value in instances],
    }


def _subset() -> ScoreFreeSubset:
    payload = canonical_json_bytes(_source_payload())
    return parse_score_free_subset(
        payload,
        expected_subset_sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_component_holdout_is_exact_deterministic_and_endpoint_disjoint() -> None:
    first = build_component_holdout(_subset(), holdout_pair_count=3, seed=29)
    second = build_component_holdout(_subset(), holdout_pair_count=3, seed=29)
    assert first == second
    assert first["train"]["instance_count"] == 5
    assert first["holdout"]["instance_count"] == 3
    assert first["separation_audit"] == {
        "instance_overlap_count": 0,
        "antibody_overlap_count": 0,
        "antigen_overlap_count": 0,
        "contact_or_score_fields_accessed": False,
        "validation_or_test_data_accessed": False,
    }
    train = first["train"]["instances"]
    holdout = first["holdout"]["instances"]
    assert not {value["antibody_id"] for value in train}.intersection(
        value["antibody_id"] for value in holdout
    )
    assert not {value["antigen_id"] for value in train}.intersection(
        value["antigen_id"] for value in holdout
    )


def test_component_builder_and_exact_subset_sum_fail_closed() -> None:
    components = connected_components(_instances())
    assert sorted(value.pair_count for value in components) == [1, 1, 1, 1, 2, 2]
    with pytest.raises(DevelopmentHoldoutError, match="positive"):
        select_exact_holdout_components(
            components,
            holdout_pair_count=0,
            seed=29,
        )
    with pytest.raises(DevelopmentHoldoutError, match="smaller"):
        select_exact_holdout_components(
            components,
            holdout_pair_count=8,
            seed=29,
        )
    only_twos = tuple(value for value in components if value.pair_count == 2)
    with pytest.raises(DevelopmentHoldoutError, match="cannot realize"):
        select_exact_holdout_components(
            only_twos,
            holdout_pair_count=1,
            seed=29,
        )


def test_parser_rejects_tampering_forbidden_fields_and_non_train_scope() -> None:
    payload = _source_payload()
    encoded = canonical_json_bytes(payload)
    with pytest.raises(DevelopmentHoldoutError, match="SHA-256 mismatch"):
        parse_score_free_subset(encoded, expected_subset_sha256="f" * 64)
    payload["contact_density"] = 0.1
    encoded = canonical_json_bytes(payload)
    with pytest.raises(DevelopmentHoldoutError, match="forbidden"):
        parse_score_free_subset(
            encoded,
            expected_subset_sha256=hashlib.sha256(encoded).hexdigest(),
        )
    payload = _source_payload()
    payload["source_split"] = "validation"
    encoded = canonical_json_bytes(payload)
    with pytest.raises(DevelopmentHoldoutError, match="not train-only"):
        parse_score_free_subset(
            encoded,
            expected_subset_sha256=hashlib.sha256(encoded).hexdigest(),
        )


def test_receipt_binds_assignment_and_complete_source_inventory() -> None:
    assignment = build_component_holdout(_subset(), holdout_pair_count=3, seed=29)
    sources = {
        name: str(index + 1) * 64
        for index, name in enumerate(sorted(REQUIRED_IMPLEMENTATION_SOURCES))
    }
    receipt = build_holdout_receipt(
        assignment,
        implementation_source_sha256=sources,
    )
    assert receipt["assignment_content_sha256"] == assignment[
        "assignment_content_sha256"
    ]
    assert receipt["assignment_file_sha256"] == hashlib.sha256(
        canonical_json_bytes(assignment)
    ).hexdigest()
    assert receipt["assignment_file_sha256"] != receipt[
        "assignment_content_sha256"
    ]
    assert receipt["confirmatory_authority_granted"] is False
    assert receipt["validation_or_test_data_accessed"] is False
    assert receipt["score_or_label_fields_accessed"] is False
    with pytest.raises(DevelopmentHoldoutError, match="keys differ"):
        build_holdout_receipt(
            assignment,
            implementation_source_sha256={"partial": "a" * 64},
        )


def test_published_assignment_and_receipt_parse_with_all_anchors() -> None:
    assignment = build_component_holdout(_subset(), holdout_pair_count=3, seed=29)
    sources = {
        name: str(index + 1) * 64
        for index, name in enumerate(sorted(REQUIRED_IMPLEMENTATION_SOURCES))
    }
    receipt = build_holdout_receipt(
        assignment,
        implementation_source_sha256=sources,
    )
    assignment_bytes = canonical_json_bytes(assignment)
    receipt_bytes = canonical_json_bytes(receipt)
    parsed = parse_component_holdout_assignment(
        assignment_bytes,
        receipt_bytes,
        expected_assignment_file_sha256=hashlib.sha256(
            assignment_bytes
        ).hexdigest(),
        expected_assignment_content_sha256=assignment[
            "assignment_content_sha256"
        ],
        expected_receipt_file_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        expected_source_subset_sha256=_subset().subset_sha256,
        expected_policy_sha256=_subset().policy_sha256,
    )
    assert len(parsed.train_instances) == 5
    assert len(parsed.holdout_instances) == 3
    assert not {
        value.antibody_id for value in parsed.train_instances
    }.intersection(value.antibody_id for value in parsed.holdout_instances)


def test_assignment_parser_rejects_receipt_and_partition_tampering() -> None:
    assignment = build_component_holdout(_subset(), holdout_pair_count=3, seed=29)
    sources = {
        name: str(index + 1) * 64
        for index, name in enumerate(sorted(REQUIRED_IMPLEMENTATION_SOURCES))
    }
    receipt = build_holdout_receipt(
        assignment,
        implementation_source_sha256=sources,
    )

    def parse(changed_assignment: object, changed_receipt: object) -> None:
        assignment_bytes = canonical_json_bytes(changed_assignment)
        receipt_bytes = canonical_json_bytes(changed_receipt)
        parse_component_holdout_assignment(
            assignment_bytes,
            receipt_bytes,
            expected_assignment_file_sha256=hashlib.sha256(
                assignment_bytes
            ).hexdigest(),
            expected_assignment_content_sha256=assignment[
                "assignment_content_sha256"
            ],
            expected_receipt_file_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
            expected_source_subset_sha256=_subset().subset_sha256,
            expected_policy_sha256=_subset().policy_sha256,
        )

    changed_assignment = json.loads(canonical_json_bytes(assignment))
    changed_assignment["train"]["instances"][0]["antibody_id"] = (
        changed_assignment["holdout"]["instances"][0]["antibody_id"]
    )
    with pytest.raises(DevelopmentHoldoutError, match="content digest mismatch"):
        parse(changed_assignment, receipt)
    changed_receipt = json.loads(canonical_json_bytes(receipt))
    changed_receipt["confirmatory_authority_granted"] = True
    with pytest.raises(DevelopmentHoldoutError, match="forbidden authority"):
        parse(assignment, changed_receipt)


def test_strict_parser_rejects_duplicate_json_keys() -> None:
    raw = canonical_json_bytes(_source_payload()).rstrip(b"\n")
    duplicate = raw[:-1] + b',"source_split":"train"}'
    with pytest.raises(DevelopmentHoldoutError, match="duplicate JSON key"):
        parse_score_free_subset(
            duplicate,
            expected_subset_sha256=hashlib.sha256(duplicate).hexdigest(),
        )


def test_cli_publication_is_create_only(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = repository_root / "scripts" / "build_pairscale_development_holdout.py"
    subset_path = tmp_path / "PAIRSCALE_DEVELOPMENT_SUBSET.json"
    output_root = tmp_path / "published"
    subset_path.write_bytes(canonical_json_bytes(_source_payload()))
    command = [
        sys.executable,
        str(script),
        "--subset",
        str(subset_path),
        "--expected-subset-sha256",
        hashlib.sha256(subset_path.read_bytes()).hexdigest(),
        "--expected-builder-sha256",
        hashlib.sha256(script.read_bytes()).hexdigest(),
        "--output-root",
        str(output_root),
        "--holdout-pair-count",
        "3",
        "--seed",
        "29",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    summary = json.loads(completed.stdout)
    assert summary["train_instance_count"] == 5
    assert summary["holdout_instance_count"] == 3
    assert (output_root / "DEVELOPMENT_COMPONENT_HOLDOUT.json").is_file()
    assert (output_root / "HOLDOUT_BUILD_RECEIPT.json").is_file()
    failed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "create-only" in failed.stderr
