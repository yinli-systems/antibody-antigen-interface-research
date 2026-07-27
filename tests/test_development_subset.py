from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from interaction_first.development_authority import (
    AuthorityCounts,
    DevelopmentPolicy,
    canonical_json_bytes,
    membership_sha256,
)
from interaction_first.development_subset import (
    AUTHORIZATION_FORMAT,
    DATA_ACCESS_SCOPE,
    AuthorizedInstance,
    DevelopmentSubsetError,
    REQUIRED_IMPLEMENTATION_SOURCES,
    TrainAuthorizationInventory,
    build_development_subsets,
    build_subset_receipt,
    parse_train_authorization,
)


def _instances(count: int = 20) -> tuple[AuthorizedInstance, ...]:
    return tuple(
        AuthorizedInstance(
            instance_id=f"instance-{index:03d}",
            antibody_id=f"antibody-{index // 2:03d}",
            antigen_id=f"antigen-{index // 3:03d}",
        )
        for index in range(count)
    )


def _inventory(count: int = 20) -> TrainAuthorizationInventory:
    instances = _instances(count)
    positive = sorted(
        {(value.antibody_id, value.antigen_id) for value in instances}
    )
    return TrainAuthorizationInventory(
        fold_id="authority-train",
        instances=instances,
        positive_pairs=tuple(positive),
        excluded_pairs=(),
    )


def _policy(inventory: TrainAuthorizationInventory) -> DevelopmentPolicy:
    return DevelopmentPolicy(
        protocol_sha256="1" * 64,
        authority_manifest_sha256="2" * 64,
        cache_authorization_manifest_sha256="3" * 64,
        cache_manifest_sha256="4" * 64,
        authority_membership_sha256=membership_sha256(inventory.instance_ids),
        counts=AuthorityCounts(
            source_instances=len(inventory.instances),
            exact_pairs=len(inventory.instances),
            antibody_entities=len({value.antibody_id for value in inventory.instances}),
            antigen_entities=len({value.antigen_id for value in inventory.instances}),
        ),
    )


def _build():
    inventory = _inventory()
    policy = _policy(inventory)
    return build_development_subsets(
        inventory=inventory,
        parent_authorization_sha256="3" * 64,
        policy=policy,
        expected_policy_sha256=policy.policy_sha256,
        sizes=(4, 8, 12),
        seed=17,
        requested_at_utc="2026-07-27T17:00:00Z",
        evaluated_at_utc="2026-07-27T17:01:00Z",
    )


def test_nested_selection_is_deterministic_and_authorized() -> None:
    first = _build()
    second = _build()
    assert [value.subset_sha256 for value in first] == [
        value.subset_sha256 for value in second
    ]
    assert [value.requested_size for value in first] == [4, 8, 12]
    for subset in first:
        assert subset.decision.authorized
        assert len(subset.instances) == subset.requested_size
        assert subset.authorization_payload["split_role"] == "train"
    assert {value.instance_id for value in first[0].instances}.issubset(
        value.instance_id for value in first[1].instances
    )
    assert {value.instance_id for value in first[1].instances}.issubset(
        value.instance_id for value in first[2].instances
    )


def test_pair_filter_includes_only_selected_endpoint_universe() -> None:
    subsets = _build()
    for subset in subsets:
        antibodies = {value.antibody_id for value in subset.instances}
        antigens = {value.antigen_id for value in subset.instances}
        assert all(
            antibody in antibodies and antigen in antigens
            for antibody, antigen in subset.positive_pairs
        )
        observed = {
            (value.antibody_id, value.antigen_id) for value in subset.instances
        }
        assert observed.issubset(subset.positive_pairs)


def test_parser_round_trip_and_forbidden_field_injection() -> None:
    inventory = _inventory()
    payload = {
        "format_version": AUTHORIZATION_FORMAT,
        "data_access_scope": DATA_ACCESS_SCOPE,
        "split_role": "train",
        "fold_id": inventory.fold_id,
        "instances": [value.canonical_payload() for value in inventory.instances],
        "positive_pairs": [list(value) for value in inventory.positive_pairs],
        "excluded_pairs": [],
    }
    assert parse_train_authorization(canonical_json_bytes(payload)) == inventory
    payload["contact_density"] = 0.25
    with pytest.raises(DevelopmentSubsetError, match="extra"):
        parse_train_authorization(canonical_json_bytes(payload))


def test_parser_rejects_validation_duplicate_unknown_endpoint_and_order() -> None:
    inventory = _inventory()
    base = {
        "format_version": AUTHORIZATION_FORMAT,
        "data_access_scope": DATA_ACCESS_SCOPE,
        "split_role": "train",
        "fold_id": inventory.fold_id,
        "instances": [value.canonical_payload() for value in inventory.instances],
        "positive_pairs": [list(value) for value in inventory.positive_pairs],
        "excluded_pairs": [],
    }
    changed = dict(base, split_role="validation")
    with pytest.raises(DevelopmentSubsetError, match="not train"):
        parse_train_authorization(canonical_json_bytes(changed))
    changed = dict(base, instances=[*base["instances"], base["instances"][0]])
    with pytest.raises(DevelopmentSubsetError, match="duplicate"):
        parse_train_authorization(canonical_json_bytes(changed))
    changed = dict(base, positive_pairs=[["antibody-000", "unknown"]])
    with pytest.raises(DevelopmentSubsetError, match="endpoints"):
        parse_train_authorization(canonical_json_bytes(changed))
    changed = dict(base, instances=list(reversed(base["instances"])))
    with pytest.raises(DevelopmentSubsetError, match="sorted"):
        parse_train_authorization(canonical_json_bytes(changed))


def test_wrong_parent_policy_count_membership_and_sizes_fail_closed() -> None:
    inventory = _inventory()
    policy = _policy(inventory)
    common = dict(
        inventory=inventory,
        parent_authorization_sha256="3" * 64,
        policy=policy,
        expected_policy_sha256=policy.policy_sha256,
        sizes=(4,),
        seed=17,
        requested_at_utc="2026-07-27T17:00:00Z",
        evaluated_at_utc="2026-07-27T17:01:00Z",
    )
    with pytest.raises(DevelopmentSubsetError, match="frozen policy"):
        build_development_subsets(
            **dict(common, parent_authorization_sha256="9" * 64)
        )
    with pytest.raises(DevelopmentSubsetError, match="exceeds"):
        build_development_subsets(**dict(common, sizes=(100,)))
    with pytest.raises(DevelopmentSubsetError, match="duplicates"):
        build_development_subsets(**dict(common, sizes=(4, 4)))
    changed_policy = DevelopmentPolicy(
        protocol_sha256=policy.protocol_sha256,
        authority_manifest_sha256=policy.authority_manifest_sha256,
        cache_authorization_manifest_sha256=policy.cache_authorization_manifest_sha256,
        cache_manifest_sha256=policy.cache_manifest_sha256,
        authority_membership_sha256="f" * 64,
        counts=policy.counts,
    )
    with pytest.raises(DevelopmentSubsetError, match="membership differs"):
        build_development_subsets(
            **dict(
                common,
                policy=changed_policy,
                expected_policy_sha256=changed_policy.policy_sha256,
            )
        )


def test_receipt_binds_every_subset_and_denies_confirmatory_authority() -> None:
    subsets = _build()
    receipt = build_subset_receipt(
        subsets,
        implementation_source_sha256={
            name: str(index + 1) * 64
            for index, name in enumerate(sorted(REQUIRED_IMPLEMENTATION_SOURCES))
        },
    )
    assert receipt["confirmatory_authority_granted"] is False
    assert receipt["validation_or_test_data_accessed"] is False
    assert receipt["score_or_label_fields_accessed"] is False
    assert [value["requested_size"] for value in receipt["subsets"]] == [4, 8, 12]
    digest_payload = dict(receipt)
    claimed = digest_payload.pop("receipt_sha256")
    from interaction_first.development_authority import canonical_sha256

    assert claimed == canonical_sha256(digest_payload)

    with pytest.raises(DevelopmentSubsetError, match="keys differ"):
        build_subset_receipt(
            subsets,
            implementation_source_sha256={"builder": "a" * 64},
        )


def test_cli_create_only_publication(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = repository_root / "scripts" / "build_pairscale_development_subsets.py"
    inventory = _inventory()
    authority_path = tmp_path / "AUTHORIZED_FOLD_INVENTORY.json"
    policy_path = tmp_path / "policy.json"
    output_root = tmp_path / "published"
    authority_payload = {
        "format_version": AUTHORIZATION_FORMAT,
        "data_access_scope": DATA_ACCESS_SCOPE,
        "split_role": "train",
        "fold_id": inventory.fold_id,
        "instances": [value.canonical_payload() for value in inventory.instances],
        "positive_pairs": [list(value) for value in inventory.positive_pairs],
        "excluded_pairs": [],
    }
    authority_path.write_bytes(canonical_json_bytes(authority_payload))
    authority_sha256 = hashlib.sha256(authority_path.read_bytes()).hexdigest()
    base_policy = _policy(inventory)
    policy = DevelopmentPolicy(
        protocol_sha256=base_policy.protocol_sha256,
        authority_manifest_sha256=base_policy.authority_manifest_sha256,
        cache_authorization_manifest_sha256=authority_sha256,
        cache_manifest_sha256=base_policy.cache_manifest_sha256,
        authority_membership_sha256=base_policy.authority_membership_sha256,
        counts=base_policy.counts,
    )
    policy_path.write_bytes(canonical_json_bytes(policy.canonical_payload()))
    command = [
        sys.executable,
        str(script),
        "--authority-inventory",
        str(authority_path),
        "--expected-authority-sha256",
        authority_sha256,
        "--policy",
        str(policy_path),
        "--expected-policy-sha256",
        policy.policy_sha256,
        "--expected-builder-sha256",
        hashlib.sha256(script.read_bytes()).hexdigest(),
        "--output-root",
        str(output_root),
        "--sizes",
        "4",
        "8",
        "--seed",
        "17",
        "--requested-at-utc",
        "2026-07-27T17:00:00Z",
        "--evaluated-at-utc",
        "2026-07-27T17:01:00Z",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "receipt_sha256" in completed.stdout
    assert (output_root / "n4" / "AUTHORIZED_FOLD_INVENTORY.json").is_file()
    assert (output_root / "n8" / "DEVELOPMENT_AUTHORITY_DECISION.json").is_file()
    assert (output_root / "SUBSET_BUILD_RECEIPT.json").is_file()
    failed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "create-only" in failed.stderr


def test_canonical_32_64_256_outputs_are_cross_hash_seed_identical(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = repository_root / "scripts" / "build_pairscale_development_subsets.py"
    inventory = _inventory(300)
    authority_path = tmp_path / "AUTHORIZED_FOLD_INVENTORY.json"
    policy_path = tmp_path / "policy.json"
    authority_payload = {
        "format_version": AUTHORIZATION_FORMAT,
        "data_access_scope": DATA_ACCESS_SCOPE,
        "split_role": "train",
        "fold_id": inventory.fold_id,
        "instances": [value.canonical_payload() for value in inventory.instances],
        "positive_pairs": [list(value) for value in inventory.positive_pairs],
        "excluded_pairs": [],
    }
    authority_path.write_bytes(canonical_json_bytes(authority_payload))
    authority_sha256 = hashlib.sha256(authority_path.read_bytes()).hexdigest()
    base_policy = _policy(inventory)
    policy = DevelopmentPolicy(
        protocol_sha256=base_policy.protocol_sha256,
        authority_manifest_sha256=base_policy.authority_manifest_sha256,
        cache_authorization_manifest_sha256=authority_sha256,
        cache_manifest_sha256=base_policy.cache_manifest_sha256,
        authority_membership_sha256=base_policy.authority_membership_sha256,
        counts=base_policy.counts,
    )
    policy_path.write_bytes(canonical_json_bytes(policy.canonical_payload()))
    output_roots: list[Path] = []
    for hash_seed in ("0", "811"):
        output_root = tmp_path / f"published-{hash_seed}"
        output_roots.append(output_root)
        command = [
            sys.executable,
            str(script),
            "--authority-inventory",
            str(authority_path),
            "--expected-authority-sha256",
            authority_sha256,
            "--policy",
            str(policy_path),
            "--expected-policy-sha256",
            policy.policy_sha256,
            "--expected-builder-sha256",
            hashlib.sha256(script.read_bytes()).hexdigest(),
            "--output-root",
            str(output_root),
            "--sizes",
            "32",
            "64",
            "256",
            "--seed",
            "17",
            "--requested-at-utc",
            "2026-07-27T17:00:00Z",
            "--evaluated-at-utc",
            "2026-07-27T17:01:00Z",
        ]
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = hash_seed
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    observed = []
    for output_root in output_roots:
        observed.append(
            {
                str(path.relative_to(output_root)): path.read_bytes()
                for path in sorted(output_root.rglob("*"))
                if path.is_file()
            }
        )
    assert observed[0] == observed[1]
    receipt = json.loads(
        observed[0]["SUBSET_BUILD_RECEIPT.json"].decode("ascii")
    )
    assert [value["requested_size"] for value in receipt["subsets"]] == [
        32,
        64,
        256,
    ]
