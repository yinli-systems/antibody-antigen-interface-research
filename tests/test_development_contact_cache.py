from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from interaction_first.contacts import SparseContactMap, save_contact_map
from interaction_first.development_authority import (
    AuthorityCounts,
    DevelopmentPolicy,
    canonical_json_bytes,
    membership_sha256,
)
from interaction_first.development_contact_cache import (
    DevelopmentContactCacheError,
    materialize_development_contact_cache,
)
from interaction_first.development_contact_dataset import (
    DevelopmentContactDataset,
    DevelopmentContactDatasetError,
)
from interaction_first.development_subset import (
    AUTHORIZATION_FORMAT,
    DATA_ACCESS_SCOPE,
    AuthorizedInstance,
    TrainAuthorizationInventory,
    build_development_subsets,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        "scripts/materialize_pairscale_development_contact_cache.py": "1" * 64,
        "src/interaction_first/contacts.py": "2" * 64,
        "src/interaction_first/development_contact_cache.py": "3" * 64,
    }


def _fixture(tmp_path: Path):
    instances = tuple(
        AuthorizedInstance(
            instance_id=f"instance-{index:03d}",
            antibody_id=f"antibody-{index:03d}",
            antigen_id=f"antigen-{index:03d}",
        )
        for index in range(4)
    )
    inventory = TrainAuthorizationInventory(
        fold_id="authority-train",
        instances=instances,
        positive_pairs=tuple(
            (value.antibody_id, value.antigen_id) for value in instances
        ),
        excluded_pairs=(),
    )
    parent_root = tmp_path / "parent"
    examples = parent_root / "examples"
    examples.mkdir(parents=True)
    parent_records = []
    for index, value in enumerate(instances):
        payload = f"authenticated-tensor-{index}".encode("ascii")
        artifact = examples / f"artifact-{index:03d}.pt"
        artifact.write_bytes(payload)
        parent_records.append(
            {
                **value.canonical_payload(),
                "artifact_path": f"examples/{artifact.name}",
                "artifact_sha256": _sha256(payload),
                "artifact_size_bytes": len(payload),
                "antibody_length": 3,
                "antigen_length": 4,
                "contact_supervision": False,
            }
        )
    parent_manifest = {
        "format_version": "frontier_fold_local_cache_manifest_v1",
        "data_access_scope": "train_only_fold_local",
        "split_role": "train",
        "fold_id": "authority-train",
        "whole_pool_materialized": False,
        "authorization_manifest_sha256": "a" * 64,
        "antibody_input_dim": 5,
        "antigen_input_dim": 7,
        "instances": parent_records,
        "positive_pairs": [list(value) for value in inventory.positive_pairs],
        "excluded_pairs": [],
    }
    parent_manifest_bytes = canonical_json_bytes(parent_manifest)
    (parent_root / "FOLD_LOCAL_CACHE_MANIFEST.json").write_bytes(
        parent_manifest_bytes
    )
    policy = DevelopmentPolicy(
        protocol_sha256="4" * 64,
        authority_manifest_sha256="5" * 64,
        cache_authorization_manifest_sha256="6" * 64,
        cache_manifest_sha256=_sha256(parent_manifest_bytes),
        authority_membership_sha256=membership_sha256(inventory.instance_ids),
        counts=AuthorityCounts(
            source_instances=4,
            exact_pairs=4,
            antibody_entities=4,
            antigen_entities=4,
        ),
    )
    subsets = build_development_subsets(
        inventory=inventory,
        parent_authorization_sha256="6" * 64,
        policy=policy,
        expected_policy_sha256=policy.policy_sha256,
        sizes=(2,),
        seed=17,
        requested_at_utc="2026-07-27T13:00:00Z",
        evaluated_at_utc="2026-07-27T13:01:00Z",
    )
    subset = subsets[0]
    authorization_bytes = canonical_json_bytes(subset.authorization_payload)
    subset_bytes = canonical_json_bytes(subset.subset_payload)
    decision_bytes = canonical_json_bytes(subset.decision.canonical_payload())
    contact_root = tmp_path / "contacts"
    contact_root.mkdir()
    for value in subset.instances:
        contact = SparseContactMap.from_rows(
            instance_id=value.instance_id,
            antibody_length=3,
            antigen_length=4,
            antibody_indices=[0, 0, 1, 1],
            antigen_indices=[0, 1, 0, 1],
            labels=[1, 0, 0, 1],
        )
        save_contact_map(contact, contact_root / f"{value.instance_id}.npz")
    return {
        "parent_cache_root": parent_root,
        "parent_cache_manifest_sha256": _sha256(parent_manifest_bytes),
        "subset_authorization_bytes": authorization_bytes,
        "subset_authorization_sha256": _sha256(authorization_bytes),
        "subset_manifest_bytes": subset_bytes,
        "subset_manifest_sha256": _sha256(subset_bytes),
        "decision_bytes": decision_bytes,
        "decision_sha256": subset.decision.decision_sha256,
        "expected_policy_sha256": policy.policy_sha256,
        "contact_root": contact_root,
        "contact_summary_sha256": "7" * 64,
        "output_root": tmp_path / "published",
        "implementation_source_sha256": _source_hashes(),
    }


def test_materializes_only_authorized_embeddings_and_contacts(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    receipt = materialize_development_contact_cache(**arguments)
    root = arguments["output_root"]
    manifest = json.loads((root / "PAIRSCALE_CONTACT_CACHE_MANIFEST.json").read_text())
    assert receipt["instance_count"] == 2
    assert receipt["candidate_pair_count"] == 8
    assert receipt["positive_pair_count"] == 4
    assert receipt["validation_or_test_data_accessed"] is False
    assert receipt["confirmatory_authority_granted"] is False
    assert manifest["whole_pool_materialized"] is False
    assert len(manifest["instances"]) == 2
    assert len(list((root / "embeddings").glob("*.pt"))) == 2
    assert len(list((root / "contacts").glob("*.npz"))) == 2
    assert all(not path.stat().st_mode & 0o222 for path in root.rglob("*"))


def test_training_dataset_verifies_all_external_cache_anchors(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    materialize_development_contact_cache(**arguments)
    root = arguments["output_root"]
    manifest_sha256 = _sha256(
        (root / "PAIRSCALE_CONTACT_CACHE_MANIFEST.json").read_bytes()
    )
    receipt_sha256 = _sha256((root / "CONTACT_CACHE_BUILD_RECEIPT.json").read_bytes())
    dataset = DevelopmentContactDataset(
        root,
        expected_manifest_sha256=manifest_sha256,
        expected_receipt_file_sha256=receipt_sha256,
        expected_policy_sha256=arguments["expected_policy_sha256"],
        expected_decision_sha256=arguments["decision_sha256"],
        expected_subset_manifest_sha256=arguments["subset_manifest_sha256"],
        expected_authorization_manifest_sha256=(
            arguments["subset_authorization_sha256"]
        ),
    )
    assert len(dataset) == 2
    assert dataset.antibody_input_dim == 5
    assert dataset.antigen_input_dim == 7
    assert sum(record.positive_pair_count for record in dataset.records) == 4

    with pytest.raises(DevelopmentContactDatasetError, match="external anchor"):
        DevelopmentContactDataset(
            root,
            expected_manifest_sha256="f" * 64,
            expected_receipt_file_sha256=receipt_sha256,
            expected_policy_sha256=arguments["expected_policy_sha256"],
            expected_decision_sha256=arguments["decision_sha256"],
            expected_subset_manifest_sha256=arguments["subset_manifest_sha256"],
            expected_authorization_manifest_sha256=(
                arguments["subset_authorization_sha256"]
            ),
        )


def test_create_only_and_all_external_digests_fail_closed(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    materialize_development_contact_cache(**arguments)
    with pytest.raises(DevelopmentContactCacheError, match="create-only"):
        materialize_development_contact_cache(**arguments)

    for field in (
        "parent_cache_manifest_sha256",
        "subset_authorization_sha256",
        "subset_manifest_sha256",
        "decision_sha256",
        "expected_policy_sha256",
    ):
        changed = _fixture(tmp_path / field)
        changed[field] = "f" * 64
        with pytest.raises(DevelopmentContactCacheError):
            materialize_development_contact_cache(**changed)


def test_unknown_subset_endpoint_and_embedding_tamper_fail(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    parent = arguments["parent_cache_root"]
    manifest_path = parent / "FOLD_LOCAL_CACHE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    selected = json.loads(arguments["subset_authorization_bytes"])
    selected["instances"][0]["antibody_id"] = "changed-antibody"
    arguments["subset_authorization_bytes"] = canonical_json_bytes(selected)
    arguments["subset_authorization_sha256"] = _sha256(
        arguments["subset_authorization_bytes"]
    )
    with pytest.raises(DevelopmentContactCacheError):
        materialize_development_contact_cache(**arguments)

    arguments = _fixture(tmp_path / "tamper")
    parent = arguments["parent_cache_root"]
    manifest = json.loads(
        (parent / "FOLD_LOCAL_CACHE_MANIFEST.json").read_text()
    )
    selected_ids = {
        value["instance_id"]
        for value in json.loads(arguments["subset_authorization_bytes"])["instances"]
    }
    record = next(
        value
        for value in manifest["instances"]
        if value["instance_id"] in selected_ids
    )
    (parent / record["artifact_path"]).write_bytes(b"tampered")
    with pytest.raises(DevelopmentContactCacheError, match="embedding"):
        materialize_development_contact_cache(**arguments)


def test_forbidden_decision_flag_and_contact_length_mismatch_fail(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    decision = json.loads(arguments["decision_bytes"])
    decision["test_data_access_authorized"] = True
    arguments["decision_bytes"] = canonical_json_bytes(decision)
    arguments["decision_sha256"] = hashlib.sha256(
        arguments["decision_bytes"]
    ).hexdigest()
    with pytest.raises(DevelopmentContactCacheError, match="forbidden"):
        materialize_development_contact_cache(**arguments)

    arguments = _fixture(tmp_path / "length")
    selected_id = json.loads(arguments["subset_authorization_bytes"])["instances"][0][
        "instance_id"
    ]
    replacement = SparseContactMap.from_rows(
        instance_id=selected_id,
        antibody_length=2,
        antigen_length=4,
        antibody_indices=[0, 1],
        antigen_indices=[0, 1],
        labels=[1, 0],
    )
    save_contact_map(
        replacement,
        arguments["contact_root"] / f"{selected_id}.npz",
    )
    with pytest.raises(DevelopmentContactCacheError, match="lengths differ"):
        materialize_development_contact_cache(**arguments)
