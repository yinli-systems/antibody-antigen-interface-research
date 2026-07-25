"""Tests for the strict fold-local real-data boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from frontier_test_cache import build_fold_local_cache, write_json  # noqa: E402
from interaction_first.frontier_data import (  # noqa: E402
    AUTHORIZATION_FILENAME,
    CACHE_MANIFEST_FILENAME,
    FoldLocalFrontierDataset,
    collective_verification_receipt_sha256,
    load_fold_local_cache_contract,
)


def test_exact_authorization_equal_cache_constructs_frontier_examples(
    tmp_path: Path,
) -> None:
    cache_root, authorization_path, _, _ = build_fold_local_cache(tmp_path)

    dataset = FoldLocalFrontierDataset(
        cache_root,
        authorization_path,
        memory_cache_size=2,
    )
    example = dataset[0]
    summary = dataset.verify_all_examples()

    assert len(dataset) == 3
    assert example.instance_id == "instance-000"
    assert example.antibody_embeddings.shape == (2, 7)
    assert example.antigen_embeddings.shape == (3, 5)
    assert example.antibody_chain_ids.tolist() == [0, 1]
    assert example.contact_target.shape == (3, 2)
    assert summary == {
        "instance_count": 3,
        "contact_supervised_instance_count": 2,
        "antibody_count": 3,
        "antigen_count": 3,
        "positive_pair_count": 4,
        "excluded_pair_count": 1,
    }
    assert len(dataset.contract.manifest_sha256) == 64
    assert len(dataset.contract.authorization_manifest_sha256) == 64


def test_cache_inventory_mismatch_fails_before_artifact_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, authorization_path, _, cache_manifest = build_fold_local_cache(tmp_path)
    cache_manifest["instances"][0]["antibody_id"] = "different-antibody"
    write_json(cache_root / CACHE_MANIFEST_FILENAME, cache_manifest)
    called = False

    def forbidden_load(path: Path) -> object:
        nonlocal called
        called = True
        raise AssertionError(f"Tensor artifact must not be opened: {path}")

    monkeypatch.setattr(
        "interaction_first.frontier_data._safe_torch_load",
        forbidden_load,
    )
    with pytest.raises(ValueError, match="authorization-equal"):
        FoldLocalFrontierDataset(cache_root, authorization_path)
    assert not called


def test_extra_file_symlink_and_whole_pool_claim_are_rejected(
    tmp_path: Path,
) -> None:
    extra_root = tmp_path / "extra"
    cache_root, authorization_path, _, _ = build_fold_local_cache(extra_root)
    (cache_root / "canonical_manifest.csv").write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file inventory"):
        load_fold_local_cache_contract(cache_root, authorization_path)

    symlink_root = tmp_path / "symlink"
    cache_root, authorization_path, _, _ = build_fold_local_cache(symlink_root)
    artifact = cache_root / "examples" / "artifact-000.pt"
    target = symlink_root / "target.pt"
    artifact.replace(target)
    artifact.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        load_fold_local_cache_contract(cache_root, authorization_path)

    whole_root = tmp_path / "whole"
    cache_root, authorization_path, _, cache_manifest = build_fold_local_cache(
        whole_root
    )
    cache_manifest["whole_pool_materialized"] = True
    write_json(cache_root / CACHE_MANIFEST_FILENAME, cache_manifest)
    with pytest.raises(ValueError, match="whole_pool_materialized=false"):
        load_fold_local_cache_contract(cache_root, authorization_path)


def test_authorization_must_use_locked_name_and_remain_outside_cache(
    tmp_path: Path,
) -> None:
    cache_root, authorization_path, authorization, cache_manifest = (
        build_fold_local_cache(tmp_path)
    )
    renamed = tmp_path / "canonical_manifest.json"
    authorization_path.replace(renamed)
    with pytest.raises(ValueError, match=AUTHORIZATION_FILENAME):
        FoldLocalFrontierDataset(cache_root, renamed)

    write_json(authorization_path, authorization)
    cache_manifest["authorization_manifest_sha256"] = hashlib.sha256(
        authorization_path.read_bytes()
    ).hexdigest()
    write_json(cache_root / CACHE_MANIFEST_FILENAME, cache_manifest)
    inside = cache_root / AUTHORIZATION_FILENAME
    write_json(inside, authorization)
    with pytest.raises(ValueError, match="outside"):
        FoldLocalFrontierDataset(cache_root, inside)


def test_artifact_tamper_is_rejected_before_tensor_loading(
    tmp_path: Path,
) -> None:
    cache_root, authorization_path, _, _ = build_fold_local_cache(tmp_path)
    dataset = FoldLocalFrontierDataset(cache_root, authorization_path)
    artifact = cache_root / "examples" / "artifact-000.pt"
    original = artifact.read_bytes()
    artifact.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))

    with pytest.raises(ValueError, match="changed after verification"):
        _ = dataset[0]


def test_payload_identity_dimension_and_contact_contracts_fail_closed(
    tmp_path: Path,
) -> None:
    cache_root, authorization_path, _, cache_manifest = build_fold_local_cache(tmp_path)
    artifact = cache_root / "examples" / "artifact-000.pt"
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    payload["antibody_id"] = "payload-mismatch"
    torch.save(payload, artifact)
    cache_manifest["instances"][0]["artifact_size_bytes"] = artifact.stat().st_size
    cache_manifest["instances"][0]["artifact_sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    write_json(cache_root / CACHE_MANIFEST_FILENAME, cache_manifest)

    dataset = FoldLocalFrontierDataset(cache_root, authorization_path)
    with pytest.raises(ValueError, match="mismatched antibody_id"):
        _ = dataset[0]


def test_collective_verification_receipt_authorizes_equal_metadata(
    tmp_path: Path,
) -> None:
    cache_root, authorization_path, _, _ = build_fold_local_cache(
        tmp_path,
        instance_count=3,
    )
    verifier = FoldLocalFrontierDataset(
        cache_root,
        authorization_path,
        verify_artifact_hashes=True,
        memory_cache_size=3,
    )
    follower = FoldLocalFrontierDataset(
        cache_root,
        authorization_path,
        verify_artifact_hashes=False,
        memory_cache_size=3,
    )
    verifier.verify_all_examples()
    receipt = verifier.collective_verification_receipt()
    receipt_sha256 = collective_verification_receipt_sha256(receipt)

    assert (
        follower.accept_collective_verification_receipt(
            receipt,
            expected_receipt_sha256=receipt_sha256,
        )
        == 3
    )
    assert follower._verified_artifact_indices == {0, 1, 2}
    assert receipt["producer_rank_observed"] == 0
    assert receipt["world_size_observed"] == 1
    assert verifier.clear_memory_cache() == 3
    assert not verifier._memory_cache


def test_collective_verification_rejects_receipt_drift(tmp_path: Path) -> None:
    cache_root, authorization_path, _, _ = build_fold_local_cache(
        tmp_path,
        instance_count=2,
    )
    verifier = FoldLocalFrontierDataset(cache_root, authorization_path)
    follower = FoldLocalFrontierDataset(
        cache_root,
        authorization_path,
        verify_artifact_hashes=False,
    )
    verifier.verify_all_examples()
    receipt = verifier.collective_verification_receipt()
    changed = {**receipt, "instance_count": 99}

    with pytest.raises(ValueError, match="differs from local metadata"):
        follower.accept_collective_verification_receipt(changed)


def test_external_manifest_anchors_reject_reissued_metadata(tmp_path: Path) -> None:
    cache_root, authorization_path, _, cache_manifest = build_fold_local_cache(
        tmp_path
    )
    authorization_digest = hashlib.sha256(authorization_path.read_bytes()).hexdigest()
    cache_manifest_path = cache_root / CACHE_MANIFEST_FILENAME
    cache_digest = hashlib.sha256(cache_manifest_path.read_bytes()).hexdigest()

    contract = load_fold_local_cache_contract(
        cache_root,
        authorization_path,
        expected_authorization_manifest_sha256=authorization_digest,
        expected_cache_manifest_sha256=cache_digest,
    )
    assert contract.authorization_manifest_sha256 == authorization_digest
    assert contract.manifest_sha256 == cache_digest
    assert contract.authorization_external_anchor_verified
    assert contract.cache_external_anchor_verified

    cache_manifest["fold_id"] = "fold-reissued"
    write_json(cache_manifest_path, cache_manifest)
    with pytest.raises(ValueError, match="external anchor"):
        load_fold_local_cache_contract(
            cache_root,
            authorization_path,
            expected_authorization_manifest_sha256=authorization_digest,
            expected_cache_manifest_sha256=cache_digest,
        )


def test_memory_cache_does_not_expose_mutable_tensor_state(tmp_path: Path) -> None:
    cache_root, authorization_path, _, _ = build_fold_local_cache(tmp_path)
    dataset = FoldLocalFrontierDataset(
        cache_root,
        authorization_path,
        memory_cache_size=1,
    )
    first = dataset[0]
    expected = first.antibody_embeddings.clone()
    first.antibody_embeddings.add_(1000.0)

    repeated = dataset[0]
    torch.testing.assert_close(repeated.antibody_embeddings, expected)
    assert repeated.antibody_embeddings.data_ptr() != first.antibody_embeddings.data_ptr()
