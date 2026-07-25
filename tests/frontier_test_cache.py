"""Synthetic fold-local cache construction for focused unit tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from interaction_first.frontier_data import (
    AUTHORIZATION_FILENAME,
    AUTHORIZATION_FORMAT,
    CACHE_MANIFEST_FILENAME,
    CACHE_MANIFEST_FORMAT,
    EXAMPLE_ARTIFACT_FORMAT,
    TRAIN_ONLY_SCOPE,
)


def _json_bytes(value: Any) -> bytes:
    """Serialize one deterministic test manifest."""
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    """Hash one small test artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fold_local_cache(
    root: Path,
    *,
    instance_count: int = 3,
    antibody_dim: int = 7,
    antigen_dim: int = 5,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    """Create a valid strict cache and return both mutable manifest objects."""
    if instance_count <= 0:
        raise ValueError("instance_count must be positive")
    cache_root = root / "fold_cache"
    examples_root = cache_root / "examples"
    examples_root.mkdir(parents=True)
    authorization_path = root / AUTHORIZATION_FILENAME
    identities = [
        {
            "instance_id": f"instance-{index:03d}",
            "antibody_id": f"antibody-{index:03d}",
            "antigen_id": f"antigen-{index:03d}",
        }
        for index in range(instance_count)
    ]
    positive_pairs = sorted(
        [[row["antibody_id"], row["antigen_id"]] for row in identities]
    )
    if instance_count >= 2:
        positive_pairs.append(
            [identities[0]["antibody_id"], identities[1]["antigen_id"]]
        )
        positive_pairs.sort()
    excluded_pairs = []
    if instance_count >= 3:
        excluded_pairs = [
            [identities[1]["antibody_id"], identities[2]["antigen_id"]]
        ]
    authorization = {
        "format_version": AUTHORIZATION_FORMAT,
        "data_access_scope": TRAIN_ONLY_SCOPE,
        "split_role": "train",
        "fold_id": "fold-0-train",
        "instances": identities,
        "positive_pairs": positive_pairs,
        "excluded_pairs": excluded_pairs,
    }
    authorization_path.write_bytes(_json_bytes(authorization))

    cache_records = []
    for index, identity in enumerate(identities):
        antibody_length = 2 + index % 2
        antigen_length = 3 + index % 2
        contact_target = None
        contact_mask = None
        if index % 2 == 0:
            contact_target = torch.zeros(
                antigen_length,
                antibody_length,
                dtype=torch.uint8,
            )
            contact_target[0, 0] = 1
            contact_mask = torch.ones_like(contact_target, dtype=torch.bool)
        artifact_path = examples_root / f"artifact-{index:03d}.pt"
        torch.save(
            {
                "format_version": EXAMPLE_ARTIFACT_FORMAT,
                **identity,
                "antibody_embeddings": torch.arange(
                    antibody_length * antibody_dim,
                    dtype=torch.float32,
                ).reshape(antibody_length, antibody_dim),
                "antigen_embeddings": torch.arange(
                    antigen_length * antigen_dim,
                    dtype=torch.float32,
                ).reshape(antigen_length, antigen_dim),
                "antibody_chain_ids": torch.tensor(
                    [0] + [1] * (antibody_length - 1),
                    dtype=torch.long,
                ),
                "contact_target": contact_target,
                "contact_mask": contact_mask,
            },
            artifact_path,
        )
        cache_records.append(
            {
                **identity,
                "artifact_path": f"examples/{artifact_path.name}",
                "artifact_sha256": _sha256_file(artifact_path),
                "artifact_size_bytes": artifact_path.stat().st_size,
                "antibody_length": antibody_length,
                "antigen_length": antigen_length,
                "contact_supervision": contact_target is not None,
            }
        )
    cache_manifest = {
        "format_version": CACHE_MANIFEST_FORMAT,
        "data_access_scope": TRAIN_ONLY_SCOPE,
        "split_role": "train",
        "fold_id": "fold-0-train",
        "whole_pool_materialized": False,
        "authorization_manifest_sha256": _sha256_file(authorization_path),
        "antibody_input_dim": antibody_dim,
        "antigen_input_dim": antigen_dim,
        "instances": cache_records,
        "positive_pairs": positive_pairs,
        "excluded_pairs": excluded_pairs,
    }
    (cache_root / CACHE_MANIFEST_FILENAME).write_bytes(_json_bytes(cache_manifest))
    return cache_root, authorization_path, authorization, cache_manifest


def write_json(path: Path, value: Any) -> None:
    """Rewrite a test manifest deterministically."""
    path.write_bytes(_json_bytes(value))
