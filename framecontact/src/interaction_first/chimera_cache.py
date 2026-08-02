"""Hash-bound adapter cache dataset for strict CHIMERA external scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .esm_embeddings import embedding_path, load_embedding, sequence_sha256
from .sabdab2_structure import sha256_file


CACHE_SCHEMA = "chimera_strict_adapter_cache_v1"
EXPECTED_KEYS = {
    "instance_id",
    "antibody_sequence",
    "antigen_sequence",
    "antibody_backbone",
    "antigen_backbone",
    "antibody_chain_ids",
    "antigen_chain_ids",
    "antibody_frame_valid_mask",
    "antigen_frame_valid_mask",
    "antibody_missing_frame_mask",
    "antigen_missing_frame_mask",
    "antibody_labels",
    "antigen_labels",
}


def load_cache_manifest(cache_root: Path, *, expected_sha256: str | None = None) -> dict:
    path = cache_root / "ADAPTER_CACHE_MANIFEST.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("adapter cache manifest is missing or unsafe")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("adapter cache manifest SHA-256 differs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unexpected adapter cache schema")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != int(payload.get("record_count", -1)):
        raise ValueError("adapter cache record count differs")
    identifiers = [record.get("instance_id") for record in records]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("adapter cache has an invalid instance ID")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("adapter cache has duplicate instance IDs")
    return payload


class ChimeraAdapterCacheDataset(Dataset):
    def __init__(
        self,
        cache_root: Path,
        embedding_root: Path,
        *,
        expected_manifest_sha256: str,
        instance_ids: Iterable[str] | None = None,
    ) -> None:
        self.cache_root = cache_root
        self.embedding_root = embedding_root
        manifest = load_cache_manifest(
            cache_root, expected_sha256=expected_manifest_sha256
        )
        selected = None if instance_ids is None else set(instance_ids)
        records = [
            record
            for record in manifest["records"]
            if selected is None or record["instance_id"] in selected
        ]
        if selected is not None and {record["instance_id"] for record in records} != selected:
            raise ValueError("requested cache instances are incomplete")
        if not records:
            raise ValueError("adapter cache selection is empty")
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self.cache_root / record["artifact_path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("adapter artifact is missing or unsafe")
        if sha256_file(path) != record["artifact_sha256"]:
            raise ValueError("adapter artifact SHA-256 differs")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != EXPECTED_KEYS:
                raise ValueError("adapter artifact schema differs")
            values = {name: archive[name].copy() for name in archive.files}
        instance_id = str(values["instance_id"].item())
        antibody_sequence = str(values["antibody_sequence"].item())
        antigen_sequence = str(values["antigen_sequence"].item())
        if instance_id != record["instance_id"]:
            raise ValueError("adapter artifact instance ID differs")
        ab_length, ag_length = len(antibody_sequence), len(antigen_sequence)
        for partner, length in (("antibody", ab_length), ("antigen", ag_length)):
            backbone = values[f"{partner}_backbone"]
            chain_ids = values[f"{partner}_chain_ids"]
            valid = values[f"{partner}_frame_valid_mask"]
            missing = values[f"{partner}_missing_frame_mask"]
            labels = values[f"{partner}_labels"]
            if backbone.shape != (length, 3, 3) or backbone.dtype != np.float32:
                raise ValueError("adapter backbone schema differs")
            if chain_ids.shape != (length,) or chain_ids.dtype != np.int64:
                raise ValueError("adapter chain ID schema differs")
            if valid.shape != (length,) or valid.dtype != np.bool_:
                raise ValueError("adapter valid-frame mask schema differs")
            if missing.shape != (length,) or missing.dtype != np.bool_:
                raise ValueError("adapter missing-frame mask schema differs")
            if labels.shape != (length,) or labels.dtype != np.float32:
                raise ValueError("adapter label schema differs")
            if not np.array_equal(valid | missing, np.ones(length, dtype=np.bool_)):
                raise ValueError("adapter frame masks do not partition the residue axis")
            if np.any(valid & missing) or not np.isin(labels, (0.0, 1.0)).all():
                raise ValueError("adapter masks or labels are invalid")
        ab_key = sequence_sha256(antibody_sequence)
        ag_key = sequence_sha256(antigen_sequence)
        return {
            "instance_id": instance_id,
            "antibody_features": load_embedding(
                embedding_path(self.embedding_root, ab_key),
                expected_key=ab_key,
                expected_length=ab_length,
            ),
            "antigen_features": load_embedding(
                embedding_path(self.embedding_root, ag_key),
                expected_key=ag_key,
                expected_length=ag_length,
            ),
            "antibody_backbone": torch.from_numpy(values["antibody_backbone"]),
            "antigen_backbone": torch.from_numpy(values["antigen_backbone"]),
            "antibody_chain_ids": torch.from_numpy(values["antibody_chain_ids"]),
            "antigen_chain_ids": torch.from_numpy(values["antigen_chain_ids"]),
            "antibody_frame_valid_mask": torch.from_numpy(
                values["antibody_frame_valid_mask"]
            ),
            "antigen_frame_valid_mask": torch.from_numpy(
                values["antigen_frame_valid_mask"]
            ),
            "antibody_missing_frame_mask": torch.from_numpy(
                values["antibody_missing_frame_mask"]
            ),
            "antigen_missing_frame_mask": torch.from_numpy(
                values["antigen_missing_frame_mask"]
            ),
            "antibody_labels": torch.from_numpy(values["antibody_labels"]),
            "antigen_labels": torch.from_numpy(values["antigen_labels"]),
        }
