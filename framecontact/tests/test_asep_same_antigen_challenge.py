"""Contracts for the label-aware AsEP same-antigen challenge builder."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from interaction_first.asep_external import DEVELOPMENT_ARTIFACT_FORMAT
from interaction_first.sabdab2_structure import sha256_file


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/build_asep_same_antigen_challenge.py"
    )
    spec = importlib.util.spec_from_file_location("asep_same_antigen_challenge", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path, *, prohibited_column: bool = False):
    cache = tmp_path / "cache"
    artifacts = cache / "artifacts/aa"
    artifacts.mkdir(parents=True)
    records = []
    for identifier, labels in (
        ("case_a", [1.0, 0.0, 0.0, 1.0]),
        ("case_b", [0.0, 1.0, 0.0, 1.0]),
    ):
        path = artifacts / f"{identifier}.pt"
        torch.save(
            {
                "format_version": DEVELOPMENT_ARTIFACT_FORMAT,
                "instance_id": identifier,
                "role": "val",
                "antigen_full_selection": torch.tensor(
                    [True, True, False, True, True]
                ),
                "antigen_labels": torch.tensor(labels),
            },
            path,
        )
        records.append(
            {
                "instance_id": identifier,
                "role": "val",
                "artifact_path": str(path.relative_to(cache)),
                "artifact_sha256": sha256_file(path),
            }
        )
    manifest = cache / "ASEP_DEVELOPMENT_CACHE_MANIFEST.json"
    manifest.write_text(json.dumps({"records": records}))
    metadata = tmp_path / "metadata.csv"
    fields = [
        "instance_id",
        "role",
        "antigen_cluster_id",
        "antibody_cluster_id",
    ]
    if prohibited_column:
        fields.append("epitope_count")
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, identifier in enumerate(("case_a", "case_b")):
            row = {
                "instance_id": identifier,
                "role": "validation",
                "antigen_cluster_id": "exact:antigen",
                "antibody_cluster_id": f"exact:antibody-{index}",
            }
            if prohibited_column:
                row["epitope_count"] = "2"
            writer.writerow(row)
    return cache, manifest, metadata


def test_same_antigen_challenge_maps_labels_to_full_axis(tmp_path: Path) -> None:
    module = _module()
    cache, manifest, metadata = _fixture(tmp_path)
    result = module.build_challenge(
        cache_root=cache,
        cache_manifest_sha256=sha256_file(manifest),
        metadata_csv=metadata,
        metadata_sha256=sha256_file(metadata),
        role="validation",
    )
    assert result["pair_count"] == 1
    entry = result["entries"][0]
    assert entry["common_antigen_full_indices"] == [0, 1, 3, 4]
    assert entry["first_epitope_full_indices"] == [0, 4]
    assert entry["second_epitope_full_indices"] == [1, 4]
    assert entry["epitope_jaccard"] == pytest.approx(1 / 3)
    assert result["authority"]["official_test_materialized"] is False


def test_same_antigen_challenge_rejects_label_bearing_metadata(tmp_path: Path) -> None:
    module = _module()
    cache, manifest, metadata = _fixture(tmp_path, prohibited_column=True)
    with pytest.raises(ValueError, match="prohibited task column"):
        module.build_challenge(
            cache_root=cache,
            cache_manifest_sha256=sha256_file(manifest),
            metadata_csv=metadata,
            metadata_sha256=sha256_file(metadata),
            role="validation",
        )
