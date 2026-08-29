#!/usr/bin/env python3
"""Freeze a label-aware same-antigen/different-antibody development challenge."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.asep_external import DEVELOPMENT_ARTIFACT_FORMAT  # noqa: E402
from interaction_first.sabdab2_structure import (  # noqa: E402
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--role", choices=("train", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _artifact_axes(
    cache_root: Path,
    record: dict[str, Any],
) -> tuple[set[int], set[int]]:
    path = cache_root / record["artifact_path"]
    authenticate_file(path, str(record["artifact_sha256"]), "AsEP challenge artifact")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if (
        artifact.get("format_version") != DEVELOPMENT_ARTIFACT_FORMAT
        or artifact.get("instance_id") != record["instance_id"]
        or artifact.get("role") != record["role"]
    ):
        raise ValueError("AsEP challenge artifact metadata differs")
    selection = artifact["antigen_full_selection"]
    labels = artifact["antigen_labels"]
    if (
        not torch.is_tensor(selection)
        or selection.dtype != torch.bool
        or selection.ndim != 1
        or not torch.is_tensor(labels)
        or labels.ndim != 1
        or int(selection.sum()) != int(labels.shape[0])
        or not bool(((labels == 0) | (labels == 1)).all())
    ):
        raise ValueError("AsEP antigen selection/label axis is malformed")
    observed = torch.where(selection)[0]
    positive = observed[labels.bool()]
    return set(observed.tolist()), set(positive.tolist())


def build_challenge(
    *,
    cache_root: Path,
    cache_manifest_sha256: str,
    metadata_csv: Path,
    metadata_sha256: str,
    role: str,
) -> dict[str, Any]:
    manifest_path = cache_root / "ASEP_DEVELOPMENT_CACHE_MANIFEST.json"
    authenticate_file(manifest_path, cache_manifest_sha256, "AsEP cache manifest")
    authenticate_file(metadata_csv, metadata_sha256, "AsEP partner metadata")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {record["instance_id"]: record for record in manifest["records"]}
    with metadata_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(
            token in column.lower()
            for column in reader.fieldnames
            for token in ("label", "contact", "epitope", "paratope", "prediction")
        ):
            raise ValueError("AsEP partner metadata contains a prohibited task column")
        metadata = [row for row in reader if row["role"] == role]
    if not metadata:
        raise ValueError("AsEP same-antigen challenge role is empty")
    expected_record_role = "val" if role == "validation" else role
    if any(
        row["instance_id"] not in records
        or records[row["instance_id"]]["role"] != expected_record_role
        for row in metadata
    ):
        raise ValueError("AsEP challenge metadata and cache roles differ")

    by_antigen: dict[str, list[dict[str, str]]] = {}
    for row in metadata:
        by_antigen.setdefault(row["antigen_cluster_id"], []).append(row)
    eligible_groups = {
        cluster: rows
        for cluster, rows in by_antigen.items()
        if len(rows) >= 2 and len({row["antibody_cluster_id"] for row in rows}) >= 2
    }
    axes: dict[str, tuple[set[int], set[int]]] = {}
    for rows in eligible_groups.values():
        for row in rows:
            identifier = row["instance_id"]
            if identifier not in axes:
                axes[identifier] = _artifact_axes(cache_root, records[identifier])

    entries: list[dict[str, Any]] = []
    for antigen_cluster, rows in sorted(eligible_groups.items()):
        for first, second in combinations(sorted(rows, key=lambda row: row["instance_id"]), 2):
            if first["antibody_cluster_id"] == second["antibody_cluster_id"]:
                continue
            first_observed, first_positive = axes[first["instance_id"]]
            second_observed, second_positive = axes[second["instance_id"]]
            common = first_observed & second_observed
            first_common_positive = first_positive & common
            second_common_positive = second_positive & common
            if (
                not first_common_positive
                or not second_common_positive
                or first_common_positive == second_common_positive
                or len(common - first_common_positive) == 0
                or len(common - second_common_positive) == 0
            ):
                continue
            union = first_common_positive | second_common_positive
            intersection = first_common_positive & second_common_positive
            entries.append(
                {
                    "antigen_cluster_id": antigen_cluster,
                    "first_instance_id": first["instance_id"],
                    "second_instance_id": second["instance_id"],
                    "first_antibody_cluster_id": first["antibody_cluster_id"],
                    "second_antibody_cluster_id": second["antibody_cluster_id"],
                    "common_antigen_full_indices": sorted(common),
                    "first_epitope_full_indices": sorted(first_common_positive),
                    "second_epitope_full_indices": sorted(second_common_positive),
                    "common_observed_count": len(common),
                    "first_epitope_count": len(first_common_positive),
                    "second_epitope_count": len(second_common_positive),
                    "epitope_intersection_count": len(intersection),
                    "epitope_union_count": len(union),
                    "epitope_jaccard": len(intersection) / len(union),
                }
            )
    if not entries:
        raise ValueError("AsEP same-antigen challenge has no eligible pair")
    return {
        "format_version": "asep_same_antigen_different_antibody_challenge_v1",
        "role": role,
        "authority": {
            "cache_manifest_sha256": cache_manifest_sha256,
            "metadata_sha256": metadata_sha256,
            "official_test_materialized": False,
            "model_predictions_or_scores_read": False,
        },
        "selection_rule": {
            "same_antigen": "exact full antigen sequence SHA-256 within one role",
            "different_antibody": "different exact paired H/L sequence SHA-256",
            "axis": "intersection of observed surface residues mapped to zero-based full-antigen indices",
            "labels": "both instances contain positive and negative residues on the common axis, and their positive sets differ",
            "pair_retention": "all unordered pairs satisfying the rule; no model score or performance threshold",
        },
        "eligible_antigen_group_count_before_label_rule": len(eligible_groups),
        "pair_count": len(entries),
        "entries": entries,
        "authorized_use": (
            "Development-only counterfactual case study. For each fixed antigen structure, "
            "swap only the antibody and require scores to move toward the swapped antibody's "
            "known epitope on the common residue axis."
        ),
        "prohibited_use": (
            "This label-aware panel may not select checkpoints, hyperparameters, thresholds, "
            "architectures, exclusions, or official-test cases."
        ),
    }


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite challenge: {arguments.output}")
    challenge = build_challenge(
        cache_root=arguments.cache_root,
        cache_manifest_sha256=arguments.cache_manifest_sha256,
        metadata_csv=arguments.metadata_csv,
        metadata_sha256=arguments.metadata_sha256,
        role=arguments.role,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{arguments.output.name}.",
        dir=arguments.output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json_bytes(challenge))
    try:
        temporary.replace(arguments.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": sha256_file(arguments.output),
                "pair_count": challenge["pair_count"],
                "eligible_antigen_group_count": challenge[
                    "eligible_antigen_group_count_before_label_rule"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
