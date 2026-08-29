#!/usr/bin/env python3
"""Create a hash-bound, label-free summary of one antigen-fold manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


ROLES = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_unique(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or key not in reader.fieldnames:
            raise ValueError(f"{path} lacks {key}")
        result: dict[str, dict[str, str]] = {}
        for row in reader:
            identifier = str(row.get(key, "")).strip()
            if not identifier or identifier in result:
                raise ValueError(f"invalid or duplicate {key}: {identifier!r}")
            result[identifier] = {
                str(name): "" if value is None else str(value).strip()
                for name, value in row.items()
            }
    return result


def _quantile(sorted_values: list[int], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot summarize an empty vector")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _length_summary(values: Iterable[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p05": _quantile(ordered, 0.05),
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.5),
        "p75": _quantile(ordered, 0.75),
        "p95": _quantile(ordered, 0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _buckets(values: Iterable[int], boundaries: tuple[int, ...]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for value in values:
        lower = 0
        for upper in boundaries:
            if value <= upper:
                result[f"{lower}-{upper}"] += 1
                break
            lower = upper + 1
        else:
            result[f">{boundaries[-1]}"] += 1
    return dict(result)


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    split = _read_unique(arguments.split, "instance_id")
    metadata = _read_unique(arguments.metadata, "instance_id")
    if set(split) != set(metadata):
        raise ValueError(
            "split and metadata instance sets differ: "
            f"split_only={len(set(split) - set(metadata))}, "
            f"metadata_only={len(set(metadata) - set(split))}"
        )
    for identifier, row in split.items():
        role = row.get("role")
        if role not in ROLES:
            raise ValueError(f"{identifier}: invalid role {role!r}")
        meta = metadata[identifier]
        for split_name, metadata_name in (
            ("antigen_sequence_sha256", "antigen_sequence_sha256"),
            ("antibody_sequence_sha256", "antibody_sequence_sha256"),
        ):
            if row[split_name] != meta[metadata_name]:
                raise ValueError(f"{identifier}: sequence authority mismatch")

    by_role: dict[str, dict[str, object]] = {}
    for role in ROLES:
        identifiers = sorted(
            identifier for identifier, row in split.items() if row["role"] == role
        )
        rows = [split[identifier] for identifier in identifiers]
        meta = [metadata[identifier] for identifier in identifiers]
        antigen_lengths = [int(row["antigen_length"]) for row in meta]
        antibody_lengths = [int(row["antibody_length"]) for row in meta]
        by_role[role] = {
            "samples": len(identifiers),
            "unique_antigen_sequences": len(
                {row["antigen_sequence_sha256"] for row in rows}
            ),
            "unique_antibody_sequences": len(
                {row["antibody_sequence_sha256"] for row in rows}
            ),
            "unique_antigen_clusters": len(
                {row["antigen_cluster_id"] for row in rows}
            ),
            "unique_antibody_clusters": len(
                {row["antibody_cluster_id"] for row in rows}
            ),
            "unique_pdb_assemblies": len({row["pdb_assembly_id"] for row in rows}),
            "unique_documented_cognate_groups": len(
                {row["documented_cognate_group_id"] for row in meta}
            ),
            "antigen_length": _length_summary(antigen_lengths),
            "antibody_length": _length_summary(antibody_lengths),
            "antigen_length_buckets": _buckets(
                antigen_lengths, (49, 99, 199, 499, 999)
            ),
            "antibody_length_buckets": _buckets(
                antibody_lengths, (119, 179, 239, 299)
            ),
            "antigen_class_counts": dict(
                sorted(Counter(row["antigen_class"] for row in meta).items())
            ),
            "antibody_class_counts": dict(
                sorted(Counter(row["antibody_class"] for row in meta).items())
            ),
        }

    cross_role_overlap: dict[str, int] = {}
    for field in (
        "antigen_sequence_sha256",
        "antibody_sequence_sha256",
        "antigen_cluster_id",
        "antibody_cluster_id",
        "pdb_assembly_id",
    ):
        roles_by_value: dict[str, set[str]] = {}
        for row in split.values():
            roles_by_value.setdefault(row[field], set()).add(row["role"])
        cross_role_overlap[field] = sum(
            len(roles) > 1 for roles in roles_by_value.values()
        )

    payload = {
        "format_version": "framecontact_antigen_fold_cohort_summary_v1",
        "authority": {
            "split_path": str(arguments.split),
            "split_sha256": _sha256(arguments.split),
            "metadata_path": str(arguments.metadata),
            "metadata_sha256": _sha256(arguments.metadata),
            "task_labels_read": False,
            "model_predictions_read": False,
        },
        "instance_count": len(split),
        "roles": by_role,
        "cross_role_overlap_value_counts": cross_role_overlap,
        "definitions": {
            "sample": "one SAbDab2 antibody-antigen complex instance",
            "unique_partner": "reported separately by exact sequence SHA-256 and metadata cluster ID",
            "length": "authenticated residue count in the isolated partner structure adapter",
            "quantile": "linear interpolation at (n-1)*q on sorted lengths",
        },
        "claim_boundary": (
            "Label-free cohort composition only; this artifact contains no model "
            "performance result and does not open the test role for evaluation."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_canonical(payload) + b"\n")
    print(json.dumps({"output": str(arguments.output), "sha256": _sha256(arguments.output)}))


if __name__ == "__main__":
    main()
