#!/usr/bin/env python3
"""Materialize frozen development-only OOD and temporal evaluation slices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


EXPECTED_FORMAT = "abpairfm_development_transfer_matrix_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_development_rows(path: Path) -> tuple[list[dict[str, str]], int]:
    """Retain only train/validation rows and count skipped sealed-role rows."""
    retained: list[dict[str, str]] = []
    sealed_count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            role = row["role"]
            if role in {"train", "validation"}:
                retained.append(row)
            else:
                sealed_count += 1
    return retained, sealed_count


def _ids(rows: Iterable[dict[str, str]]) -> list[str]:
    result = sorted(row["instance_id"] for row in rows)
    if len(result) != len(set(result)):
        raise RuntimeError("Development slice contains duplicate instance IDs")
    return result


def _axis_slices(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    train = [row for row in rows if row["role"] == "train"]
    validation = [row for row in rows if row["role"] == "validation"]
    train_antigens = {row["antigen_cluster_id"] for row in train}
    train_antibodies = {row["antibody_cluster_id"] for row in train}
    buckets: dict[str, list[dict[str, str]]] = {
        "seen_antigen__seen_antibody": [],
        "seen_antigen__unseen_antibody": [],
        "unseen_antigen__seen_antibody": [],
        "unseen_antigen__unseen_antibody": [],
    }
    for row in validation:
        antigen = (
            "seen_antigen"
            if row["antigen_cluster_id"] in train_antigens
            else "unseen_antigen"
        )
        antibody = (
            "seen_antibody"
            if row["antibody_cluster_id"] in train_antibodies
            else "unseen_antibody"
        )
        buckets[f"{antigen}__{antibody}"].append(row)
    return {name: _ids(bucket) for name, bucket in sorted(buckets.items())}


def main() -> None:
    arguments = _arguments()
    if sha256_file(arguments.protocol) != arguments.expected_protocol_sha256:
        raise RuntimeError("Development transfer protocol SHA-256 differs")
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("format_version") != EXPECTED_FORMAT
        or protocol.get("status") != "FROZEN_BEFORE_TRANSFER_EVALUATION"
        or protocol.get("test_access") is not False
        or protocol.get("authorization", {}).get("test_access") is not False
    ):
        raise RuntimeError("Development transfer protocol authorization differs")

    sources: dict[str, tuple[list[dict[str, str]], int]] = {}
    for name in ("standard", "double_cold", "temporal"):
        anchor = protocol["source_anchors"][name]
        path = Path(anchor["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Split source is not a regular file: {name}")
        if sha256_file(path) != anchor["sha256"]:
            raise RuntimeError(f"Split source SHA-256 differs: {name}")
        sources[name] = _read_development_rows(path)
    for name in ("contact_cache_manifest", "split_receipt"):
        anchor = protocol["source_anchors"][name]
        if sha256_file(Path(anchor["path"])) != anchor["sha256"]:
            raise RuntimeError(f"Bound authority SHA-256 differs: {name}")

    standard_rows, standard_sealed = sources["standard"]
    double_rows, double_sealed = sources["double_cold"]
    temporal_rows, temporal_sealed = sources["temporal"]
    standard_axes = _axis_slices(standard_rows)
    temporal_axes = _axis_slices(temporal_rows)
    slices = {
        "double_cold_validation": _ids(
            row for row in double_rows if row["role"] == "validation"
        ),
        "standard_double_unseen": standard_axes[
            "unseen_antigen__unseen_antibody"
        ],
        "standard_unseen_antibody_only": standard_axes[
            "seen_antigen__unseen_antibody"
        ],
        "standard_unseen_antigen_only": standard_axes[
            "unseen_antigen__seen_antibody"
        ],
        "standard_validation": _ids(
            row for row in standard_rows if row["role"] == "validation"
        ),
        "temporal_validation": _ids(
            row for row in temporal_rows if row["role"] == "validation"
        ),
    }
    for name, instance_ids in slices.items():
        expected = protocol["split_slices"][name]["expected_count"]
        if len(instance_ids) != expected:
            raise RuntimeError(
                f"Development slice count differs for {name}: {len(instance_ids)} != {expected}"
            )

    output: dict[str, Any] = {
        "authority": {
            "contact_labels_read": False,
            "predictions_read": False,
            "protocol_sha256": arguments.expected_protocol_sha256,
            "sealed_role_ids_retained": False,
            "sealed_role_row_counts_skipped": {
                "double_cold": double_sealed,
                "standard": standard_sealed,
                "temporal": temporal_sealed,
            },
            "task_labels_read": False,
        },
        "claim_boundary": protocol["claim_boundary"],
        "format_version": "abpairfm_development_transfer_manifest_v1",
        "slices": {
            name: {
                "count": len(instance_ids),
                "instance_ids": instance_ids,
                "instance_ids_sha256": hashlib.sha256(
                    ("\n".join(instance_ids) + "\n").encode("utf-8")
                ).hexdigest(),
            }
            for name, instance_ids in sorted(slices.items())
        },
        "standard_validation_axis_counts": {
            name: len(instance_ids) for name, instance_ids in standard_axes.items()
        },
        "status": "PASS_DEVELOPMENT_ONLY_MANIFEST",
        "temporal_validation_axis_counts": {
            name: len(instance_ids) for name, instance_ids in temporal_axes.items()
        },
        "test_access": False,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
