#!/usr/bin/env python3
"""Verify every authenticated SAbDab2 structure/contact cohort artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_structure import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--structure-manifest-sha256", required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--contact-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_manifest(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} manifest SHA-256 differs")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} manifest") from error
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} manifest records are empty or invalid")
    return payload


def _inventory(root: Path, records: list[dict[str, Any]], suffix: str) -> None:
    expected = {
        str((root / str(record["artifact_path"])).resolve()) for record in records
    }
    actual = {
        str(path.resolve()) for path in (root / "artifacts").rglob(f"*{suffix}")
        if not path.name.startswith("._")
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Artifact inventory differs: missing={missing[:3]}, extra={extra[:3]}"
        )
    sidecars = sorted(str(path.relative_to(root)) for path in root.rglob("._*"))
    if sidecars:
        raise ValueError(f"AppleDouble sidecars are prohibited: {sidecars[:3]}")


def _verify_structure(root: Path, record: dict[str, Any]) -> tuple[int, int, int, int]:
    import torch

    path = root / str(record["artifact_path"])
    if path.stat().st_size != int(record["artifact_size_bytes"]):
        raise ValueError(f"Structure artifact size differs: {record['instance_id']}")
    if sha256_file(path) != record["artifact_sha256"]:
        raise ValueError(f"Structure artifact digest differs: {record['instance_id']}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("instance_id") != record["instance_id"]:
        raise ValueError(f"Structure artifact identity differs: {record['instance_id']}")
    antibody_length = int(record["antibody_length"])
    antigen_length = int(record["antigen_length"])
    expected_shapes = {
        "antibody_backbone": (antibody_length, 3, 3),
        "antigen_backbone": (antigen_length, 3, 3),
        "antibody_chain_ids": (antibody_length,),
        "antigen_chain_ids": (antigen_length,),
        "antibody_frame_valid_mask": (antibody_length,),
        "antigen_frame_valid_mask": (antigen_length,),
        "antibody_missing_frame_mask": (antibody_length,),
        "antigen_missing_frame_mask": (antigen_length,),
    }
    for field, shape in expected_shapes.items():
        value = payload.get(field)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(
                f"Structure field {field!r} differs: {record['instance_id']}"
            )
    antibody_valid = int(payload["antibody_frame_valid_mask"].sum().item())
    antigen_valid = int(payload["antigen_frame_valid_mask"].sum().item())
    if antibody_valid != int(record["antibody_valid_frame_count"]):
        raise ValueError(f"Antibody valid-frame count differs: {record['instance_id']}")
    if antigen_valid != int(record["antigen_valid_frame_count"]):
        raise ValueError(f"Antigen valid-frame count differs: {record['instance_id']}")
    if not torch.equal(
        payload["antibody_missing_frame_mask"],
        ~payload["antibody_frame_valid_mask"],
    ) or not torch.equal(
        payload["antigen_missing_frame_mask"],
        ~payload["antigen_frame_valid_mask"],
    ):
        raise ValueError(f"Missing-frame complement differs: {record['instance_id']}")
    return antibody_length, antigen_length, antibody_valid, antigen_valid


def _verify_contact(
    root: Path,
    record: dict[str, Any],
    *,
    antibody_length: int,
    antigen_length: int,
) -> int:
    import numpy as np

    path = root / str(record["artifact_path"])
    if sha256_file(path) != record["artifact_sha256"]:
        raise ValueError(f"Contact artifact digest differs: {record['instance_id']}")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "format_version",
            "instance_id",
            "antibody_length",
            "antigen_length",
            "candidate_pair_count",
            "positive_flat_indices",
        }
        if set(payload.files) != required:
            raise ValueError(f"Contact schema differs: {record['instance_id']}")
        if str(payload["instance_id"].item()) != record["instance_id"]:
            raise ValueError(f"Contact identity differs: {record['instance_id']}")
        observed_antibody = int(payload["antibody_length"].item())
        observed_antigen = int(payload["antigen_length"].item())
        candidate_count = int(payload["candidate_pair_count"].item())
        positives = payload["positive_flat_indices"].astype(np.int64, copy=True)
    if (observed_antibody, observed_antigen) != (antibody_length, antigen_length):
        raise ValueError(f"Structure/contact axes differ: {record['instance_id']}")
    if candidate_count != antibody_length * antigen_length or candidate_count != int(
        record["candidate_pair_count"]
    ):
        raise ValueError(f"Candidate-pair count differs: {record['instance_id']}")
    if len(positives) != int(record["positive_pair_count"]):
        raise ValueError(f"Positive-pair count differs: {record['instance_id']}")
    if len(positives):
        if positives[0] < 0 or positives[-1] >= candidate_count:
            raise ValueError(f"Contact index out of range: {record['instance_id']}")
        if len(np.unique(positives)) != len(positives):
            raise ValueError(f"Duplicate contact index: {record['instance_id']}")
    return len(positives)


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite verification receipt: {arguments.output}")
    structure_path = arguments.structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    contact_path = arguments.contact_root / "CONTACT_CACHE_MANIFEST.json"
    structure = _read_manifest(
        structure_path, arguments.structure_manifest_sha256, "structure"
    )
    contact = _read_manifest(contact_path, arguments.contact_manifest_sha256, "contact")
    structure_records = structure["records"]
    contact_records = contact["records"]
    _inventory(arguments.structure_root, structure_records, ".pt")
    _inventory(arguments.contact_root, contact_records, ".npz")
    contact_by_id = {record["instance_id"]: record for record in contact_records}
    if len(contact_by_id) != len(contact_records):
        raise ValueError("Contact manifest contains duplicate instance identifiers")
    if {record["instance_id"] for record in structure_records} != set(contact_by_id):
        raise ValueError("Structure and contact instance inventories differ")
    if contact.get("authority", {}).get("structure_manifest_sha256") != (
        arguments.structure_manifest_sha256
    ):
        raise ValueError("Contact manifest is not bound to the structure manifest")

    totals = {
        "instances": 0,
        "antibody_residues": 0,
        "antigen_residues": 0,
        "antibody_valid_frames": 0,
        "antigen_valid_frames": 0,
        "candidate_pairs": 0,
        "positive_contacts": 0,
    }
    for structure_record in structure_records:
        identifier = structure_record["instance_id"]
        lengths = _verify_structure(arguments.structure_root, structure_record)
        positives = _verify_contact(
            arguments.contact_root,
            contact_by_id[identifier],
            antibody_length=lengths[0],
            antigen_length=lengths[1],
        )
        totals["instances"] += 1
        totals["antibody_residues"] += lengths[0]
        totals["antigen_residues"] += lengths[1]
        totals["antibody_valid_frames"] += lengths[2]
        totals["antigen_valid_frames"] += lengths[3]
        totals["candidate_pairs"] += lengths[0] * lengths[1]
        totals["positive_contacts"] += positives

    receipt = {
        "format_version": "sabdab2_structure_contact_cache_verification_v1",
        "authority": {
            "structure_manifest_sha256": arguments.structure_manifest_sha256,
            "contact_manifest_sha256": arguments.contact_manifest_sha256,
        },
        "totals": totals,
        "checks": {
            "manifest_inventories_equal": True,
            "all_manifest_artifacts_present": True,
            "no_extra_artifacts": True,
            "no_appledouble_sidecars": True,
            "every_artifact_sha256_verified": True,
            "every_structure_weights_only_loaded": True,
            "every_contact_allow_pickle_false_loaded": True,
            "structure_contact_axes_equal": True,
            "contact_manifest_bound_to_structure_manifest": True,
        },
        "task_labels_read": False,
        "claim_boundary": (
            "Artifact integrity and schema verification only. Contact targets are "
            "authenticated for storage consistency but are not used for model fitting, "
            "selection, prediction, or metric computation."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{arguments.output.name}.",
        dir=arguments.output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json_bytes(receipt))
    try:
        temporary.replace(arguments.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(arguments.output)}, indent=2))


if __name__ == "__main__":
    main()
