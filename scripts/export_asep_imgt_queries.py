#!/usr/bin/env python3
"""Export all authenticated AsEP-development chains on fixed IMGT positions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for source_root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "framecontact/src"):
    while str(source_root) in sys.path:
        sys.path.remove(str(source_root))
    sys.path.insert(0, str(source_root))

from antibody_pretraining.downstream_overlap import (
    DevelopmentChainReference,
    sequence_sha256,
)
from antibody_pretraining.imgt_query_export import (
    authenticate_anarcii_runtime,
    export_numbered_development_queries,
)
from antibody_pretraining.near_identity_overlap import reference_key
from interaction_first.asep_external import (
    DEVELOPMENT_ARTIFACT_FORMAT,
    DEVELOPMENT_CACHE_FORMAT,
)
from interaction_first.sabdab2_structure import authenticate_file


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot parse JSON evidence: {path}") from error
    if type(value) is not dict:
        raise ValueError(f"JSON evidence is not an object: {path}")
    return value


def load_asep_development_sequences(
    *, cache_root: Path, manifest_file_sha256: str
) -> tuple[list[tuple[DevelopmentChainReference, str]], dict[str, Any]]:
    """Read only authenticated development heavy/light sequence fields."""
    manifest_path = cache_root / "ASEP_DEVELOPMENT_CACHE_MANIFEST.json"
    authenticate_file(manifest_path, manifest_file_sha256, "AsEP cache manifest")
    manifest = _strict_json(manifest_path)
    records = manifest.get("records")
    if (
        manifest.get("format_version") != DEVELOPMENT_CACHE_FORMAT
        or manifest.get("checks", {}).get("official_test_members_materialized") is not False
        or type(records) is not list
        or len(records) != manifest.get("record_count")
    ):
        raise ValueError("AsEP development cache is not eligible for query export")
    sequences: list[tuple[DevelopmentChainReference, str]] = []
    role_counts = {"train": 0, "val": 0}
    for record in records:
        if type(record) is not dict or record.get("role") not in role_counts:
            raise ValueError("AsEP development cache record is invalid")
        artifact_path = cache_root / str(record.get("artifact_path", ""))
        authenticate_file(
            artifact_path,
            str(record.get("artifact_sha256", "")),
            "AsEP cache artifact",
        )
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if (
            artifact.get("format_version") != DEVELOPMENT_ARTIFACT_FORMAT
            or artifact.get("instance_id") != record.get("instance_id")
            or artifact.get("role") != record.get("role")
        ):
            raise ValueError("AsEP cache artifact identity differs")
        role_counts[str(record["role"])] += 1
        for chain, field in (("heavy", "heavy_sequence"), ("light", "light_sequence")):
            value = artifact.get(field)
            if type(value) is not str:
                raise ValueError(f"AsEP artifact {field} is not text")
            canonical = value.strip().upper()
            if not canonical:
                if chain == "light":
                    continue
                raise ValueError("AsEP heavy chain is empty")
            reference = DevelopmentChainReference(
                chain=chain,
                instance_id=str(record["instance_id"]),
                role=str(record["role"]),
                sequence_length=len(canonical),
                sequence_sha256=sequence_sha256(canonical),
            )
            reference.validate()
            sequences.append((reference, canonical))
    if role_counts != manifest.get("role_counts"):
        raise ValueError("AsEP development role counts differ")
    identities = [(reference.instance_id, reference.chain) for reference, _ in sequences]
    if len(identities) != len(set(identities)):
        raise ValueError("AsEP development chain identity is duplicated")
    return sequences, {
        "artifact_count": len(records),
        "cache_manifest_file_sha256": manifest_file_sha256,
        "cache_manifest_payload_sha256": hashlib.sha256(
            _canonical_json_bytes(manifest)
        ).hexdigest(),
        "official_test_members_materialized": False,
        "role_counts": role_counts,
        "task_label_values_used": False,
        "task_label_bearing_artifacts_deserialized": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asep-cache-root", type=Path, required=True)
    parser.add_argument("--asep-cache-manifest-file-sha256", required=True)
    parser.add_argument("--anarcii-wheel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-query-count", type=int, default=3_106)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ncpu", type=int, default=8)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.batch_size <= 0 or arguments.ncpu <= 0:
        raise ValueError("ANARCII execution settings must be positive")
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite query export: {arguments.output_root}")
    runtime = authenticate_anarcii_runtime(arguments.anarcii_wheel)
    sequences, asep_authority = load_asep_development_sequences(
        cache_root=arguments.asep_cache_root,
        manifest_file_sha256=arguments.asep_cache_manifest_file_sha256,
    )
    if len(sequences) != arguments.expected_query_count:
        raise ValueError("AsEP development query count differs before numbering")
    from anarcii import Anarcii

    numberer = Anarcii(
        seq_type="antibody",
        mode="accuracy",
        cpu=True,
        batch_size=arguments.batch_size,
        ncpu=arguments.ncpu,
        verbose=False,
    )
    inputs = [
        (reference_key(reference), sequence) for reference, sequence in sequences
    ]
    numbered = numberer.number(inputs)
    runtime["execution"] = {
        "batch_size": arguments.batch_size,
        "ncpu": arguments.ncpu,
        "query_count": len(inputs),
    }
    source = {
        "asep_development": asep_authority,
        "module_sha256": _sha256(
            REPOSITORY_ROOT / "src/antibody_pretraining/imgt_query_export.py"
        ),
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    receipt = export_numbered_development_queries(
        output_root=arguments.output_root,
        sequences=sequences,
        numbered=numbered,
        runtime_provenance=runtime,
        expected_query_count=arguments.expected_query_count,
        source_provenance=source,
    )
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
