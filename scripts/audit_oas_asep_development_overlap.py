#!/usr/bin/env python3
"""Audit exact OAS overlap for AsEP train/validation antibody chains."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for source_root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "framecontact/src"):
    while str(source_root) in sys.path:
        sys.path.remove(str(source_root))
    sys.path.insert(0, str(source_root))

from antibody_pretraining.downstream_overlap import (
    DevelopmentChainReference,
    audit_exact_development_overlap,
    sequence_sha256,
)
from antibody_pretraining.production_corpus import load_production_corpus
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
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence is not an object: {path}")
    return value


def load_asep_development_references(
    *, cache_root: Path, manifest_file_sha256: str
) -> tuple[list[DevelopmentChainReference], dict[str, Any]]:
    """Load only label-free sequence fields from an authenticated development cache."""
    manifest_path = cache_root / "ASEP_DEVELOPMENT_CACHE_MANIFEST.json"
    authenticate_file(manifest_path, manifest_file_sha256, "AsEP cache manifest")
    manifest = _strict_json(manifest_path)
    records = manifest.get("records")
    if (
        manifest.get("format_version") != DEVELOPMENT_CACHE_FORMAT
        or manifest.get("checks", {}).get("official_test_members_materialized")
        is not False
        or not isinstance(records, list)
        or len(records) != manifest.get("record_count")
    ):
        raise ValueError("AsEP development cache is not eligible for overlap audit")
    references: list[DevelopmentChainReference] = []
    role_counts = {"train": 0, "val": 0}
    for record in records:
        if not isinstance(record, dict) or record.get("role") not in role_counts:
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
        for chain, key in (("heavy", "heavy_sequence"), ("light", "light_sequence")):
            value = artifact.get(key)
            if not isinstance(value, str):
                raise ValueError(f"AsEP artifact {key} is not text")
            canonical = value.strip().upper()
            if not canonical:
                if chain == "light":
                    continue
                raise ValueError("AsEP heavy chain is empty")
            references.append(
                DevelopmentChainReference(
                    chain=chain,
                    instance_id=str(record["instance_id"]),
                    role=str(record["role"]),
                    sequence_length=len(canonical),
                    sequence_sha256=sequence_sha256(canonical),
                )
            )
    if role_counts != manifest.get("role_counts"):
        raise ValueError("AsEP development role counts differ")
    return references, {
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
    parser.add_argument("--production-corpus-root", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--manifest-payload-sha256", required=True)
    parser.add_argument("--corpus-protocol-file-sha256", required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--independent-audit-file-sha256", required=True)
    parser.add_argument("--independent-audit-payload-sha256", required=True)
    parser.add_argument("--asep-cache-root", type=Path, required=True)
    parser.add_argument("--asep-cache-manifest-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-open-arrays", type=int, default=128)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite overlap audit: {arguments.output}")
    references, asep_authority = load_asep_development_references(
        cache_root=arguments.asep_cache_root,
        manifest_file_sha256=arguments.asep_cache_manifest_file_sha256,
    )
    corpus_arguments = {
        "output_root": arguments.production_corpus_root,
        "expected_manifest_file_sha256": arguments.manifest_file_sha256,
        "expected_manifest_payload_sha256": arguments.manifest_payload_sha256,
        "expected_protocol_file_sha256": arguments.corpus_protocol_file_sha256,
        "independent_audit_path": arguments.independent_audit,
        "expected_independent_audit_file_sha256": (
            arguments.independent_audit_file_sha256
        ),
        "expected_independent_audit_payload_sha256": (
            arguments.independent_audit_payload_sha256
        ),
        "verify_array_file_hashes": True,
        "maximum_open_arrays": arguments.maximum_open_arrays,
    }
    train = load_production_corpus(split="train", **corpus_arguments)
    validation = load_production_corpus(split="validation", **corpus_arguments)
    try:
        result = audit_exact_development_overlap(
            references=references,
            train_corpus=train,
            validation_corpus=validation,
        )
    finally:
        train.close()
        validation.close()
    result["inputs"] = {
        "asep_development": asep_authority,
        "corpus_independent_audit_file_sha256": (
            arguments.independent_audit_file_sha256
        ),
        "corpus_independent_audit_payload_sha256": (
            arguments.independent_audit_payload_sha256
        ),
        "corpus_manifest_file_sha256": arguments.manifest_file_sha256,
        "corpus_manifest_payload_sha256": arguments.manifest_payload_sha256,
        "corpus_protocol_file_sha256": arguments.corpus_protocol_file_sha256,
        "source": {
            "module_sha256": _sha256(
                REPOSITORY_ROOT / "src/antibody_pretraining/downstream_overlap.py"
            ),
            "script_sha256": _sha256(Path(__file__)),
        },
    }
    result["test_access"] = {
        "asep_development_sequence_fields_accessed": True,
        "asep_official_test_accessed": False,
        "external_test_accessed": False,
        "frozen_test_accessed": False,
        "model_predictions_or_scores_accessed": False,
    }
    unsigned = dict(result)
    result["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=arguments.output.parent,
        prefix=f".{arguments.output.name}.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(result))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, arguments.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
