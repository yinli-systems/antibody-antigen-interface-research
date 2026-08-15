#!/usr/bin/env python3
"""Freeze a source-bound protocol for exact and gap-aware OAS decontamination."""

from __future__ import annotations

import argparse
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

from antibody_pretraining.decontaminated_corpus import (
    PROTOCOL_FORMAT,
    canonical_json_bytes,
    file_sha256,
    load_signed_json,
    publish_signed_json,
)


MANIFEST_FORMAT = "oas_production_corpus_manifest_v1"
AUDIT_FORMAT = "oas_production_corpus_independent_audit_v1"
EXCLUSION_FORMAT = "oas_near_identity_exclusion_compilation_v1"
TEST_ACCESS = {
    "downstream_metrics_used": False,
    "sabdab2_heldout_accessed": False,
}
AUTHORIZATION = {
    "large_scale_pretraining": False,
    "production_corpus_use": False,
    "scientific_claim": False,
}
AUDIT_AUTHORIZATION = {
    "independent_replay": True,
    **AUTHORIZATION,
}


class DecontaminationProtocolError(RuntimeError):
    """Report an incomplete exclusion graph or mutable source program."""


def _source_hash(path: Path, label: str) -> str:
    path = Path(path).expanduser().absolute()
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DecontaminationProtocolError(label + " is absent or symbolic")
    return file_sha256(path)


def freeze_protocol(
    *,
    source_root: Path,
    exclusion_root: Path,
    output: Path,
    filter_controller: Path,
    filter_core: Path,
    finalizer: Path,
    independent_auditor: Path,
    task_count: int = 32,
    bucket_count: int = 256,
) -> Mapping[str, Any]:
    """Authenticate all inputs and freeze a non-authorizing execution contract."""
    if (
        isinstance(task_count, bool)
        or isinstance(bucket_count, bool)
        or not 1 <= task_count <= bucket_count <= 256
    ):
        raise DecontaminationProtocolError("Decontamination topology differs")
    source_root = Path(source_root).expanduser().absolute()
    exclusion_root = Path(exclusion_root).expanduser().absolute()
    output = Path(output).expanduser().absolute()
    for root, label in ((source_root, "Source corpus"), (exclusion_root, "Exclusion")):
        if root.is_symlink() or not root.is_dir():
            raise DecontaminationProtocolError(label + " root is absent or symbolic")
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise DecontaminationProtocolError("Protocol output is invalid or already exists")
    manifest, manifest_file, manifest_payload = load_signed_json(
        source_root / "corpus_manifest.json",
        digest_field="manifest_sha256",
        label="Source corpus manifest",
    )
    audit, audit_file, audit_payload = load_signed_json(
        source_root / "independent_audit_receipt.json",
        digest_field="receipt_sha256",
        label="Source corpus independent audit",
    )
    exclusion, exclusion_file, exclusion_payload = load_signed_json(
        exclusion_root / "receipt.json",
        digest_field="receipt_sha256",
        label="Near-identity exclusion receipt",
    )
    if (
        manifest.get("format_version") != MANIFEST_FORMAT
        or manifest.get("bucket_count") != bucket_count
        or manifest.get("authorization") != AUTHORIZATION
        or manifest.get("test_access") != TEST_ACCESS
        or not isinstance(manifest.get("map_identity"), Mapping)
        or audit.get("format_version") != AUDIT_FORMAT
        or audit.get("authorization") != AUDIT_AUTHORIZATION
        or audit.get("test_access") != TEST_ACCESS
        or audit.get("inputs", {}).get("manifest_file_sha256") != manifest_file
        or audit.get("inputs", {}).get("manifest_payload_sha256") != manifest_payload
        or exclusion.get("format_version") != EXCLUSION_FORMAT
        or exclusion.get("test_access") != TEST_ACCESS
        or exclusion.get("claim_boundary", {}).get(
            "near_identity_exclusion_list_complete"
        )
        is not True
        or exclusion.get("claim_boundary", {}).get("original_training_corpus_filtered")
        is not False
    ):
        raise DecontaminationProtocolError("Decontamination evidence graph differs")
    exclusion_output = exclusion.get("output")
    if not isinstance(exclusion_output, Mapping):
        raise DecontaminationProtocolError("Exclusion output inventory is absent")
    exclusion_records = exclusion_output.get("excluded_sequence_sha256_records")
    exclusion_stream = exclusion_root / "excluded_sequence_sha256.bin"
    if (
        isinstance(exclusion_records, bool)
        or not isinstance(exclusion_records, int)
        or exclusion_records < 0
        or exclusion_output.get("excluded_sequence_sha256_record_bytes") != 32
        or exclusion_stream.is_symlink()
        or not exclusion_stream.is_file()
        or exclusion_stream.stat().st_mode & 0o222
        or exclusion_stream.stat().st_size != exclusion_records * 32
        or exclusion_stream.stat().st_size
        != exclusion_output.get("excluded_sequence_sha256_bytes")
        or file_sha256(exclusion_stream)
        != exclusion_output.get("excluded_sequence_sha256_file_sha256")
    ):
        raise DecontaminationProtocolError("Exclusion digest stream differs")
    sources = {
        "filter_controller": _source_hash(filter_controller, "Filter controller"),
        "filter_core": _source_hash(filter_core, "Filter core"),
        "finalizer": _source_hash(finalizer, "Finalizer"),
        "independent_auditor": _source_hash(
            independent_auditor, "Independent auditor"
        ),
    }
    protocol = {
        "authorization": AUTHORIZATION,
        "execution": {
            "bucket_count": bucket_count,
            "task_count": task_count,
        },
        "format_version": PROTOCOL_FORMAT,
        "inputs": {
            "exclusion_receipt_file_sha256": exclusion_file,
            "exclusion_receipt_payload_sha256": exclusion_payload,
            "source_audit_file_sha256": audit_file,
            "source_audit_payload_sha256": audit_payload,
            "source_manifest_file_sha256": manifest_file,
            "source_manifest_payload_sha256": manifest_payload,
        },
        "map_identity": manifest["map_identity"],
        "policy": {
            "excluded_splits": ["train", "validation"],
            "exclusion_key": "full_sequence_sha256",
            "exclusion_receipt_claim": "exact_plus_gap_aware_near_identity",
            "filter_after_global_deduplication": True,
            "independent_full_array_replay_required": True,
        },
        "source_sha256": sources,
        "test_access": TEST_ACCESS,
    }
    publish_signed_json(output, protocol, digest_field="protocol_sha256")
    return load_signed_json(
        output,
        digest_field="protocol_sha256",
        label="Frozen decontamination protocol",
    )[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclusion-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter-controller", type=Path, required=True)
    parser.add_argument("--filter-core", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--independent-auditor", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=32)
    parser.add_argument("--bucket-count", type=int, default=256)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    protocol = freeze_protocol(
        source_root=arguments.source_root,
        exclusion_root=arguments.exclusion_root,
        output=arguments.output,
        filter_controller=arguments.filter_controller,
        filter_core=arguments.filter_core,
        finalizer=arguments.finalizer,
        independent_auditor=arguments.independent_auditor,
        task_count=arguments.task_count,
        bucket_count=arguments.bucket_count,
    )
    sys.stdout.buffer.write(canonical_json_bytes(protocol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
