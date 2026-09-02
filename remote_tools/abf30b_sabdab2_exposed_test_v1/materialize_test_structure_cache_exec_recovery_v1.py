#!/usr/bin/env python3
"""Recover the pinned test structure materializer with a registered exec module."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import types
from typing import Any


SOURCE_SHA256 = "4326a5af92f1ff17102db0f10570d80f9d258bfc9fd1d88425554bd183b0b3fe"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def take_option(name: str) -> str:
    index = sys.argv.index(name)
    value = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    return value


def main() -> None:
    protocol_path = Path(take_option("--prep-protocol"))
    protocol_sha = take_option("--prep-protocol-sha256")
    recovery_path = Path(take_option("--exec-recovery-protocol"))
    recovery_sha = take_option("--exec-recovery-protocol-sha256")
    source_path = Path(take_option("--pinned-materializer"))
    if (
        sha256_file(protocol_path) != protocol_sha
        or sha256_file(recovery_path) != recovery_sha
        or sha256_file(source_path) != SOURCE_SHA256
    ):
        raise ValueError("protocol, recovery, or pinned materializer SHA differs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_BEFORE_EXPOSED_TEST_STRUCTURE_GENERATION_AND_LABEL_ARTIFACT_READ"
        or protocol.get("task_labels_read_during_freeze") is not False
        or recovery.get("status") != "FROZEN_AFTER_STRUCTURE_EXEC_FAILURE_BEFORE_LABEL_ARTIFACT_READ"
        or recovery.get("scientific_protocol_changed") is not False
    ):
        raise ValueError("preparation or recovery protocol differs")
    output_index = sys.argv.index("--output-root")
    output_root = Path(sys.argv[output_index + 1])
    source = source_path.read_text(encoding="utf-8")
    replacements = (
        ('ALLOWED_ROLES = frozenset({"train", "validation"})', 'ALLOWED_ROLES = frozenset({"test"})'),
        ('sdh_plan.get("allowed_roles") != ["train", "validation"]', 'sdh_plan.get("allowed_roles") != ["test"]'),
        ('sdh_plan.get("test_access") is not False', 'sdh_plan.get("test_access") is not True'),
        ('temporary_root / "DEVELOPMENT_SPLIT_VIEW.csv"', 'temporary_root / "TEST_SPLIT_VIEW.csv"'),
        ('"test_structures_resolved": False', '"test_structures_resolved": True'),
        ('"development_split_view_path": "DEVELOPMENT_SPLIT_VIEW.csv"', '"test_split_view_path": "TEST_SPLIT_VIEW.csv"'),
        ('"development_split_view_sha256": split_view_sha256', '"test_split_view_sha256": split_view_sha256'),
        ('"PASS_DEVELOPMENT_ONLY_PREDICTED_STRUCTURE_CACHE"', '"PASS_EXPOSED_TEST_PREDICTED_STRUCTURE_CACHE"'),
    )
    for old, new in replacements:
        if old not in source:
            raise RuntimeError(f"pinned materializer patch anchor missing: {old}")
        source = source.replace(old, new)
    module_name = "sabdab2_exposed_test_materializer_exec_recovery_v1"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = None
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(source_path), "exec"), module.__dict__)
        module.main()
    finally:
        sys.modules.pop(module_name, None)
    manifest_path = output_root / "STRUCTURE_CACHE_MANIFEST.json"
    receipt_path = output_root / "STRUCTURE_CACHE_RECEIPT.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        manifest.get("counts", {}).get("records") != 549
        or manifest.get("counts", {}).get("roles") != {"test": 549}
        or manifest.get("authority", {}).get("experimental_bound_complex_coordinates_read") is not False
        or manifest.get("authority", {}).get("cross_partner_coordinates_or_distances_used") is not False
        or manifest.get("authority", {}).get("task_labels_read") is not False
        or manifest.get("authority", {}).get("test_structures_resolved") is not True
        or receipt.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("test predicted-structure cache audit differs")
    binding_sha = publish(output_root / "TEST_STRUCTURE_CACHE_BINDING_V1.json", {
        "format_version": "abf30b_sabdab2_exposed_test_structure_binding_v1",
        "status": "PASS_EXPOSED_TEST_PREDICTED_STRUCTURE_CACHE",
        "prep_protocol_sha256": protocol_sha,
        "exec_recovery_protocol_sha256": recovery_sha,
        "pinned_materializer_sha256": SOURCE_SHA256,
        "manifest_sha256": sha256_file(manifest_path),
        "receipt_sha256": sha256_file(receipt_path),
        "record_count": 549,
        "experimental_bound_coordinates_as_model_input": False,
        "cross_partner_geometry": False,
        "task_labels_read": False,
        "test_sequence_metadata_read": True,
        "temporal_holdout_loaded": False,
    })
    print(json.dumps({"status": "PASS_EXPOSED_TEST_PREDICTED_STRUCTURE_CACHE", "binding_sha256": binding_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
