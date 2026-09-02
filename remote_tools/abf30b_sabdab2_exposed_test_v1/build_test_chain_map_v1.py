#!/usr/bin/env python3
"""Run the pinned ANARCII chain-map builder for the frozen test cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


SOURCE_SHA256 = "71ecb69f11e44c024590395e4758bbae63ca4661202f03beedff3f5fb05668aa"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def take_option(name: str) -> str:
    index = sys.argv.index(name); value = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    return value


def main() -> None:
    source_path = Path(take_option("--pinned-builder"))
    sequence_sha = take_option("--sequence-view-sha256")
    sequence_receipt_sha = take_option("--sequence-receipt-sha256")
    structure_sha = take_option("--structure-manifest-sha256")
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("pinned chain-map builder SHA differs")
    source = source_path.read_text(encoding="utf-8")
    replacements = (
        ('SEQUENCE_VIEW_SHA256 = "27e0beb755c3d8e4b9910d4ad2ca979161668a1e1fe7b4d09bf6a60251e07468"', f'SEQUENCE_VIEW_SHA256 = "{sequence_sha}"'),
        ('SEQUENCE_RECEIPT_SHA256 = "36a9249d5d1cec4065292e79d1efae8623987a4643675557f32252bcde7752ee"', f'SEQUENCE_RECEIPT_SHA256 = "{sequence_receipt_sha}"'),
        ('STRUCTURE_MANIFEST_SHA256 = "bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614"', f'STRUCTURE_MANIFEST_SHA256 = "{structure_sha}"'),
        ('EXPECTED_RECORDS = 3114', 'EXPECTED_RECORDS = 549'),
        ('EXPECTED_TRAIN = 2565', 'EXPECTED_TRAIN = 0'),
        ('EXPECTED_VALIDATION = 549', 'EXPECTED_VALIDATION = 0'),
        ('"FROZEN_BEFORE_CHAIN_TYPING_DCP_RESTORE_AND_MODEL_SPECIFIC_MEASUREMENT"', '"FROZEN_BEFORE_EXPOSED_TEST_STRUCTURE_GENERATION_AND_LABEL_ARTIFACT_READ"'),
        ('or sequence_receipt.get("test_loaded") is not False\n        or sequence_receipt.get("test_rows_emitted") is not False', 'or sequence_receipt.get("test_access") is not True\n        or sequence_receipt.get("task_labels_loaded") is not False'),
        ('role_counts: dict[str, int] = {"train": 0, "validation": 0}', 'role_counts: dict[str, int] = {"test": 0}'),
        ('role_counts != {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION}', 'role_counts != {"test": EXPECTED_RECORDS}'),
        ('"format_version": "abf30b_sabdab2_development_chain_map_v1"', '"format_version": "abf30b_sabdab2_exposed_test_chain_map_v1"'),
        ('"train": role_counts["train"],\n            "validation": role_counts["validation"],', '"test": role_counts["test"],'),
        ('"old_sabdab2_test_loaded": False', '"old_sabdab2_test_loaded": True'),
        ('"format_version": "abf30b_sabdab2_development_chain_map_receipt_v1"', '"format_version": "abf30b_sabdab2_exposed_test_chain_map_receipt_v1"'),
        ('"test_loaded": False', '"test_loaded": True'),
    )
    for old, new in replacements:
        if old not in source:
            raise RuntimeError(f"pinned chain-map patch anchor missing: {old}")
        source = source.replace(old, new)
    namespace = {"__name__": "abf30b_exposed_test_chain_map", "__file__": str(source_path)}
    exec(compile(source, str(source_path), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
