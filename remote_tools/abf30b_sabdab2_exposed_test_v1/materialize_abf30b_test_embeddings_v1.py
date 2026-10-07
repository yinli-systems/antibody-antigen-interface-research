#!/usr/bin/env python3
"""Run the pinned ABF-30B mixed-embedding materializer for test sequences."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


SOURCE_SHA256 = "b176a6608572c2c5fa42d74a7ac32fd339d335bd904466316cdc5b340c7c7a7f"


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
    source_path = Path(take_option("--pinned-materializer"))
    sequence_sha = take_option("--sequence-view-sha256")
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("pinned embedding materializer SHA differs")
    source = source_path.read_text(encoding="utf-8")
    replacements = (
        ('SEQUENCE_VIEW_SHA256 = "27e0beb755c3d8e4b9910d4ad2ca979161668a1e1fe7b4d09bf6a60251e07468"', f'SEQUENCE_VIEW_SHA256 = "{sequence_sha}"'),
        ('EXPECTED_RECORDS = 3114', 'EXPECTED_RECORDS = 549'),
        ('"FROZEN_BEFORE_CHAIN_TYPING_DCP_RESTORE_AND_MODEL_SPECIFIC_MEASUREMENT"', '"FROZEN_BEFORE_EXPOSED_TEST_STRUCTURE_GENERATION_AND_LABEL_ARTIFACT_READ"'),
        ('row["role"] not in {"train", "validation"}', 'row["role"] != "test"'),
        ('"PASS_COMPLETE_DEVELOPMENT_MIXED_PARTNER_EMBEDDINGS"', '"PASS_COMPLETE_EXPOSED_TEST_MIXED_PARTNER_EMBEDDINGS"'),
        ('"old_sabdab2_test_loaded": False', '"old_sabdab2_test_loaded": True'),
        ('"test_loaded": False', '"test_loaded": True'),
    )
    for old, new in replacements:
        if old not in source:
            raise RuntimeError(f"pinned embedding patch anchor missing: {old}")
        source = source.replace(old, new)
    namespace = {"__name__": "abf30b_exposed_test_embeddings", "__file__": str(source_path)}
    exec(compile(source, str(source_path), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
