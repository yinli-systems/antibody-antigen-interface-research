#!/usr/bin/env python3
"""Run the pinned evaluator with the authenticated test-only split view."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import types


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def take_option(name: str) -> str:
    index = sys.argv.index(name)
    value = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    return value


def main() -> None:
    recovery_path = Path(take_option("--execution-recovery-protocol"))
    recovery_sha = take_option("--execution-recovery-protocol-sha256")
    original_path = Path(take_option("--pinned-evaluator"))
    if sha256_file(recovery_path) != recovery_sha:
        raise ValueError("execution recovery SHA differs")
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    if (
        recovery.get("status") != "FROZEN_AFTER_PRELABEL_SPLIT_VIEW_FAILURE_FOR_SAME_OPENING"
        or recovery.get("same_opening_retry") is not True
        or recovery.get("scientific_protocol_changed") is not False
        or recovery.get("test_split_view_sha256") != "84acf74ce3debc3e176c996fd04b75510daf84183060d5d8353fc17758705a46"
    ):
        raise ValueError("execution recovery authority differs")
    source = original_path.read_text(encoding="utf-8")
    original_sha = sha256_file(original_path)
    if original_sha != "805ad46bf5f1c49b377c4b6f56e7c5143fcdac6b4f68a53f20da0ccd40f0dde3":
        raise ValueError("pinned evaluator SHA differs")
    old = '''    base = Sabdab2StructureEmbeddingDataset(
        args.structure_root, args.embedding_root, None,
        split_role="test", split_path=args.split,
        split_sha256=protocol["data"]["split_sha256"],
    )'''
    new = '''    test_split_path = args.structure_root / "TEST_SPLIT_VIEW.csv"
    structure_authority = json.loads(structure_manifest.read_text(encoding="utf-8"))["authority"]
    test_split_sha256 = structure_authority["test_split_view_sha256"]
    require(test_split_path, test_split_sha256)
    if test_split_sha256 != execution_recovery["test_split_view_sha256"]:
        raise ValueError("test split execution-recovery binding differs")
    base = Sabdab2StructureEmbeddingDataset(
        args.structure_root, args.embedding_root, None,
        split_role="test", split_path=test_split_path,
        split_sha256=test_split_sha256,
    )'''
    if source.count(old) != 1:
        raise RuntimeError("pinned evaluator split-view patch anchor differs")
    source = source.replace(old, new)
    audit_anchor = '            "cumulative_test_access_ordinal": 3,'
    if source.count(audit_anchor) != 1:
        raise RuntimeError("pinned evaluator result audit anchor differs")
    source = source.replace(
        audit_anchor,
        audit_anchor + '\n            "split_view_execution_recovery_protocol_sha256": execution_recovery_sha,',
    )
    receipt_anchor = '''        "selection_performed_on_test": False, "fresh_blind_test": False,
        "cumulative_test_access_ordinal": 3,'''
    if source.count(receipt_anchor) != 1:
        raise RuntimeError("pinned evaluator receipt anchor differs")
    source = source.replace(
        receipt_anchor,
        receipt_anchor + '\n        "split_view_execution_recovery_protocol_sha256": execution_recovery_sha,',
    )
    module_name = "abf30b_exposed_test_split_view_recovery_component"
    module = types.ModuleType(module_name)
    module.__file__ = str(original_path)
    module.__package__ = None
    module.__dict__["execution_recovery"] = recovery
    module.__dict__["execution_recovery_sha"] = recovery_sha
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(original_path), "exec"), module.__dict__)
        module.main()
    finally:
        sys.modules.pop(module_name, None)


if __name__ == "__main__":
    main()
