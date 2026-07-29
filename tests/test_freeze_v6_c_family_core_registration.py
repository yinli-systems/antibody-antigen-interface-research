"""Tests for the create-only V6-C registration freezer."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from interaction_first.v6_c_family_core_registration import SOURCE_LABELS


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts" / "freeze_v6_c_family_core_registration.py"


def _load_freezer() -> object:
    specification = importlib.util.spec_from_file_location(
        "v6_c_freezer_fixture",
        FREEZER,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_source_inventory_is_complete_and_content_addressed() -> None:
    module = _load_freezer()
    inventory = module._source_inventory(ROOT)
    assert set(inventory) == SOURCE_LABELS
    for label, reference in inventory.items():
        path = ROOT / reference["path"]
        assert path.is_file()
        assert not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            reference["sha256"]
        ), label


def test_receipt_reference_requires_valid_self_hash(
    tmp_path: Path,
) -> None:
    module = _load_freezer()
    unsigned = {
        "terminal_status": "PROVEN_INFEASIBLE",
        "training_authorized": False,
    }
    claimed = module.audit.sha256_payload(unsigned)
    receipt = {**unsigned, "receipt_sha256": claimed}
    path = tmp_path / "receipt.json"
    payload = module.audit.canonical_json_bytes(receipt)
    path.write_bytes(payload)
    reference = module._verified_receipt_reference(
        path,
        unsigned,
        "fixture",
    )
    assert reference["receipt_sha256"] == claimed
    path.write_bytes(
        module.audit.canonical_json_bytes(
            {**unsigned, "receipt_sha256": "a" * 64}
        )
    )
    with pytest.raises(ValueError, match="self hash"):
        module._verified_receipt_reference(
            path,
            unsigned,
            "fixture",
        )


def test_freezer_and_source_are_ascii() -> None:
    module = _load_freezer()
    assert FREEZER.read_bytes().isascii()
    assert set(module.SOURCE_PATHS) == SOURCE_LABELS
