"""Checks for high-confidence STRING endpoint receipt publication."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_string_v12_high_confidence_endpoints.py"


def _module():
    specification = importlib.util.spec_from_file_location("high_confidence_endpoints", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_endpoint_receipt_publisher_is_create_only(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "receipt.json"
    module.publish_create_only(output, {"status": "PASS"})
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.publish_create_only(output, {"status": "PASS"})
