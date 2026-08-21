"""Tests for pinned public-corpus materialization contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/materialize_abpairfm_public_download.py"
PLAN = ROOT / "data_registry/download_plans/intact_2026_01_09_micluster.json"


def _module():
    specification = importlib.util.spec_from_file_location("public_download", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_intact_plan_is_download_only_and_internally_consistent() -> None:
    module = _module()
    plan = module._validate_plan(json.loads(PLAN.read_text(encoding="utf-8")))
    assert plan["authorization"]["stage_a_training"] is False
    assert plan["expected_selected_file_count"] == 3
    assert sum(item["size_bytes"] for item in plan["selected_files"]) == 897632080


def test_plan_rejects_traversal_and_summary_mismatch() -> None:
    module = _module()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["selected_files"][0]["path"] = "../escape"
    with pytest.raises(ValueError, match="flat file"):
        module._validate_plan(plan)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["expected_selected_bytes"] += 1
    with pytest.raises(ValueError, match="byte count"):
        module._validate_plan(plan)


def test_receipt_publish_is_create_only(tmp_path: Path) -> None:
    module = _module()
    receipt = tmp_path / "receipt.json"
    module._publish_create_only(receipt, {"status": "PASS"})
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module._publish_create_only(receipt, {"status": "PASS"})


def test_byte_ranges_are_exact_and_nonoverlapping() -> None:
    module = _module()
    ranges = module.byte_ranges(17, 4)
    assert ranges == [(0, 4), (5, 8), (9, 12), (13, 16)]
