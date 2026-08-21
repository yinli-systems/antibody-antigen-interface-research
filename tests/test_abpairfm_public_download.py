"""Tests for pinned public-corpus materialization contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/materialize_abpairfm_public_download.py"
PLAN = ROOT / "data_registry/download_plans/intact_2026_01_09_micluster.json"
STRING_PLAN = ROOT / "data_registry/download_plans/string_v12_physical_full_and_sequences.json"


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


def test_string_plan_pins_full_physical_evidence_and_sequences() -> None:
    module = _module()
    plan = module._validate_plan(json.loads(STRING_PLAN.read_text(encoding="utf-8")))
    assert plan["authorization"]["stage_a_training"] is False
    assert plan["license"] == "CC-BY-4.0"
    assert plan["expected_selected_file_count"] == 2
    assert plan["expected_selected_bytes"] == 28_531_049_373
    assert {item["path"] for item in plan["selected_files"]} == {
        "protein.physical.links.full.v12.0.txt.gz",
        "protein.sequences.v12.0.fa.gz",
    }


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


def test_if_range_etag_is_strongly_quoted() -> None:
    module = _module()
    assert module._if_range_value("abc-123") == '"abc-123"'
    assert module._if_range_value('"abc-123"') == '"abc-123"'
    assert module._if_range_value('W/"abc-123"') == 'W/"abc-123"'


def test_range_response_parser_selects_final_redirect_block(tmp_path: Path) -> None:
    module = _module()
    headers = tmp_path / "headers.txt"
    headers.write_text(
        "HTTP/1.1 302 Found\r\nLocation: https://example.test/file\r\n\r\n"
        "HTTP/1.1 206 Partial Content\r\n"
        "Content-Range: bytes 0-4/17\r\nETag: \"abc\"\r\n\r\n",
        encoding="iso-8859-1",
    )
    status, values = module._response_headers(headers)
    assert status == 206
    assert values["content-range"] == "bytes 0-4/17"
    assert values["etag"] == '"abc"'
