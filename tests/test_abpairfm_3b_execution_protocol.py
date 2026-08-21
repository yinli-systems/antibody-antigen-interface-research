"""Lock the scale, SOTA anchors, and promotion boundaries for 3B execution."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/abpairfm_3b_execution_v1.json"
PINDER_LICENSE = ROOT / "data_registry/sources/pinder_2024_02/license_receipt.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_execution_protocol_keeps_billion_token_and_unique_pair_scale() -> None:
    protocol = _load(PROTOCOL)
    stage_a = protocol["data"]["stage_a"]
    stage_b = protocol["data"]["stage_b"]
    assert stage_a["consumed_residue_tokens"] == 5_000_000_000
    assert stage_a["minimum_post_dedup_sequence_pairs"] == 2_500_000
    assert stage_a["maximum_post_dedup_sequence_pairs"] == 25_000_000
    assert stage_a["pinder"]["structure_views_are_not_distinct_sequence_pairs"] is True
    assert stage_b["consumed_residue_tokens"] == 2_000_000_000
    assert stage_b["pair_deficit"] == 3_922_501
    assert stage_b["training_remains_blocked_below_minimum"] is True


def test_execution_protocol_freezes_efficiency_and_quality_gates() -> None:
    protocol = _load(PROTOCOL)
    systems = protocol["systems"]
    assert systems["partition"] == "gpu_4090"
    assert systems["four_gpu_candidate"]["reshard_after_forward"] is False
    assert systems["hsdp_2x4_promotion"]["minimum_speedup_over_four_gpu"] == 1.4
    assert systems["real_data_gate"]["finite_update_count"] == 200
    assert protocol["public_research_anchors"]["mint"]["training_pair_count"] == 95_800_000
    assert protocol["public_research_anchors"]["pplm"]["pair_count"] == 3_300_000
    assert protocol["validation"]["test_access_before_teacher_approval"] is False


def test_pinder_license_receipt_is_narrow_and_attribution_preserving() -> None:
    receipt = _load(PINDER_LICENSE)
    assert receipt["authorization"]["internal_academic_research_training"] is True
    assert receipt["authorization"]["raw_bundle_redistribution"] is False
    assert receipt["authorization"]["test_archive_access"] is False
    assert receipt["evidence"]["wwpdb"]["license"] == "CC0-1.0"
    assert receipt["evidence"]["alphafold_database"]["license"] == "CC-BY-4.0"
