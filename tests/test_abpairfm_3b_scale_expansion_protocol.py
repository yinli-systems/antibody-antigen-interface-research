"""Contract checks for the prospective ABPairFM 3B scale-pool expansion."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "abpairfm_3b_scale_expansion_v2.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_v2_preserves_the_frozen_v1_training_budget() -> None:
    protocol = _protocol()
    qualification = protocol["stage_a_qualification"]
    assert qualification["unchanged_from_v1"] is True
    assert qualification["minimum_post_dedup_sequence_pairs"] == 2_500_000
    assert qualification["maximum_post_dedup_sequence_pairs"] == 25_000_000
    assert qualification["consumed_residue_tokens"] == 5_000_000_000
    assert protocol["supersession"]["v1_protocol_remains_frozen"] is True


def test_scale_pool_is_mint_calibrated_but_not_training_authorization() -> None:
    protocol = _protocol()
    mint = protocol["public_scale_anchors"]["mint"]
    targets = protocol["stage_a_scale_pool"]["eligible_pool_targets"]
    assert mint["raw_physical_interactions"] == 2_400_000_000
    assert mint["training_pair_count_after_validation_cluster_exclusion"] == 95_800_000
    assert targets["mint_equivalent_stretch_endpoint_cluster_pairs"] == 95_800_000
    assert protocol["authorization"]["scale_pool_materialization"] is True
    assert protocol["authorization"]["production_pretraining"] is False


def test_scale_pool_keeps_development_and_test_boundaries_closed() -> None:
    protocol = _protocol()
    controls = protocol["stage_a_scale_pool"]["leakage_controls"]
    assert controls["exclude_every_pair_touching_a_frozen_development_cluster"] is True
    assert set(controls["frozen_development_panels"]) == {
        "asep_frozen_development",
        "sabdab2_frozen_development",
    }
    assert controls["test_bytes_permitted"] is False
    assert not any(protocol["test_access"].values())
