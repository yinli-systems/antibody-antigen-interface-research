"""Contract checks for the checkpoint RNG-stashing systems gate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "abpairfm_3b_checkpoint_rng_gate_v1.json"


def test_checkpoint_rng_gate_is_multigpu_and_fail_closed() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    frozen = protocol["frozen_comparison"]
    gates = protocol["promotion_gates"]
    assert frozen["world_size"] == 8
    assert frozen["microbatch_pairs_per_rank"] == 7
    assert frozen["fsdp2"]["mesh_shape_replicate_by_shard"] == [2, 4]
    assert frozen["model_internal_random_operations"] == []
    assert gates["maximum_peak_allocated_memory_fraction"] == 0.92
    assert gates["minimum_residue_tokens_per_second_speedup"] == 1.02
    assert protocol["authorization"]["production_pretraining"] is False
    assert protocol["authorization"]["real_data_gate"] is False
    assert protocol["authorization"]["test_access"] is False
    assert protocol["runtime_boundaries"]["hp_4090_permitted"] is False
    assert protocol["runtime_boundaries"]["external_dependency_install_permitted"] is False
