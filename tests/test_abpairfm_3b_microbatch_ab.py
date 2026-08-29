"""Tests for the same-node ABPairFM 3B HSDP microbatch decision."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_abpairfm_3b_microbatch_ab.py"


def _module():
    specification = importlib.util.spec_from_file_location("microbatch_ab", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _receipt(*, microbatch: int, throughput: float, memory: int = 15_000) -> dict:
    return {
        "architecture": {"parameter_count": 2_839_005_633},
        "format_version": "abpairfm_3b_fsdp2_systems_smoke_v1",
        "fsdp2": {
            "mesh_shape_replicate_by_shard": [2, 4],
            "reshard_after_forward": False,
        },
        "initialization": {"checkpoint_sha256": "a" * 64},
        "microbatch_pairs_per_rank": microbatch,
        "precision": "bfloat16",
        "ranks": [
            {
                "hostname": "node-a",
                "max_memory_allocated_bytes": memory,
                "total_memory_bytes": 25_000,
            }
            for _ in range(8)
        ],
        "sequence_tokens_per_rank": 1024,
        "status": "PASS",
        "synthetic_data": True,
        "test_access": {
            "asep_test_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
        "throughput": {
            "measured_step_count": 20,
            "residue_tokens_per_second_mean": throughput,
            "step_seconds_max_by_step": [2.0] * 20,
            "warmup_step_count": 5,
        },
        "torch": {"cuda_version": "12.8", "version": "2.7.1"},
        "world_size": 8,
    }


def test_promotes_a_faster_stable_microbatch() -> None:
    module = _module()
    result = module.summarize(
        _receipt(microbatch=1, throughput=3_700.0),
        _receipt(microbatch=2, throughput=5_000.0),
        minimum_throughput_speedup=1.1,
        maximum_p95_to_median=1.2,
        maximum_memory_fraction=0.92,
    )
    assert result["decision"] == "PROMOTE_MICROBATCH_2"
    assert all(result["gates"].values())


def test_rejects_an_over_memory_candidate_without_hiding_measurement() -> None:
    module = _module()
    result = module.summarize(
        _receipt(microbatch=1, throughput=3_700.0),
        _receipt(microbatch=3, throughput=6_000.0, memory=24_000),
        minimum_throughput_speedup=1.1,
        maximum_p95_to_median=1.2,
        maximum_memory_fraction=0.92,
    )
    assert result["status"] == "PASS"
    assert result["decision"] == "REJECT_MICROBATCH_3"
    assert result["gates"]["memory_fraction"] is False
