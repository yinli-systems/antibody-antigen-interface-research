"""Tests for the same-node FSDP/HSDP scaling promotion receipt."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/summarize_abpairfm_3b_scaling_ab.py"


def _module():
    specification = importlib.util.spec_from_file_location("scaling_ab", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _receipt(*, world_size: int, throughput: float, mesh: list[int]) -> dict:
    return {
        "architecture": {"parameter_count": 2_839_005_633},
        "format_version": "abpairfm_3b_fsdp2_systems_smoke_v1",
        "fsdp2": {
            "mesh_shape_replicate_by_shard": mesh,
            "reshard_after_forward": False,
        },
        "initialization": {"checkpoint_sha256": "a" * 64},
        "precision": "bfloat16",
        "ranks": [
            {
                "hostname": "node-a",
                "max_memory_allocated_bytes": 15_000,
                "total_memory_bytes": 25_000,
            }
            for _ in range(world_size)
        ],
        "sequence_tokens_per_rank": 1024,
        "status": "PASS",
        "synthetic_data": True,
        "test_access": {
            "asep_test_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
        "throughput": {
            "measured_step_count": 5,
            "residue_tokens_per_second_mean": throughput,
            "step_seconds_max_by_step": [2.0, 2.02, 2.01, 2.03, 2.04],
            "warmup_step_count": 2,
        },
        "torch": {"cuda_version": "12.8", "version": "2.7.1"},
        "world_size": world_size,
    }


def test_promotes_only_when_all_frozen_gates_pass() -> None:
    module = _module()
    baseline = _receipt(world_size=4, throughput=1_300.0, mesh=[1, 4])
    candidate = _receipt(world_size=8, throughput=3_300.0, mesh=[2, 4])
    result = module.summarize(
        baseline,
        candidate,
        minimum_speedup=1.4,
        maximum_p95_to_median=1.2,
        maximum_memory_fraction=0.92,
    )
    assert result["decision"] == "PROMOTE_HSDP_2X4"
    assert result["gates"] == {
        "memory_fraction": True,
        "step_time_stability": True,
        "throughput_speedup": True,
    }
    assert result["comparison"]["eight_over_four_gpu_speedup"] == 3_300 / 1_300


def test_records_rejection_without_hiding_valid_measurement() -> None:
    module = _module()
    baseline = _receipt(world_size=4, throughput=1_300.0, mesh=[1, 4])
    candidate = _receipt(world_size=8, throughput=1_500.0, mesh=[2, 4])
    result = module.summarize(
        baseline,
        candidate,
        minimum_speedup=1.4,
        maximum_p95_to_median=1.2,
        maximum_memory_fraction=0.92,
    )
    assert result["status"] == "PASS"
    assert result["decision"] == "REJECT_HSDP_2X4"
    assert result["gates"]["throughput_speedup"] is False
