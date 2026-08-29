"""Tests for the frozen ABPairFM checkpoint RNG-stashing decision."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_abpairfm_3b_checkpoint_rng_ab.py"


def _module():
    specification = importlib.util.spec_from_file_location("checkpoint_rng_ab", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _receipt(
    *,
    preserve_rng_state: bool,
    throughput: float,
    loss_shift: float = 0.0,
    gradient_scale: float = 1.0,
    memory: int = 20_000,
) -> dict:
    return {
        "architecture": {
            "activation_checkpointing": True,
            "activation_checkpoint_policy": "full",
            "activation_checkpoint_preserve_rng_state": preserve_rng_state,
            "attention": "pair_aware_lifted_flash_sdpa",
            "embed_dimension": 2560,
            "head_count": 40,
            "layer_count": 36,
            "parameter_count": 2_839_005_633,
        },
        "format_version": "abpairfm_3b_fsdp2_systems_smoke_v1",
        "fsdp2": {
            "mesh_shape_replicate_by_shard": [2, 4],
            "reshard_after_forward": False,
        },
        "initialization": {"checkpoint_sha256": "a" * 64},
        "microbatch_pairs_per_rank": 7,
        "precision": "bfloat16",
        "ranks": [
            {
                "hostname": "node-a",
                "max_memory_allocated_bytes": memory,
                "rank": rank,
                "selective_checkpoint_dispatch": {
                    "flash_must_save": 0,
                    "prefer_recompute": 0,
                },
                "steps": [
                    {
                        "elapsed_seconds": 2.0,
                        "global_mean_loss": 2.5 + loss_shift,
                        "gradient_norm": 1.2 * gradient_scale,
                        "step": step,
                    }
                    for step in range(20)
                ],
                "total_memory_bytes": 25_000,
            }
            for rank in range(8)
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
        "torch": {
            "cuda_version": "12.8",
            "flash_sdpa_forced": True,
            "version": "2.7.1",
        },
        "world_size": 8,
    }


def _summarize(baseline: dict, candidate: dict) -> dict:
    return _module().summarize(
        baseline,
        candidate,
        minimum_throughput_speedup=1.02,
        maximum_p95_to_median=1.2,
        maximum_memory_fraction=0.92,
        maximum_loss_absolute_delta=5e-4,
        maximum_gradient_norm_relative_delta=1e-3,
    )


def test_promotes_a_faster_numerically_matching_candidate() -> None:
    result = _summarize(
        _receipt(preserve_rng_state=True, throughput=7_000.0),
        _receipt(preserve_rng_state=False, throughput=7_210.0),
    )
    assert result["decision"] == "PROMOTE_NO_RNG_STASH_TO_REAL_DATA_GATE"
    assert all(result["gates"].values())
    assert result["authorization"]["production_pretraining"] is False


def test_rejects_candidate_that_exceeds_memory_even_when_faster() -> None:
    result = _summarize(
        _receipt(preserve_rng_state=True, throughput=7_000.0),
        _receipt(preserve_rng_state=False, throughput=8_000.0, memory=24_000),
    )
    assert result["decision"] == "REJECT_NO_RNG_STASH"
    assert result["gates"]["memory_fraction"] is False


def test_rejects_candidate_with_numerical_drift() -> None:
    result = _summarize(
        _receipt(preserve_rng_state=True, throughput=7_000.0),
        _receipt(
            preserve_rng_state=False,
            throughput=7_500.0,
            loss_shift=0.01,
            gradient_scale=1.01,
        ),
    )
    assert result["decision"] == "REJECT_NO_RNG_STASH"
    assert result["gates"]["loss_parity"] is False
    assert result["gates"]["gradient_norm_parity"] is False
