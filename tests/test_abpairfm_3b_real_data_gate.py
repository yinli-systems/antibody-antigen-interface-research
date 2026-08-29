"""Non-CUDA checks for the ABPairFM 3B real-data gate."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_abpairfm_3b_real_data_gate_v2.py"


def _module():
    specification = importlib.util.spec_from_file_location("real_data_gate", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_pair_crop_is_deterministic_and_preserves_both_chains() -> None:
    module = _module()
    first = module._crop_pair("A" * 900, "C" * 600, maximum_residues=1020, seed=17)
    second = module._crop_pair("A" * 900, "C" * 600, maximum_residues=1020, seed=17)
    assert first == second
    assert len(first[0]) + len(first[1]) == 1020
    assert min(map(len, first)) >= 20


def test_real_microbatch_masks_both_chains_and_pads() -> None:
    module = _module()
    records = [
        {"sequence1": "A" * 60, "sequence2": "C" * 40},
        {"sequence1": "G" * 30, "sequence2": "V" * 50},
    ]
    tokens, chain_ids, labels, residues, swapped = module._build_real_microbatch(
        records,
        [0, 1],
        maximum_tokens=128,
        seed=1701,
    )
    assert tokens.shape == chain_ids.shape == labels.shape == (2, 128)
    assert residues == 180
    assert 0 <= swapped <= 2
    for batch_index in range(2):
        assert labels[batch_index, chain_ids[batch_index].eq(0)].ne(-100).any()
        assert labels[batch_index, chain_ids[batch_index].eq(1)].ne(-100).any()
        assert labels[batch_index, chain_ids[batch_index].eq(-1)].eq(-100).all()
        assert tokens[batch_index, chain_ids[batch_index].eq(-1)].eq(1).all()


def test_real_microbatch_can_shorten_padding_without_changing_targets() -> None:
    module = _module()
    records = [
        {"sequence1": "A" * 60, "sequence2": "C" * 40},
        {"sequence1": "G" * 30, "sequence2": "V" * 50},
    ]
    full = module._build_real_microbatch(
        records, [0, 1], maximum_tokens=128, seed=1701
    )
    short = module._build_real_microbatch(
        records, [0, 1], maximum_tokens=128, padded_tokens=112, seed=1701
    )
    assert short[0].shape == short[1].shape == short[2].shape == (2, 112)
    for full_tensor, short_tensor in zip(full[:3], short[:3], strict=True):
        torch.testing.assert_close(short_tensor, full_tensor[:, :112])
    assert short[3:] == full[3:]


def test_length_bucket_schedule_is_complete_deterministic_and_shape_safe() -> None:
    module = _module()
    records = [
        {
            "sequence1": "A" * (24 + index * 3),
            "sequence2": "C" * (20 + index * 5),
        }
        for index in range(16)
    ]
    first = module._build_length_bucket_schedule(
        records,
        global_microbatch_size=4,
        maximum_tokens=256,
        seed=1701,
        token_boundaries=(64, 128, 192, 256),
        query_boundaries=(32, 64, 128, 192, 256),
    )
    second = module._build_length_bucket_schedule(
        records,
        global_microbatch_size=4,
        maximum_tokens=256,
        seed=1701,
        token_boundaries=(64, 128, 192, 256),
        query_boundaries=(32, 64, 128, 192, 256),
    )
    assert first == second
    assert sorted(index for plan in first for index in plan.indices) == list(range(16))
    for plan in first:
        for index in plan.indices:
            token_count, query_count, _ = module._pair_shape(
                records[index], index, maximum_tokens=256, seed=1701
            )
            assert token_count <= plan.padded_tokens
            assert query_count <= plan.query_capacity


def test_seed_is_stable_and_changes_across_fields() -> None:
    module = _module()
    assert module._seed(1701, 1, "mask") == module._seed(1701, 1, "mask")
    assert module._seed(1701, 1, "mask") != module._seed(1701, 1, "crop")


def test_microbatch_seven_recovery_preserves_frozen_optimizer_gate() -> None:
    module = _module()
    module._validate(
        Namespace(
            accumulation_steps=2,
            expected_world_size=8,
            learning_rate=1e-6,
            maximum_tokens=1024,
            microbatch_pairs=7,
            progress_interval=10,
            seed=1701,
            shard_group_size=4,
            updates=200,
            weight_decay=0.01,
        )
    )
