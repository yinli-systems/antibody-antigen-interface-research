"""Fast checks for the frozen 3B Stage-A0 production campaign."""

from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_protocol_authorizes_only_stage_a0_and_seals_test() -> None:
    value = json.loads(
        (ROOT / "protocols/abpairfm_3b_stage_a0_campaign_v1.json").read_text()
    )
    assert value["authorization"] == {
        "production_stage_a0_pair_mlm": True,
        "production_stage_a_full_objective": False,
        "test_access": False,
    }
    assert value["corpus"]["materialized_unique_pair_target"] == 2_500_000
    assert value["corpus"]["source"]["strict_development_gate"] == {
        "exclude_any_pair_touching_hit_cluster": True,
        "minimum_bidirectional_coverage": 0.8,
        "minimum_sequence_identity": 0.3,
    }
    assert all(value["prohibited"].values())


def test_stage_a0_schedule_warms_then_decays() -> None:
    runner = _module(
        "stage_a0_runner", ROOT / "scripts/run_abpairfm_3b_stage_a0_segment.py"
    )
    assert runner._learning_rate(1, warmup=100, target=1000, maximum=1e-6) == 1e-8
    assert runner._learning_rate(100, warmup=100, target=1000, maximum=1e-6) == 1e-6
    assert runner._learning_rate(1000, warmup=100, target=1000, maximum=1e-6) == 0.0


def test_successor_requires_complete_checkpoint_anchor() -> None:
    runner = _module(
        "stage_a0_runner_validate",
        ROOT / "scripts/run_abpairfm_3b_stage_a0_segment.py",
    )
    base = dict(
        accumulation_steps=2,
        expected_campaign_protocol_sha256="a" * 64,
        expected_checkpoint_sha256="b" * 64,
        expected_world_size=8,
        input_checkpoint_root=None,
        input_generation=None,
        expected_input_content_sha256=None,
        maximum_learning_rate=1e-6,
        maximum_tokens=1024,
        microbatch_pairs=7,
        progress_interval=10,
        seed=1701,
        shard_group_size=4,
        target_residue_tokens=5_000_000_000,
        updates=200,
        warmup_residue_tokens=100_000_000,
        weight_decay=0.01,
    )
    runner._validate(Namespace(**base, segment_index=0))
    with pytest.raises(ValueError, match="requires an authenticated checkpoint"):
        runner._validate(Namespace(**base, segment_index=1))
