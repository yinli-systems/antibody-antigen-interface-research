"""Validate the preregistered FSDP2 phase-two engineering settings."""

from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict[str, object]:
    """Load the machine-readable protocol without mutating it."""
    return json.loads(
        (
            REPOSITORY_ROOT
            / "protocols/frontier_fsdp2_dcp_engineering_v1.json"
        ).read_text(encoding="utf-8")
    )


def test_bf16_tolerances_are_frozen_before_execution() -> None:
    """Require explicit BF16 tolerances instead of post-result tuning."""
    numerical = _protocol()["numerical_gate"]
    assert isinstance(numerical, dict)
    assert numerical["BF16_tolerances_frozen_before_execution"] is True
    assert numerical["BF16_absolute_tolerance"] == 0.005
    assert numerical["BF16_relative_tolerance"] == 0.02


def test_profile_budget_is_frozen_at_fifty_plus_five_hundred_steps() -> None:
    """Keep initialization out of the authoritative throughput window."""
    scale = _protocol()["scale_gate"]
    assert isinstance(scale, dict)
    assert scale["profile_warmup_steps"] == 50
    assert scale["profile_measured_steps"] == 500


def test_launcher_receipts_and_forwards_numerical_tolerances() -> None:
    """Bind the numerical gate to both the environment and Python process."""
    launcher = (
        REPOSITORY_ROOT / "slurm/frontier_fsdp2_dcp_smoke.sbatch"
    ).read_text(encoding="utf-8")
    for name in (
        "FRONTIER_ABSOLUTE_TOLERANCE",
        "FRONTIER_RELATIVE_TOLERANCE",
        "--absolute-tolerance",
        "--relative-tolerance",
        "absolute_tolerance=$ABSOLUTE_TOLERANCE",
        "relative_tolerance=$RELATIVE_TOLERANCE",
    ):
        assert name in launcher
    assert "#SBATCH --mem=" not in launcher


def test_cross_world_replay_is_bidirectional_and_fixed_batch() -> None:
    """Require both topology directions under identical global examples."""
    replay = _protocol()["cross_world_replay"]
    assert isinstance(replay, dict)
    assert replay["directions"] == ["one_to_eight", "eight_to_one"]
    assert replay["fixed_global_batch_size"] == 8
    assert replay["full_checkpoint_state_must_be_exact"] is True
    assert replay["source_result_requires_external_sha256"] is True


def test_cross_world_launcher_is_public_partition_compatible() -> None:
    """Keep public GPU memory policy and source-result anchoring explicit."""
    launcher = (
        REPOSITORY_ROOT / "slurm/frontier_fsdp2_reshard.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --mem=" not in launcher
    assert "FRONTIER_SOURCE_SAVE_RESULT_SHA256" in launcher
    assert "--expected-save-result-sha256" in launcher
    assert "synthetic_fsdp2_cross_world_reshard" in launcher


def test_profile_launcher_locks_the_authoritative_window() -> None:
    """Require the registered warmup and measured step counts in the launcher."""
    launcher = (
        REPOSITORY_ROOT / "slurm/frontier_fsdp2_profile.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --mem=" not in launcher
    assert "--warmup-steps 50" in launcher
    assert "--measured-steps 500" in launcher
    assert "synthetic_fsdp2_warmed_profile" in launcher


def test_multinode_scaling_contract_is_fixed_batch_and_gated() -> None:
    """Freeze the 8-to-32-to-64 strong-scaling comparison before execution."""
    scaling = _protocol()["strong_scaling_gate"]
    assert isinstance(scaling, dict)
    assert scaling["baseline_world_size"] == 8
    assert scaling["fixed_global_batch_size"] == 64
    assert scaling["local_batch_by_world_size"] == {"8": 8, "32": 2, "64": 1}
    assert scaling["minimum_efficiency"] == {"32": 0.8, "64": 0.7}
    assert len(scaling["prerequisites"]) == 3


def test_multinode_launcher_uses_torchrun_rendezvous_and_public_policy() -> None:
    """Require explicit node ranks and no forbidden memory override."""
    launcher = (
        REPOSITORY_ROOT / "slurm/frontier_fsdp2_multinode_profile.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --mem=" not in launcher
    assert "--node-rank=\"$SLURM_PROCID\"" in launcher
    assert "--rdzv-endpoint=\"$MASTER_ADDR:$MASTER_PORT\"" in launcher
    assert "synthetic_fsdp2_multinode_strong_scaling_profile" in launcher
