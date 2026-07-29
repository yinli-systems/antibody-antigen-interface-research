"""Unit tests for the synthetic R2 FSDP2/DCP smoke launcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from frontier_fsdp2_dcp_smoke import (
    SOURCE_PATHS,
    _assert_difference,
    _source_digest,
    _state_difference,
    _synthetic_batch,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_source_digest_binds_only_existing_reviewed_files() -> None:
    observed = _source_digest(REPOSITORY_ROOT)
    assert len(observed) == 64
    assert len(set(SOURCE_PATHS)) == len(SOURCE_PATHS)
    for relative in SOURCE_PATHS:
        assert (REPOSITORY_ROOT / relative).is_file()


def test_synthetic_batch_is_stateless_and_rank_partitionable() -> None:
    full = _synthetic_batch(range(4), seed=101, device=torch.device("cpu"))
    first = _synthetic_batch(range(2), seed=101, device=torch.device("cpu"))
    second = _synthetic_batch(range(2, 4), seed=101, device=torch.device("cpu"))
    for name in full:
        observed = torch.cat((first[name], second[name]), dim=0)
        torch.testing.assert_close(observed, full[name], rtol=0.0, atol=0.0)
    assert full["antibody_features"].shape == (4, 7, 24)
    assert full["antigen_features"].shape == (4, 9, 32)
    assert full["contact_target"].sum().item() == 8


def test_state_difference_reports_and_gates_drift() -> None:
    expected = {"weight": torch.tensor([0.0, 2.0])}
    observed = {"weight": torch.tensor([1e-6, 2.001])}
    difference = _state_difference(
        observed,
        expected,
        absolute_tolerance=0.01,
        relative_tolerance=0.01,
    )
    assert difference["tensor_count"] == 1
    assert difference["element_count"] == 2
    assert difference["max_absolute"] == pytest.approx(0.001, abs=1e-7)
    assert difference["max_scaled_error"] < 1.0
    _assert_difference(
        difference,
        absolute_tolerance=0.01,
        relative_tolerance=0.01,
        label="fixture",
    )
    with pytest.raises(RuntimeError, match="exceeded parity tolerance"):
        _assert_difference(
            _state_difference(
                observed,
                expected,
                absolute_tolerance=1e-5,
                relative_tolerance=1e-5,
            ),
            absolute_tolerance=1e-5,
            relative_tolerance=1e-5,
            label="fixture",
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        _state_difference(
            {"weight": torch.tensor([float("nan")])},
            {"weight": torch.tensor([0.0])},
            absolute_tolerance=1e-5,
            relative_tolerance=1e-5,
        )


def test_fsdp2_slurm_launcher_is_fail_closed() -> None:
    content = (REPOSITORY_ROOT / "slurm/frontier_fsdp2_dcp_smoke.sbatch").read_text(
        encoding="utf-8"
    )
    required = (
        "set -euo pipefail",
        "umask 077",
        "FRONTIER_SOURCE_SHA256",
        "FRONTIER_RUNTIME_MANIFEST_SHA256",
        "--expected-world-size",
        "--expected-source-sha256",
        "--expected-launch-manifest-sha256",
        "--parallelism fsdp2",
        "--synthetic-only",
        "nvidia-smi topo -m",
        "chmod -R a-w",
    )
    for marker in required:
        assert marker in content
    assert "env |" not in content
    assert "grep '^NCCL_'" not in content


def test_machine_protocol_preserves_scientific_firewall() -> None:
    protocol = json.loads(
        (
            REPOSITORY_ROOT
            / "protocols/frontier_fsdp2_dcp_engineering_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["format"] == "frontier_fsdp2_dcp_engineering_protocol_v1"
    assert protocol["authorization"] == {
        "biological_training": False,
        "engineering_synthetic_only": True,
        "frozen_test_access": False,
        "model_selection": False,
        "outer_evaluation": False,
    }
    assert protocol["parallelism"]["strategy"] == "FSDP2"
    assert protocol["checkpoint"]["same_world_size_continuation_required"] is True
    assert protocol["scale_gate"]["thirty_two_GPU_authorized"] is False
