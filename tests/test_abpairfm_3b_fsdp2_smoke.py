"""Focused non-CUDA checks for the 3B FSDP2 systems runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_abpairfm_3b_fsdp2_smoke.py"


def _module():
    specification = importlib.util.spec_from_file_location("abpairfm_fsdp2_smoke", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_synthetic_pair_has_two_chains_and_targets() -> None:
    module = _module()
    tokens, chain_ids, labels, residues = module._synthetic_pair(
        rank=2,
        sequence_tokens=64,
        seed=17,
        device=torch.device("cpu"),
    )
    assert tokens.shape == chain_ids.shape == labels.shape == (1, 64)
    assert chain_ids[0, :32].eq(0).all()
    assert chain_ids[0, 32:].eq(1).all()
    assert labels[0, :32].ne(-100).any()
    assert labels[0, 32:].ne(-100).any()
    assert residues == 60


def test_synthetic_pair_microbatch_counts_every_residue() -> None:
    module = _module()
    tokens, chain_ids, labels, residues = module._synthetic_pair(
        rank=0,
        sequence_tokens=64,
        microbatch_pairs=3,
        seed=17,
        device=torch.device("cpu"),
    )
    assert tokens.shape == chain_ids.shape == labels.shape == (3, 64)
    for batch_index in range(3):
        assert labels[batch_index, :32].ne(-100).any()
        assert labels[batch_index, 32:].ne(-100).any()
    assert residues == 3 * 60


def test_microbatch_argument_is_bounded() -> None:
    module = _module()
    arguments = type(
        "Arguments",
        (),
        {
            "expected_world_size": 4,
            "sequence_tokens": 64,
            "microbatch_pairs": 0,
            "warmup_steps": 0,
            "measured_steps": 1,
            "seed": 17,
            "shard_group_size": 0,
        },
    )()
    with pytest.raises(ValueError, match="microbatch-pairs"):
        module._validate(arguments)


def test_systems_receipt_is_finite_and_create_only(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "receipt.json"
    payload = {"status": "PASS", "throughput": 1.25}
    module._publish_create_only(output, payload)
    assert json.loads(output.read_text()) == payload
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module._publish_create_only(output, payload)


def test_reshard_argument_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--checkpoint",
            "checkpoint.pt",
            "--expected-size",
            "1",
            "--expected-sha256",
            "0" * 64,
            "--output",
            "receipt.json",
            "--expected-world-size",
            "4",
            "--no-reshard-after-forward",
        ],
    )
    assert module._arguments().reshard_after_forward is False


def test_selective_activation_checkpoint_policy_can_be_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--checkpoint",
            "checkpoint.pt",
            "--expected-size",
            "1",
            "--expected-sha256",
            "0" * 64,
            "--output",
            "receipt.json",
            "--expected-world-size",
            "8",
            "--activation-checkpoint-policy",
            "selective_attention",
        ],
    )
    assert module._arguments().activation_checkpoint_policy == "selective_attention"


def test_hsdp_mesh_shape_is_explicit_and_divisible() -> None:
    module = _module()
    assert module._mesh_shape(4, 0) is None
    assert module._mesh_shape(4, 4) is None
    assert module._mesh_shape(8, 4) == (2, 4)
    with pytest.raises(ValueError, match="divide"):
        module._mesh_shape(8, 3)
    with pytest.raises(ValueError, match=r"\[1, world-size\]"):
        module._mesh_shape(8, 9)
