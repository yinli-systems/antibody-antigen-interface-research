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
