"""CPU checks for the full-model DCP recovery launcher contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_abpairfm_3b_dcp_gate.py"


def _module():
    specification = importlib.util.spec_from_file_location("abpairfm_3b_dcp_gate", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_source_digest_is_stable_and_reviewed() -> None:
    module = _module()
    observed = module._source_sha256(ROOT)
    assert len(observed) == 64
    assert len(set(module.SOURCE_PATHS)) == len(module.SOURCE_PATHS)
    for relative in module.SOURCE_PATHS:
        assert (ROOT / relative).is_file()


def test_canonical_local_state_digest_detects_model_and_optimizer_changes() -> None:
    module = _module()
    torch.manual_seed(17)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(3, 4)).sum().backward()
    optimizer.step()
    first = module._canonical_local_state_digests(model, optimizer)
    assert first == module._canonical_local_state_digests(model, optimizer)
    with torch.no_grad():
        model.weight.add_(1.0)
    second = module._canonical_local_state_digests(model, optimizer)
    assert second["model"] != first["model"]
    assert second["optimizer"] == first["optimizer"]


def test_frozen_gate_rejects_smaller_or_non_four_gpu_run() -> None:
    module = _module()
    arguments = module._arguments(
        [
            "run",
            "--mode",
            "save",
            "--project-root",
            str(ROOT),
            "--checkpoint",
            "checkpoint.pt",
            "--expected-checkpoint-size",
            "1",
            "--expected-checkpoint-sha256",
            "0" * 64,
            "--expected-source-sha256",
            "1" * 64,
            "--protocol",
            "protocol.json",
            "--expected-protocol-sha256",
            "2" * 64,
            "--output-root",
            "output",
            "--expected-world-size",
            "8",
        ]
    )
    with pytest.raises(ValueError, match="exactly four"):
        module._validate_run(arguments)


def test_slurm_gate_uses_only_gpu_4090_and_fresh_processes() -> None:
    content = (ROOT / "scripts/submit_abpairfm_3b_dcp_gate.sbatch").read_text()
    assert "#SBATCH -p gpu_4090" in content
    assert "hp_4090" not in content
    assert content.count("torch.distributed.run") == 2
    assert "--mode save" in content
    assert "--mode restore" in content
    assert "--mem" not in content
    assert "set -euo pipefail" in content
    assert "trap freeze_evidence EXIT" in content
    assert "abpairfm_3b_dcp_recovery_gate_v3.json" in content
