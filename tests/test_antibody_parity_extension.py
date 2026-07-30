"""Test the training-excluded 600-update parity extension wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/run_antibody_pretraining_parity_extension.py"
EXTENSION = (
    ROOT / "protocols/antibody_ddp_scientific_parity_extension_600_seed17_v1.json"
)
PARITY = ROOT / "protocols/antibody_ddp_scientific_parity_gate_v2.json"
SPLIT = ROOT / "results/pretraining/antibody_ddp_parity_split_v2"


def _load():
    specification = importlib.util.spec_from_file_location("parity_extension", WRAPPER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _base_arguments(output: Path, *, steps: int = 600) -> list[str]:
    return [
        "--parity-protocol",
        str(PARITY),
        "--parity-split-root",
        str(SPLIT),
        "--output",
        str(output),
        "--steps",
        str(steps),
        "--run-seed",
        "17",
        "--skip-checkpoint",
    ]


def test_extension_authenticates_parent_split_and_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept only the exact eight-GPU 600-update non-authorizing extension."""
    module = _load()
    monkeypatch.setenv("WORLD_SIZE", "8")
    value, digest, output = module._validate_extension(
        EXTENSION,
        _base_arguments(tmp_path / "run"),
    )
    assert value["execution"]["parent_updates_per_run"] == 500
    assert value["execution"]["run_seed"] == 17
    assert value["execution"]["updates_per_run"] == 600
    assert len(digest) == 64
    assert output == tmp_path / "run"
    assert value["authorization"]["production_pretraining"] is False


def test_extension_rejects_unregistered_step_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the failed-attempt lesson instead of silently changing steps."""
    module = _load()
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(module.ParityExtensionError, match="bindings differ"):
        module._validate_extension(
            EXTENSION,
            _base_arguments(tmp_path / "run", steps=599),
        )


def test_wrapper_import_does_not_load_torch_or_base_runner() -> None:
    """Keep protocol validation independently testable without the GPU runtime."""
    source = WRAPPER.read_text(encoding="ascii")
    prefix = source.split("def main()", maxsplit=1)[0]
    assert "import torch" not in prefix
    assert "import run_antibody_pretraining_optimized" not in prefix
