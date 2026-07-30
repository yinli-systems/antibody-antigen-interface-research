"""Tests for the V6-D controller, freezer inventory, and launcher."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from interaction_first.v6_d_two_cp_core_registration import SOURCE_LABELS


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts" / "freeze_v6_d_two_cp_core_registration.py"
CONTROLLER = ROOT / "scripts" / "run_v6_d_two_cp_core_audit.py"
LAUNCHER = ROOT / "scripts" / "submit_v6_d_two_cp_core_audit.sbatch"


def _load(path: Path, name: str) -> object:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_v6_d_source_inventory_is_complete_and_content_addressed() -> None:
    module = _load(FREEZER, "v6_d_freezer_fixture")
    inventory = module._source_inventory(ROOT)
    assert set(inventory) == SOURCE_LABELS
    for reference in inventory.values():
        path = ROOT / reference["path"]
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]


def test_v6_d_controller_rejects_noncanonical_registration() -> None:
    module = _load(CONTROLLER, "v6_d_controller_fixture")
    for payload in (b'{"a":1,"a":1}\n', b'{ "a": 1 }\n'):
        try:
            module._strict_registration(payload)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed registration was accepted")


def test_v6_d_launcher_is_hash_bound_and_hides_gpu() -> None:
    payload = LAUNCHER.read_bytes()
    assert payload.isascii()
    text = payload.decode("ascii")
    for required in (
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --time=24:00:00",
        "export CUDA_VISIBLE_DEVICES=",
        "export OMP_NUM_THREADS=1",
        "export PYTHONHASHSEED=0",
        "V6D_REGISTRATION_SHA256",
        "V6D_CONTROLLER_SHA256",
        'sha256sum "$CONTROLLER"',
        'sha256sum "$REGISTRATION"',
    ):
        assert required in text
    assert "testdata" not in text.lower()
    assert "--mem" not in text
