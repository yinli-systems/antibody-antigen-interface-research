"""Tests for the V6-C controller and Slurm isolation launcher."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts" / "run_v6_c_family_core_audit.py"
LAUNCHER = ROOT / "scripts" / "submit_v6_c_family_core_audit.sbatch"


def _load_controller() -> object:
    specification = importlib.util.spec_from_file_location(
        "v6_c_controller_fixture",
        CONTROLLER,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_controller_rejects_duplicate_and_noncanonical_registration() -> None:
    module = _load_controller()
    try:
        module._strict_registration(b'{"a":1,"a":1}\n')
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate registration was accepted")
    try:
        module._strict_registration(b'{ "a": 1 }\n')
    except ValueError as error:
        assert "canonical" in str(error)
    else:
        raise AssertionError("noncanonical registration was accepted")


def test_controller_passes_only_authenticated_arguments(
    tmp_path: Path,
    monkeypatch: object,
    capsys: object,
) -> None:
    module = _load_controller()
    registration = {"frozen": True}
    payload = module.canonical_json_bytes(registration)
    registration_path = tmp_path / "registration.json"
    registration_path.write_bytes(payload)
    output = tmp_path / "output"
    observed = {}

    monkeypatch.setattr(
        module,
        "validate_family_core_registration",
        lambda value: value,
    )

    def execute(**keyword: object) -> dict[str, object]:
        observed.update(keyword)
        return {"terminal_status": "V6_C_FIXTURE"}

    monkeypatch.setattr(module, "execute_family_core_audit", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(CONTROLLER),
            "--registration",
            str(registration_path),
            "--registration-sha256",
            hashlib.sha256(payload).hexdigest(),
            "--controller-sha256",
            hashlib.sha256(CONTROLLER.read_bytes()).hexdigest(),
            "--output-directory",
            str(output),
        ],
    )
    assert module.main() == 0
    assert observed["registration"] == registration
    assert observed["output_directory"] == output
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["terminal_status"] == "V6_C_FIXTURE"
    assert stdout["split_gate0_authorized"] is False
    assert stdout["training_authorized"] is False
    assert stdout["test_access_authorized"] is False
    assert stdout["threshold_change_authorized"] is False


def test_launcher_hides_gpu_and_binds_all_content() -> None:
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
        "V6C_REGISTRATION_SHA256",
        "V6C_CONTROLLER_SHA256",
        "sha256sum \"$CONTROLLER\"",
        "sha256sum \"$REGISTRATION\"",
        "--controller-sha256 \"$CONTROLLER_SHA256\"",
    ):
        assert required in text
    assert "testdata" not in text.lower()
    assert "--mem" not in text
