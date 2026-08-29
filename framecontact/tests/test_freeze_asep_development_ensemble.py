"""Fail-closed contracts for freezing the AsEP five-seed ensemble."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "freeze_asep_development_ensemble.py"
    )
    spec = importlib.util.spec_from_file_location("freeze_asep_ensemble", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(root: Path, *, seed: int = 17, backend: str = "asep") -> None:
    module = _module()
    root.mkdir(parents=True)
    result = {
        "format_version": module.RESULT_FORMAT,
        "experiment": "framecontact",
        "authority": {
            "dataset_backend": backend,
            "structure_manifest_sha256": None,
            "contact_manifest_sha256": None,
            "split_manifest_sha256": None,
            "test_loaded": False,
            "external_test_loaded": False,
        },
        "configuration": {"seed": seed},
    }
    result_path = root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    checkpoint_path = root / "FRAMECONTACT_SELECTED.pt"
    torch.save(
        {
            "format_version": module.RESULT_FORMAT,
            "seed": seed,
            "experiment": "framecontact",
            "selected_epoch": 3,
            "model_state_dict": {},
        },
        checkpoint_path,
    )
    receipt = {
        "format_version": module.RESULT_FORMAT,
        "result_sha256": _sha256(result_path),
        "selected_checkpoint_sha256": _sha256(checkpoint_path),
        "seed": seed,
        "selected_epoch": 3,
        "test_loaded": False,
        "external_test_loaded": False,
    }
    (root / "RUN_RECEIPT.json").write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )


def test_authenticated_run_accepts_hash_bound_development_only_run(
    tmp_path: Path,
) -> None:
    module = _module()
    root = tmp_path / "run"
    _write_run(root)
    authenticated = module.authenticated_development_run(root)
    assert authenticated["checkpoint"]["seed"] == 17
    assert authenticated["result"]["authority"]["dataset_backend"] == "asep"


def test_authenticated_run_rejects_test_opening_and_wrong_backend(
    tmp_path: Path,
) -> None:
    module = _module()
    root = tmp_path / "run"
    _write_run(root, backend="sabdab2")
    with pytest.raises(ValueError, match="not AsEP-ensemble eligible"):
        module.authenticated_development_run(root)

    _write_run(tmp_path / "opened")
    receipt_path = tmp_path / "opened" / "RUN_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["test_loaded"] = True
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt failed"):
        module.authenticated_development_run(tmp_path / "opened")


def test_member_calibration_requires_positive_finite_validation_values() -> None:
    module = _module()
    valid = {
        "selected_checkpoint_validation": {
            "validation_temperature": 2.0,
            "paratope": {"validation_temperature": 3.0},
            "epitope": {"validation_temperature": 4.0},
        }
    }
    assert module.member_calibration(valid) == {
        "contact": 2.0,
        "paratope": 3.0,
        "epitope": 4.0,
    }
    valid["selected_checkpoint_validation"]["epitope"]["validation_temperature"] = 0
    with pytest.raises(ValueError, match="finite and positive"):
        module.member_calibration(valid)
