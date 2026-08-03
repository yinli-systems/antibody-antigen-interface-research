from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest
import torch

from interaction_first.reciprocal_partner_panel import (
    RECIPROCAL_PARTNER_PANEL_FORMAT,
)
from interaction_first.sabdab2_structure import sha256_file


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS / "evaluate_framecontact_train_audit.py"


def _module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("framecontact_train_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _panel(path: Path, *, role: str = "train", label_free: bool = True) -> str:
    path.write_text(
        json.dumps(
            {
                "format_version": RECIPROCAL_PARTNER_PANEL_FORMAT,
                "role": role,
                "authority": {"task_label_columns_read": not label_free},
            }
        ),
        encoding="utf-8",
    )
    return sha256_file(path)


def test_training_panel_must_be_hash_bound_label_free_and_train(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "panel.json"
    digest = _panel(path)
    assert module._authenticated_training_panel(path, digest)["role"] == "train"

    with pytest.raises(ValueError, match="SHA-256"):
        module._authenticated_training_panel(path, "0" * 64)
    digest = _panel(path, role="validation")
    with pytest.raises(ValueError, match="label-free train authority"):
        module._authenticated_training_panel(path, digest)
    digest = _panel(path, label_free=False)
    with pytest.raises(ValueError, match="label-free train authority"):
        module._authenticated_training_panel(path, digest)


def test_train_audit_reuses_only_finite_validation_calibration() -> None:
    module = _module()
    development = {
        "selected_checkpoint_validation": {
            "validation_temperature": 2.0,
            "contact": {"threshold_selected_on_validation": 0.1},
            "paratope": {
                "validation_temperature": 3.0,
                "threshold_selected_on_validation": 0.2,
            },
            "epitope": {
                "validation_temperature": 4.0,
                "threshold_selected_on_validation": 0.3,
            },
        }
    }
    frozen = module._frozen_validation_calibration(development)
    assert frozen == {
        "temperature": 2.0,
        "paratope_temperature": 3.0,
        "epitope_temperature": 4.0,
        "contact_threshold": 0.1,
        "paratope_threshold": 0.2,
        "epitope_threshold": 0.3,
    }
    development["selected_checkpoint_validation"]["validation_temperature"] = math.inf
    with pytest.raises(ValueError, match="non-finite"):
        module._frozen_validation_calibration(development)


def test_development_run_for_train_audit_must_remain_test_unopened(
    tmp_path: Path,
) -> None:
    module = _module()
    result_path = tmp_path / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = tmp_path / "FRAMECONTACT_SELECTED.pt"
    receipt_path = tmp_path / "RUN_RECEIPT.json"
    result_path.write_text(json.dumps({"configuration": {}}), encoding="utf-8")
    torch.save({"model_state_dict": {}}, checkpoint_path)
    receipt = {
        "test_loaded": False,
        "external_test_loaded": False,
        "result_sha256": sha256_file(result_path),
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    development, checkpoint, observed = module._authenticated_run(tmp_path)
    assert development["configuration"] == {}
    assert checkpoint["model_state_dict"] == {}
    assert observed["test_loaded"] is False

    receipt["test_loaded"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="not eligible"):
        module._authenticated_run(tmp_path)
