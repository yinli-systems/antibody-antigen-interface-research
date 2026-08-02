"""Fail-closed tests for validation-only AsEP architecture selection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "freeze_asep_development_selection.py"
    )
    spec = importlib.util.spec_from_file_location("asep_selection", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ensemble(root: Path, experiment: str, selection: float) -> None:
    module = _module()
    root.mkdir(parents=True)
    result = {
        "format_version": module.ENSEMBLE_FORMAT,
        "experiment": experiment,
        "seeds": [17, 42, 73, 101, 137],
        "ensemble_rule": {"member_weighting": "uniform 1/5"},
        "authority": {
            "dataset_backend": "asep",
            "asep_cache_manifest_sha256": "a" * 64,
            "protocol_file_sha256": "b" * 64,
            "protocol_receipt_sha256": "c" * 64,
            "training_partner_panel_sha256": "d" * 64,
            "validation_partner_panel_sha256": "e" * 64,
            "test_loaded": False,
            "external_test_loaded": False,
            "members": [],
        },
        "validation": {
            "selection_metric": selection,
            "selection_metric_rule": "minimum_direction_cluster_weighted_margin",
            "specificity_score_source": "interaction_only",
            "contact": {"mean_per_complex_auprc": 0.1 + selection},
        },
    }
    result_path = root / module.RESULT_NAME
    result_path.write_text(json.dumps(result), encoding="utf-8")
    (root / module.RECEIPT_NAME).write_text(
        json.dumps(
            {
                "format_version": module.ENSEMBLE_FORMAT,
                "result_sha256": _sha256(result_path),
                "test_loaded": False,
                "external_test_loaded": False,
            }
        ),
        encoding="utf-8",
    )


def _write_protocol(root: Path, architectures: list[str]) -> tuple[Path, Path]:
    module = _module()
    protocol = root / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema_version": module.PROTOCOL_FORMAT,
                "architecture_matrix": architectures,
                "model_selection": {"test_role_forbidden": True},
            }
        ),
        encoding="utf-8",
    )
    receipt = root / "protocol_receipt.json"
    receipt.write_text(
        json.dumps({"protocol_file_sha256": _sha256(protocol)}), encoding="utf-8"
    )
    return protocol, receipt


def test_selection_requires_complete_matrix_and_uses_validation_only(tmp_path: Path) -> None:
    module = _module()
    architectures = [f"experiment_{index}" for index in range(8)]
    roots = []
    for index, experiment in enumerate(architectures):
        root = tmp_path / experiment
        _write_ensemble(root, experiment, selection=float(index) / 100)
        roots.append(root)
    protocol, protocol_receipt = _write_protocol(tmp_path, architectures)
    output = tmp_path / "selection"
    module.freeze_selection(
        ensemble_roots=roots,
        protocol=protocol,
        protocol_receipt=protocol_receipt,
        protocol_receipt_sha256=_sha256(protocol_receipt),
        output_root=output,
    )
    result = json.loads((output / "ASEP_DEVELOPMENT_SELECTION.json").read_text())
    assert result["selected_experiment"] == "experiment_7"
    assert result["authority"]["test_loaded"] is False
    assert len(result["all_candidates"]) == 8
    with pytest.raises(FileExistsError, match="overwrite"):
        module.freeze_selection(
            ensemble_roots=roots,
            protocol=protocol,
            protocol_receipt=protocol_receipt,
            protocol_receipt_sha256=_sha256(protocol_receipt),
            output_root=output,
        )


def test_selection_rejects_opened_or_incomplete_ensemble(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "bad"
    _write_ensemble(root, "bad", 0.1)
    receipt_path = root / module.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt["test_loaded"] = True
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="ineligible"):
        module.authenticated_ensemble(root)
