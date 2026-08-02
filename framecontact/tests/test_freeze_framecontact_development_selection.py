"""Fail-closed tests for SAbDab2 validation-only architecture selection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/freeze_framecontact_development_selection.py"
    )
    spec = importlib.util.spec_from_file_location("sabdab2_selection", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(root: Path) -> tuple[Path, Path]:
    module = _module()
    protocol = root / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema_version": "framecontact_primary_protocol_v5",
                "development_matrix": list(module.ARCHITECTURE_MATRIX),
                "model_selection": {
                    "seeds": list(module.FROZEN_SEEDS),
                    "temperature_and_thresholds": "validation only",
                },
            }
        ),
        encoding="utf-8",
    )
    receipt = root / "protocol-receipt.json"
    receipt.write_text(
        json.dumps({"protocol_file_sha256": _sha(protocol)}), encoding="utf-8"
    )
    return protocol, receipt


def _ensemble(
    root: Path,
    experiment: str,
    selection: float,
    *,
    protocol_sha: str,
    receipt_sha: str,
) -> None:
    module = _module()
    root.mkdir()
    authority = {
        "structure_manifest_sha256": "a" * 64,
        "contact_manifest_sha256": "b" * 64,
        "embedding_manifest_sha256": "c" * 64,
        "split_manifest_sha256": "d" * 64,
        "validation_partner_panel_sha256": "e" * 64,
        "protocol_file_sha256": protocol_sha,
        "protocol_receipt_sha256": receipt_sha,
        "test_loaded": False,
        "external_test_loaded": False,
        "members": [{"seed": seed} for seed in module.FROZEN_SEEDS],
    }
    result = root / module.RESULT_NAME
    result.write_text(
        json.dumps(
            {
                "format_version": module.ENSEMBLE_FORMAT,
                "experiment": experiment,
                "seeds": list(module.FROZEN_SEEDS),
                "ensemble_rule": {"member_weighting": "uniform 1/5"},
                "authority": authority,
                "validation": {
                    "selection_metric": selection,
                    "selection_metric_rule": "minimum_direction_cluster_weighted_margin",
                    "specificity_score_source": "interaction_only",
                    "contact": {"mean_per_complex_auprc": 0.1 + selection},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / module.RECEIPT_NAME).write_text(
        json.dumps(
            {
                "format_version": module.ENSEMBLE_FORMAT,
                "result_sha256": _sha(result),
                "test_loaded": False,
                "external_test_loaded": False,
            }
        ),
        encoding="utf-8",
    )


def test_selection_requires_all_eight_and_never_reads_test(tmp_path: Path) -> None:
    module = _module()
    protocol, receipt = _protocol(tmp_path)
    roots = []
    for index, experiment in enumerate(module.ARCHITECTURE_MATRIX):
        root = tmp_path / experiment
        _ensemble(
            root,
            experiment,
            index / 100,
            protocol_sha=_sha(protocol),
            receipt_sha=_sha(receipt),
        )
        roots.append(root)
    output = tmp_path / "selection"
    module.freeze_selection(
        ensemble_roots=roots,
        protocol=protocol,
        protocol_receipt=receipt,
        protocol_receipt_sha256=_sha(receipt),
        output_root=output,
    )
    result = json.loads((output / "SABDAB2_DEVELOPMENT_SELECTION.json").read_text())
    assert result["selected_experiment"] == "framecontact_interleaved"
    assert len(result["all_candidates"]) == 8
    assert result["authority"]["test_loaded"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        module.freeze_selection(
            ensemble_roots=roots,
            protocol=protocol,
            protocol_receipt=receipt,
            protocol_receipt_sha256=_sha(receipt),
            output_root=output,
        )


def test_selection_rejects_an_opened_ensemble(tmp_path: Path) -> None:
    module = _module()
    protocol, receipt = _protocol(tmp_path)
    root = tmp_path / "sequence_dense"
    _ensemble(
        root,
        "sequence_dense",
        0.1,
        protocol_sha=_sha(protocol),
        receipt_sha=_sha(receipt),
    )
    ensemble_receipt = root / module.RECEIPT_NAME
    payload = json.loads(ensemble_receipt.read_text())
    payload["test_loaded"] = True
    ensemble_receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="ineligible"):
        module.authenticated_ensemble(root)
