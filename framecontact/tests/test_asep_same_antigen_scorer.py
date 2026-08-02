"""Metrics contracts for the same-antigen partner-swap scorer."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluate_asep_same_antigen_challenge.py"
    )
    spec = importlib.util.spec_from_file_location("asep_same_antigen_scorer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_directional_assignment_requires_targeted_migration() -> None:
    module = _module()
    result = module.directional_assignment_metrics(
        np.asarray([0.9, 0.1, 0.2, 0.8]),
        np.asarray([0.1, 0.9, 0.8, 0.2]),
        np.asarray([1, 0, 0, 1]),
        np.asarray([0, 1, 1, 0]),
    )
    assert result["own_retention_margin"] > 0
    assert result["target_migration_margin"] > 0
    assert result["assignment_margin"] > 0
    assert result["directional_success"] is True


def test_nonzero_change_is_not_sufficient_without_correct_direction() -> None:
    module = _module()
    result = module.directional_assignment_metrics(
        np.asarray([0.9, 0.1, 0.2, 0.8]),
        np.asarray([0.8, 0.2, 0.1, 0.9]),
        np.asarray([1, 0, 0, 1]),
        np.asarray([0, 1, 1, 0]),
    )
    assert result["mean_absolute_prediction_shift"] > 0
    assert result["directional_success"] is False


def test_cluster_bootstrap_is_cluster_weighted_and_deterministic() -> None:
    module = _module()
    records = [
        {"antigen_cluster_id": "a", "assignment_margin": 1.0},
        {"antigen_cluster_id": "a", "assignment_margin": 3.0},
        {"antigen_cluster_id": "b", "assignment_margin": 0.0},
    ]
    first = module._cluster_bootstrap(records, replicates=1000, seed=17)
    second = module._cluster_bootstrap(records, replicates=1000, seed=17)
    assert first == second
    assert first["mean_cluster_assignment_margin"] == pytest.approx(1.0)
    assert first["cluster_count"] == 2


def test_directional_assignment_rejects_invalid_probability() -> None:
    module = _module()
    with pytest.raises(ValueError, match="probabilities are invalid"):
        module.directional_assignment_metrics(
            np.asarray([1.1, 0.0]),
            np.asarray([0.5, 0.5]),
            np.asarray([1, 0]),
            np.asarray([0, 1]),
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_development_run(root: Path, seed: int, module) -> dict[str, object]:
    root.mkdir(parents=True)
    result = {
        "format_version": module.RESULT_FORMAT,
        "experiment": "framecontact",
        "authority": {
            "dataset_backend": "asep",
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
            "selected_epoch": 2,
            "model_state_dict": {},
        },
        checkpoint_path,
    )
    receipt = {
        "format_version": module.RESULT_FORMAT,
        "seed": seed,
        "selected_epoch": 2,
        "result_sha256": _sha256(result_path),
        "selected_checkpoint_sha256": _sha256(checkpoint_path),
        "test_loaded": False,
        "external_test_loaded": False,
    }
    (root / "RUN_RECEIPT.json").write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )
    return receipt


def test_ensemble_authentication_binds_all_five_member_hashes(tmp_path: Path) -> None:
    module = _module()
    run_roots = []
    members = []
    for seed in module.FROZEN_SEEDS:
        root = tmp_path / f"seed-{seed}"
        receipt = _write_development_run(root, seed, module)
        run_roots.append(root)
        members.append(
            {
                "seed": seed,
                "development_result_sha256": receipt["result_sha256"],
                "selected_checkpoint_sha256": receipt[
                    "selected_checkpoint_sha256"
                ],
            }
        )
    ensemble_root = tmp_path / "ensemble"
    ensemble_root.mkdir()
    result = {
        "format_version": module.ENSEMBLE_FORMAT,
        "experiment": "framecontact",
        "seeds": list(module.FROZEN_SEEDS),
        "authority": {
            "dataset_backend": "asep",
            "test_loaded": False,
            "external_test_loaded": False,
            "members": members,
        },
    }
    result_path = ensemble_root / "FRAMECONTACT_ASEP_DEVELOPMENT_ENSEMBLE.json"
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    receipt = {
        "format_version": module.ENSEMBLE_FORMAT,
        "result_sha256": _sha256(result_path),
        "test_loaded": False,
        "external_test_loaded": False,
    }
    (ensemble_root / "ENSEMBLE_RECEIPT.json").write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )
    observed, _, authenticated = module._authenticated_ensemble(
        ensemble_root, run_roots
    )
    assert observed["authority"]["members"] == members
    assert [item["checkpoint"]["seed"] for item in authenticated] == list(
        module.FROZEN_SEEDS
    )

    observed["authority"]["members"][0]["selected_checkpoint_sha256"] = "0" * 64
    result_path.write_text(json.dumps(observed, sort_keys=True), encoding="utf-8")
    receipt["result_sha256"] = _sha256(result_path)
    (ensemble_root / "ENSEMBLE_RECEIPT.json").write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="member runs differ"):
        module._authenticated_ensemble(ensemble_root, run_roots)
