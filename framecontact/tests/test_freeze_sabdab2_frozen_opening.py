"""Synthetic end-to-end authority test for the SAbDab2 one-shot opening."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


def _module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / "freeze_sabdab2_frozen_test_opening.py"
    spec = importlib.util.spec_from_file_location("sabdab2_opening", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _runs(root: Path, experiment: str) -> tuple[list[Path], list[dict]]:
    from scripts.train_framecontact import RESULT_FORMAT

    roots = []
    members = []
    for seed in (17, 42, 73, 101, 137):
        run = root / f"run-{seed}"
        run.mkdir()
        result_path = run / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
        _json(
            result_path,
            {
                "format_version": RESULT_FORMAT,
                "experiment": experiment,
                "configuration": {"seed": seed},
                "authority": {
                    "test_loaded": False,
                    "external_test_loaded": False,
                },
            },
        )
        checkpoint_path = run / "FRAMECONTACT_SELECTED.pt"
        torch.save(
            {
                "format_version": RESULT_FORMAT,
                "seed": seed,
                "experiment": experiment,
                "specificity_score_source": "interaction_only",
                "model_state_dict": {},
            },
            checkpoint_path,
        )
        receipt = {
            "format_version": RESULT_FORMAT,
            "result_sha256": _sha(result_path),
            "selected_checkpoint_sha256": _sha(checkpoint_path),
            "test_loaded": False,
            "external_test_loaded": False,
        }
        _json(run / "RUN_RECEIPT.json", receipt)
        roots.append(run)
        members.append(
            {
                "seed": seed,
                "development_result_sha256": receipt["result_sha256"],
                "selected_checkpoint_sha256": receipt[
                    "selected_checkpoint_sha256"
                ],
            }
        )
    return roots, members


def _ensemble(
    root: Path,
    experiment: str,
    selection_metric: float,
    authority: dict,
) -> None:
    from interaction_first.sabdab2_frozen_test import ENSEMBLE_FORMAT, FROZEN_SEEDS

    root.mkdir()
    result_path = root / "FRAMECONTACT_DEVELOPMENT_ENSEMBLE.json"
    _json(
        result_path,
        {
            "format_version": ENSEMBLE_FORMAT,
            "experiment": experiment,
            "seeds": list(FROZEN_SEEDS),
            "ensemble_rule": {"member_weighting": "uniform 1/5"},
            "authority": authority,
            "validation": {
                "selection_metric": selection_metric,
                "selection_metric_rule": (
                    "minimum_direction_cluster_weighted_margin"
                ),
                "specificity_score_source": "interaction_only",
                "contact": {"mean_per_complex_auprc": 0.2},
            },
        },
    )
    _json(
        root / "ENSEMBLE_RECEIPT.json",
        {
            "format_version": ENSEMBLE_FORMAT,
            "result_sha256": _sha(result_path),
            "test_loaded": False,
            "external_test_loaded": False,
        },
    )


def test_freeze_opening_binds_selection_data_and_fixed_diagnostics(
    tmp_path: Path,
) -> None:
    module = _module()
    from interaction_first.reciprocal_partner_panel import (
        RECIPROCAL_PARTNER_PANEL_FORMAT,
    )
    from interaction_first.sabdab2_frozen_test import (
        ARCHITECTURE_MATRIX,
        FROZEN_REQUIRED_INTERVENTIONS,
        OPENING_FORMAT,
        authenticate_sabdab2_test_opening,
    )
    from interaction_first.same_antigen_challenge import (
        LABEL_FREE_SAME_ANTIGEN_CHALLENGE_FORMAT,
    )
    from scripts.freeze_framecontact_development_selection import freeze_selection

    manifests = {}
    for name in ("structure", "contact", "embedding"):
        path = tmp_path / f"{name}.json"
        _json(path, {"name": name})
        manifests[name] = (path, _sha(path))
    split = tmp_path / "antigen_fold.csv"
    identifiers = [f"complex-{index:02d}" for index in range(33)]
    with split.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("instance_id", "role"))
        writer.writeheader()
        writer.writerows(
            {"instance_id": identifier, "role": "test"}
            for identifier in identifiers
        )
    split_sha = _sha(split)

    protocol = tmp_path / "protocol.json"
    _json(
        protocol,
        {
            "schema_version": "framecontact_primary_protocol_v5",
            "development_matrix": list(ARCHITECTURE_MATRIX),
            "model_selection": {
                "seeds": [17, 42, 73, 101, 137],
                "temperature_and_thresholds": "validation only",
            },
        },
    )
    protocol_receipt = tmp_path / "protocol-receipt.json"
    _json(protocol_receipt, {"protocol_file_sha256": _sha(protocol)})
    protocol_receipt_sha = _sha(protocol_receipt)

    selected_experiment = "framecontact_interleaved"
    run_roots, members = _runs(tmp_path, selected_experiment)
    common_authority = {
        "structure_manifest_sha256": manifests["structure"][1],
        "contact_manifest_sha256": manifests["contact"][1],
        "embedding_manifest_sha256": manifests["embedding"][1],
        "split_manifest_sha256": split_sha,
        "validation_partner_panel_sha256": "d" * 64,
        "protocol_file_sha256": _sha(protocol),
        "protocol_receipt_sha256": protocol_receipt_sha,
        "test_loaded": False,
        "external_test_loaded": False,
        "members": members,
    }
    ensemble_roots = []
    for index, experiment in enumerate(ARCHITECTURE_MATRIX):
        root = tmp_path / f"ensemble-{experiment}"
        _ensemble(
            root,
            experiment,
            1.0 if experiment == selected_experiment else -float(index),
            common_authority,
        )
        ensemble_roots.append(root)
    selection_root = tmp_path / "selection"
    freeze_selection(
        ensemble_roots=ensemble_roots,
        protocol=protocol,
        protocol_receipt=protocol_receipt,
        protocol_receipt_sha256=protocol_receipt_sha,
        output_root=selection_root,
    )
    selection_receipt_sha = _sha(selection_root / "SELECTION_RECEIPT.json")

    panel = tmp_path / "test-panel.json"
    _json(
        panel,
        {
            "format_version": RECIPROCAL_PARTNER_PANEL_FORMAT,
            "role": "test",
            "panel_size_per_direction": 31,
            "authority": {"task_label_columns_read": False},
            "entries": [{"instance_id": identifier} for identifier in identifiers],
        },
    )
    same_antigen = tmp_path / "same-antigen.json"
    _json(
        same_antigen,
        {
            "format_version": LABEL_FREE_SAME_ANTIGEN_CHALLENGE_FORMAT,
            "role": "test",
            "authority": {
                "split_sha256": split_sha,
                "task_labels_read": False,
                "model_predictions_read": False,
                "same_antigen_rule": "exact antigen_sequence_sha256",
            },
            "entries": [
                {
                    "focal_instance_id": identifiers[0],
                    "alternates": [{"instance_id": identifiers[1]}],
                }
            ],
        },
    )
    state = tmp_path / "test-state"
    state.mkdir()
    output = tmp_path / "opening.json"
    result = module.freeze_opening(
        argparse.Namespace(
            selection_root=selection_root,
            selection_receipt_sha256=selection_receipt_sha,
            run_roots=run_roots,
            structure_manifest=manifests["structure"][0],
            structure_manifest_sha256=manifests["structure"][1],
            contact_manifest=manifests["contact"][0],
            contact_manifest_sha256=manifests["contact"][1],
            embedding_manifest=manifests["embedding"][0],
            embedding_manifest_sha256=manifests["embedding"][1],
            split_manifest=split,
            split_manifest_sha256=split_sha,
            test_partner_panel=panel,
            test_partner_panel_sha256=_sha(panel),
            same_antigen_challenge=same_antigen,
            same_antigen_challenge_sha256=_sha(same_antigen),
            panel_size=31,
            bootstrap_replicates=10_000,
            bootstrap_seed=20_260_730,
            coordinate_noise=(0.10, 0.25, 0.50),
            evaluator_source=Path(__file__).resolve().parents[1]
            / "scripts/evaluate_framecontact_sabdab2_ensemble_test.py",
            test_state_root=state,
            output=output,
        )
    )
    opening = authenticate_sabdab2_test_opening(output, result["sha256"])
    assert opening["format_version"] == OPENING_FORMAT
    assert opening["selected_experiment"] == selected_experiment
    assert opening["test_instance_count"] == len(identifiers)
    assert tuple(opening["required_interventions"]) == FROZEN_REQUIRED_INTERVENTIONS
    assert not (state / "SABDAB2_TEST_SCORE_CONSUMED.json").exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        module.freeze_opening(
            argparse.Namespace(output=output, panel_size=31)
        )


def test_opening_rejects_mutable_panel_size(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(ValueError, match="exactly 31"):
        module.freeze_opening(
            argparse.Namespace(output=tmp_path / "opening.json", panel_size=7)
        )
