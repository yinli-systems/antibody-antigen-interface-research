"""Synthetic end-to-end authority test for the AsEP opening receipt."""

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
    path = scripts / "freeze_asep_official_test_opening.py"
    spec = importlib.util.spec_from_file_location("asep_opening", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _runs(root: Path) -> tuple[list[Path], list[dict]]:
    from scripts.train_framecontact import RESULT_FORMAT

    runs = []
    authorities = []
    for seed in (17, 42, 73, 101, 137):
        run = root / f"run-{seed}"
        run.mkdir()
        result = {
            "format_version": RESULT_FORMAT,
            "experiment": "framecontact_interleaved",
            "configuration": {"seed": seed},
            "authority": {
                "dataset_backend": "asep",
                "test_loaded": False,
                "external_test_loaded": False,
                "structure_manifest_sha256": None,
                "contact_manifest_sha256": None,
                "split_manifest_sha256": None,
            },
        }
        result_path = run / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
        _write_json(result_path, result)
        checkpoint_path = run / "FRAMECONTACT_SELECTED.pt"
        torch.save(
            {
                "format_version": RESULT_FORMAT,
                "seed": seed,
                "experiment": "framecontact_interleaved",
                "selected_epoch": 3,
                "specificity_score_source": "interaction_only",
                "model_state_dict": {},
            },
            checkpoint_path,
        )
        receipt = {
            "format_version": RESULT_FORMAT,
            "seed": seed,
            "selected_epoch": 3,
            "result_sha256": _sha(result_path),
            "selected_checkpoint_sha256": _sha(checkpoint_path),
            "test_loaded": False,
            "external_test_loaded": False,
        }
        _write_json(run / "RUN_RECEIPT.json", receipt)
        runs.append(run)
        authorities.append(
            {
                "seed": seed,
                "development_result_sha256": receipt["result_sha256"],
                "selected_checkpoint_sha256": receipt[
                    "selected_checkpoint_sha256"
                ],
            }
        )
    return runs, authorities


def _selection(root: Path, members: list[dict]) -> tuple[Path, str, str]:
    from interaction_first.asep_frozen_test import ENSEMBLE_FORMAT, SELECTION_FORMAT

    ensemble = root / "ensemble"
    ensemble.mkdir()
    ensemble_result = ensemble / "FRAMECONTACT_ASEP_DEVELOPMENT_ENSEMBLE.json"
    _write_json(
        ensemble_result,
        {
            "format_version": ENSEMBLE_FORMAT,
            "experiment": "framecontact_interleaved",
            "seeds": [17, 42, 73, 101, 137],
            "ensemble_rule": {"member_weighting": "uniform 1/5"},
            "authority": {
                "dataset_backend": "asep",
                "test_loaded": False,
                "external_test_loaded": False,
                "members": members,
            },
            "validation": {
                "selection_metric": 0.1,
                "selection_metric_rule": "minimum_direction_cluster_weighted_margin",
                "specificity_score_source": "interaction_only",
                "contact": {"mean_per_complex_auprc": 0.2},
            },
        },
    )
    ensemble_receipt = ensemble / "ENSEMBLE_RECEIPT.json"
    _write_json(
        ensemble_receipt,
        {
            "format_version": ENSEMBLE_FORMAT,
            "result_sha256": _sha(ensemble_result),
            "test_loaded": False,
            "external_test_loaded": False,
        },
    )
    selection = root / "selection"
    selection.mkdir()
    selection_result = selection / "ASEP_DEVELOPMENT_SELECTION.json"
    _write_json(
        selection_result,
        {
            "format_version": SELECTION_FORMAT,
            "selected_experiment": "framecontact_interleaved",
            "selection_rule": {
                "primary": "maximize minimum_direction_cluster_weighted_margin",
                "first_tie_break": "maximize mean_per_complex_contact_auprc",
                "test_role_forbidden": True,
            },
            "selected_ensemble": {
                "root": str(ensemble),
                "result_sha256": _sha(ensemble_result),
                "receipt_sha256": _sha(ensemble_receipt),
                "selection_metric": 0.1,
                "contact_mean_per_complex_auprc": 0.2,
            },
            "all_candidates": [
                {
                    "experiment": experiment,
                    "root": str(ensemble) if index == 0 else f"unused-{index}",
                    "result_sha256": (
                        _sha(ensemble_result) if index == 0 else f"{index}" * 64
                    ),
                    "receipt_sha256": (
                        _sha(ensemble_receipt) if index == 0 else f"{index}" * 64
                    ),
                    "selection_metric": 0.1 if index == 0 else -float(index),
                    "contact_mean_per_complex_auprc": 0.2 if index == 0 else 0.0,
                }
                for index, experiment in enumerate(
                    (
                        "framecontact_interleaved",
                        "sequence_dense",
                        "spatial_topology",
                        "distance",
                        "distance_direction",
                        "full_pose",
                        "frozen_frame_ipa",
                        "framecontact",
                    )
                )
            ],
            "authority": {"test_loaded": False, "external_test_loaded": False},
        },
    )
    selection_receipt = selection / "SELECTION_RECEIPT.json"
    _write_json(
        selection_receipt,
        {
            "format_version": SELECTION_FORMAT,
            "result_sha256": _sha(selection_result),
            "selected_experiment": "framecontact_interleaved",
            "selected_ensemble_result_sha256": _sha(ensemble_result),
            "test_loaded": False,
            "external_test_loaded": False,
        },
    )
    return selection, _sha(selection_receipt), _sha(ensemble_result)


def _panel(path: Path, identifiers: list[str], scope: str, label_receipt: str) -> str:
    entries = []
    for index, identifier in enumerate(identifiers):
        candidates = [value for value in identifiers if value != identifier][:31]
        entries.append(
            {
                "instance_id": identifier,
                "antigen_cluster_id": f"ag-{index}",
                "antibody_cluster_id": f"ab-{index}",
                "wrong_antigen_instance_ids": candidates,
                "wrong_antibody_instance_ids": list(reversed(candidates)),
            }
        )
    _write_json(
        path,
        {
            "format_version": "framecontact_reciprocal_partner_panel_v2",
            "role": "test",
            "panel_size_per_direction": 31,
            "authority": {
                "panel_scope": scope,
                "label_free_test_receipt_sha256": label_receipt,
                "task_label_columns_read": False,
                "official_test_graph_tensors_opened": False,
                "model_predictions_or_scores_read": False,
            },
            "entries": entries,
        },
    )
    return _sha(path)


def test_freeze_opening_binds_complete_prelabel_authority(tmp_path: Path) -> None:
    module = _module()
    from interaction_first.asep_authority import INNER_AUDIT_FORMAT, SPLIT_MANIFEST_FORMAT
    from interaction_first.asep_decontamination import DECONTAMINATION_FORMAT
    from interaction_first.asep_frozen_test import LABEL_FREE_TEST_FORMAT, OPENING_FORMAT

    runs, members = _runs(tmp_path)
    selection, selection_receipt_sha, ensemble_result_sha = _selection(
        tmp_path, members
    )
    identifiers = [f"{index:04x}_0P" for index in range(33)]
    clean_ids = identifiers[:32]

    split_root = tmp_path / "splits"
    split_root.mkdir()
    split_manifest = split_root / "epitope_group.csv"
    with split_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("dataset_index", "instance_id", "role")
        )
        writer.writeheader()
        writer.writerows(
            {
                "dataset_index": index,
                "instance_id": identifier,
                "role": "test",
            }
            for index, identifier in enumerate(identifiers)
        )
    split_receipt = split_root / "ASEP_SPLIT_MANIFEST_RECEIPT.json"
    _write_json(
        split_receipt,
        {
            "format_version": SPLIT_MANIFEST_FORMAT,
            "authority": {
                "task_labels_read": False,
                "model_predictions_or_scores_read": False,
            },
            "instance_count": len(identifiers),
            "manifests": {
                "epitope_group": {
                    "path": split_manifest.name,
                    "sha256": _sha(split_manifest),
                    "role_counts": {"train": 0, "val": 0, "test": len(identifiers)},
                }
            },
        },
    )

    inner = tmp_path / "inner.json"
    _write_json(
        inner,
        {
            "format_version": INNER_AUDIT_FORMAT,
            "authority": {
                "task_labels_read": False,
                "model_predictions_or_scores_read": False,
            },
            "checks": {"split_payload_opened": False},
            "graph_archive": {"sha256": "a" * 64},
            "structure_archive": {"sha256": "b" * 64},
        },
    )

    label_root = tmp_path / "label-free"
    label_root.mkdir()
    label_manifest = label_root / "ASEP_LABEL_FREE_TEST_METADATA.csv"
    with label_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("instance_id", "role"))
        writer.writeheader()
        writer.writerows(
            {"instance_id": identifier, "role": "test"}
            for identifier in identifiers
        )
    label_receipt = label_root / "ASEP_LABEL_FREE_TEST_METADATA_RECEIPT.json"
    _write_json(
        label_receipt,
        {
            "format_version": LABEL_FREE_TEST_FORMAT,
            "authority": {
                "inner_audit_sha256": _sha(inner),
                "split_receipt_sha256": _sha(split_receipt),
                "split_manifest_sha256": _sha(split_manifest),
                "selection_receipt_sha256": selection_receipt_sha,
                "selected_ensemble_result_sha256": ensemble_result_sha,
            },
            "manifest_sha256": _sha(label_manifest),
            "test_instance_count": len(identifiers),
            "label_free_test_feature_open_count": 1,
            "test_label_open_count": 0,
            "checks": {
                "official_test_graph_tensors_opened": False,
                "official_test_contact_or_residue_labels_read": False,
                "model_predictions_or_scores_read": False,
            },
        },
    )

    decon_root = tmp_path / "decontamination"
    decon_root.mkdir()
    decon_result = decon_root / "ASEP_SABDAB2_DECONTAMINATION.json"
    _write_json(
        decon_result,
        {
            "format_version": DECONTAMINATION_FORMAT,
            "policy": {
                "antigen_structural_rule": "reciprocal_min(query_tm,target_tm) >= 0.50"
            },
            "authority": {
                "label_free_test_receipt_sha256": _sha(label_receipt),
                "label_free_test_manifest_sha256": _sha(label_manifest),
                "test_graph_tensors_opened": False,
                "test_labels_read": False,
                "model_predictions_or_scores_read": False,
            },
            "counts": {
                "asep_test_instances": len(identifiers),
                "cross_structure_pairs": 3300,
                "decontaminated_primary_instances": len(clean_ids),
            },
            "decontaminated_primary_instance_ids": clean_ids,
            "records": [
                {
                    "instance_id": identifier,
                    "contaminated": identifier not in clean_ids,
                    "eligible_for_decontaminated_primary": identifier in clean_ids,
                    "flags": {
                        "exact_pdb_identity": identifier not in clean_ids,
                        "exact_antigen_pdb_seqres": False,
                        "exact_antibody_pdb_seqres": False,
                        "antigen_structural_similarity": False,
                    },
                }
                for identifier in identifiers
            ],
        },
    )
    decon_receipt = decon_root / "DECONTAMINATION_RECEIPT.json"
    _write_json(
        decon_receipt,
        {
            "format_version": DECONTAMINATION_FORMAT,
            "result_sha256": _sha(decon_result),
            "test_labels_read": False,
            "model_predictions_or_scores_read": False,
            "decontaminated_primary_instance_count": len(clean_ids),
        },
    )
    full_panel = tmp_path / "full-panel.json"
    clean_panel = tmp_path / "clean-panel.json"
    full_panel_sha = _panel(
        full_panel, identifiers, "full_official_test", _sha(label_receipt)
    )
    clean_panel_sha = _panel(
        clean_panel,
        clean_ids,
        "sabdab2_decontaminated_primary",
        _sha(label_receipt),
    )
    output = tmp_path / "opening.json"
    test_state_root = tmp_path / "test-state"
    test_state_root.mkdir()
    result = module.freeze_opening(
        argparse.Namespace(
            output=output,
            panel_size=31,
            bootstrap_replicates=10_000,
            bootstrap_seed=20_260_730,
            selection_root=selection,
            selection_receipt_sha256=selection_receipt_sha,
            run_roots=runs,
            inner_audit=inner,
            inner_audit_sha256=_sha(inner),
            split_root=split_root,
            split_receipt_sha256=_sha(split_receipt),
            split_method="epitope_group",
            label_free_test_root=label_root,
            label_free_test_receipt_sha256=_sha(label_receipt),
            decontamination_root=decon_root,
            decontamination_receipt_sha256=_sha(decon_receipt),
            full_test_panel=full_panel,
            full_test_panel_sha256=full_panel_sha,
            clean_test_panel=clean_panel,
            clean_test_panel_sha256=clean_panel_sha,
            evaluator_source=Path(__file__).resolve().parents[1]
            / "scripts/evaluate_asep_frozen_test.py",
            test_state_root=test_state_root,
        )
    )
    opening = json.loads(output.read_text())
    assert result["sha256"] == _sha(output)
    assert opening["format_version"] == OPENING_FORMAT
    assert opening["eligible_to_open"] is True
    assert opening["test_graph_or_label_open_count_before"] == 0
    assert opening["official_test_instance_count"] == 33
    assert opening["decontaminated_primary_instance_count"] == 32
    assert len(opening["members"]) == 5
    assert opening["test_state_root"] == str(test_state_root.resolve())
    assert opening["score_consumption_token_name"] == "ASEP_TEST_SCORE_CONSUMED.json"
    assert opening["evaluation_source_dependencies"][
        "scripts/evaluate_asep_frozen_test.py"
    ] == _sha(Path(__file__).resolve().parents[1] / "scripts/evaluate_asep_frozen_test.py")


def test_opening_rejects_mutable_confirmatory_defaults(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(ValueError, match="exactly 31"):
        module.freeze_opening(
            argparse.Namespace(
                output=tmp_path / "opening.json",
                panel_size=1,
                bootstrap_replicates=10_000,
                bootstrap_seed=20_260_730,
            )
        )
