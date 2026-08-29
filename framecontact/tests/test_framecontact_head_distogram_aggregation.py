"""End-to-end receipt and promotion-gate tests for the head/distogram matrix."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "aggregate_framecontact_head_distogram_multiseed.py"
    )
    spec = importlib.util.spec_from_file_location(
        "aggregate_framecontact_head_distogram_test_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _metric_block(value: float) -> dict:
    fixed = {
        "precision": value,
        "recall": value,
        "f1": value,
        "mcc": value,
        "predicted_positive_ratio": value,
    }
    return {
        "auprc": value,
        "mean_per_complex_auprc": value,
        "brier": 1.0 - value,
        "ece_15": 1.0 - value,
        "threshold_selected_on_validation": value,
        "precision": value,
        "recall": value,
        "f1": value,
        "mcc": value,
        "predicted_positive_ratio": value,
        "fixed_raw_logit_threshold_zero": fixed,
    }


def _validation(value: float, recall: float) -> dict:
    contact = _metric_block(value)
    contact["fixed_raw_logit_threshold_zero"]["recall"] = recall
    contact["ranked_pair_macro"] = {
        "lmin_over_5": {"mean_precision": value, "mean_recall": recall}
    }
    return {
        "selection_metric": value,
        "contact": contact,
        "paratope": _metric_block(value),
        "epitope": _metric_block(value),
    }


def _write_protocol(module, path: Path, value: dict) -> tuple[Path, str]:
    _write_json(path, value)
    digest = module.sha256_file(path)
    receipt = path.with_name(f"{path.stem}-receipt.json")
    _write_json(receipt, {"protocol_file_sha256": digest})
    return receipt, digest


def _write_run(
    module,
    *,
    root: Path,
    condition: str,
    seed: int,
    protocol_sha256: str,
    pair_geometry_sha256: str | None,
    value: float,
    recall: float,
) -> None:
    run = root / condition / f"seed-{seed}"
    run.mkdir(parents=True)
    checkpoint = run / "FRAMECONTACT_SELECTED.pt"
    checkpoint.write_bytes(f"{condition}:{seed}".encode())
    authority = {
        "dataset_backend": "sabdab2",
        "structure_manifest_sha256": "structure",
        "contact_manifest_sha256": "contact",
        "embedding_manifest_sha256": "embedding",
        "training_partner_panel_sha256": "train-panel",
        "validation_partner_panel_sha256": "validation-panel",
        "split_manifest_sha256": "split",
        "pair_geometry_manifest_sha256": pair_geometry_sha256,
        "protocol_file_sha256": protocol_sha256,
        "test_loaded": False,
        "external_test_loaded": False,
    }
    result = {
        "format_version": module.RUN_FORMAT,
        "experiment": condition,
        "configuration": {"seed": seed},
        "authority": authority,
        "selection": {"best_epoch": 3},
        "history": [
            {
                "epoch": epoch,
                "validation": _validation(value - 0.01 * (3 - epoch), recall),
            }
            for epoch in (1, 2, 3)
        ],
        "selected_checkpoint_validation": _validation(value, recall),
    }
    result_path = run / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    _write_json(result_path, result)
    _write_json(
        run / "RUN_RECEIPT.json",
        {
            "format_version": module.RUN_FORMAT,
            "seed": seed,
            "test_loaded": False,
            "external_test_loaded": False,
            "result_sha256": module.sha256_file(result_path),
            "selected_checkpoint_sha256": module.sha256_file(checkpoint),
        },
    )


def test_aggregate_authenticates_all_runs_and_applies_paired_gates(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    geometry_sha256 = "pair-geometry"
    binary_receipt, binary_sha256 = _write_protocol(
        module,
        tmp_path / "binary-protocol.json",
        {
            "development_matrix": [
                "framecontact_unary_only",
                "framecontact_contact_marginal",
                "framecontact_contact_hybrid",
            ]
        },
    )
    distogram_receipt, distogram_sha256 = _write_protocol(
        module,
        tmp_path / "distogram-protocol.json",
        {
            "development_matrix": [
                "framecontact_distogram",
                "framecontact_distogram_triangle",
            ],
            "evidence_boundary_at_freeze": {
                "current_sabdab2_test_is_confirmatory": False
            },
            "data": {"pair_geometry_manifest_sha256": geometry_sha256},
        },
    )
    binary_root = tmp_path / "binary"
    distogram_root = tmp_path / "distogram"
    for seed in module.SEEDS:
        _write_run(
            module,
            root=binary_root,
            condition="framecontact_contact_hybrid",
            seed=seed,
            protocol_sha256=binary_sha256,
            pair_geometry_sha256=None,
            value=0.20,
            recall=0.60,
        )
        _write_run(
            module,
            root=distogram_root,
            condition="framecontact_distogram",
            seed=seed,
            protocol_sha256=distogram_sha256,
            pair_geometry_sha256=geometry_sha256,
            value=0.30,
            recall=0.60,
        )
        _write_run(
            module,
            root=distogram_root,
            condition="framecontact_distogram_triangle",
            seed=seed,
            protocol_sha256=distogram_sha256,
            pair_geometry_sha256=geometry_sha256,
            value=0.40,
            recall=0.58,
        )
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(module.__file__),
            "--binary-run-root",
            str(binary_root),
            "--distogram-run-root",
            str(distogram_root),
            "--binary-protocol",
            str(tmp_path / "binary-protocol.json"),
            "--binary-protocol-receipt",
            str(binary_receipt),
            "--distogram-protocol",
            str(tmp_path / "distogram-protocol.json"),
            "--distogram-protocol-receipt",
            str(distogram_receipt),
            "--output",
            str(output),
        ],
    )
    module.main()
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["matrix_complete"] is True
    assert summary["promotion_gates"]["distogram_vs_binary_hybrid"]["passed"]
    assert summary["promotion_gates"]["triangle_vs_distogram"]["passed"]
    assert summary["recall_retention_ratio_triangle_vs_distogram"]["five_seed"][
        "mean"
    ] > 0.95
    trajectory = summary["conditions"]["framecontact_distogram"]["epoch_trajectory"]
    assert trajectory["last_complete_five_seed_epoch"] == 3
    assert trajectory["complete_five_seed_epochs"]["3"][
        "contact_fixed_raw_precision"
    ]["count"] == 5
