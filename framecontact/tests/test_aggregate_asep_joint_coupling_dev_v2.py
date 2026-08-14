"""Tests for the development-only joint-coupling v2 aggregate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/aggregate_asep_joint_coupling_dev_v2.py"
PROTOCOL = ROOT / "protocols/framecontact_asep_joint_coupling_dev_v2.json"
PROTOCOL_RECEIPT = (
    ROOT / "protocols/FRAMECONTACT_ASEP_JOINT_COUPLING_DEV_V2_RECEIPT.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module():
    spec = importlib.util.spec_from_file_location("joint_v2_aggregate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(value: float) -> dict[str, object]:
    return {
        "auprc": value,
        "mean_per_complex_auprc": value + 0.001,
        "per_complex_auprc_by_instance": {
            "case_a": value,
            "case_b": value + 0.002,
        },
        "auroc": 0.8 + value,
        "precision": value + 0.01,
        "recall": value + 0.02,
        "f1": value + 0.015,
        "mcc": value + 0.005,
    }


def _selected(value: float) -> dict[str, object]:
    directions = {
        direction: {
            "cluster_weighted_cognate_compatibility_margin": value - 0.02,
            "top1_retrieval_accuracy": 0.5 + value,
            "mean_reciprocal_rank": 0.6 + value,
        }
        for direction in (
            "fixed_antibody_wrong_antigen",
            "fixed_antigen_wrong_antibody",
        )
    }
    return {
        "contact": _metrics(value),
        "epitope": _metrics(value + 0.1),
        "paratope": _metrics(value + 0.2),
        "partner_retrieval_by_direction": directions,
    }


def _write_matrix(root: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_sha = _sha(PROTOCOL)
    receipt_sha = _sha(PROTOCOL_RECEIPT)
    for condition_index, condition in enumerate(protocol["development_matrix"]):
        for seed_index, seed in enumerate(protocol["model_selection"]["seeds"]):
            run = root / condition / f"seed-{seed}"
            run.mkdir(parents=True)
            value = 0.01 * condition_index + 0.001 * seed_index
            selected = _selected(value)
            history = []
            for epoch in (1, 2):
                epoch_selected = _selected(value + 0.0001 * epoch)
                history.append(
                    {
                        "epoch": epoch,
                        "validation": epoch_selected,
                    }
                )
            checkpoint_sha = hashlib.sha256(
                f"{condition}:{seed}".encode()
            ).hexdigest()
            result = {
                "format_version": "framecontact_development_training_v1",
                "experiment": condition,
                "configuration": {"seed": seed},
                "authority": {
                    "dataset_backend": "asep",
                    "asep_cache_manifest_sha256": "a" * 64,
                    "training_partner_panel_sha256": "b" * 64,
                    "validation_partner_panel_sha256": "c" * 64,
                    "protocol_file_sha256": protocol_sha,
                    "protocol_receipt_sha256": receipt_sha,
                    "test_loaded": False,
                    "external_test_loaded": False,
                },
                "selection": {"best_epoch": 2},
                "model": {"selected_state_sha256": checkpoint_sha},
                "selected_checkpoint_validation": selected,
                "history": history,
            }
            result_path = run / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            receipt = {
                "result_sha256": _sha(result_path),
                "selected_checkpoint_sha256": checkpoint_sha,
                "seed": seed,
                "selected_epoch": 2,
                "test_loaded": False,
                "external_test_loaded": False,
            }
            (run / "RUN_RECEIPT.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )


def test_paired_seed_bootstrap_is_deterministic_and_explicitly_seed_level() -> None:
    module = _module()
    result = module._paired_seed_bootstrap(
        [0.0] * 5,
        [0.1] * 5,
        resamples=1000,
        seed=7,
    )
    assert result["right_minus_left_mean"] == 0.1
    assert result["confidence_interval_95"] == [0.1, 0.1]
    assert result["resampling_unit"] == "seed"
    assert "not a per-complex bootstrap" in result["caution"]


def test_paired_complex_bootstrap_is_deterministic_and_uses_complexes() -> None:
    module = _module()
    left = [
        {
            "seed": seed,
            "per_complex_auprc": {
                "contact": {"case_a": 0.1, "case_b": 0.2},
            },
        }
        for seed in (17, 42, 73, 101, 137)
    ]
    right = [
        {
            "seed": seed,
            "per_complex_auprc": {
                "contact": {"case_a": 0.11, "case_b": 0.21},
            },
        }
        for seed in (17, 42, 73, 101, 137)
    ]
    result = module._paired_complex_bootstrap(
        left,
        right,
        group="contact",
        resamples=1000,
        seed=7,
    )
    assert abs(result["right_minus_left_mean"] - 0.01) < 1e-12
    assert result["validation_complex_count"] == 2
    assert (
        result["resampling_unit"]
        == "validation_complex_after_mean_paired_delta_across_seeds"
    )


def test_aggregate_authenticates_all_30_development_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_matrix(runs)
    output = tmp_path / "aggregate.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runs-root",
            str(runs),
            "--protocol",
            str(PROTOCOL),
            "--protocol-receipt",
            str(PROTOCOL_RECEIPT),
            "--output",
            str(output),
            "--bootstrap-resamples",
            "1000",
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert list(payload["conditions"]) == protocol["development_matrix"]
    assert len(payload["paired_seed_comparisons"]) == 5
    comparison = payload["paired_seed_comparisons"][
        "framecontact_joint_a4_coupled_soft_crop__to__framecontact_joint_a5_graph_spatial"
    ]
    assert abs(comparison["contact_auprc"]["right_minus_left_mean"] - 0.01) < 1e-12
    complex_comparison = comparison["contact_per_complex_auprc"]
    assert abs(complex_comparison["right_minus_left_mean"] - 0.01) < 1e-12
    assert complex_comparison["validation_complex_count"] == 2
    assert (
        complex_comparison["resampling_unit"]
        == "validation_complex_after_mean_paired_delta_across_seeds"
    )
    assert payload["matched_validation_trajectory"]["matched_epochs"] == [1, 2]
    receipt = json.loads(
        output.with_name(output.name + ".receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["run_count"] == 30
    assert receipt["official_test_loaded"] is False
    assert receipt["frozen_test_loaded"] is False
    assert receipt["external_test_loaded"] is False


def test_aggregate_fails_closed_if_any_run_reports_test_access(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_matrix(runs)
    receipt_path = (
        runs
        / "framecontact_joint_a0_pair_only"
        / "seed-17"
        / "RUN_RECEIPT.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["test_loaded"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "aggregate.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runs-root",
            str(runs),
            "--protocol",
            str(PROTOCOL),
            "--protocol-receipt",
            str(PROTOCOL_RECEIPT),
            "--output",
            str(output),
            "--bootstrap-resamples",
            "1000",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "Run receipt failed authentication" in completed.stderr
    assert not output.exists()


def test_aggregate_fails_closed_if_per_complex_axis_differs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_matrix(runs)
    result_path = (
        runs
        / "framecontact_joint_a5_graph_spatial"
        / "seed-17"
        / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["selected_checkpoint_validation"]["contact"][
        "per_complex_auprc_by_instance"
    ] = {"case_a": 0.05, "case_c": 0.052}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    receipt_path = result_path.with_name("RUN_RECEIPT.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["result_sha256"] = _sha(result_path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "aggregate.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runs-root",
            str(runs),
            "--protocol",
            str(PROTOCOL),
            "--protocol-receipt",
            str(PROTOCOL_RECEIPT),
            "--output",
            str(output),
            "--bootstrap-resamples",
            "1000",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "Paired complex bootstrap instance axes differ" in completed.stderr
    assert not output.exists()
