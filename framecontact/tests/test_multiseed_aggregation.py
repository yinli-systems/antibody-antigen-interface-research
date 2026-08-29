"""End-to-end checks for strict multi-seed result aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


EXPERIMENTS = (
    "sequence_dense",
    "sequence_local",
    "spatial_topology",
    "distance",
    "distance_direction",
    "full_pose",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_multiseed_aggregator_validates_and_summarizes_complete_matrix(
    tmp_path: Path,
) -> None:
    seeds = (17, 42, 73)
    labels = np.asarray([1, 0, 0, 1, 0, 0], dtype=np.float32)
    offsets = np.asarray([0, 3, 6], dtype=np.int64)
    for seed_index, seed in enumerate(seeds):
        for experiment_index, experiment in enumerate(EXPERIMENTS):
            root = tmp_path / "runs" / f"seed_{seed}" / experiment
            root.mkdir(parents=True)
            probability_shift = 0.002 * seed_index + 0.01 * experiment_index
            probabilities = np.asarray(
                [0.75, 0.2, 0.1, 0.7, 0.3, 0.15], dtype=np.float32
            )
            probabilities = np.clip(probabilities + probability_shift, 0.0, 1.0)
            prediction_path = root / f"{experiment}_validation_predictions.npz"
            np.savez_compressed(
                prediction_path,
                instance_ids=np.asarray(["complex_a", "complex_b"], dtype=np.str_),
                paratope_offsets=offsets,
                paratope_labels=labels,
                paratope_probabilities=probabilities,
                epitope_offsets=offsets,
                epitope_labels=labels,
                epitope_probabilities=probabilities,
            )
            metric = 0.4 + 0.01 * experiment_index + 0.001 * seed_index
            axis_metrics = {
                "auprc": metric,
                "auroc": metric + 0.2,
                "mcc_at_0_5": metric - 0.1,
            }
            validation = {
                "macro_auprc": metric,
                "macro_auroc": metric + 0.2,
                "weighted_bce": 1.0 - metric,
                "paratope": axis_metrics,
                "epitope": axis_metrics,
            }
            result = {
                "format_version": "sabdab2_self_attention_factorized_ablation_v2",
                "configuration": {"seed": seed},
                "data": {
                    "structure_cache_manifest_sha256": "a" * 64,
                    "validation_instance_ids_sha256": "b" * 64,
                },
                "experiments": [
                    {
                        "experiment": experiment,
                        "experiment_spec": {
                            "geometry_feature_mode": "none",
                            "attention_graph_mode": "dense",
                        },
                        "parameter_count": 1234,
                        "initial_state_sha256": f"{seed:064x}",
                        "history": [{"validation": validation}],
                        "development_best_epoch": 1,
                        "validation_predictions": prediction_path.name,
                        "validation_predictions_sha256": _sha256(prediction_path),
                    }
                ],
            }
            (root / "SABDAB2_SELF_ATTENTION_ABLATION.json").write_text(
                json.dumps(result), encoding="utf-8"
            )

    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "aggregate_multiseed_ablation.py"
            ),
            "--run-root",
            str(tmp_path / "runs"),
            "--output",
            str(output),
            "--bootstrap-replicates",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["format_version"] == "sabdab2_self_attention_multiseed_summary_v1"
    assert summary["model_parameter_count"] == 1234
    assert summary["data"]["validation_complex_count"] == 2
    assert len(summary["paired_comparisons"]) == 6
    assert (tmp_path / "summary.json.receipt.json").is_file()
