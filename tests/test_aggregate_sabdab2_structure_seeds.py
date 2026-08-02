"""Tests for strict multi-seed aggregation of SAbDab2 ablations."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "aggregate_sabdab2_structure_seeds.py"
MODES = (
    "dense_sequence",
    "topology_only",
    "distance",
    "distance_direction",
    "full_pose",
)


def _run_payload(seed: int, offset: float) -> dict[str, object]:
    modes = []
    for index, mode in enumerate(MODES):
        value = offset + index / 100.0
        modes.append(
            {
                "mode": mode,
                "history": [
                    {
                        "validation": {
                            "macro_auprc": value,
                            "macro_auroc": value + 0.1,
                            "paratope": {"auprc": value + 0.2},
                            "epitope": {"auprc": value - 0.2},
                        }
                    }
                ],
            }
        )
    return {
        "format_version": "sabdab2_self_attention_geometry_ablation_v2",
        "claim_boundary": "Development only; frozen test sealed.",
        "data": {"structure_cache_manifest_sha256": "a" * 64},
        "configuration": {
            "seed": seed,
            "epochs": 1,
            "modes": list(MODES),
        },
        "runtime": {
            "device": "cuda",
            "torch": "2.7.1",
            "cuda": "12.8",
            "gpu": "NVIDIA A100-PCIE-40GB",
        },
        "model_parameter_count": 123,
        "modes": modes,
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_reports_fixed_final_epoch_mean_and_sample_std(tmp_path: Path) -> None:
    first = tmp_path / "seed_1.json"
    second = tmp_path / "seed_2.json"
    output = tmp_path / "aggregate.json"
    _write(first, _run_payload(1, 0.4))
    _write(second, _run_payload(2, 0.6))

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--inputs",
            str(first),
            str(second),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["seed_count"] == 2
    assert result["seeds"] == [1, 2]
    assert result["scientific_contract"]["mode_order"] == list(MODES)
    assert result["metrics"]["dense_sequence"]["macro_auprc"]["mean"] == 0.5
    assert result["metrics"]["full_pose"]["macro_auprc"]["values"] == [
        0.44,
        0.64,
    ]
    assert "No frozen test" in result["claim_boundary"]


def test_aggregate_rejects_duplicate_seed_values(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(first, _run_payload(1, 0.4))
    _write(second, _run_payload(1, 0.6))

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--inputs",
            str(first),
            str(second),
            "--output",
            str(tmp_path / "aggregate.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Seed values must be unique" in completed.stderr
