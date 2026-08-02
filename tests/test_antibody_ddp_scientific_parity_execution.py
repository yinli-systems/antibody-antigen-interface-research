"""Validate the frozen nine-run DDP scientific-parity execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ROOT / "protocols/antibody_ddp_scientific_parity_execution_v2_1.json"


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_execution_binds_all_inputs_and_the_counterbalanced_run_order() -> None:
    """Reject drift in code, split membership, protocols, or seed order."""
    payload = json.loads(EXECUTION.read_text(encoding="ascii"))
    assert payload["format_version"] == "antibody_ddp_scientific_parity_execution_v2_1"
    assert payload["allocation"] == {
        "gpu_count": 8,
        "hardware": "NVIDIA GeForce RTX 5090",
        "node_count": 1,
        "same_slurm_job_for_all_runs": True,
    }
    assert payload["run_order"] == [
        "seed17_safe_eager",
        "seed17_fp32_compiled",
        "seed17_bf16_selected",
        "seed42_bf16_selected",
        "seed42_safe_eager",
        "seed42_fp32_compiled",
        "seed73_fp32_compiled",
        "seed73_bf16_selected",
        "seed73_safe_eager",
    ]
    inputs = payload["inputs"]
    expected_files = {
        "gate_protocol_sha256": "protocols/antibody_ddp_scientific_parity_gate_v2.json",
        "efficiency_protocol_sha256": "protocols/antibody_efficiency_gate_v1.json",
        "communication_protocol_sha256": "protocols/antibody_ddp_communication_optimization_v1.json",
        "model_family_sha256": "protocols/antibody_model_family_v1.json",
        "training_protocol_sha256": "protocols/antibody_training_v1.json",
    }
    for key, relative in expected_files.items():
        assert inputs[key] == _sha256(relative)
    for name, digest in inputs["split_files"].items():
        assert digest == _sha256(
            f"results/pretraining/antibody_ddp_parity_split_v2/{name}"
        )
    assert inputs["split_receipt_file_sha256"] == _sha256(
        "results/pretraining/antibody_ddp_parity_split_v2/receipt.json"
    )
    assert payload["implementation"]["run_antibody_pretraining_optimized.py"] == _sha256(
        "scripts/run_antibody_pretraining_optimized.py"
    )


def test_execution_remains_non_authorizing_before_results() -> None:
    """Keep scale and scientific claims closed until the gate is evaluated."""
    payload = json.loads(EXECUTION.read_text(encoding="ascii"))
    assert not any(payload["authorization"].values())
    assert payload["execution"]["updates_per_run"] == 500
    assert payload["execution"]["checkpoint"] == "disabled_for_gate"
    assert payload["test_access"] == {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }
