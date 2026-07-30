"""Validate the production-batch eight-GPU systems experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_V1 = ROOT / "protocols/antibody_production_batch_efficiency_v1.json"
PROTOCOL_V2 = ROOT / "protocols/antibody_production_batch_efficiency_v2.json"
PROTOCOL = ROOT / "protocols/antibody_production_batch_efficiency_v3.json"
EXTENSION = (
    ROOT / "protocols/antibody_ddp_scientific_parity_extension_600_seed17_v1.json"
)
SLURM = ROOT / "slurm/antibody_production_batch_efficiency.sbatch"


def test_protocol_uses_production_batching_with_safe_fp32_reduction() -> None:
    """Increase accumulation while excluding the failed BF16 communication hook."""
    value = json.loads(PROTOCOL.read_text(encoding="ascii"))
    assert value["format_version"] == "antibody_production_batch_efficiency_protocol_v3"
    assert value["parent_protocol_file_sha256"] == hashlib.sha256(
        PROTOCOL_V2.read_bytes()
    ).hexdigest()
    assert value["batch"] == {
        "batch_quantum_sequences": 512,
        "global_residue_token_target_per_update": 262_144,
        "micro_batch_per_gpu": 16,
        "updates_per_run": 600,
    }
    assert value["optimization"]["ddp_gradient_reduction_dtype"] == "float32"
    assert value["optimization"]["experimental_bf16_ddp_hook_allowed"] is False
    assert value["measurement"]["crossover_run_order"] == [
        "safe_eager_r1",
        "fp32_compiled_r1",
        "fp32_compiled_r2",
        "safe_eager_r2",
    ]
    assert value["authorization"]["production_pretraining"] is False
    assert value["test_access"]["sabdab2_heldout_accessed"] is False
    acceptance = value["acceptance"]
    assert acceptance["bootstrap_iterations"] == 10_000
    assert acceptance["maximum_masked_cross_entropy_relative_increase"] == 0.01
    assert acceptance["maximum_masked_accuracy_absolute_decrease"] == 0.005


def test_protocol_binds_every_source_and_evaluation_input() -> None:
    """Detect changes to the trainer, model, data panel, or systems semantics."""
    value = json.loads(PROTOCOL.read_text(encoding="ascii"))
    bindings = {
        "communication_protocol_sha256": (
            "protocols/antibody_ddp_communication_optimization_v1.json"
        ),
        "efficiency_protocol_sha256": "protocols/antibody_efficiency_gate_v1.json",
        "mfu_protocol_sha256": "protocols/antibody_foundation_scale_v2.json",
        "model_family_sha256": "protocols/antibody_model_family_v1.json",
        "parity_evaluation_indices_sha256": (
            "results/pretraining/antibody_ddp_parity_split_v2/"
            "evaluation_indices.npy"
        ),
        "parity_extension_protocol_sha256": (
            "protocols/antibody_ddp_scientific_parity_extension_600_seed17_v1.json"
        ),
        "parity_protocol_sha256": "protocols/antibody_ddp_scientific_parity_gate_v2.json",
        "parity_split_receipt_file_sha256": (
            "results/pretraining/antibody_ddp_parity_split_v2/receipt.json"
        ),
        "parity_training_indices_sha256": (
            "results/pretraining/antibody_ddp_parity_split_v2/training_indices.npy"
        ),
        "runner_sha256": "scripts/run_antibody_pretraining_optimized.py",
        "runner_wrapper_sha256": (
            "scripts/run_antibody_pretraining_parity_extension.py"
        ),
        "training_protocol_sha256": "protocols/antibody_training_v1.json",
    }
    for key, relative in bindings.items():
        assert value["inputs"][key] == hashlib.sha256(
            (ROOT / relative).read_bytes()
        ).hexdigest()


def test_extension_preserves_split_and_non_authorizing_boundary() -> None:
    """Extend only the systems run length while preserving the frozen panel."""
    value = json.loads(EXTENSION.read_text(encoding="ascii"))
    assert value["execution"] == {
        "gpu_count": 8,
        "parent_updates_per_run": 500,
        "run_seed": 17,
        "updates_per_run": 600,
    }
    assert value["parent_parity_protocol_sha256"] == hashlib.sha256(
        (ROOT / "protocols/antibody_ddp_scientific_parity_gate_v2.json").read_bytes()
    ).hexdigest()
    assert value["authorization"] == {
        "model_selection": False,
        "production_pretraining": False,
        "scientific_claim": False,
    }
    assert value["test_access"]["sabdab2_heldout_accessed"] is False


def test_protocol_history_preserves_both_fail_closed_attempts() -> None:
    """Keep each pre-update incompatibility as immutable parent evidence."""
    v2 = json.loads(PROTOCOL_V2.read_text(encoding="ascii"))
    v3 = json.loads(PROTOCOL.read_text(encoding="ascii"))
    assert v2["parent_protocol_file_sha256"] == hashlib.sha256(
        PROTOCOL_V1.read_bytes()
    ).hexdigest()
    assert v2["correction"]["failed_attempt"]["job_id"] == 1_342_810
    assert v3["correction"]["failed_attempt"]["job_id"] == 1_342_862
    assert v3["measurement"]["run_seed"] == 17


def test_slurm_launcher_is_single_node_eight_gpu_crossover() -> None:
    """Run four same-allocation measurements with the frozen safe candidate."""
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --gres=gpu:8" in source
    assert "--target-tokens 262144" in source
    assert "--batch-quantum 512" in source
    assert "--steps 600" in source
    assert "antibody_production_batch_efficiency_v3.json" in source
    assert "--parity-extension-protocol" in source
    assert source.count("--ddp-communication-dtype float32") == 1
    assert "--ddp-communication-dtype bfloat16" not in source
    assert "run_mode safe_eager_r1 safe_eager" in source
    assert "run_mode fp32_compiled_r2 fp32_compiled" in source
    assert "--skip-checkpoint" in source
