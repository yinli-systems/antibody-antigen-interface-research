"""Test production-batch crossover audit helpers and fixed boundaries."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_antibody_production_batch_efficiency.py"


def _load_module():
    specification = importlib.util.spec_from_file_location("production_batch_audit", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _receipt() -> dict[str, object]:
    trace = []
    consumed = 0
    for update in range(1, 601):
        consumed += 262_144
        trace.append(
            {
                "consumed_residue_tokens": consumed,
                "gradient_norm": 1.0,
                "learning_rate": 1.0e-4,
                "loss": 2.0,
                "micro_steps": 16,
                "residue_tokens": 262_144,
                "seconds": 1.0,
                "selected_targets": 39_000,
                "sequence_count": 2_560,
                "update": update,
            }
        )
    return {"trace": trace}


def test_throughput_uses_production_window_and_dense_peak() -> None:
    """Exclude 100 warmup updates and report useful-token approximate MFU."""
    module = _load_module()
    summary = module._throughput(_receipt(), 100)
    assert summary["measured_updates"] == 500
    assert summary["nonpadding_residue_tokens"] == 500 * 262_144
    assert summary["residue_tokens_per_second"] == 262_144
    expected = (
        6
        * module.PARAMETER_COUNT
        * 262_144
        / (8 * module.PEAK_DENSE_BF16_TFLOPS * 1.0e12)
    )
    assert math.isclose(summary["approximate_6n_mfu"], expected)


def test_bootstrap_is_deterministic_and_paired() -> None:
    """Use PCG64DXSM over identical sample keys for noninferiority bounds."""
    module = _load_module()
    keys = np.arange(32, dtype=np.uint64)
    reference = (keys, np.full(32, 2.0), np.full(32, 0.5))
    candidate = (keys, np.full(32, 2.01), np.full(32, 0.499))
    first = module._bootstrap_noninferiority(
        reference,
        candidate,
        iterations=100,
        confidence=0.95,
        seed=17,
    )
    second = module._bootstrap_noninferiority(
        reference,
        candidate,
        iterations=100,
        confidence=0.95,
        seed=17,
    )
    assert first == second
    assert math.isclose(first["cross_entropy_relative_increase_upper"], 0.005)
    assert math.isclose(first["masked_accuracy_decrease_upper"], 0.001)


def test_optimization_contract_excludes_bf16_gradient_hook() -> None:
    """Require FP32 DDP reduction for both reference and compiled candidate."""
    module = _load_module()
    eager = module._expected_optimization("safe_eager")
    compiled = module._expected_optimization("fp32_compiled")
    assert eager["ddp_communication_dtype"] == "float32"
    assert compiled["ddp_communication_dtype"] == "float32"
    assert eager["ddp_batched_grad_copy"] is False
    assert compiled["ddp_batched_grad_copy"] is False
    assert compiled["compile_blocks"] is True


def test_rank_state_digest_requires_all_eight_replicas() -> None:
    """Reject a state receipt if any rank differs from the replicated digest."""
    module = _load_module()
    digest = "a" * 64
    receipt = {
        "final_model_rank_sha256": [digest] * 8,
        "final_model_sha256": digest,
        "initial_model_rank_sha256": [digest] * 8,
    }
    assert module._rank_state_digest(receipt, "final_model") == digest
    assert module._rank_state_digest(receipt, "initial_model") == digest
    receipt["final_model_rank_sha256"][-1] = "b" * 64
    with pytest.raises(module.ProductionBatchAuditError, match="parity differs"):
        module._rank_state_digest(receipt, "final_model")
