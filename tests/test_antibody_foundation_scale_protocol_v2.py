"""Validate the quality-preserving foundation-model scale protocol."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/antibody_foundation_scale_v2.json"


def _load() -> dict[str, object]:
    """Load the frozen machine-readable scale protocol."""
    return json.loads(PROTOCOL.read_text(encoding="ascii"))


def test_protocol_uses_dense_bf16_peak_without_sparse_inflation() -> None:
    """Keep approximate MFU tied to the documented dense hardware peak."""
    value = _load()
    mfu = value["mfu"]
    assert mfu["per_gpu_dense_bf16_tflops"] == 209.5
    assert mfu["sparsity_assumed"] is False
    assert mfu["profiler_exact_claim"] is False
    assert mfu["estimate_name"] == "approximate_6n_mfu"


def test_protocol_excludes_failed_bf16_gradient_compression() -> None:
    """Retain BF16 compute while protecting the FP32 main-gradient path."""
    value = _load()
    precision = value["precision"]
    assert precision["compute"] == "bfloat16"
    assert precision["ddp_gradient_reduction"] == "float32"
    assert precision["experimental_bf16_ddp_hook_allowed"] is False


def test_protocol_requires_data_and_scale_gates_before_production() -> None:
    """Reject parameter-count or allocation-only promotion claims."""
    value = _load()
    assert value["authorization"]["production_pretraining"] is False
    assert value["data"]["complete_production_materialization_required"] is True
    ladder = value["scale_ladder"]
    assert [stage["world_size"] for stage in ladder] == [8, 16, 32, 64]
    assert [stage["minimum_efficiency"] for stage in ladder] == [1.0, 0.85, 0.8, 0.7]
    assert value["test_access"]["sabdab2_heldout_accessed"] is False
